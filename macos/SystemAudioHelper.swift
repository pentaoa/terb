import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

final class AudioCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "com.pentaoa.terb.audio-helper")
    private let output = FileHandle.standardOutput

    func start() async throws {
        if !CGPreflightScreenCaptureAccess() {
            let granted = CGRequestScreenCaptureAccess()
            guard granted else {
                fputs("permission-denied\n", stderr)
                throw ExitCode.permissionDenied
            }
        }

        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            fputs("no-display\n", stderr)
            throw ExitCode.noDisplay
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = 48_000
        configuration.channelCount = 2
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
        fputs("ready\n", stderr)
    }

    func stop() async {
        guard let stream else { return }
        try? await stream.stopCapture()
        self.stream = nil
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("stopped: \(error.localizedDescription)\n", stderr)
        Foundation.exit(4)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio, sampleBuffer.isValid else { return }
        let samples = extractStereoSamples(from: sampleBuffer)
        guard !samples.isEmpty else { return }

        samples.withUnsafeBytes { rawBuffer in
            if let base = rawBuffer.baseAddress {
                output.write(Data(bytes: base, count: rawBuffer.count))
            }
        }
    }

    private func extractStereoSamples(from sampleBuffer: CMSampleBuffer) -> [Float] {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            return []
        }

        let audioDescription = streamDescription.pointee
        let channelCount = max(Int(audioDescription.mChannelsPerFrame), 1)
        let bitsPerChannel = Int(audioDescription.mBitsPerChannel)
        let isFloat = audioDescription.mFormatFlags & kAudioFormatFlagIsFloat != 0
        let isSignedInteger = audioDescription.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0
        let isNonInterleaved = audioDescription.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        let bufferCount = isNonInterleaved ? channelCount : 1

        let audioBufferList = AudioBufferList.allocate(maximumBuffers: max(bufferCount, 1))
        defer {
            audioBufferList.unsafeMutablePointer.deallocate()
        }

        var blockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: audioBufferList.unsafeMutablePointer,
            bufferListSize: AudioBufferList.sizeInBytes(maximumBuffers: max(bufferCount, 1)),
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )

        guard status == noErr else { return [] }

        var channels: [[Float]] = []
        for audioBuffer in audioBufferList {
            guard let data = audioBuffer.mData else { continue }
            channels.append(convert(data: data, byteCount: Int(audioBuffer.mDataByteSize), bitsPerChannel: bitsPerChannel, isFloat: isFloat, isSignedInteger: isSignedInteger))
        }

        guard !channels.isEmpty else { return [] }

        if isNonInterleaved {
            let frameCount = channels.map(\.count).min() ?? 0
            guard frameCount > 0 else { return [] }

            let left = channels[0]
            let right = channels.count > 1 ? channels[1] : channels[0]
            var stereo: [Float] = []
            stereo.reserveCapacity(frameCount * 2)
            for index in 0..<frameCount {
                stereo.append(left[index])
                stereo.append(right[index])
            }
            return stereo
        }

        let combined = channels.flatMap { $0 }
        if channelCount == 1 {
            var stereo: [Float] = []
            stereo.reserveCapacity(combined.count * 2)
            for sample in combined {
                stereo.append(sample)
                stereo.append(sample)
            }
            return stereo
        }

        var stereo: [Float] = []
        stereo.reserveCapacity((combined.count / channelCount) * 2)
        var index = 0
        while index + channelCount <= combined.count {
            stereo.append(combined[index])
            stereo.append(combined[index + 1])
            index += channelCount
        }
        return stereo
    }

    private func convert(data: UnsafeMutableRawPointer, byteCount: Int, bitsPerChannel: Int, isFloat: Bool, isSignedInteger: Bool) -> [Float] {
        if isFloat && bitsPerChannel == 32 {
            let count = byteCount / MemoryLayout<Float>.size
            let pointer = data.assumingMemoryBound(to: Float.self)
            return (0..<count).map { pointer[$0] }
        }

        if isFloat && bitsPerChannel == 64 {
            let count = byteCount / MemoryLayout<Double>.size
            let pointer = data.assumingMemoryBound(to: Double.self)
            return (0..<count).map { Float(pointer[$0]) }
        }

        if isSignedInteger && bitsPerChannel == 16 {
            let count = byteCount / MemoryLayout<Int16>.size
            let pointer = data.assumingMemoryBound(to: Int16.self)
            return (0..<count).map { Float(pointer[$0]) / Float(Int16.max) }
        }

        if isSignedInteger && bitsPerChannel == 32 {
            let count = byteCount / MemoryLayout<Int32>.size
            let pointer = data.assumingMemoryBound(to: Int32.self)
            return (0..<count).map { Float(pointer[$0]) / Float(Int32.max) }
        }

        return []
    }
}

enum ExitCode: Error {
    case permissionDenied
    case noDisplay
}

@main
struct TerbAudioHelper {
    static func main() async {
        let capture = AudioCapture()
        do {
            try await capture.start()
            while true {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        } catch ExitCode.permissionDenied {
            Foundation.exit(2)
        } catch ExitCode.noDisplay {
            Foundation.exit(3)
        } catch {
            fputs("capture-error: \(error.localizedDescription)\n", stderr)
            Foundation.exit(4)
        }
    }
}
