import AppKit
import CoreAudio
import Foundation

func propertyAddress(
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
    element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
}

func audioProcessObjectIDs() -> [AudioObjectID] {
    var address = propertyAddress(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    let sizeStatus = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        0,
        nil,
        &size
    )
    guard sizeStatus == noErr, size >= MemoryLayout<AudioObjectID>.size else {
        return []
    }

    var processes = [AudioObjectID](
        repeating: AudioObjectID(kAudioObjectUnknown),
        count: Int(size) / MemoryLayout<AudioObjectID>.size
    )
    let dataStatus = processes.withUnsafeMutableBufferPointer { buffer in
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &size,
            buffer.baseAddress!
        )
    }

    guard dataStatus == noErr else {
        return []
    }

    return processes.filter { $0 != AudioObjectID(kAudioObjectUnknown) }
}

func audioProcessUInt32Property(_ objectID: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
    var address = propertyAddress(selector)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value)
    return status == noErr ? value : nil
}

func audioProcessPID(_ objectID: AudioObjectID) -> pid_t? {
    var address = propertyAddress(kAudioProcessPropertyPID)
    var value = pid_t(0)
    var size = UInt32(MemoryLayout<pid_t>.size)
    let status = AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value)
    return status == noErr ? value : nil
}

func audioProcessBundleID(_ objectID: AudioObjectID) -> String? {
    var address = propertyAddress(kAudioProcessPropertyBundleID)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &value) { pointer in
        AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, pointer)
    }
    guard status == noErr, let value else {
        return nil
    }

    return value.takeRetainedValue() as String
}

func audioProcessIsRunningOutput(_ objectID: AudioObjectID) -> Bool {
    if let output = audioProcessUInt32Property(objectID, kAudioProcessPropertyIsRunningOutput) {
        return output != 0
    }
    if let running = audioProcessUInt32Property(objectID, kAudioProcessPropertyIsRunning) {
        return running != 0
    }
    return false
}

func launchableBundleID(_ bundleID: String?) -> String? {
    guard let bundleID, !bundleID.isEmpty else {
        return nil
    }
    guard NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) != nil else {
        return nil
    }
    return bundleID
}

func candidateBundleIDs(for objectID: AudioObjectID) -> [String] {
    var candidates: [String] = []
    if let bundleID = launchableBundleID(audioProcessBundleID(objectID)) {
        candidates.append(bundleID)
    }
    if let pid = audioProcessPID(objectID),
       let appBundleID = NSRunningApplication(processIdentifier: pid)?.bundleIdentifier,
       let bundleID = launchableBundleID(appBundleID) {
        candidates.append(bundleID)
    }
    return candidates
}

@main
struct TerbAudioProcessHelper {
    static func main() {
        var seen = Set<String>()
        var bundles: [String] = []

        for objectID in audioProcessObjectIDs() where audioProcessIsRunningOutput(objectID) {
            for bundleID in candidateBundleIDs(for: objectID) where !seen.contains(bundleID) {
                seen.insert(bundleID)
                bundles.append(bundleID)
            }
        }

        guard let data = try? JSONSerialization.data(withJSONObject: bundles) else {
            Foundation.exit(1)
        }
        FileHandle.standardOutput.write(data)
    }
}
