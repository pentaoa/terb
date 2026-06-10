# Terb

Terb is a restrained terminal spectrum visualizer for macOS system audio. The TUI follows the same Rust stack as `toptimer`: `ratatui`, `crossterm`, JSON config, and a single terminal-first binary.

System audio capture on macOS is handled by a small Swift helper compiled during `cargo build`. The helper uses ScreenCaptureKit and streams interleaved stereo `f32` PCM samples to the Rust TUI.

## Requirements

- macOS 14 or newer
- Swift compiler from Xcode or Xcode Command Line Tools
- Rust stable

## Run

```bash
cargo run
```

On first capture, macOS may ask for Screen & System Audio Recording permission. If permission is denied, open System Settings and allow the terminal or Terb helper, then start capture again.

## Keys

- Start screen: full-screen ASCII main menu.
- `↑/↓` or `j/k`: move menu selection or sidebar setting selection.
- `Enter`: activate the selected main-menu item.
- `Space`: start/stop capture.
- In Spectrum, `←/→` or `h/l`: adjust the selected settings row while the spectrum keeps updating.
- In Spectrum, `s`, `p`, `t`, `m`, and `w`: show or hide Settings, Pipeline, Toolbar, Master, and Waveform modules.
- In Spectrum or Settings, `-` and `=`: decrease or increase visual audio delay in 10 ms steps.
- `S`: open the compact full-screen settings panel. This is useful when the terminal is too narrow and the modules are hidden.
- `q` or `Esc` in Spectrum: return to the main menu.
- `?`: help.
- `q` or `Esc` on the main menu: quit.

Refresh rate is adjustable in Settings: 12, 24, 30, 45, 60, 90, 120, 144, 165, or 240 Hz.

Spectrum rendering and processing are also adjustable from the sidebar or settings panel:
renderer mode (blocks, Braille, or CAVA-style stepped characters), frequency bands,
FFT size, analysis hop, refresh rate, high-shelf compensation, shelf gain,
adaptive sensitivity, noise reduction, BPM analysis, height curve, curve power,
spectrum trail, trail decay, accent display, accent threshold, and limiter ceiling.
The full settings panel shows the current value, valid range, and step for the selected row.
Key adjustable ranges are: 8-256 base frequency bands, 512-16384 FFT size, 64-4096 analysis hop, 0-2000 ms audio delay, 2-100% attack, 0-99.5% release, 0-36 dB shelf gain, 0-95% noise reduction, 0.25-2.50 curve power, 20-99.5% trail decay, 2-98% accent threshold, and 35-100% limiter ceiling.
The spectrum uses dB-style band normalization plus visualizer-style adaptive sensitivity. A physically mapped spectrum can sit low and rarely touch the top, which is normal for real audio dynamics, but Terb's autosens now follows the CAVA-style rule of slowly increasing gain while bars do not peak and reducing gain faster when peaks approach the ceiling.
Accent display detects sudden spectrum-energy lifts using the adjustable accent threshold. Trace mode waits for the rise to settle, captures a low-pass-smoothed peak envelope, glides it upward over 0.1 seconds with a height-proportional offset, and renders an interpolated Braille line that fades toward the background over 0.5 seconds. Note-name mode instead fades compact note labels into empty spectrum cells over 0.1 seconds and fades them out over 0.5 seconds; the same note class keeps the same distribution.
Static themes now color each spectrum column vertically by level instead of splitting colors by low and high frequency position.
Spring is the default soft palette, built from cream, pale cyan, rose, and light rose; Vintage remains a restrained four-color calibrated theme built around muted green, parchment, clay, and wine tones.
Music-reactive themes include Aurora, Sonic Texture, Miku, Square Album, and Circle Album. Aurora follows frequency position, spectrum energy, centroid, and transient flux. Sonic Texture samples a pitch-stable two-dimensional color field inside each Braille cell. Miku loops the bundled Tenor GIF in the center of the spectrum, contain-scaled so it never crops or overflows; the background is darkened and spectrum hits brighten the sampled GIF colors. Album themes poll system now-playing artwork first, fall back to Music or Spotify, always render the cover in Braille, darken non-spectrum regions, and brighten spectrum hits using the sampled cover colors. Square Album contain-scales the cover across the full spectrum canvas. Circle Album crops the cover into a centered disc and refreshes its slow roll with a rotating radial sweep rather than rotating every point globally at once. Miku playback starts at 5fps and adds 0.2x speed for every accent trigger in the last 3 seconds, with no cap. Their color drivers are time-smoothed so pitch and transient changes move without harsh jumps.
The limiter ceiling controls analysis headroom; the meter still maps that ceiling to full height.
Audio delay is also part of the processing view, so it can be adjusted live from the keyboard or the settings list.
The pipeline module exposes the main DAW-style stages: capture, sync delay, pre-analysis windowing, FFT, detector shaping, post-processing, and tempo estimation. BPM uses a live onset-strength envelope from wideband spectral flux rather than simple low-frequency peak counting.

The Spectrum view is module-based: large terminals can show settings, pipeline, toolbar, waveform, and a right-side stereo master meter at the same time. Small terminals automatically hide side modules and keep the spectrum readable.
The waveform module always renders with Braille subpixels, sampling two virtual columns and four virtual rows per terminal cell.
The master meter also renders with Braille subpixels, softly fading from the border/background color near the bottom toward the current theme accent near the top.

Config is stored at `~/.config/terb/config.json`.
