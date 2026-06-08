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

Refresh rate is adjustable in Settings: 24, 30, 45, 60, 90, 120, or 144 Hz.

Spectrum rendering and processing are also adjustable from the sidebar or settings panel:
renderer mode (blocks or Braille), frequency bands,
FFT size, high-shelf compensation, shelf gain, height curve, curve power,
spectrum trail, trail decay, accent trace, accent threshold, and limiter ceiling.
Accent trace detects sudden spectrum-energy lifts using the adjustable accent threshold, waits for the rise to settle, captures a low-pass-smoothed peak envelope, glides it upward from one to five terminal cells over 0.1 seconds, and renders an interpolated Braille line that fades toward the background over 0.5 seconds. A new accent replaces the previous visible trace.
Music-reactive themes include Aurora, Sonic Texture, Noise Warp, and Miku. Aurora follows frequency position, spectrum energy, centroid, and transient flux. Sonic Texture samples a pitch-stable two-dimensional color field inside each Braille cell, while Noise Warp layers fBM noise, contour lines, and domain warping for broader procedural texture. Miku loops the bundled Tenor GIF in the center of the spectrum, contain-scaled so it never crops or overflows; the background is darkened and spectrum hits brighten the sampled GIF colors. Miku playback starts at 5fps and adds 0.2x speed for every accent trigger in the last 3 seconds, with no cap. Their color drivers are time-smoothed so pitch and transient changes move without harsh jumps.
The limiter ceiling controls analysis headroom; the meter still maps that ceiling to full height.
Audio delay is also part of the processing view, so it can be adjusted live from the keyboard or the settings list.

The Spectrum view is module-based: large terminals can show settings, pipeline, toolbar, waveform, and a right-side stereo master meter at the same time. Small terminals automatically hide side modules and keep the spectrum readable.
The waveform module always renders with Braille subpixels, sampling two virtual columns and four virtual rows per terminal cell.
The master meter also renders with Braille subpixels, softly fading from the border/background color near the bottom toward the current theme accent near the top.

Config is stored at `~/.config/terb/config.json`.
