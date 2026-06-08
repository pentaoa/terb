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

Refresh rate is adjustable in Settings: 24, 30, 45, 60, 90, or 120 Hz.

Spectrum rendering and processing are also adjustable from the sidebar or settings panel:
renderer mode (blocks or Braille), frequency bands,
FFT size, high-shelf compensation, shelf gain, height curve, curve power, and limiter ceiling.
The limiter ceiling controls analysis headroom; the meter still maps that ceiling to full height.
Audio delay is also part of the processing view, so it can be adjusted live from the keyboard or the settings list.

The Spectrum view is module-based: large terminals can show settings, pipeline, toolbar, waveform, and a right-side stereo master meter at the same time. Small terminals automatically hide side modules and keep the spectrum readable.

Config is stored at `~/.config/terb/config.json`.
