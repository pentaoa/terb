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
- `S`: open the compact full-screen settings panel. This is useful when the terminal is too narrow and the modules are hidden.
- `q` or `Esc` in Spectrum: return to the main menu.
- `?`: help.
- `q` or `Esc` on the main menu: quit.

Refresh rate is adjustable in Settings: 12, 24, 30, 45, 60, 90, 120, 144, 165, or 240 Hz.
Low Latency is the default analysis preset (1024-point FFT, 256-sample hop, 90 Hz refresh); Balanced and Precision remain available when steadier low-frequency resolution matters more than response time.

Spectrum rendering and processing are also adjustable from the sidebar or settings panel:
renderer mode (blocks, Braille, or CAVA-style stepped characters), frequency bands,
FFT size, analysis hop, refresh rate, high-shelf compensation, shelf gain,
adaptive sensitivity, noise reduction, BPM analysis, height curve, curve power,
spectrum trail, trail decay, accent display, accent threshold, and limiter ceiling.
The full settings panel shows the current value, valid range, and step for the selected row.
Key adjustable ranges are: 8-256 base frequency bands, 512-16384 FFT size, 64-4096 analysis hop, 2-100% attack, 0-99.5% release, 0-36 dB shelf gain, 0-95% noise reduction, 0.25-2.50 curve power, 20-99.5% trail decay, 2-98% accent threshold, and 35-100% limiter ceiling.
The spectrum uses dB-style band normalization plus visualizer-style adaptive sensitivity. A physically mapped spectrum can sit low and rarely touch the top, which is normal for real audio dynamics, but Terb's autosens now follows the CAVA-style rule of slowly increasing gain while bars do not peak and reducing gain faster when peaks approach the ceiling.
Accent display detects sudden spectrum-energy lifts using the adjustable accent threshold. Trace mode waits for the rise to settle, captures a low-pass-smoothed peak envelope, glides it upward over 0.1 seconds with a height-proportional offset, and renders an interpolated Braille line that fades toward the background over 0.5 seconds. Note-name mode instead fades compact note labels into empty spectrum cells over 0.1 seconds and fades them out over 0.5 seconds; the same note class keeps the same distribution.
Themes are deliberately static and restrained. Spring is the default soft palette, Vintage uses muted green, parchment, clay, and wine tones, and Mono stays neutral. Spectrum color changes only with bar height.
The limiter ceiling controls analysis headroom; the meter still maps that ceiling to full height.
The pipeline module exposes the main stages: capture, pre-analysis windowing, FFT, detector shaping, post-processing, and tempo estimation. Incoming audio is analyzed immediately without an added visual delay. BPM uses a live onset-strength envelope from wideband spectral flux, a rolling autocorrelation window, and continuity-aware tempo selection rather than simple low-frequency peak counting.

The Spectrum view is module-based: large terminals can show settings, pipeline, toolbar, waveform, and a right-side stereo master meter at the same time. Small terminals automatically hide side modules and keep the spectrum readable.
The waveform module always renders with Braille subpixels, sampling two virtual columns and four virtual rows per terminal cell.
The master meter also renders with Braille subpixels, softly fading from the border/background color near the bottom toward the current theme accent near the top.

Config is stored at `~/.config/terb/config.json`.
