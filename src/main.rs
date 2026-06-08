use std::{
    collections::VecDeque,
    env, fs, io,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event as CEvent, KeyCode, KeyEvent},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, size as terminal_size, BeginSynchronizedUpdate,
        EndSynchronizedUpdate, EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    prelude::{CrosstermBackend, Frame, Terminal},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, List, ListItem, ListState, Paragraph, Widget, Wrap},
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const LANGUAGES: &[(&str, &str)] = &[("zh", "中文"), ("en", "English"), ("ja", "日本語")];
const THEMES: &[ThemeId] = &[
    ThemeId::System,
    ThemeId::Graphite,
    ThemeId::Ocean,
    ThemeId::Aurora,
    ThemeId::SonicTexture,
    ThemeId::NoiseWarp,
    ThemeId::Amber,
    ThemeId::Mono,
];
const SPECTRUM_RENDERERS: &[SpectrumRenderer] =
    &[SpectrumRenderer::Blocks, SpectrumRenderer::Braille];
const MENU_ITEMS: &[&str] = &[
    "menu_spectrum",
    "menu_toggle",
    "menu_settings",
    "menu_help",
    "menu_quit",
];
const SETTING_COUNT: usize = 20;
const REFRESH_RATES: &[u16] = &[24, 30, 45, 60, 90, 120, 144];
const ANALYSIS_HOPS: &[usize] = &[128, 256, 512, 1024, 2048];
const MIN_FREQUENCY: f32 = 35.0;
const MAX_FREQUENCY: f32 = 18_000.0;
const DEFAULT_HIGH_SHELF_DB: f32 = 6.0;
const DEFAULT_VISUAL_CURVE: f32 = 0.88;
const DEFAULT_CEILING: f32 = 0.88;
const DEFAULT_TRAIL_DECAY: f32 = 0.88;
const DEFAULT_ACCENT_TRACE_THRESHOLD: f32 = 0.50;
const ACCENT_TRACE_LIFETIME_MS: u64 = 500;
const ACCENT_TRACE_COOLDOWN_MS: u64 = 90;
const ACCENT_TRACE_VERTICAL_OFFSET_CELLS: usize = 5;
const ACCENT_TRACE_SETTLE_FRAMES: usize = 1;
const ACCENT_TRACE_MAX_CAPTURE_FRAMES: usize = 8;
const ACCENT_TRACE_RISE_EPSILON: f32 = 0.012;
const DEFAULT_AUDIO_DELAY_MS: u16 = 0;
const DEFAULT_ATTACK: f32 = 0.82;
const DEFAULT_RELEASE: f32 = 0.48;
const DEFAULT_ANALYSIS_HOP: usize = 256;
const AUDIO_DELAY_STEP_MS: i32 = 10;
const MAX_AUDIO_DELAY_MS: i32 = 500;
const FFT_SIZES: &[usize] = &[1024, 2048, 4096, 8192];
const MAX_ANALYSIS_BARS: usize = 1024;
const AUDIO_READ_FRAMES: usize = 512;
const VISUAL_NOISE_FLOOR: f32 = 0.025;
const SILENCE_GATE: f32 = 0.000_12;
const WAVEFORM_SAMPLES: usize = 1024;
const WAVEFORM_TARGET_PEAK: f32 = 0.72;
const BRAILLE_DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
const TITLE_ART: &[&str] = &[
    " _            _     ",
    "| |_ ___ _ __| |__  ",
    "| __/ _ \\ '__| '_ \\ ",
    "| ||  __/ |  | |_) |",
    " \\__\\___|_|  |_.__/ ",
];

fn main() -> io::Result<()> {
    let config = Config::load();
    let mut app = App::new(config);
    run(&mut app)
}

fn run(app: &mut App) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_app(&mut terminal, app);
    app.stop_capture();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    let mut last_tick = Instant::now();

    loop {
        let (width, height) = terminal_size()?;
        if let Some(target) = visual_bar_count(app, Rect::new(0, 0, width, height)) {
            app.set_visual_bar_count(target);
        }
        app.drain_audio_events();
        execute!(terminal.backend_mut(), BeginSynchronizedUpdate)?;
        let draw_result = terminal.draw(|frame| draw(frame, app)).map(|_| ());
        let sync_result = execute!(terminal.backend_mut(), EndSynchronizedUpdate);
        draw_result?;
        sync_result?;

        let tick_rate = app.frame_duration();
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_millis(0));

        if event::poll(timeout)? {
            if let CEvent::Key(key) = event::read()? {
                if handle_key(app, key) {
                    return Ok(());
                }
            }
        }

        let elapsed = last_tick.elapsed();
        if elapsed >= tick_rate {
            app.tick(elapsed);
            last_tick = Instant::now();
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match app.screen {
        Screen::Menu => handle_menu_key(app, key),
        Screen::Spectrum => handle_spectrum_key(app, key),
        Screen::Settings => handle_settings_key(app, key),
        Screen::Help => handle_help_key(app, key),
    }
}

fn handle_menu_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => return true,
        KeyCode::Up | KeyCode::Char('k') => app.prev_menu(),
        KeyCode::Down | KeyCode::Char('j') => app.next_menu(),
        KeyCode::Char(' ') => app.toggle_capture(),
        KeyCode::Enter => match MENU_ITEMS[app.menu_index] {
            "menu_spectrum" => app.screen = Screen::Spectrum,
            "menu_toggle" => app.toggle_capture(),
            "menu_settings" => app.screen = Screen::Settings,
            "menu_help" => app.screen = Screen::Help,
            "menu_quit" => return true,
            _ => {}
        },
        KeyCode::Char('s') => app.screen = Screen::Settings,
        KeyCode::Char('?') => app.screen = Screen::Help,
        _ => {}
    }
    false
}

fn handle_spectrum_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => app.screen = Screen::Menu,
        KeyCode::Char(' ') => app.toggle_capture(),
        KeyCode::Char('S') => app.screen = Screen::Settings,
        KeyCode::Char('s') => app.toggle_settings_panel(),
        KeyCode::Char('p') => app.toggle_pipeline_panel(),
        KeyCode::Char('t') => app.toggle_toolbar_panel(),
        KeyCode::Char('m') => app.toggle_master_panel(),
        KeyCode::Char('w') => app.toggle_waveform_panel(),
        KeyCode::Char('-') => app.adjust_audio_delay(-1),
        KeyCode::Char('=') | KeyCode::Char('+') => app.adjust_audio_delay(1),
        KeyCode::Char('?') => app.screen = Screen::Help,
        KeyCode::Up | KeyCode::Char('k') => app.prev_setting(),
        KeyCode::Down | KeyCode::Char('j') => app.next_setting(),
        KeyCode::Left | KeyCode::Char('h') => app.adjust_setting(-1),
        KeyCode::Right | KeyCode::Char('l') => app.adjust_setting(1),
        _ => {}
    }
    false
}

fn handle_settings_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            app.save_config();
            app.screen = if app.audio.is_some() {
                Screen::Spectrum
            } else {
                Screen::Menu
            };
        }
        KeyCode::Up | KeyCode::Char('k') => app.prev_setting(),
        KeyCode::Down | KeyCode::Char('j') => app.next_setting(),
        KeyCode::Left | KeyCode::Char('h') => app.adjust_setting(-1),
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => app.adjust_setting(1),
        KeyCode::Char('-') => app.adjust_audio_delay(-1),
        KeyCode::Char('=') | KeyCode::Char('+') => app.adjust_audio_delay(1),
        KeyCode::Char(' ') => app.toggle_capture(),
        KeyCode::Char('?') => app.screen = Screen::Help,
        _ => {}
    }
    false
}

fn handle_help_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => app.screen = Screen::Menu,
        _ => {}
    }
    false
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum Lang {
    Zh,
    En,
    Ja,
}

impl Lang {
    fn code(self) -> &'static str {
        match self {
            Lang::Zh => "zh",
            Lang::En => "en",
            Lang::Ja => "ja",
        }
    }

    fn from_code(code: &str) -> Self {
        match code {
            "en" => Lang::En,
            "ja" => Lang::Ja,
            _ => Lang::Zh,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum ThemeId {
    System,
    Graphite,
    Ocean,
    Aurora,
    // Legacy theme ids are kept only so older config files can be migrated.
    PitchClass,
    ChromaBands,
    PitchMemory,
    HarmonicComb,
    SonicTexture,
    NoiseWarp,
    Amber,
    Mono,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SpectrumRenderer {
    Blocks,
    Braille,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum AnalysisPreset {
    LowLatency,
    Balanced,
    Precision,
    Custom,
}

#[derive(Clone, Copy)]
struct AnalysisPresetSpec {
    fft_size: usize,
    hop_size: usize,
    refresh_hz: u16,
    attack: f32,
    release: f32,
}

impl AnalysisPreset {
    fn title_key(self) -> &'static str {
        match self {
            AnalysisPreset::LowLatency => "preset_low_latency",
            AnalysisPreset::Balanced => "preset_balanced",
            AnalysisPreset::Precision => "preset_precision",
            AnalysisPreset::Custom => "preset_custom",
        }
    }

    fn spec(self) -> Option<AnalysisPresetSpec> {
        match self {
            AnalysisPreset::LowLatency => Some(AnalysisPresetSpec {
                fft_size: 1024,
                hop_size: 256,
                refresh_hz: 90,
                attack: DEFAULT_ATTACK,
                release: DEFAULT_RELEASE,
            }),
            AnalysisPreset::Balanced => Some(AnalysisPresetSpec {
                fft_size: 2048,
                hop_size: 512,
                refresh_hz: 60,
                attack: 0.72,
                release: 0.62,
            }),
            AnalysisPreset::Precision => Some(AnalysisPresetSpec {
                fft_size: 8192,
                hop_size: 2048,
                refresh_hz: 45,
                attack: 0.64,
                release: 0.76,
            }),
            AnalysisPreset::Custom => None,
        }
    }
}

const ANALYSIS_PRESETS: &[AnalysisPreset] = &[
    AnalysisPreset::LowLatency,
    AnalysisPreset::Balanced,
    AnalysisPreset::Precision,
    AnalysisPreset::Custom,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    version: u8,
    settings: Settings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Settings {
    language: String,
    theme: ThemeId,
    #[serde(default = "default_smoothing")]
    smoothing: f32,
    #[serde(default = "default_analysis_preset")]
    analysis_preset: AnalysisPreset,
    #[serde(default = "default_attack")]
    attack: f32,
    #[serde(default = "default_release")]
    release: f32,
    bar_count: usize,
    #[serde(default = "default_spectrum_renderer")]
    renderer: SpectrumRenderer,
    #[serde(default = "default_fft_size")]
    fft_size: usize,
    #[serde(default = "default_analysis_hop")]
    analysis_hop: usize,
    #[serde(default = "default_refresh_hz")]
    refresh_hz: u16,
    #[serde(default = "default_audio_delay_ms")]
    audio_delay_ms: u16,
    #[serde(default = "default_high_shelf_enabled")]
    high_shelf_enabled: bool,
    #[serde(default = "default_high_shelf_db")]
    high_shelf_db: f32,
    #[serde(default = "default_visual_curve_enabled")]
    visual_curve_enabled: bool,
    #[serde(default = "default_visual_curve")]
    visual_curve: f32,
    #[serde(default = "default_ceiling")]
    ceiling: f32,
    #[serde(default = "default_trail_enabled")]
    trail_enabled: bool,
    #[serde(default = "default_trail_decay")]
    trail_decay: f32,
    #[serde(default = "default_accent_trace_enabled")]
    accent_trace_enabled: bool,
    #[serde(default = "default_accent_trace_threshold")]
    accent_trace_threshold: f32,
    #[serde(default = "default_show_settings_panel")]
    show_settings_panel: bool,
    #[serde(default = "default_show_pipeline_panel")]
    show_pipeline_panel: bool,
    #[serde(default = "default_show_toolbar_panel")]
    show_toolbar_panel: bool,
    #[serde(default = "default_show_master_panel")]
    show_master_panel: bool,
    #[serde(default = "default_show_waveform_panel")]
    show_waveform_panel: bool,
}

fn default_spectrum_renderer() -> SpectrumRenderer {
    SpectrumRenderer::Blocks
}

fn default_refresh_hz() -> u16 {
    90
}

fn default_audio_delay_ms() -> u16 {
    DEFAULT_AUDIO_DELAY_MS
}

fn default_fft_size() -> usize {
    8192
}

fn default_smoothing() -> f32 {
    0.72
}

fn default_analysis_preset() -> AnalysisPreset {
    AnalysisPreset::Precision
}

fn default_attack() -> f32 {
    DEFAULT_ATTACK
}

fn default_release() -> f32 {
    DEFAULT_RELEASE
}

fn default_analysis_hop() -> usize {
    DEFAULT_ANALYSIS_HOP
}

fn default_high_shelf_enabled() -> bool {
    false
}

fn default_high_shelf_db() -> f32 {
    DEFAULT_HIGH_SHELF_DB
}

fn default_visual_curve_enabled() -> bool {
    true
}

fn default_visual_curve() -> f32 {
    DEFAULT_VISUAL_CURVE
}

fn default_ceiling() -> f32 {
    DEFAULT_CEILING
}

fn default_trail_enabled() -> bool {
    true
}

fn default_trail_decay() -> f32 {
    DEFAULT_TRAIL_DECAY
}

fn default_accent_trace_enabled() -> bool {
    true
}

fn default_accent_trace_threshold() -> f32 {
    DEFAULT_ACCENT_TRACE_THRESHOLD
}

fn default_show_settings_panel() -> bool {
    true
}

fn default_show_pipeline_panel() -> bool {
    true
}

fn default_show_toolbar_panel() -> bool {
    true
}

fn default_show_master_panel() -> bool {
    true
}

fn default_show_waveform_panel() -> bool {
    false
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            settings: Settings {
                language: "zh".to_string(),
                theme: ThemeId::System,
                smoothing: default_smoothing(),
                analysis_preset: default_analysis_preset(),
                attack: default_attack(),
                release: default_release(),
                bar_count: 72,
                renderer: default_spectrum_renderer(),
                fft_size: default_fft_size(),
                analysis_hop: default_analysis_hop(),
                refresh_hz: default_refresh_hz(),
                audio_delay_ms: default_audio_delay_ms(),
                high_shelf_enabled: default_high_shelf_enabled(),
                high_shelf_db: default_high_shelf_db(),
                visual_curve_enabled: default_visual_curve_enabled(),
                visual_curve: default_visual_curve(),
                ceiling: default_ceiling(),
                trail_enabled: default_trail_enabled(),
                trail_decay: default_trail_decay(),
                accent_trace_enabled: default_accent_trace_enabled(),
                accent_trace_threshold: default_accent_trace_threshold(),
                show_settings_panel: default_show_settings_panel(),
                show_pipeline_panel: default_show_pipeline_panel(),
                show_toolbar_panel: default_show_toolbar_panel(),
                show_master_panel: default_show_master_panel(),
                show_waveform_panel: default_show_waveform_panel(),
            },
        }
    }
}

impl Config {
    fn load() -> Self {
        let path = config_path();
        fs::read_to_string(path)
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default()
    }

    fn save(&self) {
        let path = config_path();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, data);
        }
    }
}

impl Settings {
    fn normalize(&mut self) {
        self.bar_count = self.bar_count.clamp(32, 120);
        self.fft_size = nearest_fft_size(self.fft_size);
        self.analysis_hop = nearest_hop_size(self.analysis_hop, self.fft_size);
        self.refresh_hz = nearest_refresh_rate(self.refresh_hz);
        self.attack = self.attack.clamp(0.10, 0.98);
        self.release = self.release.clamp(0.05, 0.95);
        self.smoothing = self.smoothing.clamp(0.20, 0.92);
        self.audio_delay_ms = (self.audio_delay_ms as i32).clamp(0, MAX_AUDIO_DELAY_MS) as u16;
        self.high_shelf_db = self.high_shelf_db.clamp(0.0, 18.0);
        self.visual_curve = self.visual_curve.clamp(0.55, 1.35);
        self.ceiling = self.ceiling.clamp(0.70, 0.98);
        self.trail_decay = self.trail_decay.clamp(0.65, 0.98);
        self.accent_trace_threshold = self.accent_trace_threshold.clamp(0.10, 0.90);
        if is_retired_theme(self.theme) {
            self.theme = ThemeId::SonicTexture;
        }
    }

    fn apply_analysis_preset(&mut self, preset: AnalysisPreset) {
        self.analysis_preset = preset;
        if let Some(spec) = preset.spec() {
            self.fft_size = spec.fft_size;
            self.analysis_hop = spec.hop_size;
            self.refresh_hz = spec.refresh_hz;
            self.attack = spec.attack;
            self.release = spec.release;
        }
        self.normalize();
    }

    fn mark_custom_analysis(&mut self) {
        self.analysis_preset = AnalysisPreset::Custom;
    }
}

fn nearest_fft_size(value: usize) -> usize {
    FFT_SIZES
        .iter()
        .copied()
        .min_by_key(|size| size.abs_diff(value))
        .unwrap_or(default_fft_size())
}

fn is_retired_theme(theme: ThemeId) -> bool {
    matches!(
        theme,
        ThemeId::PitchClass | ThemeId::ChromaBands | ThemeId::PitchMemory | ThemeId::HarmonicComb
    )
}

fn nearest_hop_size(value: usize, fft_size: usize) -> usize {
    ANALYSIS_HOPS
        .iter()
        .copied()
        .filter(|hop| *hop <= fft_size)
        .min_by_key(|hop| hop.abs_diff(value))
        .unwrap_or(default_analysis_hop().min(fft_size))
}

fn nearest_refresh_rate(value: u16) -> u16 {
    REFRESH_RATES
        .iter()
        .copied()
        .min_by_key(|rate| rate.abs_diff(value))
        .unwrap_or(default_refresh_hz())
}

fn config_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config/terb/config.json")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Menu,
    Spectrum,
    Settings,
    Help,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureState {
    Idle,
    Starting,
    Running,
    PermissionNeeded,
    Failed,
}

struct App {
    config: Config,
    lang: Lang,
    theme_id: ThemeId,
    screen: Screen,
    menu_index: usize,
    setting_index: usize,
    capture_state: CaptureState,
    status: String,
    spectrum: Vec<f32>,
    spectrum_trail: Vec<f32>,
    pending_accent_trace: Option<PendingAccentTrace>,
    accent_traces: Vec<AccentTrace>,
    accent_energy_baseline: f32,
    accent_trace_cooldown: Duration,
    color_state: VisualColorState,
    level: f32,
    master_left: f32,
    master_right: f32,
    waveform: Vec<f32>,
    analyzer: SpectrumAnalyzer,
    visual_bar_count: usize,
    audio: Option<AudioProcess>,
    capture_id: u64,
    rx: Receiver<AudioEvent>,
    tx: Sender<AudioEvent>,
    last_samples_at: Option<Instant>,
    delayed_audio: VecDeque<DelayedAudio>,
}

#[derive(Clone, Debug)]
struct PendingAccentTrace {
    envelope: Vec<f32>,
    peak_energy: f32,
    settle_frames: usize,
    capture_frames: usize,
}

impl PendingAccentTrace {
    fn new(bars: &[f32], energy: f32) -> Self {
        Self {
            envelope: bars.to_vec(),
            peak_energy: energy,
            settle_frames: 0,
            capture_frames: 1,
        }
    }

    fn update(&mut self, bars: &[f32], energy: f32) -> bool {
        merge_envelope_max(&mut self.envelope, bars);
        self.capture_frames += 1;

        if energy > self.peak_energy + ACCENT_TRACE_RISE_EPSILON {
            self.peak_energy = energy;
            self.settle_frames = 0;
        } else {
            self.settle_frames += 1;
        }

        self.settle_frames >= ACCENT_TRACE_SETTLE_FRAMES
            || self.capture_frames >= ACCENT_TRACE_MAX_CAPTURE_FRAMES
    }
}

#[derive(Clone, Debug)]
struct AccentTrace {
    envelope: Vec<f32>,
    age: Duration,
}

impl AccentTrace {
    fn fade(&self) -> f32 {
        let lifetime = ACCENT_TRACE_LIFETIME_MS as f32 / 1000.0;
        let age = self.age.as_secs_f32();
        (1.0 - age / lifetime).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug)]
struct AccentTraceRender {
    envelope: Vec<f32>,
    fade: f32,
}

#[derive(Clone, Copy, Debug)]
struct AccentTraceOverlay {
    mask: u8,
    color: Color,
    visibility: f32,
}

#[derive(Clone, Copy, Debug)]
struct AccentTriggerThresholds {
    peak: f32,
    initial_energy: f32,
    energy: f32,
    flux: f32,
    rise: f32,
    ratio: f32,
}

#[derive(Clone, Copy, Debug)]
struct VisualColorState {
    centroid: f32,
    energy: f32,
    flux: f32,
    phase: f32,
    chroma: [f32; 12],
    dominant_pitch: usize,
    pitch_hue: f32,
    pitch_confidence: f32,
    dominant_frequency: f32,
    pitch_memory: [f32; 12],
}

impl Default for VisualColorState {
    fn default() -> Self {
        Self {
            centroid: 0.45,
            energy: 0.0,
            flux: 0.0,
            phase: 0.0,
            chroma: [0.0; 12],
            dominant_pitch: 0,
            pitch_hue: pitch_class_hue(0),
            pitch_confidence: 0.0,
            dominant_frequency: 220.0,
            pitch_memory: [0.0; 12],
        }
    }
}

impl VisualColorState {
    fn update(&mut self, bars: &[f32], previous: &[f32], level: f32) {
        if bars.is_empty() {
            return;
        }

        let mut weighted_total = 0.0_f32;
        let mut weight = 0.0_f32;
        let mut positive_flux = 0.0_f32;
        let mut chroma = [0.0_f32; 12];
        let mut dominant_frequency = self.dominant_frequency;
        let mut dominant_weight = 0.0_f32;
        let max_index = (bars.len().saturating_sub(1)).max(1) as f32;

        for (index, value) in bars.iter().copied().enumerate() {
            let value = value.clamp(0.0, 1.0);
            let position = index as f32 / max_index;
            let frequency = frequency_for_position(position);
            let emphasis = value.powf(1.2);
            weighted_total += position * emphasis;
            weight += emphasis;
            let pitch_class = pitch_class_for_frequency(frequency);
            let melody_weight = emphasis * melody_frequency_weight(frequency);
            chroma[pitch_class] += melody_weight;
            if melody_weight > dominant_weight {
                dominant_weight = melody_weight;
                dominant_frequency = frequency;
            }

            let old = previous.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            positive_flux += (value - old).max(0.0);
        }

        let centroid = if weight > 0.000_1 {
            weighted_total / weight
        } else {
            self.centroid
        };
        let energy = (bars.iter().copied().sum::<f32>() / bars.len() as f32)
            .max(level)
            .clamp(0.0, 1.0);
        let flux = (positive_flux / bars.len() as f32 * 3.0).clamp(0.0, 1.0);
        let chroma_peak = chroma.iter().copied().fold(0.0_f32, f32::max).max(0.000_1);
        for (current, next) in self.chroma.iter_mut().zip(chroma.into_iter()) {
            *current = lerp(*current, (next / chroma_peak).clamp(0.0, 1.0), 0.14);
        }
        let (dominant_pitch, confidence) = strongest_chroma(&self.chroma);

        self.centroid = lerp(self.centroid, centroid, 0.12);
        self.energy = lerp(self.energy, energy, 0.16);
        self.flux = if flux > self.flux {
            lerp(self.flux, flux, 0.30)
        } else {
            lerp(self.flux, flux, 0.12)
        };
        self.phase = wrap_unit(self.phase + 0.003 + self.energy * 0.007 + self.flux * 0.010);
        self.dominant_pitch = dominant_pitch;
        self.pitch_hue = lerp_unit(self.pitch_hue, pitch_class_hue(dominant_pitch), 0.14);
        self.pitch_confidence = lerp(self.pitch_confidence, confidence, 0.20);
        self.dominant_frequency = lerp(self.dominant_frequency, dominant_frequency, 0.12);
        for (pitch, memory) in self.pitch_memory.iter_mut().enumerate() {
            let activation = self.chroma[pitch] * 0.88;
            *memory = (*memory * 0.94).max(activation).clamp(0.0, 1.0);
        }
    }
}

impl App {
    fn new(mut config: Config) -> Self {
        if config.settings.analysis_preset != AnalysisPreset::Custom {
            let preset = config.settings.analysis_preset;
            config.settings.apply_analysis_preset(preset);
        } else {
            config.settings.normalize();
        }
        let lang = Lang::from_code(&config.settings.language);
        let theme_id = config.settings.theme;
        let bar_count = config.settings.bar_count;
        let fft_size = config.settings.fft_size;
        let hop_size = config.settings.analysis_hop;
        let (tx, rx) = mpsc::channel();

        Self {
            config,
            lang,
            theme_id,
            screen: Screen::Menu,
            menu_index: 0,
            setting_index: 0,
            capture_state: CaptureState::Idle,
            status: tr(lang, "ready").to_string(),
            spectrum: vec![0.0; bar_count],
            spectrum_trail: vec![0.0; bar_count],
            pending_accent_trace: None,
            accent_traces: Vec::new(),
            accent_energy_baseline: 0.0,
            accent_trace_cooldown: Duration::from_millis(0),
            color_state: VisualColorState::default(),
            level: 0.0,
            master_left: 0.0,
            master_right: 0.0,
            waveform: vec![0.0; WAVEFORM_SAMPLES],
            analyzer: SpectrumAnalyzer::new(fft_size, 48_000.0, bar_count, hop_size),
            visual_bar_count: bar_count,
            audio: None,
            capture_id: 0,
            rx,
            tx,
            last_samples_at: None,
            delayed_audio: VecDeque::new(),
        }
    }

    fn theme(&self) -> Theme {
        theme(self.theme_id)
    }

    fn t(&self, key: &'static str) -> &'static str {
        tr(self.lang, key)
    }

    fn frame_duration(&self) -> Duration {
        let refresh_hz = self.config.settings.refresh_hz.clamp(12, 144);
        Duration::from_secs_f64(1.0 / refresh_hz as f64)
    }

    fn tick(&mut self, elapsed: Duration) {
        if self.capture_state == CaptureState::Running {
            self.advance_accent_traces(elapsed);
            if let Some(last) = self.last_samples_at {
                if last.elapsed() > Duration::from_secs(3) {
                    self.status = self.t("waiting_audio").to_string();
                }
            }
        }
    }

    fn advance_accent_traces(&mut self, elapsed: Duration) {
        let lifetime = Duration::from_millis(ACCENT_TRACE_LIFETIME_MS);
        for trace in &mut self.accent_traces {
            trace.age += elapsed;
        }
        self.accent_traces.retain(|trace| trace.age < lifetime);
        self.accent_trace_cooldown = self
            .accent_trace_cooldown
            .checked_sub(elapsed)
            .unwrap_or_default();
    }

    fn drain_audio_events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AudioEvent::Samples(capture_id, samples) => {
                    if capture_id == self.capture_id && self.audio.is_some() {
                        self.queue_audio_samples(samples);
                    }
                }
                AudioEvent::Status(capture_id, message) => {
                    if capture_id != self.capture_id || self.audio.is_none() {
                        continue;
                    }
                    if message.contains("ready") {
                        self.status = self.t("helper_ready").to_string();
                    } else if message.contains("permission-denied") {
                        self.audio = None;
                        self.delayed_audio.clear();
                        self.capture_state = CaptureState::PermissionNeeded;
                        self.status = self.t("permission_needed").to_string();
                    } else if message.contains("no-display") || message.contains("capture-error") {
                        self.audio = None;
                        self.delayed_audio.clear();
                        self.capture_state = CaptureState::Failed;
                        self.status = self.t("capture_failed").to_string();
                    }
                }
                AudioEvent::Exit(capture_id, code) => {
                    if capture_id != self.capture_id {
                        continue;
                    }
                    if code.is_none() && self.audio.is_none() {
                        continue;
                    }
                    self.audio = None;
                    self.delayed_audio.clear();
                    self.capture_state = if code == Some(2) {
                        CaptureState::PermissionNeeded
                    } else {
                        CaptureState::Failed
                    };
                    self.status = if code == Some(2) {
                        self.t("permission_needed").to_string()
                    } else {
                        self.t("capture_failed").to_string()
                    };
                }
                AudioEvent::Error(capture_id, message) => {
                    if capture_id != self.capture_id || self.audio.is_none() {
                        continue;
                    }
                    self.delayed_audio.clear();
                    self.capture_state = CaptureState::Failed;
                    self.status = message;
                }
            }
        }
        self.flush_audio_delay();
    }

    fn queue_audio_samples(&mut self, samples: AudioSamples) {
        self.delayed_audio.push_back(DelayedAudio {
            received_at: Instant::now(),
            samples,
        });

        while self.delayed_audio.len() > 256 {
            self.delayed_audio.pop_front();
        }

        self.flush_audio_delay();
    }

    fn flush_audio_delay(&mut self) {
        let delay = self.audio_delay_duration();
        loop {
            let Some(front) = self.delayed_audio.front() else {
                break;
            };
            if front.received_at.elapsed() < delay {
                break;
            }
            if let Some(delayed) = self.delayed_audio.pop_front() {
                self.process_audio_samples(delayed.samples);
            }
        }
    }

    fn process_audio_samples(&mut self, samples: AudioSamples) {
        self.capture_state = CaptureState::Running;
        self.last_samples_at = Some(Instant::now());
        self.level = audio_level(&samples.mono);
        self.master_left = samples.left_level;
        self.master_right = samples.right_level;
        self.update_waveform(&samples.mono);
        let pipeline = SpectrumPipeline::from_settings(&self.config.settings);
        if let Some(bars) = self.analyzer.consume(
            &samples.mono,
            self.config.settings.attack,
            self.config.settings.release,
            pipeline,
        ) {
            self.color_state.update(&bars, &self.spectrum, self.level);
            self.update_accent_trace_detector(&bars);
            update_spectrum_trail(&mut self.spectrum_trail, &bars, &self.config.settings);
            self.spectrum = bars;
        }
        self.status = self.t("running").to_string();
    }

    fn update_accent_trace_detector(&mut self, bars: &[f32]) {
        if bars.is_empty() {
            return;
        }

        let energy = spectrum_accent_energy(bars);
        let trigger_thresholds =
            accent_trigger_thresholds(self.config.settings.accent_trace_threshold);
        if !self.config.settings.accent_trace_enabled {
            self.pending_accent_trace = None;
            self.accent_energy_baseline = energy;
            return;
        }

        if self.pending_accent_trace.is_some() {
            let ready = self
                .pending_accent_trace
                .as_mut()
                .map(|pending| pending.update(bars, energy))
                .unwrap_or(false);
            if ready {
                let pending = self
                    .pending_accent_trace
                    .take()
                    .expect("pending trace exists");
                self.push_accent_trace(&pending.envelope);
                self.accent_trace_cooldown = Duration::from_millis(ACCENT_TRACE_COOLDOWN_MS);
            }
            self.update_accent_energy_baseline(energy, false);
            return;
        }

        if self.accent_energy_baseline <= 0.0 {
            let peak = bars.iter().copied().fold(0.0_f32, f32::max);
            let flux = spectrum_positive_flux(bars, &self.spectrum);
            if peak > trigger_thresholds.peak
                && energy > trigger_thresholds.initial_energy
                && flux > trigger_thresholds.flux
            {
                self.pending_accent_trace = Some(PendingAccentTrace::new(bars, energy));
            }
            self.accent_energy_baseline = energy;
            return;
        }

        let baseline = self.accent_energy_baseline;
        let peak = bars.iter().copied().fold(0.0_f32, f32::max);
        let flux = spectrum_positive_flux(bars, &self.spectrum);
        let rise = energy - baseline;
        let ratio = energy / baseline.max(0.035);
        let triggered = self.config.settings.accent_trace_enabled
            && self.accent_trace_cooldown.is_zero()
            && peak > trigger_thresholds.peak
            && energy > trigger_thresholds.energy
            && flux > trigger_thresholds.flux
            && (rise > trigger_thresholds.rise || ratio > trigger_thresholds.ratio);

        if triggered {
            self.pending_accent_trace = Some(PendingAccentTrace::new(bars, energy));
        }

        self.update_accent_energy_baseline(energy, triggered);
    }

    fn update_accent_energy_baseline(&mut self, energy: f32, triggered: bool) {
        let baseline = self.accent_energy_baseline;
        let follow = if energy > baseline {
            if triggered {
                0.06
            } else {
                0.10
            }
        } else {
            0.20
        };
        self.accent_energy_baseline = lerp(baseline, energy, follow);
    }

    fn push_accent_trace(&mut self, bars: &[f32]) {
        if bars.is_empty() {
            return;
        }

        self.accent_traces.clear();
        self.accent_traces.push(AccentTrace {
            envelope: bars.to_vec(),
            age: Duration::from_millis(0),
        });
    }

    fn audio_delay_duration(&self) -> Duration {
        Duration::from_millis(self.config.settings.audio_delay_ms as u64)
    }

    fn start_capture(&mut self) {
        if self.audio.is_some() {
            return;
        }

        self.capture_state = CaptureState::Starting;
        self.status = self.t("starting").to_string();
        self.capture_id = self.capture_id.wrapping_add(1);
        let capture_id = self.capture_id;

        match AudioProcess::spawn(self.tx.clone(), capture_id) {
            Ok(audio) => {
                self.audio = Some(audio);
            }
            Err(error) => {
                self.capture_state = CaptureState::Failed;
                self.status = format!("{}: {}", self.t("capture_failed"), error);
            }
        }
    }

    fn stop_capture(&mut self) {
        self.capture_id = self.capture_id.wrapping_add(1);
        if let Some(mut audio) = self.audio.take() {
            audio.stop();
        }
        self.capture_state = CaptureState::Idle;
        self.status = self.t("stopped").to_string();
        self.delayed_audio.clear();
    }

    fn toggle_capture(&mut self) {
        if self.audio.is_some() {
            self.stop_capture();
        } else {
            self.start_capture();
            self.screen = Screen::Spectrum;
        }
    }

    fn prev_menu(&mut self) {
        self.menu_index = self.menu_index.saturating_sub(1);
    }

    fn next_menu(&mut self) {
        self.menu_index = (self.menu_index + 1).min(MENU_ITEMS.len() - 1);
    }

    fn prev_setting(&mut self) {
        self.setting_index = self.setting_index.saturating_sub(1);
    }

    fn next_setting(&mut self) {
        self.setting_index = (self.setting_index + 1).min(SETTING_COUNT - 1);
    }

    fn set_visual_bar_count(&mut self, bar_count: usize) {
        let bar_count = bar_count.clamp(32, MAX_ANALYSIS_BARS);
        if self.visual_bar_count != bar_count {
            self.visual_bar_count = bar_count;
            self.resize_spectrum_analyzer();
        }
    }

    fn analysis_bar_count(&self) -> usize {
        self.config
            .settings
            .bar_count
            .max(self.visual_bar_count)
            .clamp(32, MAX_ANALYSIS_BARS)
    }

    fn resize_spectrum_analyzer(&mut self) {
        let bar_count = self.analysis_bar_count();
        self.spectrum = display_bars(&self.spectrum, bar_count);
        self.spectrum_trail = display_bars(&self.spectrum_trail, bar_count);
        self.analyzer.resize_bar_count(bar_count);
    }

    fn adjust_setting(&mut self, direction: i32) {
        match self.setting_index {
            0 => self.cycle_language(direction),
            1 => {
                if direction < 0 {
                    self.prev_theme();
                } else {
                    self.next_theme();
                }
            }
            2 => self.cycle_analysis_preset(direction),
            3 => {
                let delta = if direction < 0 { -0.05 } else { 0.05 };
                self.config.settings.attack =
                    (self.config.settings.attack + delta).clamp(0.10, 0.98);
                self.config.settings.mark_custom_analysis();
            }
            4 => {
                let delta = if direction < 0 { -0.05 } else { 0.05 };
                self.config.settings.release =
                    (self.config.settings.release + delta).clamp(0.05, 0.95);
                self.config.settings.mark_custom_analysis();
            }
            5 => {
                let current = self.config.settings.bar_count as i32;
                let next = (current + direction * 8).clamp(32, 120) as usize;
                if next != self.config.settings.bar_count {
                    self.config.settings.bar_count = next;
                    self.resize_spectrum_analyzer();
                }
            }
            6 => self.cycle_renderer(direction),
            7 => {
                self.cycle_fft_size(direction);
                self.config.settings.mark_custom_analysis();
                self.rebuild_analyzer();
            }
            8 => {
                self.cycle_analysis_hop(direction);
                self.config.settings.mark_custom_analysis();
                self.rebuild_analyzer();
            }
            9 => {
                self.cycle_refresh_rate(direction);
                self.config.settings.mark_custom_analysis();
            }
            10 => self.adjust_audio_delay_unsaved(direction),
            11 => {
                self.config.settings.high_shelf_enabled = !self.config.settings.high_shelf_enabled
            }
            12 => {
                self.config.settings.high_shelf_db =
                    (self.config.settings.high_shelf_db + direction as f32).clamp(0.0, 18.0);
            }
            13 => {
                self.config.settings.visual_curve_enabled =
                    !self.config.settings.visual_curve_enabled;
            }
            14 => {
                let delta = if direction < 0 { -0.04 } else { 0.04 };
                self.config.settings.visual_curve =
                    (self.config.settings.visual_curve + delta).clamp(0.55, 1.35);
            }
            15 => {
                self.config.settings.trail_enabled = !self.config.settings.trail_enabled;
                if !self.config.settings.trail_enabled {
                    self.spectrum_trail.clone_from(&self.spectrum);
                }
            }
            16 => {
                let delta = if direction < 0 { -0.03 } else { 0.03 };
                self.config.settings.trail_decay =
                    (self.config.settings.trail_decay + delta).clamp(0.65, 0.98);
            }
            17 => {
                self.config.settings.accent_trace_enabled =
                    !self.config.settings.accent_trace_enabled;
                if !self.config.settings.accent_trace_enabled {
                    self.pending_accent_trace = None;
                    self.accent_traces.clear();
                }
            }
            18 => {
                let delta = if direction < 0 { -0.05 } else { 0.05 };
                self.config.settings.accent_trace_threshold =
                    (self.config.settings.accent_trace_threshold + delta).clamp(0.10, 0.90);
            }
            19 => {
                let delta = if direction < 0 { -0.02 } else { 0.02 };
                self.config.settings.ceiling =
                    (self.config.settings.ceiling + delta).clamp(0.70, 0.98);
            }
            _ => {}
        }
        self.save_config();
    }

    fn adjust_audio_delay(&mut self, direction: i32) {
        self.adjust_audio_delay_unsaved(direction);
        self.save_config();
    }

    fn adjust_audio_delay_unsaved(&mut self, direction: i32) {
        let current = self.config.settings.audio_delay_ms as i32;
        let next = (current + direction * AUDIO_DELAY_STEP_MS).clamp(0, MAX_AUDIO_DELAY_MS) as u16;
        if next != self.config.settings.audio_delay_ms {
            self.config.settings.audio_delay_ms = next;
            self.flush_audio_delay();
        }
    }

    fn cycle_language(&mut self, direction: i32) {
        let index = LANGUAGES
            .iter()
            .position(|(code, _)| *code == self.lang.code())
            .unwrap_or(0);
        let len = LANGUAGES.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.lang = Lang::from_code(LANGUAGES[next].0);
        self.config.settings.language = self.lang.code().to_string();
        self.status = match self.capture_state {
            CaptureState::Idle => self.t("ready").to_string(),
            CaptureState::Starting => self.t("starting").to_string(),
            CaptureState::Running => self.t("running").to_string(),
            CaptureState::PermissionNeeded => self.t("permission_needed").to_string(),
            CaptureState::Failed => self.t("capture_failed").to_string(),
        };
    }

    fn prev_theme(&mut self) {
        let index = THEMES
            .iter()
            .position(|theme| *theme == self.theme_id)
            .unwrap_or(0);
        let next = (index + THEMES.len() - 1) % THEMES.len();
        self.set_theme(THEMES[next]);
    }

    fn next_theme(&mut self) {
        let index = THEMES
            .iter()
            .position(|theme| *theme == self.theme_id)
            .unwrap_or(0);
        let next = (index + 1) % THEMES.len();
        self.set_theme(THEMES[next]);
    }

    fn set_theme(&mut self, theme_id: ThemeId) {
        self.theme_id = theme_id;
        self.config.settings.theme = theme_id;
        self.save_config();
    }

    fn cycle_analysis_preset(&mut self, direction: i32) {
        let current = self.config.settings.analysis_preset;
        let index = ANALYSIS_PRESETS
            .iter()
            .position(|preset| *preset == current)
            .unwrap_or(0);
        let len = ANALYSIS_PRESETS.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config
            .settings
            .apply_analysis_preset(ANALYSIS_PRESETS[next]);
        self.rebuild_analyzer();
    }

    fn cycle_refresh_rate(&mut self, direction: i32) {
        let current = self.config.settings.refresh_hz;
        let index = REFRESH_RATES
            .iter()
            .position(|value| *value == current)
            .unwrap_or_else(|| {
                REFRESH_RATES
                    .iter()
                    .position(|value| *value >= current)
                    .unwrap_or(REFRESH_RATES.len() - 1)
            });
        let len = REFRESH_RATES.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config.settings.refresh_hz = REFRESH_RATES[next];
    }

    fn cycle_fft_size(&mut self, direction: i32) {
        let current = nearest_fft_size(self.config.settings.fft_size);
        let index = FFT_SIZES
            .iter()
            .position(|value| *value == current)
            .unwrap_or(1);
        let len = FFT_SIZES.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config.settings.fft_size = FFT_SIZES[next];
        self.config.settings.analysis_hop = nearest_hop_size(
            self.config.settings.analysis_hop,
            self.config.settings.fft_size,
        );
    }

    fn cycle_analysis_hop(&mut self, direction: i32) {
        let fft_size = nearest_fft_size(self.config.settings.fft_size);
        let hops: Vec<usize> = ANALYSIS_HOPS
            .iter()
            .copied()
            .filter(|hop| *hop <= fft_size)
            .collect();
        let current = nearest_hop_size(self.config.settings.analysis_hop, fft_size);
        let index = hops.iter().position(|value| *value == current).unwrap_or(1);
        let len = hops.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config.settings.analysis_hop = hops[next];
    }

    fn rebuild_analyzer(&mut self) {
        self.config.settings.normalize();
        let fft_size = self.config.settings.fft_size;
        let hop_size = self.config.settings.analysis_hop;
        let bar_count = self.analysis_bar_count();
        self.analyzer = SpectrumAnalyzer::new(fft_size, 48_000.0, bar_count, hop_size);
        self.spectrum = vec![0.0; bar_count];
        self.spectrum_trail = vec![0.0; bar_count];
    }

    fn cycle_renderer(&mut self, direction: i32) {
        let index = SPECTRUM_RENDERERS
            .iter()
            .position(|renderer| *renderer == self.config.settings.renderer)
            .unwrap_or(0);
        let len = SPECTRUM_RENDERERS.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config.settings.renderer = SPECTRUM_RENDERERS[next];
    }

    fn save_config(&self) {
        self.config.save();
    }

    fn toggle_settings_panel(&mut self) {
        self.config.settings.show_settings_panel = !self.config.settings.show_settings_panel;
        self.save_config();
    }

    fn toggle_pipeline_panel(&mut self) {
        self.config.settings.show_pipeline_panel = !self.config.settings.show_pipeline_panel;
        self.save_config();
    }

    fn toggle_toolbar_panel(&mut self) {
        self.config.settings.show_toolbar_panel = !self.config.settings.show_toolbar_panel;
        self.save_config();
    }

    fn toggle_master_panel(&mut self) {
        self.config.settings.show_master_panel = !self.config.settings.show_master_panel;
        self.save_config();
    }

    fn toggle_waveform_panel(&mut self) {
        self.config.settings.show_waveform_panel = !self.config.settings.show_waveform_panel;
        self.save_config();
    }

    fn update_waveform(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let stride = (samples.len() / 256).max(1);
        for chunk in samples.chunks(stride) {
            let sample = chunk.iter().copied().fold(0.0_f32, |current, sample| {
                if sample.abs() > current.abs() {
                    sample
                } else {
                    current
                }
            });
            self.waveform.push(sample.clamp(-1.0, 1.0));
        }

        if self.waveform.len() > WAVEFORM_SAMPLES {
            let excess = self.waveform.len() - WAVEFORM_SAMPLES;
            self.waveform.drain(0..excess);
        }
    }
}

enum AudioEvent {
    Samples(u64, AudioSamples),
    Status(u64, String),
    Exit(u64, Option<i32>),
    Error(u64, String),
}

struct AudioSamples {
    mono: Vec<f32>,
    left_level: f32,
    right_level: f32,
}

struct DelayedAudio {
    received_at: Instant,
    samples: AudioSamples,
}

struct AudioProcess {
    child: Child,
}

impl AudioProcess {
    fn spawn(tx: Sender<AudioEvent>, capture_id: u64) -> io::Result<Self> {
        let helper = helper_path()?;
        let mut child = Command::new(helper)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing helper stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing helper stderr"))?;

        let stdout_tx = tx.clone();
        thread::spawn(move || read_audio_stdout(stdout, stdout_tx, capture_id));

        let stderr_tx = tx.clone();
        thread::spawn(move || read_helper_stderr(stderr, stderr_tx, capture_id));

        Ok(Self { child })
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for AudioProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn helper_path() -> io::Result<PathBuf> {
    option_env!("TERB_AUDIO_HELPER")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "macOS audio helper is unavailable; rebuild with `cargo build` on macOS",
            )
        })
}

fn read_audio_stdout(mut stdout: impl Read, tx: Sender<AudioEvent>, capture_id: u64) {
    let mut buffer = vec![0_u8; AUDIO_READ_FRAMES * 8];
    let mut pending = Vec::new();
    loop {
        match stdout.read(&mut buffer) {
            Ok(0) => break,
            Ok(size) => {
                pending.extend_from_slice(&buffer[..size]);
                let frame_count = pending.len() / 8;
                if frame_count == 0 {
                    continue;
                }

                let bytes_to_read = frame_count * 8;
                let mut mono = Vec::with_capacity(frame_count);
                let mut left = Vec::with_capacity(frame_count);
                let mut right = Vec::with_capacity(frame_count);

                for chunk in pending[..bytes_to_read].chunks_exact(8) {
                    let left_sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let right_sample = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    left.push(left_sample);
                    right.push(right_sample);
                    mono.push((left_sample + right_sample) * 0.5);
                }

                pending.drain(0..bytes_to_read);
                let samples = AudioSamples {
                    left_level: audio_level(&left),
                    right_level: audio_level(&right),
                    mono,
                };

                if tx.send(AudioEvent::Samples(capture_id, samples)).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = tx.send(AudioEvent::Error(capture_id, error.to_string()));
                break;
            }
        }
    }
    let _ = tx.send(AudioEvent::Exit(capture_id, None));
}

fn read_helper_stderr(stderr: impl Read, tx: Sender<AudioEvent>, capture_id: u64) {
    let reader = BufReader::new(stderr);
    for line in reader.lines().map_while(Result::ok) {
        let _ = tx.send(AudioEvent::Status(capture_id, line));
    }
}

struct SpectrumAnalyzer {
    fft_size: usize,
    hop_size: usize,
    sample_rate: f32,
    bar_count: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    sample_buffer: Vec<f32>,
    smoothed: Vec<f32>,
    samples_since_analysis: usize,
    has_analysis: bool,
}

#[derive(Clone, Copy)]
struct SpectrumPipeline {
    high_shelf_enabled: bool,
    high_shelf_db: f32,
    ceiling: f32,
}

impl SpectrumPipeline {
    fn from_settings(settings: &Settings) -> Self {
        Self {
            high_shelf_enabled: settings.high_shelf_enabled,
            high_shelf_db: settings.high_shelf_db,
            ceiling: settings.ceiling,
        }
    }
}

impl SpectrumAnalyzer {
    fn new(fft_size: usize, sample_rate: f32, bar_count: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let hop_size = nearest_hop_size(hop_size, fft_size);
        let window: Vec<f32> = (0..fft_size)
            .map(|index| {
                let position = index as f32 / (fft_size - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * position).cos()
            })
            .collect();
        let window_sum = window.iter().sum::<f32>().max(1.0);

        Self {
            fft_size,
            hop_size,
            sample_rate,
            bar_count,
            fft,
            window,
            window_sum,
            sample_buffer: Vec::new(),
            smoothed: vec![0.0; bar_count],
            samples_since_analysis: 0,
            has_analysis: false,
        }
    }

    fn resize_bar_count(&mut self, bar_count: usize) {
        if self.bar_count == bar_count {
            return;
        }

        self.bar_count = bar_count;
        self.smoothed = display_bars(&self.smoothed, bar_count);
    }

    fn consume(
        &mut self,
        samples: &[f32],
        attack: f32,
        release: f32,
        pipeline: SpectrumPipeline,
    ) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }

        self.sample_buffer.extend_from_slice(samples);
        self.samples_since_analysis = self.samples_since_analysis.saturating_add(samples.len());
        let max_samples = self
            .fft_size
            .saturating_add(self.hop_size)
            .max(self.fft_size * 2);
        if self.sample_buffer.len() > max_samples {
            let excess = self.sample_buffer.len() - max_samples;
            self.sample_buffer.drain(0..excess);
        }

        if self.sample_buffer.len() < self.fft_size {
            return None;
        }

        if self.has_analysis && self.samples_since_analysis < self.hop_size {
            return None;
        }
        self.samples_since_analysis %= self.hop_size;
        self.has_analysis = true;

        let source = &self.sample_buffer[self.sample_buffer.len() - self.fft_size..];
        let source_rms =
            (source.iter().map(|sample| sample * sample).sum::<f32>() / source.len() as f32).sqrt();
        if source_rms < SILENCE_GATE {
            self.smoothed.fill(0.0);
            return Some(self.smoothed.clone());
        }

        let mean = source.iter().sum::<f32>() / source.len() as f32;
        let mut buffer: Vec<Complex<f32>> = source
            .iter()
            .zip(self.window.iter())
            .map(|(sample, window)| Complex::new((sample - mean) * window, 0.0))
            .collect();

        self.fft.process(&mut buffer);

        let half = self.fft_size / 2;
        let amplitude_scale = 2.0 / self.window_sum;
        let mut magnitudes = vec![0.0_f32; half];
        for index in 1..half {
            magnitudes[index] = buffer[index].norm() * amplitude_scale;
        }

        let bars = self.make_bars(&magnitudes, pipeline);
        let attack = attack.clamp(0.10, 0.98);
        let release = release.clamp(0.05, 0.95);

        for (smoothed, target) in self.smoothed.iter_mut().zip(bars.into_iter()) {
            if target > *smoothed {
                *smoothed = *smoothed * (1.0 - attack) + target * attack;
            } else {
                *smoothed = *smoothed * release + target * (1.0 - release);
            }
            if *smoothed < 0.002 {
                *smoothed = 0.0;
            } else {
                *smoothed = smoothed.clamp(0.0, pipeline.ceiling);
            }
        }

        Some(self.smoothed.clone())
    }

    fn make_bars(&self, magnitudes: &[f32], pipeline: SpectrumPipeline) -> Vec<f32> {
        let max_frequency = (self.sample_rate / 2.0).min(MAX_FREQUENCY);
        let ratio = max_frequency / MIN_FREQUENCY;
        let mut bars = vec![0.0; self.bar_count];

        for (bar, value) in bars.iter_mut().enumerate() {
            let lower_t = bar as f32 / self.bar_count as f32;
            let upper_t = (bar + 1) as f32 / self.bar_count as f32;
            let center_t = (lower_t + upper_t) * 0.5;
            let lower_frequency = MIN_FREQUENCY * ratio.powf(lower_t);
            let upper_frequency = MIN_FREQUENCY * ratio.powf(upper_t);
            let lower_bin = ((lower_frequency / self.sample_rate) * self.fft_size as f32).max(1.0);
            let upper_bin =
                ((upper_frequency / self.sample_rate) * self.fft_size as f32).max(lower_bin + 0.25);
            let band = sample_frequency_band(magnitudes, lower_bin, upper_bin);
            let energy = (band.rms * 0.72 + band.peak * 0.28).max(band.average);
            let high_shelf = if pipeline.high_shelf_enabled {
                pipeline.high_shelf_db * center_t.powf(1.35)
            } else {
                0.0
            };
            let low_trim = 2.5 * (1.0 - center_t / 0.18).clamp(0.0, 1.0);
            let db = 20.0 * energy.max(0.000_000_1).log10() + high_shelf - low_trim;
            let normalized = ((db + 82.0) / 72.0).clamp(0.0, 1.0);
            *value = (normalized * pipeline.ceiling).min(pipeline.ceiling);
        }

        bars
    }
}

struct BandStats {
    average: f32,
    rms: f32,
    peak: f32,
}

fn sample_frequency_band(magnitudes: &[f32], lower_bin: f32, upper_bin: f32) -> BandStats {
    if magnitudes.is_empty() {
        return BandStats {
            average: 0.0,
            rms: 0.0,
            peak: 0.0,
        };
    }

    let max_bin = (magnitudes.len() - 1) as f32;
    let lower_bin = lower_bin.clamp(1.0, max_bin);
    let upper_bin = upper_bin.clamp(lower_bin, max_bin);
    let width = (upper_bin - lower_bin).max(0.001);
    let sample_count = ((width.ceil() as usize) + 2).clamp(4, 64);
    let mut total = 0.0_f32;
    let mut squared_total = 0.0_f32;
    let mut weight_total = 0.0_f32;
    let mut peak = 0.0_f32;

    for sample in 0..sample_count {
        let position = (sample as f32 + 0.5) / sample_count as f32;
        let bin = lower_bin + width * position;
        let magnitude = sample_magnitude(magnitudes, bin);
        let weight = 1.0 + position;
        total += magnitude * weight;
        squared_total += magnitude * magnitude * weight;
        weight_total += weight;
        peak = peak.max(magnitude);
    }

    BandStats {
        average: total / weight_total.max(1.0),
        rms: (squared_total / weight_total.max(1.0)).sqrt(),
        peak,
    }
}

fn sample_magnitude(magnitudes: &[f32], bin: f32) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }

    let max_index = magnitudes.len() - 1;
    let bin = bin.clamp(0.0, max_index as f32);
    let left = bin.floor() as usize;
    let right = bin.ceil() as usize;
    let mix = bin - left as f32;
    let left_value = magnitudes[left];
    let right_value = magnitudes[right.min(max_index)];
    left_value * (1.0 - mix) + right_value * mix
}

fn audio_level(samples: &[f32]) -> f32 {
    let count = samples.len().min(4096);
    if count == 0 {
        return 0.0;
    }
    let square_sum = samples
        .iter()
        .rev()
        .take(count)
        .map(|sample| sample * sample)
        .sum::<f32>();
    let rms = (square_sum / count as f32).sqrt();
    let db = 20.0 * rms.max(0.000_001).log10();
    ((db + 60.0) / 54.0).clamp(0.0, 1.0).powf(0.85)
}

#[derive(Clone, Copy)]
struct Theme {
    title_key: &'static str,
    accent: Color,
    text: Color,
    muted: Color,
    border: Color,
    low: Color,
    mid: Color,
    high: Color,
    color_mode: ColorMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorMode {
    Static,
    Aurora,
    SonicTexture,
    NoiseWarp,
}

fn theme(id: ThemeId) -> Theme {
    match id {
        ThemeId::System => Theme {
            title_key: "theme_system",
            accent: Color::Blue,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Blue,
            mid: Color::Cyan,
            high: Color::LightBlue,
            color_mode: ColorMode::Static,
        },
        ThemeId::Graphite => Theme {
            title_key: "theme_graphite",
            accent: Color::Gray,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::DarkGray,
            mid: Color::Gray,
            high: Color::White,
            color_mode: ColorMode::Static,
        },
        ThemeId::Ocean => Theme {
            title_key: "theme_ocean",
            accent: Color::Cyan,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Blue,
            mid: Color::Cyan,
            high: Color::White,
            color_mode: ColorMode::Static,
        },
        ThemeId::Aurora => Theme {
            title_key: "theme_aurora",
            accent: Color::Cyan,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Blue,
            mid: Color::Magenta,
            high: Color::White,
            color_mode: ColorMode::Aurora,
        },
        ThemeId::PitchClass
        | ThemeId::ChromaBands
        | ThemeId::PitchMemory
        | ThemeId::HarmonicComb
        | ThemeId::SonicTexture => Theme {
            title_key: "theme_sonic_texture",
            accent: Color::LightCyan,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Blue,
            mid: Color::LightMagenta,
            high: Color::White,
            color_mode: ColorMode::SonicTexture,
        },
        ThemeId::NoiseWarp => Theme {
            title_key: "theme_noise_warp",
            accent: Color::LightMagenta,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Magenta,
            mid: Color::LightBlue,
            high: Color::White,
            color_mode: ColorMode::NoiseWarp,
        },
        ThemeId::Amber => Theme {
            title_key: "theme_amber",
            accent: Color::Yellow,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::DarkGray,
            mid: Color::Yellow,
            high: Color::White,
            color_mode: ColorMode::Static,
        },
        ThemeId::Mono => Theme {
            title_key: "theme_mono",
            accent: Color::White,
            text: Color::Gray,
            muted: Color::DarkGray,
            border: Color::DarkGray,
            low: Color::Gray,
            mid: Color::Gray,
            high: Color::White,
            color_mode: ColorMode::Static,
        },
    }
}

fn bar_color(theme: Theme, index: usize, len: usize, value: f32) -> Color {
    if value > 0.90 {
        return theme.high;
    }

    let ratio = index as f32 / len.max(1) as f32;
    if ratio < 0.45 {
        theme.low
    } else {
        theme.mid
    }
}

fn spectrum_bar_color(app: &App, index: usize, len: usize, value: f32, trail: bool) -> Color {
    spectrum_bar_color_at(app, index, len, value, value, trail)
}

fn spectrum_bar_color_at(
    app: &App,
    index: usize,
    len: usize,
    value: f32,
    height_ratio: f32,
    trail: bool,
) -> Color {
    let theme = app.theme();
    if theme.color_mode == ColorMode::Static {
        if trail {
            theme.border
        } else {
            bar_color(theme, index, len, value)
        }
    } else {
        music_color_for_position_at(
            app,
            spectrum_position(index, len),
            value,
            height_ratio,
            trail,
        )
    }
}

fn waveform_color(app: &App, value: f32) -> Color {
    let theme = app.theme();
    if theme.color_mode == ColorMode::Static {
        meter_color(theme, value)
    } else {
        music_color_for_position_at(app, 0.72, value, value, false)
    }
}

fn draw(frame: &mut Frame, app: &App) {
    let size = frame.area();
    if size.width < 36 || size.height < 10 {
        let text = Paragraph::new(app.t("too_small"))
            .alignment(Alignment::Center)
            .style(Style::default().fg(app.theme().muted));
        frame.render_widget(text, size);
        return;
    }

    match app.screen {
        Screen::Menu => draw_menu(frame, app, size),
        Screen::Spectrum => draw_spectrum_screen(frame, app, size),
        Screen::Settings => draw_settings(frame, app, size),
        Screen::Help => draw_help(frame, app, size),
    }
}

fn visual_bar_count(app: &App, area: Rect) -> Option<usize> {
    if app.screen != Screen::Spectrum || area.width < 36 || area.height < 10 {
        return None;
    }

    let chart_area = spectrum_visual_area(app, area);
    let inner_width = chart_area.width.saturating_sub(2) as usize;
    if inner_width == 0 {
        return None;
    }

    Some(match app.config.settings.renderer {
        SpectrumRenderer::Blocks => inner_width,
        SpectrumRenderer::Braille => inner_width * 2,
    })
}

fn spectrum_visual_area(app: &App, area: Rect) -> Rect {
    let settings = &app.config.settings;
    let left_height = left_module_height(settings);
    let show_left_modules = left_height > 0 && area.width >= 96 && area.height >= left_height;
    let mut content_area = area;

    if show_left_modules {
        content_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(35), Constraint::Min(28)])
            .split(area)[1];
    }

    let show_master =
        settings.show_master_panel && content_area.width >= 63 && content_area.height >= 14;
    let mut visual_area = content_area;
    if show_master {
        visual_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(24), Constraint::Length(16)])
            .split(content_area)[0];
    }

    let show_compact_footer = !show_left_modules && visual_area.height >= 9;
    let mut chart_area = if show_compact_footer {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(3)])
            .split(visual_area)[0]
    } else {
        visual_area
    };

    if settings.show_waveform_panel && chart_area.width >= 28 && chart_area.height >= 16 {
        chart_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(6)])
            .split(chart_area)[1];
    }

    chart_area
}

fn draw_menu(frame: &mut Frame, app: &App, area: Rect) {
    if area.width < 62 || area.height < 22 {
        draw_compact_menu(frame, app, area);
        return;
    }

    let theme = app.theme();
    let panel = centered_rect(74, 82, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(TITLE_ART.len() as u16 + 2),
            Constraint::Length(9),
            Constraint::Min(3),
            Constraint::Length(4),
        ])
        .split(panel);

    let art_lines: Vec<Line> = TITLE_ART
        .iter()
        .map(|line| {
            Line::from(Span::styled(
                *line,
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ))
        })
        .collect();
    let art_inner = draw_panel(frame, rows[0], theme, Some(panel_title("terb", theme)));
    frame.render_widget(
        Paragraph::new(art_lines).alignment(Alignment::Center),
        art_inner,
    );

    draw_main_menu(frame, app, rows[1]);

    let status_inner = draw_panel(
        frame,
        rows[2],
        theme,
        Some(panel_title(app.t("overview"), theme)),
    );
    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                capture_control_label(app),
                Style::default().fg(theme.accent),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{} {:>3}%", app.t("level"), (app.level * 100.0) as u16),
                Style::default().fg(theme.text),
            ),
            Span::raw("  "),
            Span::styled(
                format!(
                    "{} {}Hz",
                    app.t("refresh_rate"),
                    app.config.settings.refresh_hz
                ),
                Style::default().fg(theme.muted),
            ),
        ]),
        Line::from(Span::styled(&app.status, Style::default().fg(theme.muted))),
    ])
    .wrap(Wrap { trim: true });
    frame.render_widget(status, status_inner);

    let footer = Paragraph::new(app.t("menu_hint"))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(theme.muted));
    frame.render_widget(footer, rows[3]);
}

fn draw_compact_menu(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let panel = centered_rect(92, 90, area);
    let mut lines = Vec::new();

    if panel.height >= 15 {
        for line in TITLE_ART {
            lines.push(Line::from(Span::styled(
                *line,
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        push_compact_menu_items(app, theme, &mut lines);
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(
                "{} {}Hz · {} {:>3}%",
                app.t("refresh_rate"),
                app.config.settings.refresh_hz,
                app.t("level"),
                (app.level * 100.0) as u16
            ),
            Style::default().fg(theme.muted),
        )));
    } else if panel.height >= 12 {
        lines.push(Line::from(Span::styled(
            "terb",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        push_compact_menu_items(app, theme, &mut lines);
        lines.push(Line::from(Span::styled(
            "Enter · Space · q",
            Style::default().fg(theme.muted),
        )));
    } else {
        let label = compact_menu_label(app, MENU_ITEMS[app.menu_index]);
        lines.push(Line::from(Span::styled(
            "terb",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!("> {}", app.t(label)),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "j/k Enter q",
            Style::default().fg(theme.muted),
        )));
    }

    let inner = draw_panel(
        frame,
        panel,
        theme,
        Some(panel_title(app.t("main_menu"), theme)),
    );
    frame.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        inner,
    );
}

fn push_compact_menu_items(app: &App, theme: Theme, lines: &mut Vec<Line<'static>>) {
    for (index, key) in MENU_ITEMS.iter().enumerate() {
        let label = compact_menu_label(app, key);
        let selected = index == app.menu_index;
        let prefix = if selected { "> " } else { "  " };
        let style = if selected {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.text)
        };
        lines.push(Line::from(Span::styled(
            format!("{}{}", prefix, app.t(label)),
            style,
        )));
    }
}

fn compact_menu_label(app: &App, key: &'static str) -> &'static str {
    if key == "menu_toggle" {
        if app.audio.is_some() {
            "menu_stop"
        } else {
            "menu_start"
        }
    } else {
        key
    }
}

fn draw_main_menu(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let inner = draw_panel(
        frame,
        area,
        theme,
        Some(panel_title(app.t("main_menu"), theme)),
    );
    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .map(|key| {
            let label_key = compact_menu_label(app, key);
            ListItem::new(Line::from(Span::styled(
                app.t(label_key),
                Style::default().fg(theme.text),
            )))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.menu_index));

    let list = List::new(items).highlight_symbol("  ").highlight_style(
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );

    frame.render_stateful_widget(list, inner, &mut state);
}

fn draw_spectrum_screen(frame: &mut Frame, app: &App, area: Rect) {
    let settings = &app.config.settings;
    let left_height = left_module_height(settings);
    let show_left_modules = left_height > 0 && area.width >= 96 && area.height >= left_height;
    let mut content_area = area;

    if show_left_modules {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(35), Constraint::Min(28)])
            .split(area);
        draw_left_modules(frame, app, chunks[0]);
        content_area = chunks[1];
    }

    let show_master =
        settings.show_master_panel && content_area.width >= 63 && content_area.height >= 14;
    let mut visual_area = content_area;
    if show_master {
        let meter_width = 16;
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(24), Constraint::Length(meter_width)])
            .split(content_area);
        visual_area = chunks[0];
        draw_master_meter(frame, app, chunks[1]);
    }

    let show_compact_footer = !show_left_modules && visual_area.height >= 9;
    let chart_area = if show_compact_footer {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(3)])
            .split(visual_area);
        draw_compact_footer(frame, app, rows[1]);
        rows[0]
    } else {
        visual_area
    };

    if settings.show_waveform_panel && chart_area.width >= 28 && chart_area.height >= 16 {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(6)])
            .split(chart_area);
        draw_waveform(frame, app, rows[0]);
        draw_spectrum(frame, app, rows[1], app.t("spectrum"));
    } else {
        draw_spectrum(frame, app, chart_area, app.t("spectrum"));
    }
}

fn left_module_height(settings: &Settings) -> u16 {
    let mut height = 0;
    if settings.show_toolbar_panel {
        height += 7;
    }
    if settings.show_settings_panel {
        height += 22;
    }
    if settings.show_pipeline_panel {
        height += 8;
    }
    height
}

fn draw_left_modules(frame: &mut Frame, app: &App, area: Rect) {
    let settings = &app.config.settings;
    let mut constraints = Vec::new();

    if settings.show_toolbar_panel {
        constraints.push(Constraint::Length(7));
    }
    if settings.show_settings_panel {
        constraints.push(Constraint::Length(22));
    }
    if settings.show_pipeline_panel {
        constraints.push(Constraint::Length(8));
    }
    if constraints.is_empty() {
        return;
    }
    constraints.push(Constraint::Min(0));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    let mut index = 0;

    if settings.show_toolbar_panel {
        draw_toolbar(frame, app, chunks[index]);
        index += 1;
    }
    if settings.show_settings_panel {
        draw_settings_list(
            frame,
            app,
            chunks[index],
            module_title_line(app, 's', "settings"),
        );
        index += 1;
    }
    if settings.show_pipeline_panel {
        draw_pipeline(frame, app, chunks[index]);
    }
}

fn draw_toolbar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let inner = draw_panel(
        frame,
        area,
        theme,
        Some(module_title_line(app, 't', "toolbar")),
    );
    let lines = vec![
        Line::from(vec![
            Span::styled(
                capture_control_label(app),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("   "),
            Span::styled(
                format!(
                    "L{:>3} R{:>3}",
                    (app.master_left * 100.0) as u16,
                    (app.master_right * 100.0) as u16
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            Span::styled(app.t("theme"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(theme_label(app), Style::default().fg(theme.muted)),
        ]),
        Line::from(vec![
            Span::styled(app.t("refresh_rate"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(
                format!("{}Hz", app.config.settings.refresh_hz),
                Style::default().fg(theme.text),
            ),
            Span::raw("  "),
            Span::styled(app.t("audio_delay"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(
                format!("{}ms", app.config.settings.audio_delay_ms),
                Style::default().fg(theme.text),
            ),
        ]),
        module_toggle_line(app),
        Line::from(vec![
            Span::styled("S", Style::default().fg(theme.accent)),
            Span::styled(
                format!(" {}  ", app.t("settings")),
                Style::default().fg(theme.muted),
            ),
            Span::styled("?", Style::default().fg(theme.accent)),
            Span::styled(
                format!(" {}  ", app.t("help")),
                Style::default().fg(theme.muted),
            ),
            Span::styled("q", Style::default().fg(theme.accent)),
            Span::styled(
                format!(" {}", app.t("main_menu")),
                Style::default().fg(theme.muted),
            ),
        ]),
        Line::from(Span::styled(&app.status, Style::default().fg(theme.muted))),
    ];

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn module_toggle_line(app: &App) -> Line<'static> {
    let theme = app.theme();
    Line::from(vec![
        Span::styled(
            module_toggle_keys(app),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {}", app.t("modules")),
            Style::default().fg(theme.muted),
        ),
    ])
}

fn module_toggle_keys(app: &App) -> String {
    if app.lang == Lang::En {
        "s p t m w".to_string()
    } else {
        "[s] [p] [t] [m] [w]".to_string()
    }
}

fn draw_master_meter(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let inner = draw_panel(
        frame,
        area,
        theme,
        Some(module_title_line(app, 'm', "master")),
    );

    if inner.width < 5 || inner.height < 5 {
        return;
    }

    let chart_height = inner.height.saturating_sub(1) as usize;
    let available_width = inner.width as usize;
    let left_width = ((available_width.saturating_sub(1)) / 2).max(1);
    let right_width = available_width.saturating_sub(left_width + 1).max(1);
    let virtual_height = chart_height * 4;
    let left = app.master_left.clamp(0.0, 1.0);
    let right = app.master_right.clamp(0.0, 1.0);
    let mut lines = Vec::with_capacity(chart_height + 1);

    for row in 0..chart_height {
        let mut spans = Vec::with_capacity(left_width + right_width + 1);
        for col in 0..left_width {
            let (mask, value) = master_braille_cell(left, col, row, virtual_height);
            spans.push(Span::styled(
                master_braille_symbol(mask),
                Style::default().fg(if mask == 0 {
                    Color::Reset
                } else {
                    master_meter_color(app, value)
                }),
            ));
        }
        spans.push(Span::raw(" "));
        for col in 0..right_width {
            let (mask, value) = master_braille_cell(right, col, row, virtual_height);
            spans.push(Span::styled(
                master_braille_symbol(mask),
                Style::default().fg(if mask == 0 {
                    Color::Reset
                } else {
                    master_meter_color(app, value)
                }),
            ));
        }

        lines.push(Line::from(spans));
    }

    lines.push(Line::from(vec![
        Span::styled(
            format!("L{:>3}", (left * 100.0) as u16),
            Style::default().fg(theme.muted),
        ),
        Span::raw(" "),
        Span::styled(
            format!("R{:>3}", (right * 100.0) as u16),
            Style::default().fg(theme.muted),
        ),
    ]));

    frame.render_widget(Paragraph::new(lines), inner);
}

fn meter_color(theme: Theme, value: f32) -> Color {
    if value > 0.82 {
        theme.high
    } else if value > 0.52 {
        theme.mid
    } else {
        theme.low
    }
}

fn master_braille_cell(
    value: f32,
    _cell_col: usize,
    cell_row: usize,
    virtual_height: usize,
) -> (u8, f32) {
    let value = value.clamp(0.0, 1.0);
    let mut mask = 0;
    let mut cell_value = 0.0_f32;

    for dot_col in 0..2 {
        for dot_row in 0..4 {
            let virtual_row = cell_row * 4 + dot_row;
            let threshold = 1.0 - (virtual_row as f32 + 0.5) / virtual_height.max(1) as f32;
            if value < threshold {
                continue;
            }

            mask |= BRAILLE_DOT_BITS[dot_col][dot_row];
            cell_value = cell_value.max(threshold);
        }
    }

    (mask, cell_value)
}

fn master_meter_color(app: &App, height_ratio: f32) -> Color {
    let theme = app.theme();
    if theme.color_mode != ColorMode::Static {
        let glow = music_color_for_position(app, 0.88, 0.75, false);
        return blend_color(theme.border, glow, smoothstep(height_ratio.clamp(0.0, 1.0)));
    }

    let fade = smoothstep(height_ratio.clamp(0.0, 1.0));
    let from = color_to_rgb(theme.border);
    let to = color_to_rgb(theme.accent);
    let blend = |start: u8, end: u8| {
        (start as f32 + (end as f32 - start as f32) * fade)
            .round()
            .clamp(0.0, 255.0) as u8
    };

    Color::Rgb(
        blend(from.0, to.0),
        blend(from.1, to.1),
        blend(from.2, to.2),
    )
}

fn smoothstep(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn color_to_rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Black => (0, 0, 0),
        Color::Red => (205, 49, 49),
        Color::Green => (13, 188, 121),
        Color::Yellow => (229, 229, 16),
        Color::Blue => (36, 114, 200),
        Color::Magenta => (188, 63, 188),
        Color::Cyan => (17, 168, 205),
        Color::Gray => (160, 160, 160),
        Color::DarkGray => (82, 88, 100),
        Color::LightRed => (241, 76, 76),
        Color::LightGreen => (35, 209, 139),
        Color::LightYellow => (245, 245, 67),
        Color::LightBlue => (59, 142, 234),
        Color::LightMagenta => (214, 112, 214),
        Color::LightCyan => (41, 184, 219),
        Color::White => (229, 229, 229),
        Color::Rgb(red, green, blue) => (red, green, blue),
        Color::Indexed(index) if index >= 232 => {
            let level = 8 + (index - 232) * 10;
            (level, level, level)
        }
        Color::Indexed(_) | Color::Reset => (82, 88, 100),
    }
}

fn spectrum_position(index: usize, len: usize) -> f32 {
    if len <= 1 {
        0.0
    } else {
        index as f32 / (len - 1) as f32
    }
}

fn music_color_for_position(app: &App, position: f32, intensity: f32, trail: bool) -> Color {
    music_color_for_position_at(app, position, intensity, intensity, trail)
}

fn music_color_for_position_at(
    app: &App,
    position: f32,
    intensity: f32,
    height_ratio: f32,
    trail: bool,
) -> Color {
    let theme = app.theme();
    let state = app.color_state;
    let position = position.clamp(0.0, 1.0);
    let height_ratio = height_ratio.clamp(0.0, 1.0);

    match theme.color_mode {
        ColorMode::Static => {
            if trail {
                theme.border
            } else {
                bar_color(theme, (position * 100.0) as usize, 100, intensity)
            }
        }
        ColorMode::Aurora => aurora_color(
            theme,
            state,
            position,
            (intensity * 0.80 + height_ratio * 0.20).clamp(0.0, 1.0),
            height_ratio * 0.035 + if trail { 0.18 } else { 0.0 },
            trail,
        ),
        ColorMode::SonicTexture => {
            sonic_texture_theme_color(theme, state, position, intensity, height_ratio, trail)
        }
        ColorMode::NoiseWarp => {
            noise_warp_theme_color(theme, state, position, intensity, height_ratio, trail)
        }
    }
}

fn sonic_texture_theme_color(
    theme: Theme,
    state: VisualColorState,
    position: f32,
    intensity: f32,
    height_ratio: f32,
    trail: bool,
) -> Color {
    let dominant = state.dominant_pitch;
    let confidence = state.pitch_confidence.clamp(0.0, 1.0);
    let (left_pitch, right_pitch, mix) = regional_pitch_pair(position, 4);
    let pitch_seed = dominant as f32 * 0.055 + state.pitch_hue;
    let frequency_scale = lerp(1.8, 4.8, smoothstep(position));
    let x = position * frequency_scale + state.centroid * 0.55 + pitch_seed * 0.35;
    let y = height_ratio * (2.35 + state.energy * 1.45) + state.phase * 1.35;
    let cloud = fbm_2d(x * 1.35 + pitch_seed, y * 1.10 - pitch_seed, 0.19, 4);
    let flow = fbm_2d(
        x * 0.85 + cloud * 0.70,
        y * 1.45 + state.flux * 1.20,
        0.47 + dominant as f32 * 0.03,
        3,
    );
    let contour_phase =
        x * 0.65 + y * (1.05 + confidence * 0.55) + flow * 1.35 + pitch_seed + state.phase * 0.85;
    let contour = 0.5 + 0.5 * (contour_phase * std::f32::consts::TAU).sin();
    let surface = (1.0 - (intensity - height_ratio).abs() * 2.2).clamp(0.0, 1.0);
    let texture = (cloud * 0.42 + flow * 0.32 + contour * 0.26).clamp(0.0, 1.0);
    let texture = ((texture - 0.5) * 1.35 + 0.5).clamp(0.0, 1.0);
    let hue_shift =
        (texture - 0.5) * 0.125 + (surface - 0.5) * 0.045 + (state.centroid - 0.5) * 0.040;
    let left = pitch_color(
        theme,
        left_pitch,
        (intensity * (0.58 + state.chroma[left_pitch] * 0.18)).clamp(0.0, 1.0),
        hue_shift,
        height_ratio,
    );
    let right = pitch_color(
        theme,
        right_pitch,
        (intensity * (0.58 + state.chroma[right_pitch] * 0.18)).clamp(0.0, 1.0),
        hue_shift + 0.018,
        height_ratio,
    );
    let regional = blend_color(left, right, mix);
    let anchor = pitch_color(
        theme,
        dominant,
        (intensity * 0.70 + confidence * 0.20 + texture * 0.10).clamp(0.0, 1.0),
        hue_shift * 0.55,
        height_ratio,
    );
    let pattern_hue = wrap_unit(state.pitch_hue + hue_shift + contour * 0.055);
    let pattern = hsv_to_rgb(
        pattern_hue,
        (0.52 + confidence * 0.20 + flow * 0.22).clamp(0.42, 0.94),
        (0.24 + intensity * 0.40 + height_ratio * 0.10 + texture * 0.28 + surface * 0.15)
            .clamp(0.14, 1.0),
    );
    let color = blend_color(regional, anchor, 0.30 + confidence * 0.18);
    let color = blend_color(color, pattern, 0.32 + texture * 0.26);
    finish_music_color(
        theme,
        color,
        (intensity + texture * 0.22 + surface * 0.14).clamp(0.0, 1.0),
        height_ratio,
        trail,
    )
}

fn noise_warp_theme_color(
    theme: Theme,
    state: VisualColorState,
    position: f32,
    intensity: f32,
    height_ratio: f32,
    trail: bool,
) -> Color {
    let base_hue = rgb_hue(color_to_rgb(theme.accent));
    let coarse_x = position * (2.15 + state.centroid * 1.10);
    let coarse_y = height_ratio * (2.75 + state.energy * 1.75);
    let drift = state.phase * (1.10 + state.flux * 1.80);
    let warp_x = fbm_2d(coarse_x + 3.11, coarse_y - drift + 7.70, 0.61, 3) - 0.5;
    let warp_y = fbm_2d(coarse_x - 5.30, coarse_y + drift + 2.43, 0.83, 3) - 0.5;
    let warp_strength = 0.24 + state.energy * 0.42 + state.flux * 0.75;
    let warped_x = coarse_x + warp_x * warp_strength;
    let warped_y = coarse_y + warp_y * warp_strength;
    let field = fbm_2d(warped_x * 1.25, warped_y * 1.18, 0.29, 5);
    let ridges = (1.0 - (field * 2.0 - 1.0).abs()).powf(1.18);
    let veins = 0.5
        + 0.5
            * ((warped_x * 0.75 + warped_y * 1.45 + field * 1.90 + drift) * std::f32::consts::TAU)
                .sin();
    let texture = (field * 0.48 + ridges * 0.32 + veins * 0.20).clamp(0.0, 1.0);
    let texture = ((texture - 0.5) * 1.45 + 0.5).clamp(0.0, 1.0);
    let surface = (1.0 - (intensity - height_ratio).abs() * 1.8).clamp(0.0, 1.0);
    let hue = wrap_unit(
        base_hue
            + (state.centroid - 0.5) * 0.13
            + position * 0.12
            + (field - 0.5) * 0.30
            + (veins - 0.5) * 0.090,
    );
    let saturation =
        (0.48 + texture * 0.36 + state.energy * 0.14 + state.flux * 0.10).clamp(0.38, 0.96);
    let value = (0.20
        + intensity * 0.40
        + height_ratio * 0.09
        + texture * 0.28
        + surface * 0.16
        + state.flux * 0.08)
        .clamp(0.12, 1.0);
    let color = hsv_to_rgb(hue, saturation, value);
    let accent = aurora_color(
        theme,
        state,
        position * 0.45 + texture * 0.35,
        intensity,
        (field - 0.5) * 0.08,
        false,
    );
    let color = blend_color(color, accent, 0.14 + ridges * 0.14);
    finish_music_color(
        theme,
        color,
        (intensity + texture * 0.22 + surface * 0.12).clamp(0.0, 1.0),
        height_ratio,
        trail,
    )
}

fn finish_music_color(
    theme: Theme,
    color: Color,
    intensity: f32,
    height_ratio: f32,
    trail: bool,
) -> Color {
    let intensity = (intensity * 0.82 + height_ratio * 0.18).clamp(0.0, 1.0);
    let color = blend_color(color, theme.high, (intensity - 0.88).max(0.0) * 1.5);
    if trail {
        blend_color(theme.border, color, 0.46 + intensity * 0.20)
    } else {
        blend_color(theme.border, color, 0.30 + intensity * 0.62)
    }
}

fn pitch_color(
    theme: Theme,
    pitch_class: usize,
    intensity: f32,
    hue_shift: f32,
    height_ratio: f32,
) -> Color {
    let hue = wrap_unit(pitch_class_hue(pitch_class) + hue_shift);
    let saturation = (0.50 + intensity * 0.20 + height_ratio * 0.08).clamp(0.38, 0.82);
    let value = (0.28 + intensity * 0.48 + height_ratio * 0.18).clamp(0.18, 0.96);
    blend_color(hsv_to_rgb(hue, saturation, value), theme.accent, 0.16)
}

fn regional_pitch_pair(position: f32, regions: usize) -> (usize, usize, f32) {
    let regions = regions.max(2);
    let scaled = position.clamp(0.0, 1.0) * (regions - 1) as f32;
    let left_region = scaled.floor() as usize;
    let right_region = scaled.ceil().min((regions - 1) as f32) as usize;
    let mix = smoothstep(scaled - left_region as f32);
    let region_position = |region: usize| {
        if regions <= 1 {
            0.0
        } else {
            region as f32 / (regions - 1) as f32
        }
    };

    (
        pitch_class_for_frequency(frequency_for_position(region_position(left_region))),
        pitch_class_for_frequency(frequency_for_position(region_position(right_region))),
        mix,
    )
}

fn fbm_2d(x: f32, y: f32, seed: f32, octaves: usize) -> f32 {
    let mut value = 0.0_f32;
    let mut amplitude = 0.5_f32;
    let mut frequency = 1.0_f32;
    let mut total = 0.0_f32;

    for octave in 0..octaves.max(1) {
        value +=
            value_noise_2d(x * frequency, y * frequency, seed + octave as f32 * 9.17) * amplitude;
        total += amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    (value / total.max(0.000_1)).clamp(0.0, 1.0)
}

fn value_noise_2d(x: f32, y: f32, seed: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let xf = x - x0;
    let yf = y - y0;
    let u = smootherstep(xf);
    let v = smootherstep(yf);
    let n00 = hash_unit(x0, y0, seed);
    let n10 = hash_unit(x0 + 1.0, y0, seed);
    let n01 = hash_unit(x0, y0 + 1.0, seed);
    let n11 = hash_unit(x0 + 1.0, y0 + 1.0, seed);
    let x1 = lerp(n00, n10, u);
    let x2 = lerp(n01, n11, u);

    lerp(x1, x2, v)
}

fn hash_unit(x: f32, y: f32, seed: f32) -> f32 {
    let value = (x * 127.1 + y * 311.7 + seed * 74.7).sin() * 43_758.547;
    value - value.floor()
}

fn smootherstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn pitch_class_hue(pitch_class: usize) -> f32 {
    const FIFTHS: [usize; 12] = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5];
    let rank = FIFTHS
        .iter()
        .position(|pitch| *pitch == pitch_class % 12)
        .unwrap_or(0);
    wrap_unit(rank as f32 / 12.0 + 0.02)
}

fn blend_color(from: Color, to: Color, amount: f32) -> Color {
    let amount = amount.clamp(0.0, 1.0);
    let from = color_to_rgb(from);
    let to = color_to_rgb(to);
    let blend = |start: u8, end: u8| {
        (start as f32 + (end as f32 - start as f32) * amount)
            .round()
            .clamp(0.0, 255.0) as u8
    };

    Color::Rgb(
        blend(from.0, to.0),
        blend(from.1, to.1),
        blend(from.2, to.2),
    )
}

fn aurora_color(
    theme: Theme,
    state: VisualColorState,
    position: f32,
    intensity: f32,
    hue_offset: f32,
    trail: bool,
) -> Color {
    let base_hue = rgb_hue(color_to_rgb(theme.accent));
    let hue = wrap_unit(
        base_hue
            + position.clamp(0.0, 1.0) * 0.58
            + (state.centroid - 0.5) * 0.20
            + state.phase
            + hue_offset,
    );
    let intensity = intensity.clamp(0.0, 1.0);
    let saturation = (0.54 + state.energy * 0.26 + intensity * 0.18).clamp(0.42, 0.96);
    let value =
        (0.24 + intensity * 0.50 + state.energy * 0.22 + state.flux * 0.24).clamp(0.18, 1.0);
    let color = hsv_to_rgb(hue, saturation, value);
    let theme_mix = if trail { 0.08 } else { 0.16 };
    let color = blend_color(color, theme.accent, theme_mix);

    if trail {
        blend_color(theme.border, color, 0.58)
    } else {
        color
    }
}

fn rgb_hue((red, green, blue): (u8, u8, u8)) -> f32 {
    let red = red as f32 / 255.0;
    let green = green as f32 / 255.0;
    let blue = blue as f32 / 255.0;
    let max = red.max(green).max(blue);
    let min = red.min(green).min(blue);
    let delta = max - min;

    if delta <= f32::EPSILON {
        return 0.0;
    }

    let hue = if (max - red).abs() <= f32::EPSILON {
        (green - blue) / delta
    } else if (max - green).abs() <= f32::EPSILON {
        (blue - red) / delta + 2.0
    } else {
        (red - green) / delta + 4.0
    };

    wrap_unit(hue / 6.0)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> Color {
    let hue = wrap_unit(hue) * 6.0;
    let saturation = saturation.clamp(0.0, 1.0);
    let value = value.clamp(0.0, 1.0);
    let chroma = value * saturation;
    let x = chroma * (1.0 - ((hue % 2.0) - 1.0).abs());
    let m = value - chroma;
    let (red, green, blue) = if hue < 1.0 {
        (chroma, x, 0.0)
    } else if hue < 2.0 {
        (x, chroma, 0.0)
    } else if hue < 3.0 {
        (0.0, chroma, x)
    } else if hue < 4.0 {
        (0.0, x, chroma)
    } else if hue < 5.0 {
        (x, 0.0, chroma)
    } else {
        (chroma, 0.0, x)
    };
    let to_byte = |channel: f32| ((channel + m) * 255.0).round().clamp(0.0, 255.0) as u8;

    Color::Rgb(to_byte(red), to_byte(green), to_byte(blue))
}

fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount.clamp(0.0, 1.0)
}

fn lerp_unit(from: f32, to: f32, amount: f32) -> f32 {
    let delta = (to - from + 0.5).rem_euclid(1.0) - 0.5;
    wrap_unit(from + delta * amount.clamp(0.0, 1.0))
}

fn wrap_unit(value: f32) -> f32 {
    value.rem_euclid(1.0)
}

fn frequency_for_position(position: f32) -> f32 {
    let ratio = MAX_FREQUENCY / MIN_FREQUENCY;
    MIN_FREQUENCY * ratio.powf(position.clamp(0.0, 1.0))
}

fn pitch_class_for_frequency(frequency: f32) -> usize {
    if frequency <= 0.0 {
        return 0;
    }

    let midi = (69.0 + 12.0 * (frequency / 440.0).log2()).round() as i32;
    midi.rem_euclid(12) as usize
}

fn melody_frequency_weight(frequency: f32) -> f32 {
    let low = smoothstep(((frequency - 80.0) / 220.0).clamp(0.0, 1.0));
    let high = 1.0 - smoothstep(((frequency - 2_200.0) / 3_800.0).clamp(0.0, 1.0));
    (low * high).clamp(0.10, 1.0)
}

fn strongest_chroma(chroma: &[f32; 12]) -> (usize, f32) {
    let mut best_index = 0;
    let mut best = 0.0_f32;
    let mut total = 0.0_f32;
    for (index, value) in chroma.iter().copied().enumerate() {
        total += value;
        if value > best {
            best = value;
            best_index = index;
        }
    }

    let average_other = ((total - best) / 11.0).max(0.0);
    let confidence = if best <= 0.000_1 {
        0.0
    } else {
        ((best - average_other) / best).clamp(0.0, 1.0)
    };

    (best_index, confidence)
}

fn master_braille_symbol(mask: u8) -> String {
    if mask == 0 {
        " ".to_string()
    } else {
        braille_pattern(mask).to_string()
    }
}

fn draw_waveform(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let inner = draw_panel(
        frame,
        area,
        theme,
        Some(module_title_line(app, 'w', "waveform")),
    );

    if inner.width < 8 || inner.height < 3 {
        return;
    }

    let cell_width = inner.width as usize;
    let cell_height = inner.height as usize;
    let virtual_width = cell_width * 2;
    let virtual_height = cell_height * 4;
    let samples = display_waveform(&app.waveform, virtual_width);
    let peak = samples
        .iter()
        .fold(0.0_f32, |current, sample| {
            current.max(sample.min.abs()).max(sample.max.abs())
        })
        .max(0.000_1);
    let gain = (WAVEFORM_TARGET_PEAK / peak).clamp(1.0, 10.0);
    let mut lines = Vec::with_capacity(cell_height);

    for row in 0..cell_height {
        let mut spans = Vec::with_capacity(cell_width);
        for col in 0..cell_width {
            let (waveform_mask, value) =
                waveform_braille_cell(&samples, col, row, virtual_height, gain);
            let center_mask = waveform_centerline_mask(row, virtual_height);
            let mask = waveform_mask | center_mask;
            let color = if waveform_mask == 0 {
                theme.border
            } else {
                waveform_color(app, value)
            };
            spans.push(Span::styled(
                braille_pattern(mask).to_string(),
                Style::default().fg(color),
            ));
        }
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_spectrum(frame: &mut Frame, app: &App, area: Rect, title: &'static str) {
    let theme = app.theme();
    let inner = draw_panel(frame, area, theme, Some(panel_title(title, theme)));

    if inner.width < 10 || inner.height < 4 {
        return;
    }

    match app.config.settings.renderer {
        SpectrumRenderer::Blocks => draw_block_spectrum(frame, app, inner),
        SpectrumRenderer::Braille => draw_braille_spectrum(frame, app, inner),
    }
}

fn draw_block_spectrum(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let settings = &app.config.settings;
    let bars = display_bars(&app.spectrum, area.width as usize);
    let trail = if settings.trail_enabled {
        display_bars(&app.spectrum_trail, area.width as usize)
    } else {
        Vec::new()
    };
    let chart_height = area.height as usize;
    let virtual_width = area.width as usize * 2;
    let virtual_height = chart_height * 4;
    let accent_traces = display_accent_traces(app, virtual_width);
    let mut lines = Vec::with_capacity(chart_height);

    for row in 0..chart_height {
        let threshold = 1.0 - (row as f32 + 0.5) / chart_height as f32;
        let mut spans = Vec::with_capacity(bars.len());
        for (index, value) in bars.iter().enumerate() {
            let accent_trace = accent_trace_overlay_cell(
                app,
                &accent_traces,
                index,
                row,
                virtual_width,
                virtual_height,
            );
            let value = render_bar_value(*value, settings);
            let trail_value = trail
                .get(index)
                .map(|value| render_bar_value(*value, settings))
                .unwrap_or(0.0);
            let filled = value >= threshold;
            let trail_filled = !filled && trail_value > value && trail_value >= threshold;
            let base_symbol = if filled {
                "█".to_string()
            } else if trail_filled {
                "░".to_string()
            } else {
                " ".to_string()
            };
            let base_color = if filled {
                Some(spectrum_bar_color_at(
                    app,
                    index,
                    bars.len(),
                    value,
                    threshold,
                    false,
                ))
            } else if trail_filled {
                Some(spectrum_bar_color_at(
                    app,
                    index,
                    bars.len(),
                    trail_value,
                    threshold,
                    true,
                ))
            } else {
                None
            };
            let (symbol, color) = if let Some(overlay) = accent_trace {
                if let Some(base_color) = base_color {
                    (
                        base_symbol,
                        accent_trace_overlay_color(theme, Some(base_color), overlay),
                    )
                } else {
                    (
                        braille_pattern(overlay.mask).to_string(),
                        accent_trace_overlay_color(theme, None, overlay),
                    )
                }
            } else {
                (base_symbol, base_color.unwrap_or(Color::Reset))
            };
            spans.push(Span::styled(symbol, Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_braille_spectrum(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let settings = &app.config.settings;
    let cell_width = area.width as usize;
    let chart_height = area.height as usize;
    let virtual_width = cell_width * 2;
    let virtual_height = chart_height * 4;
    let bars: Vec<f32> = display_bars(&app.spectrum, virtual_width)
        .into_iter()
        .map(|value| render_bar_value(value, settings))
        .collect();
    let trail: Vec<f32> = if settings.trail_enabled {
        display_bars(&app.spectrum_trail, virtual_width)
            .into_iter()
            .map(|value| render_bar_value(value, settings))
            .collect()
    } else {
        Vec::new()
    };
    let accent_traces = display_accent_traces(app, virtual_width);
    let mut lines = Vec::with_capacity(chart_height);

    for row in 0..chart_height {
        let height_ratio = 1.0 - (row as f32 + 0.5) / chart_height.max(1) as f32;
        let mut spans = Vec::with_capacity(cell_width);
        for col in 0..cell_width {
            let (mask, value) = braille_bar_cell(&bars, col, row, virtual_height);
            let (trail_mask, trail_value) = braille_bar_cell(&trail, col, row, virtual_height);
            let accent_trace = accent_trace_overlay_cell(
                app,
                &accent_traces,
                col,
                row,
                virtual_width,
                virtual_height,
            );
            let combined_mask = mask | (trail_mask & !mask);
            if let Some(overlay) = accent_trace {
                if combined_mask == 0 {
                    spans.push(Span::styled(
                        braille_pattern(overlay.mask).to_string(),
                        Style::default().fg(accent_trace_overlay_color(theme, None, overlay)),
                    ));
                } else {
                    let base_color = if mask == 0 {
                        theme.border
                    } else {
                        spectrum_bar_color_at(
                            app,
                            col * 2,
                            virtual_width,
                            value.max(trail_value),
                            height_ratio,
                            false,
                        )
                    };
                    spans.push(Span::styled(
                        braille_pattern(combined_mask | overlay.mask).to_string(),
                        Style::default().fg(accent_trace_overlay_color(
                            theme,
                            Some(base_color),
                            overlay,
                        )),
                    ));
                }
            } else if combined_mask == 0 {
                spans.push(Span::raw(" "));
            } else {
                let color = if mask == 0 {
                    theme.border
                } else {
                    spectrum_bar_color_at(
                        app,
                        col * 2,
                        virtual_width,
                        value.max(trail_value),
                        height_ratio,
                        false,
                    )
                };
                spans.push(Span::styled(
                    braille_pattern(combined_mask).to_string(),
                    Style::default().fg(color),
                ));
            }
        }
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_compact_footer(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let text = Paragraph::new(Line::from(vec![
        Span::styled(
            capture_control_label(app),
            Style::default().fg(theme.accent),
        ),
        Span::raw("  "),
        Span::styled(
            format!(
                "L{:>3} R{:>3}",
                (app.master_left * 100.0) as u16,
                (app.master_right * 100.0) as u16
            ),
            Style::default().fg(theme.text),
        ),
        Span::raw("  "),
        Span::styled(app.t("compact_hint"), Style::default().fg(theme.muted)),
    ]))
    .wrap(Wrap { trim: true });
    let inner = draw_panel(frame, area, theme, None);
    frame.render_widget(text, inner);
}

fn display_bars(source: &[f32], width: usize) -> Vec<f32> {
    if width == 0 || source.is_empty() {
        return Vec::new();
    }

    if width == source.len() {
        return source.to_vec();
    }

    if width < source.len() {
        let mut bars = Vec::with_capacity(width);
        for index in 0..width {
            let start = index * source.len() / width;
            let end = ((index + 1) * source.len() / width)
                .max(start + 1)
                .min(source.len());
            let value = source[start..end].iter().copied().fold(0.0_f32, f32::max);
            bars.push(value);
        }
        return bars;
    }

    let mut bars = Vec::with_capacity(width);
    let max_source_index = source.len() - 1;
    for index in 0..width {
        let position = if width == 1 {
            0.0
        } else {
            index as f32 * max_source_index as f32 / (width - 1) as f32
        };
        let left = position.floor() as usize;
        let right = position.ceil() as usize;
        let mix = position - left as f32;
        let left_value = source[left];
        let right_value = source[right.min(max_source_index)];
        bars.push(left_value * (1.0 - mix) + right_value * mix);
    }
    bars
}

fn update_spectrum_trail(trail: &mut Vec<f32>, bars: &[f32], settings: &Settings) {
    if bars.is_empty() {
        trail.clear();
        return;
    }

    if trail.len() != bars.len() {
        *trail = display_bars(trail, bars.len());
        if trail.len() != bars.len() {
            trail.resize(bars.len(), 0.0);
        }
    }

    if !settings.trail_enabled {
        trail.clone_from_slice(bars);
        return;
    }

    let decay = settings.trail_decay.clamp(0.65, 0.98);
    for (ghost, current) in trail.iter_mut().zip(bars.iter().copied()) {
        if current >= *ghost {
            *ghost = current;
        } else {
            *ghost = (*ghost * decay).max(current);
            if *ghost < VISUAL_NOISE_FLOOR * 0.5 {
                *ghost = 0.0;
            }
        }
    }
}

fn merge_envelope_max(envelope: &mut Vec<f32>, bars: &[f32]) {
    if bars.is_empty() {
        return;
    }

    if envelope.len() != bars.len() {
        *envelope = display_bars(envelope, bars.len());
        if envelope.len() != bars.len() {
            envelope.resize(bars.len(), 0.0);
        }
    }

    for (stored, current) in envelope.iter_mut().zip(bars.iter().copied()) {
        *stored = stored.max(current.clamp(0.0, 1.0));
    }
}

fn spectrum_accent_energy(bars: &[f32]) -> f32 {
    if bars.is_empty() {
        return 0.0;
    }

    let mut square_sum = 0.0_f32;
    let mut peak = 0.0_f32;
    for value in bars.iter().copied() {
        let value = value.clamp(0.0, 1.0);
        square_sum += value * value;
        peak = peak.max(value);
    }
    let rms = (square_sum / bars.len() as f32).sqrt();
    (rms * 0.68 + peak * 0.32).clamp(0.0, 1.0)
}

fn spectrum_positive_flux(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() {
        return 0.0;
    }

    let mut flux = 0.0_f32;
    for (index, current) in current.iter().copied().enumerate() {
        let previous = previous.get(index).copied().unwrap_or(0.0);
        flux += (current.clamp(0.0, 1.0) - previous.clamp(0.0, 1.0)).max(0.0);
    }
    (flux / current.len() as f32 * 2.0).clamp(0.0, 1.0)
}

fn accent_trigger_thresholds(threshold: f32) -> AccentTriggerThresholds {
    let threshold = threshold.clamp(0.10, 0.90);
    AccentTriggerThresholds {
        peak: lerp(0.10, 0.26, threshold),
        initial_energy: lerp(0.03, 0.09, threshold),
        energy: lerp(0.025, 0.065, threshold),
        flux: lerp(0.015, 0.055, threshold),
        rise: lerp(0.010, 0.060, threshold),
        ratio: lerp(1.15, 1.75, threshold),
    }
}

fn display_accent_traces(app: &App, virtual_width: usize) -> Vec<AccentTraceRender> {
    if !app.config.settings.accent_trace_enabled || virtual_width == 0 {
        return Vec::new();
    }

    app.accent_traces
        .iter()
        .filter_map(|trace| {
            let fade = trace.fade().powf(1.35);
            if fade <= 0.0 {
                return None;
            }

            let mut envelope: Vec<f32> = display_bars(&trace.envelope, virtual_width)
                .into_iter()
                .map(|value| render_bar_value(value, &app.config.settings))
                .collect();
            smooth_accent_trace_envelope(&mut envelope);

            Some(AccentTraceRender { envelope, fade })
        })
        .collect()
}

fn smooth_accent_trace_envelope(envelope: &mut [f32]) {
    if envelope.len() < 3 {
        return;
    }

    let source = envelope.to_vec();
    let max_index = envelope.len().saturating_sub(1).max(1) as f32;
    for (index, value) in envelope.iter_mut().enumerate() {
        let position = index as f32 / max_index;
        let radius = (2.0 + position * 7.0).round() as isize;
        let mut total = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for offset in -radius..=radius {
            let sample_index =
                (index as isize + offset).clamp(0, source.len() as isize - 1) as usize;
            let distance = offset.unsigned_abs() as f32;
            let weight = 1.0 - distance / (radius as f32 + 1.0);
            total += source[sample_index] * weight;
            weight_total += weight;
        }

        *value = if weight_total > 0.0 {
            total / weight_total
        } else {
            source[index]
        };
    }
}

fn accent_trace_overlay_cell(
    app: &App,
    traces: &[AccentTraceRender],
    cell_col: usize,
    cell_row: usize,
    virtual_width: usize,
    virtual_height: usize,
) -> Option<AccentTraceOverlay> {
    let mut combined_mask = 0;
    let mut strongest = 0.0_f32;
    let theme = app.theme();
    let mut color = theme.accent;

    for trace in traces {
        let (mask, value) =
            accent_trace_braille_cell(&trace.envelope, cell_col, cell_row, virtual_height);
        if mask == 0 {
            continue;
        }

        combined_mask |= mask;
        let visibility = (trace.fade * (0.48 + value * 0.52)).clamp(0.0, 1.0);
        if visibility >= strongest {
            color = spectrum_bar_color(app, cell_col * 2, virtual_width, value.max(0.30), false);
            strongest = visibility;
        }
    }

    if combined_mask == 0 {
        None
    } else {
        Some(AccentTraceOverlay {
            mask: combined_mask,
            color,
            visibility: strongest,
        })
    }
}

fn accent_trace_overlay_color(
    theme: Theme,
    base_color: Option<Color>,
    overlay: AccentTraceOverlay,
) -> Color {
    if let Some(base_color) = base_color {
        blend_color(base_color, overlay.color, overlay.visibility * 0.58)
    } else {
        blend_color(
            accent_trace_background_color(theme),
            overlay.color,
            overlay.visibility,
        )
    }
}

fn accent_trace_background_color(theme: Theme) -> Color {
    blend_color(Color::Black, theme.border, 0.42)
}

fn accent_trace_braille_cell(
    envelope: &[f32],
    cell_col: usize,
    cell_row: usize,
    virtual_height: usize,
) -> (u8, f32) {
    let mut mask = 0;
    let mut cell_value = 0.0_f32;

    for dot_col in 0..2 {
        let bar_index = cell_col * 2 + dot_col;
        let Some(value) = envelope.get(bar_index).copied() else {
            continue;
        };
        let value = value.clamp(0.0, 1.0);
        if value <= 0.0 {
            continue;
        }

        let virtual_row = accent_trace_virtual_row(value, virtual_height);
        let mut start_row = virtual_row;
        let mut end_row = virtual_row;
        let mut segment_value = value;

        if bar_index > 0 {
            if let Some(previous) = envelope.get(bar_index - 1).copied() {
                let previous = previous.clamp(0.0, 1.0);
                if previous > 0.0 {
                    let previous_row = accent_trace_virtual_row(previous, virtual_height);
                    start_row = start_row.min(previous_row);
                    end_row = end_row.max(previous_row);
                    segment_value = segment_value.max(previous);
                }
            }
        }

        cell_value = cell_value.max(segment_value);
        for virtual_row in start_row..=end_row {
            if virtual_row / 4 == cell_row {
                mask |= BRAILLE_DOT_BITS[dot_col][virtual_row % 4];
            }
        }
    }

    (mask, cell_value)
}

fn accent_trace_virtual_row(value: f32, virtual_height: usize) -> usize {
    if virtual_height == 0 {
        return 0;
    }

    let max_row = virtual_height.saturating_sub(1) as i32;
    let offset = (ACCENT_TRACE_VERTICAL_OFFSET_CELLS * 4) as i32;
    let row = ((1.0 - value.clamp(0.0, 1.0)) * max_row as f32).round() as i32 - offset;
    row.clamp(0, max_row) as usize
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WaveformColumn {
    min: f32,
    max: f32,
}

fn display_waveform(source: &[f32], width: usize) -> Vec<WaveformColumn> {
    if width == 0 || source.is_empty() {
        return Vec::new();
    }

    if width < source.len() {
        let mut columns = Vec::with_capacity(width);
        for index in 0..width {
            let start = index * source.len() / width;
            let end = ((index + 1) * source.len() / width)
                .max(start + 1)
                .min(source.len());
            columns.push(waveform_column(&source[start..end]));
        }
        return columns;
    }

    let mut columns = Vec::with_capacity(width);
    let max_source_index = source.len() - 1;
    for index in 0..width {
        let source_index = if width == 1 {
            0
        } else {
            index * max_source_index / (width - 1)
        };
        let sample = source[source_index].clamp(-1.0, 1.0);
        columns.push(WaveformColumn {
            min: sample.min(0.0),
            max: sample.max(0.0),
        });
    }
    columns
}

fn waveform_column(samples: &[f32]) -> WaveformColumn {
    let mut min = 0.0_f32;
    let mut max = 0.0_f32;
    for sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        min = min.min(sample);
        max = max.max(sample);
    }
    WaveformColumn { min, max }
}

fn waveform_braille_cell(
    columns: &[WaveformColumn],
    cell_col: usize,
    cell_row: usize,
    virtual_height: usize,
    gain: f32,
) -> (u8, f32) {
    let mut mask = 0;
    let mut cell_value = 0.0_f32;
    let center = (virtual_height.saturating_sub(1)) as f32 * 0.5;

    for dot_col in 0..2 {
        let column_index = cell_col * 2 + dot_col;
        let Some(column) = columns.get(column_index).copied() else {
            continue;
        };
        let min = (column.min * gain).clamp(-1.0, 1.0);
        let max = (column.max * gain).clamp(-1.0, 1.0);
        let active = max.abs().max(min.abs()) > 0.002;
        cell_value = cell_value.max(max.abs()).max(min.abs());

        for dot_row in 0..4 {
            let virtual_row = cell_row * 4 + dot_row;
            let position = if center <= 0.0 {
                0.0
            } else {
                ((center - virtual_row as f32) / center).clamp(-1.0, 1.0)
            };
            if active && position >= min && position <= max {
                mask |= BRAILLE_DOT_BITS[dot_col][dot_row];
            }
        }
    }

    (mask, cell_value)
}

fn waveform_centerline_mask(cell_row: usize, virtual_height: usize) -> u8 {
    let center_row = ((virtual_height.saturating_sub(1)) as f32 * 0.5).round() as usize;
    let mut mask = 0;

    for dot_col in 0..2 {
        for dot_row in 0..4 {
            let virtual_row = cell_row * 4 + dot_row;
            if virtual_row == center_row {
                mask |= BRAILLE_DOT_BITS[dot_col][dot_row];
            }
        }
    }

    mask
}

fn render_bar_value(value: f32, settings: &Settings) -> f32 {
    let ceiling = settings.ceiling.max(0.01);
    let floor = VISUAL_NOISE_FLOOR.min(ceiling * 0.5);
    let value = ((value.clamp(0.0, ceiling) - floor) / (ceiling - floor)).clamp(0.0, 1.0);
    if settings.visual_curve_enabled {
        value.powf(settings.visual_curve)
    } else {
        value
    }
}

fn braille_bar_cell(
    bars: &[f32],
    cell_col: usize,
    cell_row: usize,
    virtual_height: usize,
) -> (u8, f32) {
    let mut mask = 0;
    let mut cell_value = 0.0_f32;

    for dot_col in 0..2 {
        let bar_index = cell_col * 2 + dot_col;
        let Some(value) = bars.get(bar_index).copied() else {
            continue;
        };
        cell_value = cell_value.max(value);

        for dot_row in 0..4 {
            let virtual_row = cell_row * 4 + dot_row;
            let threshold = 1.0 - (virtual_row as f32 + 0.5) / virtual_height.max(1) as f32;
            if value >= threshold {
                mask |= BRAILLE_DOT_BITS[dot_col][dot_row];
            }
        }
    }

    (mask, cell_value)
}

fn braille_pattern(mask: u8) -> char {
    char::from_u32(0x2800 + mask as u32).unwrap_or(' ')
}

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = if area.width < 70 || area.height < 24 {
        area
    } else {
        centered_rect(54, 76, area)
    };
    frame.render_widget(Clear, chunks);
    draw_settings_list(
        frame,
        app,
        chunks,
        panel_title(app.t("settings"), app.theme()),
    );
}

fn draw_settings_list(frame: &mut Frame, app: &App, area: Rect, title: Line<'static>) {
    let theme = app.theme();
    let inner = draw_panel(frame, area, theme, Some(title));
    let rows = vec![
        setting_line(app, "language", language_label(app.lang)),
        setting_line(app, "theme", theme_label(app)),
        setting_line(
            app,
            "analysis_preset",
            preset_label(app, app.config.settings.analysis_preset),
        ),
        setting_line(
            app,
            "attack",
            format!("{:>3}%", (app.config.settings.attack * 100.0) as u16),
        ),
        setting_line(
            app,
            "release",
            format!("{:>3}%", (app.config.settings.release * 100.0) as u16),
        ),
        setting_line(app, "bars", bar_count_label(app)),
        setting_line(app, "renderer", renderer_label(app)),
        setting_line(app, "fft_size", app.config.settings.fft_size.to_string()),
        setting_line(
            app,
            "analysis_hop",
            app.config.settings.analysis_hop.to_string(),
        ),
        setting_line(
            app,
            "refresh_rate",
            format!("{}Hz", app.config.settings.refresh_hz),
        ),
        setting_line(
            app,
            "audio_delay",
            format!("{}ms", app.config.settings.audio_delay_ms),
        ),
        setting_line(
            app,
            "high_shelf",
            on_off_label(app, app.config.settings.high_shelf_enabled),
        ),
        setting_line(
            app,
            "high_shelf_db",
            format!("{:.0}dB", app.config.settings.high_shelf_db),
        ),
        setting_line(
            app,
            "visual_curve",
            on_off_label(app, app.config.settings.visual_curve_enabled),
        ),
        setting_line(
            app,
            "curve_power",
            format!("{:.2}", app.config.settings.visual_curve),
        ),
        setting_line(
            app,
            "trail",
            on_off_label(app, app.config.settings.trail_enabled),
        ),
        setting_line(
            app,
            "trail_decay",
            format!("{:>3}%", (app.config.settings.trail_decay * 100.0) as u16),
        ),
        setting_line(
            app,
            "accent_trace",
            on_off_label(app, app.config.settings.accent_trace_enabled),
        ),
        setting_line(
            app,
            "accent_threshold",
            format!(
                "{:>3}%",
                (app.config.settings.accent_trace_threshold * 100.0).round() as u16
            ),
        ),
        setting_line(
            app,
            "ceiling",
            format!("{:>3}%", (app.config.settings.ceiling * 100.0) as u16),
        ),
    ];

    let visible_height = inner.height as usize;
    let selected = app.setting_index.min(rows.len().saturating_sub(1));
    let start = visible_list_start(selected, rows.len(), visible_height);
    let end = rows.len().min(start + visible_height);
    let visible_rows = if visible_height == 0 {
        Vec::new()
    } else {
        rows.into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect()
    };
    let items: Vec<ListItem> = visible_rows.into_iter().map(ListItem::new).collect();
    let mut state = ListState::default();
    state.select(Some(selected.saturating_sub(start)));

    let list = List::new(items).highlight_symbol("  ").highlight_style(
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, inner, &mut state);
}

fn visible_list_start(selected: usize, len: usize, visible_height: usize) -> usize {
    if visible_height == 0 || len <= visible_height {
        return 0;
    }

    let max_start = len - visible_height;
    selected
        .saturating_add(1)
        .saturating_sub(visible_height)
        .min(max_start)
}

fn draw_pipeline(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let inner = draw_panel(
        frame,
        area,
        theme,
        Some(module_title_line(app, 'p', "pipeline")),
    );
    let settings = &app.config.settings;
    let shelf = if settings.high_shelf_enabled {
        format!("HS +{:.0}dB", settings.high_shelf_db)
    } else {
        "HS bypass".to_string()
    };
    let meter = if settings.visual_curve_enabled {
        format!("pow {:.2}", settings.visual_curve)
    } else {
        "linear".to_string()
    };
    let trail = if settings.trail_enabled {
        format!("trail {:>2}%", (settings.trail_decay * 100.0) as u16)
    } else {
        "trail off".to_string()
    };
    let accent_trace = if settings.accent_trace_enabled {
        format!(
            "accent {:>2}%",
            (settings.accent_trace_threshold * 100.0).round() as u16
        )
    } else {
        "accent off".to_string()
    };
    let lines = vec![
        Line::from(vec![
            pipeline_stage(theme, "SRC"),
            Span::styled(
                format!(
                    "SCStream 48k/2ch | {}",
                    preset_label(app, settings.analysis_preset)
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "SYNC"),
            Span::styled(
                format!("delay {}ms -> mono bus", settings.audio_delay_ms),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "PRE"),
            Span::styled("mono -> DC trim -> Hann", Style::default().fg(theme.text)),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "FFT"),
            Span::styled(
                format!(
                    "{} hop {} log 35Hz-18k",
                    settings.fft_size, settings.analysis_hop
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "DET"),
            Span::styled(
                format!(
                    "RMS+peak atk {:>2} rel {:>2}",
                    (settings.attack * 100.0) as u16,
                    (settings.release * 100.0) as u16
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "PROC"),
            Span::styled(
                format!(
                    "{} | lim {:>2}% | {} | {} | {}",
                    shelf,
                    (settings.ceiling * 100.0) as u16,
                    meter,
                    trail,
                    accent_trace
                ),
                Style::default().fg(theme.text),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn pipeline_stage(theme: Theme, label: &'static str) -> Span<'static> {
    Span::styled(
        format!("{:<5}", label),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    )
}

fn setting_line(app: &App, key: &'static str, value: impl Into<String>) -> Line<'static> {
    let theme = app.theme();
    Line::from(vec![
        Span::styled(
            format!("{:<12}", app.t(key)),
            Style::default().fg(theme.text),
        ),
        Span::styled(value.into(), Style::default().fg(theme.muted)),
    ])
}

fn draw_help(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let chunks = centered_rect(76, 70, area);
    frame.render_widget(Clear, chunks);

    let text = vec![
        Line::from(Span::styled(
            app.t("help_title"),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(app.t("help_1")),
        Line::from(app.t("help_2")),
        Line::from(app.t("help_3")),
        Line::from(app.t("help_4")),
        Line::from(app.t("help_5")),
        Line::from(""),
        Line::from(Span::styled(
            app.t("permission_note"),
            Style::default().fg(theme.muted),
        )),
    ];

    let inner = draw_panel(
        frame,
        chunks,
        theme,
        Some(panel_title(app.t("help"), theme)),
    );
    frame.render_widget(Paragraph::new(text).wrap(Wrap { trim: true }), inner);
}

struct Panel {
    title: Option<Line<'static>>,
    theme: Theme,
}

impl Panel {
    fn new(title: Option<Line<'static>>, theme: Theme) -> Self {
        Self { title, theme }
    }

    fn inner(area: Rect) -> Rect {
        Rect {
            x: area.x.saturating_add(1),
            y: area.y.saturating_add(1),
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        }
    }

    fn set_cell(buf: &mut Buffer, x: u16, y: u16, symbol: &str, style: Style) {
        buf[(x, y)].set_symbol(symbol).set_style(style);
    }

    fn render_title(&self, area: Rect, buf: &mut Buffer, title: &Line<'_>) {
        if area.width < 7 || title.width() == 0 {
            return;
        }

        let border_style = border_style(self.theme);
        let text_start = area.x + 3;
        let max_width = area.width.saturating_sub(5);
        let right_limit = area.right().saturating_sub(1);

        Self::set_cell(buf, area.x + 2, area.y, "┐", border_style);
        let (end_x, _) = buf.set_line(text_start, area.y, title, max_width);
        if end_x < right_limit {
            Self::set_cell(buf, end_x, area.y, "┌", border_style);
        }
    }
}

impl Widget for Panel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let content_style = Style::default().fg(self.theme.text);
        let border_style = border_style(self.theme);
        buf.set_style(area, content_style);

        if area.width < 2 || area.height < 2 {
            return;
        }

        let x0 = area.left();
        let y0 = area.top();
        let x1 = area.right() - 1;
        let y1 = area.bottom() - 1;

        for y in y0 + 1..y1 {
            for x in x0 + 1..x1 {
                Self::set_cell(buf, x, y, " ", content_style);
            }
        }

        for x in x0 + 1..x1 {
            Self::set_cell(buf, x, y0, "─", border_style);
            Self::set_cell(buf, x, y1, "─", border_style);
        }

        for y in y0 + 1..y1 {
            Self::set_cell(buf, x0, y, "│", border_style);
            Self::set_cell(buf, x1, y, "│", border_style);
        }

        Self::set_cell(buf, x0, y0, "╭", border_style);
        Self::set_cell(buf, x1, y0, "╮", border_style);
        Self::set_cell(buf, x0, y1, "╰", border_style);
        Self::set_cell(buf, x1, y1, "╯", border_style);

        if let Some(title) = &self.title {
            self.render_title(area, buf, title);
        }
    }
}

fn draw_panel(frame: &mut Frame, area: Rect, theme: Theme, title: Option<Line<'static>>) -> Rect {
    let inner = Panel::inner(area);
    frame.render_widget(Panel::new(title, theme), area);
    inner
}

fn panel_title(title: &'static str, theme: Theme) -> Line<'static> {
    Line::from(Span::styled(
        title,
        title_style(theme).add_modifier(Modifier::BOLD),
    ))
}

fn border_style(theme: Theme) -> Style {
    Style::default().fg(theme.border)
}

fn module_title_line(app: &App, shortcut: char, title_key: &'static str) -> Line<'static> {
    let theme = app.theme();
    if app.lang == Lang::En {
        inline_hotkey_label(theme, shortcut, english_module_title(title_key))
    } else {
        Line::from(vec![
            Span::styled(
                format!("[{}]", shortcut),
                hotkey_style(theme).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                app.t(title_key).to_string(),
                title_style(theme).add_modifier(Modifier::BOLD),
            ),
        ])
    }
}

fn inline_hotkey_label(theme: Theme, shortcut: char, label: &'static str) -> Line<'static> {
    let mut chars = label.chars();
    let first = chars.next().unwrap_or(shortcut);
    if first.eq_ignore_ascii_case(&shortcut) {
        Line::from(vec![
            Span::styled(
                first.to_string(),
                hotkey_style(theme).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                chars.as_str().to_string(),
                title_style(theme).add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                shortcut.to_string(),
                hotkey_style(theme).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {}", label),
                title_style(theme).add_modifier(Modifier::BOLD),
            ),
        ])
    }
}

fn english_module_title(title_key: &'static str) -> &'static str {
    match title_key {
        "settings" => "settings",
        "pipeline" => "pipeline",
        "toolbar" => "toolbar",
        "master" => "master",
        "waveform" => "waveform",
        _ => title_key,
    }
}

fn hotkey_style(theme: Theme) -> Style {
    Style::default().fg(theme.accent)
}

fn title_style(theme: Theme) -> Style {
    Style::default().fg(theme.text)
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn language_label(lang: Lang) -> &'static str {
    LANGUAGES
        .iter()
        .find(|(code, _)| *code == lang.code())
        .map(|(_, label)| *label)
        .unwrap_or("中文")
}

fn theme_label(app: &App) -> &'static str {
    app.t(app.theme().title_key)
}

fn preset_label(app: &App, preset: AnalysisPreset) -> &'static str {
    app.t(preset.title_key())
}

fn on_off_label(app: &App, enabled: bool) -> &'static str {
    if enabled {
        app.t("on")
    } else {
        app.t("off")
    }
}

fn capture_control_label(app: &App) -> String {
    match app.capture_state {
        CaptureState::Idle => format!("Space {}", app.t("transport_idle")),
        CaptureState::Starting => format!("Space {}", app.t("transport_starting")),
        CaptureState::Running => format!("Space {}", app.t("transport_running")),
        CaptureState::PermissionNeeded => format!("Space {}", app.t("transport_permission")),
        CaptureState::Failed => format!("Space {}", app.t("transport_failed")),
    }
}

fn renderer_label(app: &App) -> &'static str {
    match app.config.settings.renderer {
        SpectrumRenderer::Blocks => app.t("renderer_blocks"),
        SpectrumRenderer::Braille => app.t("renderer_braille"),
    }
}

fn bar_count_label(app: &App) -> String {
    let configured = app.config.settings.bar_count;
    let actual = app.analyzer.bar_count;
    if actual == configured {
        configured.to_string()
    } else {
        format!("{configured}/{actual}")
    }
}

fn tr(lang: Lang, key: &'static str) -> &'static str {
    match (lang, key) {
        (Lang::Zh, "main_menu") => "主菜单",
        (Lang::Zh, "menu_spectrum") => "进入频谱",
        (Lang::Zh, "menu_toggle") => "捕获开关",
        (Lang::Zh, "menu_start") => "开始捕获",
        (Lang::Zh, "menu_stop") => "停止捕获",
        (Lang::Zh, "menu_settings") => "设置",
        (Lang::Zh, "menu_help") => "帮助",
        (Lang::Zh, "menu_quit") => "退出",
        (Lang::Zh, "subtitle") => "系统音频频谱",
        (Lang::Zh, "overview") => "概览",
        (Lang::Zh, "preview") => "预览",
        (Lang::Zh, "spectrum") => "频谱",
        (Lang::Zh, "status") => "状态",
        (Lang::Zh, "level") => "电平",
        (Lang::Zh, "config") => "配置",
        (Lang::Zh, "settings") => "设置",
        (Lang::Zh, "language") => "语言",
        (Lang::Zh, "theme") => "主题",
        (Lang::Zh, "smoothing") => "平滑",
        (Lang::Zh, "analysis_preset") => "分析预设",
        (Lang::Zh, "attack") => "Attack",
        (Lang::Zh, "release") => "Release",
        (Lang::Zh, "analysis_hop") => "Hop",
        (Lang::Zh, "bars") => "频段",
        (Lang::Zh, "renderer") => "渲染",
        (Lang::Zh, "fft_size") => "FFT",
        (Lang::Zh, "refresh_rate") => "刷新率",
        (Lang::Zh, "audio_delay") => "音频延迟",
        (Lang::Zh, "high_shelf") => "高频补偿",
        (Lang::Zh, "high_shelf_db") => "补偿强度",
        (Lang::Zh, "visual_curve") => "高度曲线",
        (Lang::Zh, "curve_power") => "曲线指数",
        (Lang::Zh, "trail") => "残影",
        (Lang::Zh, "trail_decay") => "残影衰减",
        (Lang::Zh, "accent_trace") => "重音轮廓",
        (Lang::Zh, "accent_threshold") => "重音阈值",
        (Lang::Zh, "ceiling") => "上限",
        (Lang::Zh, "pipeline") => "音频链路",
        (Lang::Zh, "toolbar") => "工具栏",
        (Lang::Zh, "master") => "master",
        (Lang::Zh, "waveform") => "波形",
        (Lang::Zh, "modules") => "模块",
        (Lang::Zh, "on") => "开",
        (Lang::Zh, "off") => "关",
        (Lang::Zh, "transport_idle") => "○ 待机",
        (Lang::Zh, "transport_starting") => "◐ 启动",
        (Lang::Zh, "transport_running") => "● 运行",
        (Lang::Zh, "transport_permission") => "! 授权",
        (Lang::Zh, "transport_failed") => "× 错误",
        (Lang::Zh, "controls") => "控制",
        (Lang::Zh, "help") => "帮助",
        (Lang::Zh, "help_title") => "Terb 终端频谱",
        (Lang::Zh, "help_1") => "↑/↓ 或 j/k 移动选择。",
        (Lang::Zh, "help_2") => "Enter 执行；Space 开始或停止捕获。",
        (Lang::Zh, "help_3") => "频谱页按 s/p/t/m/w 显示或隐藏设置、链路、工具栏、master、波形。",
        (Lang::Zh, "help_4") => "↑/↓ 选择设置，←/→ 调整；-/= 调整音频延迟；S 打开全屏设置。",
        (Lang::Zh, "help_5") => "窗口较小时模块会自动隐藏，仍可用快捷键操作；主菜单 q/Esc 退出。",
        (Lang::Zh, "permission_note") => "首次捕获会触发 macOS 屏幕与系统音频录制授权；Terb 只实时分析，不保存音频。",
        (Lang::Zh, "menu_hint") => "↑/↓ 选择 · Enter 确认 · Space 捕获 · ? 帮助 · q 退出",
        (Lang::Zh, "spectrum_hint") => "Space 捕获 · -/= 延迟 · s/p/t/m/w 模块 · S 设置 · q 菜单",
        (Lang::Zh, "sidebar_hint") => "Space 开关捕获\n-/= 音频延迟\ns/p/t/m/w 模块\n↑/↓ 选择设置\n←/→ 调整\nS 设置\nq 菜单\n? 帮助",
        (Lang::Zh, "compact_hint") => "-/= 延迟 · s/p/t/m/w 模块 · S 设置 · q 菜单",
        (Lang::Zh, "ready") => "准备就绪。",
        (Lang::Zh, "starting") => "正在启动系统音频捕获...",
        (Lang::Zh, "helper_ready") => "捕获进程已就绪，等待音频。",
        (Lang::Zh, "running") => "正在分析系统音频。",
        (Lang::Zh, "waiting_audio") => "捕获已开启，暂未收到音频；请确认系统正在播放声音。",
        (Lang::Zh, "stopped") => "已停止。",
        (Lang::Zh, "permission_needed") => "需要 macOS 授权。请在系统设置中允许屏幕与系统音频录制。",
        (Lang::Zh, "capture_failed") => "捕获失败。",
        (Lang::Zh, "state_idle") => "待机",
        (Lang::Zh, "state_starting") => "启动中",
        (Lang::Zh, "state_running") => "运行中",
        (Lang::Zh, "state_permission") => "需授权",
        (Lang::Zh, "state_failed") => "错误",
        (Lang::Zh, "too_small") => "窗口太小，请放大终端。",
        (Lang::Zh, "theme_system") => "系统",
        (Lang::Zh, "theme_graphite") => "石墨",
        (Lang::Zh, "theme_ocean") => "海蓝",
        (Lang::Zh, "theme_aurora") => "奥罗拉",
        (Lang::Zh, "theme_sonic_texture") => "音纹场",
        (Lang::Zh, "theme_noise_warp") => "流纹噪声",
        (Lang::Zh, "theme_amber") => "琥珀",
        (Lang::Zh, "theme_mono") => "单色",
        (Lang::Zh, "renderer_blocks") => "方块",
        (Lang::Zh, "renderer_braille") => "盲文",
        (Lang::Zh, "preset_low_latency") => "低延迟",
        (Lang::Zh, "preset_balanced") => "均衡",
        (Lang::Zh, "preset_precision") => "精细",
        (Lang::Zh, "preset_custom") => "自定义",

        (Lang::En, "main_menu") => "Main Menu",
        (Lang::En, "menu_spectrum") => "Open Spectrum",
        (Lang::En, "menu_toggle") => "Toggle Capture",
        (Lang::En, "menu_start") => "Start Capture",
        (Lang::En, "menu_stop") => "Stop Capture",
        (Lang::En, "menu_settings") => "Settings",
        (Lang::En, "menu_help") => "Help",
        (Lang::En, "menu_quit") => "Quit",
        (Lang::En, "subtitle") => "system-audio spectrum",
        (Lang::En, "overview") => "Overview",
        (Lang::En, "preview") => "Preview",
        (Lang::En, "spectrum") => "Spectrum",
        (Lang::En, "status") => "Status",
        (Lang::En, "level") => "Level",
        (Lang::En, "config") => "Config",
        (Lang::En, "settings") => "Settings",
        (Lang::En, "language") => "Language",
        (Lang::En, "theme") => "Theme",
        (Lang::En, "smoothing") => "Smoothing",
        (Lang::En, "analysis_preset") => "Preset",
        (Lang::En, "attack") => "Attack",
        (Lang::En, "release") => "Release",
        (Lang::En, "analysis_hop") => "Hop",
        (Lang::En, "bars") => "Bands",
        (Lang::En, "renderer") => "Render",
        (Lang::En, "fft_size") => "FFT",
        (Lang::En, "refresh_rate") => "Refresh",
        (Lang::En, "audio_delay") => "Audio Delay",
        (Lang::En, "high_shelf") => "High-shelf",
        (Lang::En, "high_shelf_db") => "Shelf Gain",
        (Lang::En, "visual_curve") => "Height Curve",
        (Lang::En, "curve_power") => "Curve Power",
        (Lang::En, "trail") => "Trail",
        (Lang::En, "trail_decay") => "Trail Decay",
        (Lang::En, "accent_trace") => "Accent Trace",
        (Lang::En, "accent_threshold") => "Accent Threshold",
        (Lang::En, "ceiling") => "Ceiling",
        (Lang::En, "pipeline") => "Pipeline",
        (Lang::En, "toolbar") => "Toolbar",
        (Lang::En, "master") => "master",
        (Lang::En, "waveform") => "Waveform",
        (Lang::En, "modules") => "modules",
        (Lang::En, "on") => "On",
        (Lang::En, "off") => "Off",
        (Lang::En, "transport_idle") => "○ Idle",
        (Lang::En, "transport_starting") => "◐ Start",
        (Lang::En, "transport_running") => "● Run",
        (Lang::En, "transport_permission") => "! Permission",
        (Lang::En, "transport_failed") => "× Error",
        (Lang::En, "controls") => "Controls",
        (Lang::En, "help") => "Help",
        (Lang::En, "help_title") => "Terb terminal spectrum",
        (Lang::En, "help_1") => "Use ↑/↓ or j/k to move.",
        (Lang::En, "help_2") => "Enter activates; Space starts or stops capture.",
        (Lang::En, "help_3") => "In Spectrum, press s/p/t/m/w to show or hide settings, pipeline, toolbar, master, and waveform.",
        (Lang::En, "help_4") => "Use ↑/↓ to select settings and ←/→ to adjust. -/= adjusts audio delay; S opens full-screen settings.",
        (Lang::En, "help_5") => "Small terminals hide modules automatically, but shortcuts still work. q/Esc quits from the main menu.",
        (Lang::En, "permission_note") => "First capture may trigger macOS Screen & System Audio Recording permission. Terb analyzes live audio only and does not save it.",
        (Lang::En, "menu_hint") => "↑/↓ select · Enter confirm · Space capture · ? help · q quit",
        (Lang::En, "spectrum_hint") => "Space capture · -/= delay · s/p/t/m/w modules · S settings · q menu",
        (Lang::En, "sidebar_hint") => "Space toggle capture\n-/= audio delay\ns/p/t/m/w modules\n↑/↓ select setting\n←/→ adjust\nS settings\nq menu\n? help",
        (Lang::En, "compact_hint") => "-/= delay · s/p/t/m/w modules · S settings · q menu",
        (Lang::En, "ready") => "Ready.",
        (Lang::En, "starting") => "Starting system-audio capture...",
        (Lang::En, "helper_ready") => "Capture helper is ready; waiting for audio.",
        (Lang::En, "running") => "Analyzing system audio.",
        (Lang::En, "waiting_audio") => "Capture is running, but no audio has arrived yet. Make sure audio is playing.",
        (Lang::En, "stopped") => "Stopped.",
        (Lang::En, "permission_needed") => "macOS permission is required. Allow Screen & System Audio Recording in System Settings.",
        (Lang::En, "capture_failed") => "Capture failed.",
        (Lang::En, "state_idle") => "Idle",
        (Lang::En, "state_starting") => "Starting",
        (Lang::En, "state_running") => "Running",
        (Lang::En, "state_permission") => "Permission",
        (Lang::En, "state_failed") => "Error",
        (Lang::En, "too_small") => "Terminal window is too small.",
        (Lang::En, "theme_system") => "System",
        (Lang::En, "theme_graphite") => "Graphite",
        (Lang::En, "theme_ocean") => "Ocean",
        (Lang::En, "theme_aurora") => "Aurora",
        (Lang::En, "theme_sonic_texture") => "Sonic Texture",
        (Lang::En, "theme_noise_warp") => "Noise Warp",
        (Lang::En, "theme_amber") => "Amber",
        (Lang::En, "theme_mono") => "Mono",
        (Lang::En, "renderer_blocks") => "Blocks",
        (Lang::En, "renderer_braille") => "Braille",
        (Lang::En, "preset_low_latency") => "Low Latency",
        (Lang::En, "preset_balanced") => "Balanced",
        (Lang::En, "preset_precision") => "Precision",
        (Lang::En, "preset_custom") => "Custom",

        (Lang::Ja, "main_menu") => "メインメニュー",
        (Lang::Ja, "menu_spectrum") => "スペクトラムを開く",
        (Lang::Ja, "menu_toggle") => "キャプチャ切替",
        (Lang::Ja, "menu_start") => "キャプチャ開始",
        (Lang::Ja, "menu_stop") => "キャプチャ停止",
        (Lang::Ja, "menu_settings") => "設定",
        (Lang::Ja, "menu_help") => "ヘルプ",
        (Lang::Ja, "menu_quit") => "終了",
        (Lang::Ja, "subtitle") => "システム音声スペクトラム",
        (Lang::Ja, "overview") => "概要",
        (Lang::Ja, "preview") => "プレビュー",
        (Lang::Ja, "spectrum") => "スペクトラム",
        (Lang::Ja, "status") => "状態",
        (Lang::Ja, "level") => "レベル",
        (Lang::Ja, "config") => "設定ファイル",
        (Lang::Ja, "settings") => "設定",
        (Lang::Ja, "language") => "言語",
        (Lang::Ja, "theme") => "テーマ",
        (Lang::Ja, "smoothing") => "平滑化",
        (Lang::Ja, "analysis_preset") => "解析プリセット",
        (Lang::Ja, "attack") => "Attack",
        (Lang::Ja, "release") => "Release",
        (Lang::Ja, "analysis_hop") => "Hop",
        (Lang::Ja, "bars") => "バンド",
        (Lang::Ja, "renderer") => "描画",
        (Lang::Ja, "fft_size") => "FFT",
        (Lang::Ja, "refresh_rate") => "更新率",
        (Lang::Ja, "audio_delay") => "音声遅延",
        (Lang::Ja, "high_shelf") => "高域補正",
        (Lang::Ja, "high_shelf_db") => "補正量",
        (Lang::Ja, "visual_curve") => "高さ曲線",
        (Lang::Ja, "curve_power") => "曲線指数",
        (Lang::Ja, "trail") => "残像",
        (Lang::Ja, "trail_decay") => "残像減衰",
        (Lang::Ja, "accent_trace") => "アクセント輪郭",
        (Lang::Ja, "accent_threshold") => "アクセント閾値",
        (Lang::Ja, "ceiling") => "上限",
        (Lang::Ja, "pipeline") => "音声チェーン",
        (Lang::Ja, "toolbar") => "ツールバー",
        (Lang::Ja, "master") => "master",
        (Lang::Ja, "waveform") => "波形",
        (Lang::Ja, "modules") => "モジュール",
        (Lang::Ja, "on") => "オン",
        (Lang::Ja, "off") => "オフ",
        (Lang::Ja, "transport_idle") => "○ 待機",
        (Lang::Ja, "transport_starting") => "◐ 起動",
        (Lang::Ja, "transport_running") => "● 実行",
        (Lang::Ja, "transport_permission") => "! 権限",
        (Lang::Ja, "transport_failed") => "× エラー",
        (Lang::Ja, "controls") => "操作",
        (Lang::Ja, "help") => "ヘルプ",
        (Lang::Ja, "help_title") => "Terb ターミナルスペクトラム",
        (Lang::Ja, "help_1") => "↑/↓ または j/k で移動します。",
        (Lang::Ja, "help_2") => "Enter で実行、Space でキャプチャ開始/停止。",
        (Lang::Ja, "help_3") => "スペクトラム画面では s/p/t/m/w で設定、チェーン、ツールバー、master、波形を表示/非表示にします。",
        (Lang::Ja, "help_4") => "↑/↓ で設定選択、←/→ で変更。-/= で音声遅延を調整、S で全画面設定。",
        (Lang::Ja, "help_5") => "小さいウィンドウではモジュールを自動で隠しますが、ショートカットは使えます。メインメニューでは q/Esc で終了します。",
        (Lang::Ja, "permission_note") => "初回キャプチャでは macOS の画面とシステム音声録音権限が必要です。Terb はリアルタイム解析のみ行い、音声を保存しません。",
        (Lang::Ja, "menu_hint") => "↑/↓ 選択 · Enter 決定 · Space キャプチャ · ? ヘルプ · q 終了",
        (Lang::Ja, "spectrum_hint") => "Space キャプチャ · -/= 遅延 · s/p/t/m/w モジュール · S 設定 · q メニュー",
        (Lang::Ja, "sidebar_hint") => "Space キャプチャ切替\n-/= 音声遅延\ns/p/t/m/w モジュール\n↑/↓ 設定選択\n←/→ 変更\nS 設定\nq メニュー\n? ヘルプ",
        (Lang::Ja, "compact_hint") => "-/= 遅延 · s/p/t/m/w モジュール · S 設定 · q メニュー",
        (Lang::Ja, "ready") => "準備完了。",
        (Lang::Ja, "starting") => "システム音声キャプチャを開始しています...",
        (Lang::Ja, "helper_ready") => "キャプチャヘルパーは準備完了。音声を待っています。",
        (Lang::Ja, "running") => "システム音声を解析中です。",
        (Lang::Ja, "waiting_audio") => "キャプチャ中ですが音声が届いていません。音声が再生中か確認してください。",
        (Lang::Ja, "stopped") => "停止しました。",
        (Lang::Ja, "permission_needed") => "macOS の権限が必要です。システム設定で画面とシステム音声録音を許可してください。",
        (Lang::Ja, "capture_failed") => "キャプチャに失敗しました。",
        (Lang::Ja, "state_idle") => "待機",
        (Lang::Ja, "state_starting") => "起動中",
        (Lang::Ja, "state_running") => "実行中",
        (Lang::Ja, "state_permission") => "権限待ち",
        (Lang::Ja, "state_failed") => "エラー",
        (Lang::Ja, "too_small") => "ターミナルウィンドウが小さすぎます。",
        (Lang::Ja, "theme_system") => "システム",
        (Lang::Ja, "theme_graphite") => "グラファイト",
        (Lang::Ja, "theme_ocean") => "オーシャン",
        (Lang::Ja, "theme_aurora") => "オーロラ",
        (Lang::Ja, "theme_sonic_texture") => "音紋フィールド",
        (Lang::Ja, "theme_noise_warp") => "ノイズワープ",
        (Lang::Ja, "theme_amber") => "アンバー",
        (Lang::Ja, "theme_mono") => "モノ",
        (Lang::Ja, "renderer_blocks") => "ブロック",
        (Lang::Ja, "renderer_braille") => "点字",
        (Lang::Ja, "preset_low_latency") => "低遅延",
        (Lang::Ja, "preset_balanced") => "バランス",
        (Lang::Ja, "preset_precision") => "精密",
        (Lang::Ja, "preset_custom") => "カスタム",

        _ => key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    fn buffer_row(buf: &Buffer, y: u16, width: u16) -> String {
        (0..width).map(|x| buf[(x, y)].symbol()).collect()
    }

    fn color_delta(left: Color, right: Color) -> u16 {
        let left = color_to_rgb(left);
        let right = color_to_rgb(right);
        (left.0.abs_diff(right.0) as u16)
            + (left.1.abs_diff(right.1) as u16)
            + (left.2.abs_diff(right.2) as u16)
    }

    #[test]
    fn settings_list_fits_twenty_rows_without_scrolling() {
        assert_eq!(visible_list_start(0, SETTING_COUNT, 20), 0);
        assert_eq!(visible_list_start(SETTING_COUNT - 1, SETTING_COUNT, 20), 0);
    }

    #[test]
    fn settings_list_scrolls_selected_row_into_view() {
        assert_eq!(visible_list_start(SETTING_COUNT - 1, SETTING_COUNT, 18), 2);
    }

    #[test]
    fn panel_renders_rounded_btop_style_title_cutout() {
        let theme = theme(ThemeId::System);
        let area = Rect::new(0, 0, 16, 3);
        let mut buf = Buffer::empty(area);

        Panel::new(Some(inline_hotkey_label(theme, 's', "settings")), theme).render(area, &mut buf);

        assert_eq!(buffer_row(&buf, 0, 16), "╭─┐settings┌───╮");
        assert_eq!(buffer_row(&buf, 1, 16), "│              │");
        assert_eq!(buffer_row(&buf, 2, 16), "╰──────────────╯");
    }

    #[test]
    fn english_module_title_highlights_inline_hotkey() {
        let mut config = Config::default();
        config.settings.language = "en".to_string();
        let app = App::new(config);
        let line = module_title_line(&app, 's', "settings");

        assert_eq!(line_text(&line), "settings");
        assert_eq!(line.spans[0].content.as_ref(), "s");
        assert_eq!(line.spans[1].content.as_ref(), "ettings");
    }

    #[test]
    fn localized_module_title_uses_bracketed_hotkey() {
        let app = App::new(Config::default());
        let line = module_title_line(&app, 'p', "pipeline");

        assert_eq!(line_text(&line), "[p] 音频链路");
    }

    #[test]
    fn display_bars_expands_to_full_width() {
        let bars = display_bars(&[0.10, 0.90], 5);

        assert_eq!(bars.len(), 5);
        assert!((bars[0] - 0.10).abs() < f32::EPSILON);
        assert!((bars[4] - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn display_bars_preserves_peaks_when_downsampling() {
        let bars = display_bars(&[0.10, 0.95, 0.20, 0.30], 2);

        assert_eq!(bars, vec![0.95, 0.30]);
    }

    #[test]
    fn display_waveform_preserves_signed_column_extents() {
        let columns = display_waveform(&[-0.20, 0.70, -0.85, 0.10], 2);

        assert_eq!(
            columns,
            vec![
                WaveformColumn {
                    min: -0.20,
                    max: 0.70
                },
                WaveformColumn {
                    min: -0.85,
                    max: 0.10
                }
            ]
        );
    }

    #[test]
    fn waveform_braille_cell_uses_signed_subpixel_columns() {
        let columns = vec![
            WaveformColumn { min: 0.0, max: 1.0 },
            WaveformColumn {
                min: -1.0,
                max: 0.0,
            },
        ];

        let (mask, value) = waveform_braille_cell(&columns, 0, 0, 4, 1.0);

        assert_eq!(mask, 0xa3);
        assert_eq!(value, 1.0);
    }

    #[test]
    fn waveform_centerline_uses_braille_dots() {
        assert_eq!(waveform_centerline_mask(0, 4), 0x24);
    }

    #[test]
    fn master_braille_cell_samples_subpixel_rows() {
        let (mask, value) = master_braille_cell(0.20, 0, 0, 4);

        assert_eq!(mask, 0xc0);
        assert!(value > 0.0);
    }

    #[test]
    fn master_braille_cell_is_solid_near_bottom() {
        let (mask, _) = master_braille_cell(0.30, 0, 3, 16);

        assert_eq!(mask, 0xff);
    }

    #[test]
    fn master_braille_cell_stays_solid_near_top() {
        let (mask, _) = master_braille_cell(1.0, 0, 0, 16);

        assert_eq!(mask, 0xff);
    }

    #[test]
    fn master_meter_color_blends_border_toward_theme_accent() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::Ocean;

        assert_eq!(master_meter_color(&app, 0.0), Color::Rgb(82, 88, 100));
        assert_eq!(master_meter_color(&app, 1.0), Color::Rgb(17, 168, 205));
        assert_ne!(master_meter_color(&app, 0.5), master_meter_color(&app, 0.0));
        assert_ne!(master_meter_color(&app, 0.5), master_meter_color(&app, 1.0));
    }

    #[test]
    fn aurora_theme_enables_music_reactive_spectrum_color() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::Aurora;
        app.color_state = VisualColorState {
            centroid: 0.20,
            energy: 0.25,
            flux: 0.10,
            phase: 0.0,
            ..VisualColorState::default()
        };
        let low_state_color = spectrum_bar_color(&app, 12, 100, 0.70, false);
        app.color_state = VisualColorState {
            centroid: 0.82,
            energy: 0.80,
            flux: 0.70,
            phase: 0.32,
            ..VisualColorState::default()
        };
        let high_state_color = spectrum_bar_color(&app, 12, 100, 0.70, false);

        assert_eq!(theme(ThemeId::Aurora).color_mode, ColorMode::Aurora);
        assert_ne!(low_state_color, high_state_color);
    }

    #[test]
    fn music_reactive_theme_modes_are_available() {
        assert_eq!(
            theme(ThemeId::SonicTexture).color_mode,
            ColorMode::SonicTexture
        );
        assert_eq!(theme(ThemeId::NoiseWarp).color_mode, ColorMode::NoiseWarp);
        assert!(!THEMES.contains(&ThemeId::PitchClass));
        assert!(!THEMES.contains(&ThemeId::ChromaBands));
        assert!(!THEMES.contains(&ThemeId::PitchMemory));
        assert!(!THEMES.contains(&ThemeId::HarmonicComb));
    }

    #[test]
    fn sonic_texture_theme_varies_inside_a_column_without_large_stripes() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::SonicTexture;
        app.color_state = VisualColorState {
            centroid: 0.34,
            energy: 0.62,
            flux: 0.18,
            phase: 0.21,
            chroma: [
                0.18, 0.26, 0.42, 0.22, 0.71, 0.34, 0.28, 0.58, 0.37, 0.24, 0.19, 0.31,
            ],
            dominant_pitch: 4,
            pitch_confidence: 0.82,
            ..VisualColorState::default()
        };

        let lower = music_color_for_position_at(&app, 0.38, 0.72, 0.24, false);
        let upper = music_color_for_position_at(&app, 0.38, 0.72, 0.82, false);
        let left = music_color_for_position_at(&app, 0.380, 0.72, 0.54, false);
        let right = music_color_for_position_at(&app, 0.386, 0.72, 0.54, false);

        assert_ne!(lower, upper);
        assert!(color_delta(left, right) < 45);
    }

    #[test]
    fn noise_warp_theme_has_smooth_height_based_texture() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::NoiseWarp;
        app.color_state = VisualColorState {
            centroid: 0.56,
            energy: 0.74,
            flux: 0.24,
            phase: 0.33,
            dominant_pitch: 9,
            pitch_confidence: 0.65,
            ..VisualColorState::default()
        };

        let lower = music_color_for_position_at(&app, 0.58, 0.76, 0.18, false);
        let upper = music_color_for_position_at(&app, 0.58, 0.76, 0.86, false);
        let left = music_color_for_position_at(&app, 0.512, 0.76, 0.62, false);
        let right = music_color_for_position_at(&app, 0.518, 0.76, 0.62, false);

        assert_ne!(lower, upper);
        assert!(color_delta(left, right) < 55);
    }

    #[test]
    fn pitch_class_for_frequency_maps_a440_to_a() {
        assert_eq!(pitch_class_for_frequency(440.0), 9);
        assert_eq!(pitch_class_for_frequency(261.63), 0);
    }

    #[test]
    fn visual_color_state_tracks_frequency_centroid() {
        let mut state = VisualColorState::default();
        let previous = vec![0.0; 4];

        state.update(&[0.0, 0.0, 0.0, 1.0], &previous, 0.0);

        assert!(state.centroid > 0.45);
        assert!(state.flux > 0.0);
    }

    #[test]
    fn visual_color_state_filters_pitch_hue_and_flux_over_time() {
        let mut state = VisualColorState::default();
        let previous = vec![0.0; 12];
        let mut bars = vec![0.0; 12];
        bars[9] = 1.0;

        state.update(&bars, &previous, 0.0);

        let target_hue = pitch_class_hue(state.dominant_pitch);
        assert_ne!(state.pitch_hue, target_hue);
        assert!(state.flux < 1.0);

        let hue_after_first_frame = state.pitch_hue;
        state.update(&bars, &bars, 0.0);

        assert!(state.flux < 0.35);
        assert_ne!(state.pitch_hue, hue_after_first_frame);
    }

    #[test]
    fn retired_theme_config_migrates_to_sonic_texture() {
        let mut config = Config::default();
        config.settings.theme = ThemeId::HarmonicComb;

        let app = App::new(config);

        assert_eq!(app.theme_id, ThemeId::SonicTexture);
        assert_eq!(app.config.settings.theme, ThemeId::SonicTexture);
    }

    #[test]
    fn stopping_capture_preserves_last_visual_frame() {
        let mut app = App::new(Config::default());
        app.level = 0.42;
        app.master_left = 0.72;
        app.master_right = 0.38;
        app.spectrum[0] = 0.64;
        app.spectrum_trail[0] = 0.80;
        app.waveform[0] = -0.50;

        app.stop_capture();

        assert!((app.level - 0.42).abs() < f32::EPSILON);
        assert!((app.master_left - 0.72).abs() < f32::EPSILON);
        assert!((app.master_right - 0.38).abs() < f32::EPSILON);
        assert!((app.spectrum[0] - 0.64).abs() < f32::EPSILON);
        assert!((app.spectrum_trail[0] - 0.80).abs() < f32::EPSILON);
        assert!((app.waveform[0] + 0.50).abs() < f32::EPSILON);
    }

    #[test]
    fn stale_audio_samples_after_stop_do_not_mutate_waveform() {
        let mut app = App::new(Config::default());
        let stale_capture_id = app.capture_id;
        app.capture_id = app.capture_id.wrapping_add(1);
        app.waveform.fill(0.0);
        app.tx
            .send(AudioEvent::Samples(
                stale_capture_id,
                AudioSamples {
                    mono: vec![1.0; 512],
                    left_level: 1.0,
                    right_level: 1.0,
                },
            ))
            .expect("event should enqueue");

        app.drain_audio_events();

        assert!(app.waveform.iter().all(|sample| *sample == 0.0));
        assert!(app.capture_state == CaptureState::Idle);
    }

    #[test]
    fn render_bar_value_maps_ceiling_to_full_height() {
        let settings = Settings {
            ceiling: 0.88,
            ..Config::default().settings
        };

        assert!((render_bar_value(0.88, &settings) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn render_bar_value_removes_noise_floor() {
        let settings = Config::default().settings;

        assert_eq!(render_bar_value(VISUAL_NOISE_FLOOR * 0.5, &settings), 0.0);
        assert!(render_bar_value(VISUAL_NOISE_FLOOR * 2.0, &settings) > 0.0);
    }

    #[test]
    fn spectrum_trail_decays_after_peak() {
        let settings = Settings {
            trail_decay: 0.80,
            ..Config::default().settings
        };
        let mut trail = vec![0.0];

        update_spectrum_trail(&mut trail, &[1.0], &settings);
        update_spectrum_trail(&mut trail, &[0.25], &settings);

        assert!((trail[0] - 0.80).abs() < f32::EPSILON);
    }

    #[test]
    fn disabled_spectrum_trail_tracks_current_bars() {
        let settings = Settings {
            trail_enabled: false,
            ..Config::default().settings
        };
        let mut trail = vec![0.90, 0.80];

        update_spectrum_trail(&mut trail, &[0.10, 0.20], &settings);

        assert_eq!(trail, vec![0.10, 0.20]);
    }

    #[test]
    fn accent_trigger_threshold_scales_detection_strictness() {
        let low = accent_trigger_thresholds(0.10);
        let default = accent_trigger_thresholds(0.50);
        let high = accent_trigger_thresholds(0.90);

        assert!(low.peak < default.peak);
        assert!(default.peak < high.peak);
        assert!((default.ratio - 1.45).abs() < 0.001);
        assert!((default.flux - 0.035).abs() < 0.001);
    }

    #[test]
    fn accent_trace_line_shifts_envelope_up_five_cells() {
        let envelope = vec![0.50, 0.50];

        let (shifted_mask, value) = accent_trace_braille_cell(&envelope, 0, 1, 48);
        let (lower_cell_mask, _) = accent_trace_braille_cell(&envelope, 0, 6, 48);

        assert_eq!(shifted_mask, 0x09);
        assert_eq!(value, 0.50);
        assert_eq!(lower_cell_mask, 0);
    }

    #[test]
    fn accent_trace_line_interpolates_large_vertical_jumps() {
        let envelope = vec![0.90, 0.10];

        let (middle_mask, value) = accent_trace_braille_cell(&envelope, 0, 3, 40);

        assert_ne!(middle_mask & 0x38, 0);
        assert_eq!(value, 0.90);
    }

    #[test]
    fn accent_traces_fade_out_after_half_second() {
        let mut app = App::new(Config::default());

        app.push_accent_trace(&[0.80, 0.60]);
        app.advance_accent_traces(Duration::from_millis(250));

        assert_eq!(app.accent_traces.len(), 1);
        assert!(app.accent_traces[0].fade() < 0.51);

        app.advance_accent_traces(Duration::from_millis(251));

        assert!(app.accent_traces.is_empty());
    }

    #[test]
    fn pushing_new_accent_trace_replaces_previous_visible_trace() {
        let mut app = App::new(Config::default());

        app.push_accent_trace(&[0.80, 0.60]);
        app.push_accent_trace(&[0.20, 0.30]);

        assert_eq!(app.accent_traces.len(), 1);
        assert_eq!(app.accent_traces[0].envelope, vec![0.20, 0.30]);
    }

    #[test]
    fn accent_trace_envelope_smoothing_reduces_jaggedness() {
        let mut envelope = vec![0.10_f32, 0.95, 0.12, 0.90, 0.08, 0.85, 0.10];
        let before = envelope
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .sum::<f32>();

        smooth_accent_trace_envelope(&mut envelope);

        let after = envelope
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .sum::<f32>();
        assert!(after < before * 0.60);
    }

    #[test]
    fn fading_accent_overlay_preserves_underlying_spectrum_color() {
        let theme = theme(ThemeId::PitchMemory);
        let base = Color::Rgb(112, 94, 143);
        let overlay = AccentTraceOverlay {
            mask: 0x09,
            color: Color::Rgb(220, 120, 180),
            visibility: 0.0,
        };

        assert_eq!(accent_trace_overlay_color(theme, Some(base), overlay), base);
        assert_eq!(
            accent_trace_overlay_color(theme, None, overlay),
            accent_trace_background_color(theme)
        );
    }

    #[test]
    fn accent_trace_detector_waits_for_accent_to_settle() {
        let mut app = App::new(Config::default());
        app.spectrum = vec![0.0; 32];

        app.update_accent_trace_detector(&vec![0.80; 32]);

        assert!(app.pending_accent_trace.is_some());
        assert!(app.accent_traces.is_empty());

        app.spectrum = vec![0.80; 32];
        app.update_accent_trace_detector(&vec![0.55; 32]);

        assert_eq!(app.accent_traces.len(), 1);
        assert!(!app.accent_trace_cooldown.is_zero());
    }

    #[test]
    fn disabled_accent_trace_detector_skips_trace() {
        let mut app = App::new(Config::default());
        app.config.settings.accent_trace_enabled = false;
        app.spectrum = vec![0.0; 32];

        app.update_accent_trace_detector(&vec![0.80; 32]);

        assert!(app.pending_accent_trace.is_none());
        assert!(app.accent_traces.is_empty());
    }

    #[test]
    fn analyzer_gates_silence_to_zero() {
        let settings = Config::default().settings;
        let pipeline = SpectrumPipeline::from_settings(&settings);
        let mut analyzer = SpectrumAnalyzer::new(1024, 48_000.0, 32, 256);
        let bars = analyzer
            .consume(
                &vec![0.0; 2048],
                settings.attack,
                settings.release,
                pipeline,
            )
            .expect("enough samples");

        assert!(bars.iter().all(|bar| *bar == 0.0));
    }

    #[test]
    fn legacy_config_defaults_to_block_renderer() {
        let config: Config = serde_json::from_str(
            r#"{
                "version": 1,
                "settings": {
                    "language": "zh",
                    "theme": "System",
                    "smoothing": 0.72,
                    "bar_count": 72,
                    "refresh_hz": 45
                }
            }"#,
        )
        .expect("legacy config should still load");

        assert_eq!(config.settings.renderer, SpectrumRenderer::Blocks);
    }

    #[test]
    fn visual_bar_count_tracks_renderer_subpixels() {
        let mut app = App::new(Config::default());
        let area = Rect::new(0, 0, 100, 30);
        app.screen = Screen::Spectrum;

        app.config.settings.renderer = SpectrumRenderer::Blocks;
        assert_eq!(visual_bar_count(&app, area), Some(82));

        app.config.settings.renderer = SpectrumRenderer::Braille;
        assert_eq!(visual_bar_count(&app, area), Some(164));
    }

    #[test]
    fn visual_bar_count_resizes_spectrum_analyzer() {
        let mut app = App::new(Config::default());

        app.set_visual_bar_count(160);

        assert_eq!(app.visual_bar_count, 160);
        assert_eq!(app.spectrum.len(), 160);
        assert_eq!(app.analyzer.bar_count, 160);
    }

    #[test]
    fn sample_magnitude_interpolates_fractional_bins() {
        let value = sample_magnitude(&[0.0, 1.0, 3.0, 4.0], 1.5);

        assert!((value - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn braille_bar_cell_uses_standard_dot_order() {
        let (left_mask, left_value) = braille_bar_cell(&[1.0, 0.0], 0, 0, 4);
        let (right_mask, right_value) = braille_bar_cell(&[0.0, 1.0], 0, 0, 4);

        assert_eq!(left_mask, 0x47);
        assert_eq!(right_mask, 0xb8);
        assert_eq!(left_value, 1.0);
        assert_eq!(right_value, 1.0);
    }

    #[test]
    fn braille_bar_cell_fills_from_bottom() {
        let (left_mask, _) = braille_bar_cell(&[0.20, 0.0], 0, 0, 4);
        let (right_mask, _) = braille_bar_cell(&[0.0, 0.20], 0, 0, 4);

        assert_eq!(left_mask, 0x40);
        assert_eq!(right_mask, 0x80);
    }

    #[test]
    fn braille_pattern_offsets_from_unicode_braille_block() {
        assert_eq!(braille_pattern(0x01), '\u{2801}');
        assert_eq!(braille_pattern(0xff), '\u{28ff}');
    }

    #[test]
    fn analyzer_respects_analysis_hop() {
        let settings = Config::default().settings;
        let pipeline = SpectrumPipeline::from_settings(&settings);
        let mut analyzer = SpectrumAnalyzer::new(1024, 48_000.0, 32, 256);

        assert!(analyzer
            .consume(&vec![0.0; 512], settings.attack, settings.release, pipeline)
            .is_none());
        assert!(analyzer
            .consume(&vec![0.0; 512], settings.attack, settings.release, pipeline)
            .is_some());
        assert!(analyzer
            .consume(&vec![0.0; 128], settings.attack, settings.release, pipeline)
            .is_none());
        assert!(analyzer
            .consume(&vec![0.0; 128], settings.attack, settings.release, pipeline)
            .is_some());
    }

    #[test]
    fn low_latency_preset_applies_realtime_analysis_defaults() {
        let mut settings = Settings {
            analysis_preset: AnalysisPreset::Balanced,
            fft_size: 4096,
            analysis_hop: 1024,
            refresh_hz: 45,
            attack: 0.20,
            release: 0.90,
            ..Config::default().settings
        };

        settings.apply_analysis_preset(AnalysisPreset::LowLatency);

        assert_eq!(settings.fft_size, 1024);
        assert_eq!(settings.analysis_hop, 256);
        assert_eq!(settings.refresh_hz, 90);
        assert!((settings.attack - DEFAULT_ATTACK).abs() < f32::EPSILON);
        assert!((settings.release - DEFAULT_RELEASE).abs() < f32::EPSILON);
    }
}
