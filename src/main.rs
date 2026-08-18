use std::{
    env, fs, io,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

pub(crate) mod analysis;
pub(crate) mod bpm;

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
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use analysis::sample_frequency_band;
#[cfg(test)]
use analysis::sample_magnitude;
use bpm::{BpmAnalyzer, BPM_MAX, BPM_MIN, BPM_PULSE_DECAY_SECONDS};

const LANGUAGES: &[(&str, &str)] = &[("zh", "中文"), ("en", "English"), ("ja", "日本語")];
const THEMES: &[ThemeId] = &[ThemeId::Spring, ThemeId::Vintage, ThemeId::Mono];
const SPECTRUM_RENDERERS: &[SpectrumRenderer] = &[
    SpectrumRenderer::Blocks,
    SpectrumRenderer::Braille,
    SpectrumRenderer::Cava,
];
const ACCENT_DISPLAY_MODES: &[AccentDisplayMode] = &[
    AccentDisplayMode::Trace,
    AccentDisplayMode::NoteNames,
    AccentDisplayMode::Off,
];
const MENU_ITEMS: &[&str] = &[
    "menu_spectrum",
    "menu_toggle",
    "menu_settings",
    "menu_help",
    "menu_quit",
];
const REFRESH_RATES: &[u16] = &[12, 24, 30, 45, 60, 90, 120, 144, 165, 240];
const MIN_REFRESH_HZ: u16 = 12;
const MAX_REFRESH_HZ: u16 = 240;
const ANALYSIS_HOPS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
const MIN_FREQUENCY: f32 = 35.0;
const MAX_FREQUENCY: f32 = 18_000.0;
const DEFAULT_HIGH_SHELF_DB: f32 = 6.0;
const DEFAULT_NOISE_REDUCTION: f32 = 0.26;
const DEFAULT_VISUAL_CURVE: f32 = 0.88;
const DEFAULT_CEILING: f32 = 0.88;
const DEFAULT_TRAIL_DECAY: f32 = 0.88;
const DEFAULT_ACCENT_TRACE_THRESHOLD: f32 = 0.50;
const MIN_CONFIG_BARS: usize = 8;
const MAX_CONFIG_BARS: usize = 256;
const MIN_ATTACK: f32 = 0.02;
const MAX_ATTACK: f32 = 1.00;
const ATTACK_STEP: f32 = 0.02;
const MIN_RELEASE: f32 = 0.00;
const MAX_RELEASE: f32 = 0.995;
const RELEASE_STEP: f32 = 0.02;
const MIN_HIGH_SHELF_DB: f32 = 0.0;
const MAX_HIGH_SHELF_DB: f32 = 36.0;
const HIGH_SHELF_DB_STEP: f32 = 1.0;
const MIN_NOISE_REDUCTION: f32 = 0.0;
const MAX_NOISE_REDUCTION: f32 = 0.95;
const NOISE_REDUCTION_STEP: f32 = 0.05;
const MIN_VISUAL_CURVE: f32 = 0.25;
const MAX_VISUAL_CURVE: f32 = 2.50;
const VISUAL_CURVE_STEP: f32 = 0.05;
const MIN_CEILING: f32 = 0.35;
const MAX_CEILING: f32 = 1.00;
const CEILING_STEP: f32 = 0.05;
const MIN_TRAIL_DECAY: f32 = 0.20;
const MAX_TRAIL_DECAY: f32 = 0.995;
const TRAIL_DECAY_STEP: f32 = 0.05;
const MIN_ACCENT_TRACE_THRESHOLD: f32 = 0.02;
const MAX_ACCENT_TRACE_THRESHOLD: f32 = 0.98;
const ACCENT_TRACE_THRESHOLD_STEP: f32 = 0.02;
const ACCENT_TRACE_LIFETIME_MS: u64 = 500;
const ACCENT_TRACE_COOLDOWN_MS: u64 = 90;
const ACCENT_TRACE_START_OFFSET_CELLS: f32 = 1.0;
const ACCENT_TRACE_END_OFFSET_CELLS: f32 = 5.0;
const ACCENT_TRACE_OFFSET_ANIMATION_MS: u64 = 100;
const ACCENT_TRACE_SETTLE_FRAMES: usize = 1;
const ACCENT_TRACE_MAX_CAPTURE_FRAMES: usize = 8;
const ACCENT_TRACE_RISE_EPSILON: f32 = 0.012;
const ACCENT_NOTE_FADE_IN_MS: u64 = 100;
const ACCENT_NOTE_FADE_OUT_MS: u64 = 500;
const ACCENT_NOTE_COUNT: usize = 28;
const MASTER_METER_WIDTH: u16 = 16;
const MASTER_METER_CHANNEL_WIDTH: usize = 7;
const BRAILLE_LEFT_COLUMN_MASK: u8 = 0x01 | 0x02 | 0x04 | 0x40;
const BRAILLE_RIGHT_COLUMN_MASK: u8 = 0x08 | 0x10 | 0x20 | 0x80;
const DEFAULT_ATTACK: f32 = 0.82;
const DEFAULT_RELEASE: f32 = 0.48;
const DEFAULT_ANALYSIS_HOP: usize = 256;
const FFT_SIZES: &[usize] = &[512, 1024, 2048, 4096, 8192, 16_384];
const MAX_ANALYSIS_BARS: usize = 1024;
const ADAPTIVE_GAIN_MIN: f32 = 0.45;
const ADAPTIVE_GAIN_MAX: f32 = 6.0;
const ADAPTIVE_TARGET_RMS: f32 = 0.58;
const ADAPTIVE_TARGET_PEAK: f32 = 0.94;
const AUDIO_READ_FRAMES: usize = 512;
const VISUAL_NOISE_FLOOR: f32 = 0.025;
const SILENCE_GATE: f32 = 0.000_12;
const WAVEFORM_SAMPLES: usize = 1024;
const WAVEFORM_TARGET_PEAK: f32 = 0.72;
const BRAILLE_DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
const CAVA_BLOCKS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
const ACCENT_TRACE_REFERENCE_VIRTUAL_HEIGHT: f32 = 48.0;
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
        KeyCode::Char('?') => app.screen = Screen::Help,
        KeyCode::Up | KeyCode::Char('k') => app.prev_setting(),
        KeyCode::Down | KeyCode::Char('j') => app.next_setting(),
        KeyCode::BackTab => app.prev_setting_category(),
        KeyCode::Tab => app.next_setting_category(),
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
        KeyCode::BackTab => app.prev_setting_category(),
        KeyCode::Tab => app.next_setting_category(),
        KeyCode::Left | KeyCode::Char('h') => app.adjust_setting(-1),
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => app.adjust_setting(1),
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
    Spring,
    // Legacy theme ids are kept only so older config files can be migrated.
    System,
    Graphite,
    Ocean,
    Vintage,
    Aurora,
    PitchClass,
    ChromaBands,
    PitchMemory,
    HarmonicComb,
    SonicTexture,
    NoiseWarp,
    Miku,
    AlbumArt,
    AlbumArtCircle,
    Amber,
    Mono,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SpectrumRenderer {
    Blocks,
    Braille,
    Cava,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AccentDisplayMode {
    Off,
    Trace,
    NoteNames,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SettingCategory {
    General,
    Analysis,
    Processing,
    Visual,
}

const SETTING_CATEGORIES: &[SettingCategory] = &[
    SettingCategory::General,
    SettingCategory::Analysis,
    SettingCategory::Processing,
    SettingCategory::Visual,
];

#[derive(Clone, Debug)]
struct SettingRow {
    index: usize,
    key: &'static str,
    category: SettingCategory,
    value: String,
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
    #[serde(default = "default_high_shelf_enabled")]
    high_shelf_enabled: bool,
    #[serde(default = "default_high_shelf_db")]
    high_shelf_db: f32,
    #[serde(default = "default_auto_sensitivity_enabled")]
    auto_sensitivity_enabled: bool,
    #[serde(default = "default_noise_reduction")]
    noise_reduction: f32,
    #[serde(default = "default_bpm_enabled")]
    bpm_enabled: bool,
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
    #[serde(default = "default_accent_display_mode")]
    accent_display_mode: AccentDisplayMode,
    #[serde(default = "default_accent_trace_enabled", skip_serializing)]
    // Legacy config field. Old `false` values migrate to `accent_display_mode = off`.
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

fn default_fft_size() -> usize {
    1024
}

fn default_analysis_preset() -> AnalysisPreset {
    AnalysisPreset::LowLatency
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

fn default_auto_sensitivity_enabled() -> bool {
    true
}

fn default_noise_reduction() -> f32 {
    DEFAULT_NOISE_REDUCTION
}

fn default_bpm_enabled() -> bool {
    true
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

fn default_accent_display_mode() -> AccentDisplayMode {
    AccentDisplayMode::Trace
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
                theme: ThemeId::Spring,
                analysis_preset: default_analysis_preset(),
                attack: default_attack(),
                release: default_release(),
                bar_count: 72,
                renderer: default_spectrum_renderer(),
                fft_size: default_fft_size(),
                analysis_hop: default_analysis_hop(),
                refresh_hz: default_refresh_hz(),
                high_shelf_enabled: default_high_shelf_enabled(),
                high_shelf_db: default_high_shelf_db(),
                auto_sensitivity_enabled: default_auto_sensitivity_enabled(),
                noise_reduction: default_noise_reduction(),
                bpm_enabled: default_bpm_enabled(),
                visual_curve_enabled: default_visual_curve_enabled(),
                visual_curve: default_visual_curve(),
                ceiling: default_ceiling(),
                trail_enabled: default_trail_enabled(),
                trail_decay: default_trail_decay(),
                accent_display_mode: default_accent_display_mode(),
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
        self.bar_count = self.bar_count.clamp(MIN_CONFIG_BARS, MAX_CONFIG_BARS);
        self.fft_size = nearest_fft_size(self.fft_size);
        self.analysis_hop = nearest_hop_size(self.analysis_hop, self.fft_size);
        self.refresh_hz = nearest_refresh_rate(self.refresh_hz);
        self.attack = self.attack.clamp(MIN_ATTACK, MAX_ATTACK);
        self.release = self.release.clamp(MIN_RELEASE, MAX_RELEASE);
        self.high_shelf_db = self
            .high_shelf_db
            .clamp(MIN_HIGH_SHELF_DB, MAX_HIGH_SHELF_DB);
        self.noise_reduction = self
            .noise_reduction
            .clamp(MIN_NOISE_REDUCTION, MAX_NOISE_REDUCTION);
        self.visual_curve = self.visual_curve.clamp(MIN_VISUAL_CURVE, MAX_VISUAL_CURVE);
        self.ceiling = self.ceiling.clamp(MIN_CEILING, MAX_CEILING);
        self.trail_decay = self.trail_decay.clamp(MIN_TRAIL_DECAY, MAX_TRAIL_DECAY);
        self.accent_trace_threshold = self
            .accent_trace_threshold
            .clamp(MIN_ACCENT_TRACE_THRESHOLD, MAX_ACCENT_TRACE_THRESHOLD);
        if !self.accent_trace_enabled {
            self.accent_display_mode = AccentDisplayMode::Off;
            self.accent_trace_enabled = true;
        }
        if is_removed_theme(self.theme) || is_retired_theme(self.theme) {
            self.theme = ThemeId::Spring;
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
        ThemeId::Aurora
            | ThemeId::PitchClass
            | ThemeId::ChromaBands
            | ThemeId::PitchMemory
            | ThemeId::HarmonicComb
            | ThemeId::SonicTexture
            | ThemeId::Miku
            | ThemeId::AlbumArt
            | ThemeId::AlbumArtCircle
    )
}

fn is_removed_theme(theme: ThemeId) -> bool {
    matches!(
        theme,
        ThemeId::System | ThemeId::Graphite | ThemeId::Ocean | ThemeId::NoiseWarp | ThemeId::Amber
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
    accent_note_bursts: Vec<AccentNoteBurst>,
    accent_energy_baseline: f32,
    accent_trace_cooldown: Duration,
    level: f32,
    master_left: f32,
    master_right: f32,
    bpm: BpmAnalyzer,
    bpm_estimate: Option<f32>,
    bpm_confidence: f32,
    bpm_phase: f32,
    bpm_pulse: f32,
    bpm_next_beat_at: Option<Instant>,
    waveform: Vec<f32>,
    analyzer: SpectrumAnalyzer,
    visual_bar_count: usize,
    audio: Option<AudioProcess>,
    capture_id: u64,
    rx: Receiver<AudioEvent>,
    tx: Sender<AudioEvent>,
    last_samples_at: Option<Instant>,
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

    fn vertical_offset_rows(&self, virtual_height: usize) -> f32 {
        let duration = ACCENT_TRACE_OFFSET_ANIMATION_MS as f32 / 1000.0;
        let progress = (self.age.as_secs_f32() / duration.max(f32::EPSILON)).clamp(0.0, 1.0);
        let start_offset =
            scaled_accent_trace_offset_rows(ACCENT_TRACE_START_OFFSET_CELLS, virtual_height);
        let end_offset =
            scaled_accent_trace_offset_rows(ACCENT_TRACE_END_OFFSET_CELLS, virtual_height);
        lerp(start_offset, end_offset, smoothstep(progress))
    }
}

#[derive(Clone, Debug)]
struct AccentNoteBurst {
    notes: Vec<AccentNote>,
    age: Duration,
}

impl AccentNoteBurst {
    fn opacity(&self) -> f32 {
        let fade_in = ACCENT_NOTE_FADE_IN_MS as f32 / 1000.0;
        let fade_out = ACCENT_NOTE_FADE_OUT_MS as f32 / 1000.0;
        let age = self.age.as_secs_f32();
        if age < fade_in {
            (age / fade_in.max(f32::EPSILON)).clamp(0.0, 1.0)
        } else {
            (1.0 - (age - fade_in) / fade_out.max(f32::EPSILON)).clamp(0.0, 1.0)
        }
    }
}

#[derive(Clone, Debug)]
struct AccentNote {
    label: &'static str,
    pitch_class: usize,
    x: f32,
    y: f32,
    intensity: f32,
}

fn scaled_accent_trace_offset_rows(offset_cells: f32, virtual_height: usize) -> f32 {
    if virtual_height <= 1 {
        return 0.0;
    }

    let fixed_rows = offset_cells * 4.0;
    let proportional_rows =
        virtual_height as f32 * (fixed_rows / ACCENT_TRACE_REFERENCE_VIRTUAL_HEIGHT);
    proportional_rows
        .clamp(1.0, fixed_rows)
        .min((virtual_height - 1) as f32)
}

#[derive(Clone, Debug)]
struct AccentTraceRender {
    envelope: Vec<f32>,
    fade: f32,
    vertical_offset_rows: f32,
}

#[derive(Clone, Copy, Debug)]
struct AccentTraceOverlay {
    mask: u8,
    color: Color,
    visibility: f32,
}

#[derive(Clone, Copy, Debug)]
struct AccentNoteOverlay {
    label: &'static str,
    color: Color,
    opacity: f32,
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
            accent_note_bursts: Vec::new(),
            accent_energy_baseline: 0.0,
            accent_trace_cooldown: Duration::from_millis(0),
            level: 0.0,
            master_left: 0.0,
            master_right: 0.0,
            bpm: BpmAnalyzer::new(48_000.0),
            bpm_estimate: None,
            bpm_confidence: 0.0,
            bpm_phase: 0.0,
            bpm_pulse: 0.0,
            bpm_next_beat_at: None,
            waveform: vec![0.0; WAVEFORM_SAMPLES],
            analyzer: SpectrumAnalyzer::new(fft_size, 48_000.0, bar_count, hop_size),
            visual_bar_count: bar_count,
            audio: None,
            capture_id: 0,
            rx,
            tx,
            last_samples_at: None,
        }
    }

    fn theme(&self) -> Theme {
        theme(self.theme_id)
    }

    fn t(&self, key: &'static str) -> &'static str {
        tr(self.lang, key)
    }

    fn frame_duration(&self) -> Duration {
        let refresh_hz = self
            .config
            .settings
            .refresh_hz
            .clamp(MIN_REFRESH_HZ, MAX_REFRESH_HZ);
        Duration::from_secs_f64(1.0 / refresh_hz as f64)
    }

    fn tick(&mut self, elapsed: Duration) {
        self.advance_bpm_pulse(elapsed);
        if self.capture_state == CaptureState::Running {
            self.advance_accent_traces(elapsed);
            if let Some(last) = self.last_samples_at {
                if last.elapsed() > Duration::from_secs(3) {
                    self.status = self.t("waiting_audio").to_string();
                }
            }
        }
    }

    fn advance_bpm_pulse(&mut self, elapsed: Duration) {
        self.bpm_pulse =
            (self.bpm_pulse - elapsed.as_secs_f32() / BPM_PULSE_DECAY_SECONDS).clamp(0.0, 1.0);
        if !self.config.settings.bpm_enabled {
            self.bpm_phase = 0.0;
            self.bpm_pulse = 0.0;
            self.bpm_next_beat_at = None;
            return;
        }

        let Some(bpm) = self.bpm_estimate.filter(|bpm| *bpm > 1.0) else {
            self.bpm_phase = 0.0;
            self.bpm_next_beat_at = None;
            return;
        };
        let beat_period = Duration::from_secs_f32(60.0 / bpm);
        let now = Instant::now();
        let next = self.bpm_next_beat_at.get_or_insert(now + beat_period);

        while *next <= now {
            self.bpm_pulse = 1.0;
            *next += beat_period;
        }

        let until_next = next.saturating_duration_since(now).as_secs_f32();
        self.bpm_phase = (1.0 - until_next / beat_period.as_secs_f32()).clamp(0.0, 1.0);
    }

    fn advance_accent_traces(&mut self, elapsed: Duration) {
        let lifetime = Duration::from_millis(ACCENT_TRACE_LIFETIME_MS);
        for trace in &mut self.accent_traces {
            trace.age += elapsed;
        }
        self.accent_traces.retain(|trace| trace.age < lifetime);
        let note_lifetime = Duration::from_millis(ACCENT_NOTE_FADE_IN_MS + ACCENT_NOTE_FADE_OUT_MS);
        for burst in &mut self.accent_note_bursts {
            burst.age += elapsed;
        }
        self.accent_note_bursts
            .retain(|burst| burst.age < note_lifetime);
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
                        self.process_audio_samples(samples);
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
                        self.capture_state = CaptureState::PermissionNeeded;
                        self.status = self.t("permission_needed").to_string();
                    } else if message.contains("no-display") || message.contains("capture-error") {
                        self.audio = None;
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
                    self.capture_state = CaptureState::Failed;
                    self.status = message;
                }
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
        if self.config.settings.bpm_enabled {
            if let Some(estimate) = self.bpm.consume(&samples.mono) {
                self.set_bpm_estimate(estimate.bpm);
                self.bpm_confidence = estimate.confidence;
            }
        }
        let pipeline = SpectrumPipeline::from_settings(&self.config.settings);
        if let Some(bars) = self.analyzer.consume(
            &samples.mono,
            self.config.settings.attack,
            self.config.settings.release,
            pipeline,
        ) {
            self.update_accent_trace_detector(&bars);
            update_spectrum_trail(&mut self.spectrum_trail, &bars, &self.config.settings);
            self.spectrum = bars;
        }
        self.status = self.t("running").to_string();
    }

    fn set_bpm_estimate(&mut self, bpm: f32) {
        let bpm = bpm.clamp(BPM_MIN, BPM_MAX);
        let should_regrid = self
            .bpm_estimate
            .map(|previous| (previous - bpm).abs() > 3.0)
            .unwrap_or(true);
        self.bpm_estimate = Some(bpm);
        if should_regrid {
            self.bpm_next_beat_at = Some(Instant::now() + Duration::from_secs_f32(60.0 / bpm));
            self.bpm_phase = 0.0;
        }
    }

    fn clear_bpm_state(&mut self) {
        self.bpm.reset();
        self.bpm_estimate = None;
        self.bpm_confidence = 0.0;
        self.bpm_phase = 0.0;
        self.bpm_pulse = 0.0;
        self.bpm_next_beat_at = None;
    }

    fn update_accent_trace_detector(&mut self, bars: &[f32]) {
        if bars.is_empty() {
            return;
        }

        let energy = spectrum_accent_energy(bars);
        let trigger_thresholds =
            accent_trigger_thresholds(self.config.settings.accent_trace_threshold);
        if self.config.settings.accent_display_mode == AccentDisplayMode::Off {
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
                self.push_accent_visual(&pending.envelope);
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
        let triggered = self.config.settings.accent_display_mode != AccentDisplayMode::Off
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

    fn push_accent_visual(&mut self, bars: &[f32]) {
        if bars.is_empty() {
            return;
        }

        match self.config.settings.accent_display_mode {
            AccentDisplayMode::Trace => self.push_accent_trace(bars),
            AccentDisplayMode::NoteNames => self.push_accent_note_burst(bars),
            AccentDisplayMode::Off => {}
        }
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

    fn push_accent_note_burst(&mut self, bars: &[f32]) {
        if bars.is_empty() {
            return;
        }

        let pitch_class = accent_note_pitch_class(bars);
        let label = accent_note_label(pitch_class);
        let energy = spectrum_accent_energy(bars);
        let mut seed = accent_note_seed(pitch_class);
        let count = (ACCENT_NOTE_COUNT as f32 * (0.70 + energy * 0.45)).round() as usize;
        let mut notes = Vec::with_capacity(count);

        for _ in 0..count {
            let x = next_random_unit(&mut seed);
            let y = next_random_unit(&mut seed);
            let intensity = (0.48 + next_random_unit(&mut seed) * 0.52).clamp(0.0, 1.0);
            notes.push(AccentNote {
                label,
                pitch_class,
                x,
                y,
                intensity,
            });
        }

        self.accent_note_bursts.push(AccentNoteBurst {
            notes,
            age: Duration::from_millis(0),
        });
        while self.accent_note_bursts.len() > 5 {
            self.accent_note_bursts.remove(0);
        }
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
        let rows = settings_rows(self);
        self.setting_index = previous_setting_index(self.setting_index, &rows);
    }

    fn next_setting(&mut self) {
        let rows = settings_rows(self);
        self.setting_index = next_setting_index(self.setting_index, &rows);
    }

    fn prev_setting_category(&mut self) {
        let rows = settings_rows(self);
        self.setting_index = adjacent_setting_category_index(self.setting_index, &rows, -1);
    }

    fn next_setting_category(&mut self) {
        let rows = settings_rows(self);
        self.setting_index = adjacent_setting_category_index(self.setting_index, &rows, 1);
    }

    fn set_visual_bar_count(&mut self, bar_count: usize) {
        let bar_count = bar_count.clamp(MIN_CONFIG_BARS, MAX_ANALYSIS_BARS);
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
            .clamp(MIN_CONFIG_BARS, MAX_ANALYSIS_BARS)
    }

    fn resize_spectrum_analyzer(&mut self) {
        let bar_count = self.analysis_bar_count();
        self.spectrum = display_bars(&self.spectrum, bar_count);
        self.spectrum_trail = display_bars(&self.spectrum_trail, bar_count);
        self.analyzer.resize_bar_count(bar_count);
    }

    fn adjust_setting(&mut self, direction: i32) {
        let rows = settings_rows(self);
        if let Some(row) = selected_setting_row(self.setting_index, &rows) {
            self.adjust_setting_by_key(row.key, direction);
            self.save_config();
        }
    }

    fn adjust_setting_by_key(&mut self, key: &'static str, direction: i32) {
        match key {
            "language" => self.cycle_language(direction),
            "theme" => self.cycle_theme(direction),
            "analysis_preset" => self.cycle_analysis_preset(direction),
            "attack" => {
                let delta = if direction < 0 {
                    -ATTACK_STEP
                } else {
                    ATTACK_STEP
                };
                self.config.settings.attack =
                    (self.config.settings.attack + delta).clamp(MIN_ATTACK, MAX_ATTACK);
                self.config.settings.mark_custom_analysis();
            }
            "release" => {
                let delta = if direction < 0 {
                    -RELEASE_STEP
                } else {
                    RELEASE_STEP
                };
                self.config.settings.release =
                    (self.config.settings.release + delta).clamp(MIN_RELEASE, MAX_RELEASE);
                self.config.settings.mark_custom_analysis();
            }
            "bars" => {
                let current = self.config.settings.bar_count as i32;
                let next = (current + direction * 8)
                    .clamp(MIN_CONFIG_BARS as i32, MAX_CONFIG_BARS as i32)
                    as usize;
                if next != self.config.settings.bar_count {
                    self.config.settings.bar_count = next;
                    self.resize_spectrum_analyzer();
                }
            }
            "renderer" => self.cycle_renderer(direction),
            "fft_size" => {
                self.cycle_fft_size(direction);
                self.config.settings.mark_custom_analysis();
                self.rebuild_analyzer();
            }
            "analysis_hop" => {
                self.cycle_analysis_hop(direction);
                self.config.settings.mark_custom_analysis();
                self.rebuild_analyzer();
            }
            "refresh_rate" => {
                self.cycle_refresh_rate(direction);
                self.config.settings.mark_custom_analysis();
            }
            "high_shelf" => {
                self.config.settings.high_shelf_enabled = !self.config.settings.high_shelf_enabled
            }
            "high_shelf_db" => {
                let delta = if direction < 0 {
                    -HIGH_SHELF_DB_STEP
                } else {
                    HIGH_SHELF_DB_STEP
                };
                self.config.settings.high_shelf_db = (self.config.settings.high_shelf_db + delta)
                    .clamp(MIN_HIGH_SHELF_DB, MAX_HIGH_SHELF_DB);
            }
            "auto_sensitivity" => {
                self.config.settings.auto_sensitivity_enabled =
                    !self.config.settings.auto_sensitivity_enabled;
                self.analyzer.reset_adaptive_state();
            }
            "noise_reduction" => {
                let delta = if direction < 0 {
                    -NOISE_REDUCTION_STEP
                } else {
                    NOISE_REDUCTION_STEP
                };
                self.config.settings.noise_reduction = (self.config.settings.noise_reduction
                    + delta)
                    .clamp(MIN_NOISE_REDUCTION, MAX_NOISE_REDUCTION);
            }
            "bpm_analysis" => {
                self.config.settings.bpm_enabled = !self.config.settings.bpm_enabled;
                if !self.config.settings.bpm_enabled {
                    self.clear_bpm_state();
                }
            }
            "visual_curve" => {
                self.config.settings.visual_curve_enabled =
                    !self.config.settings.visual_curve_enabled;
            }
            "curve_power" => {
                let delta = if direction < 0 {
                    -VISUAL_CURVE_STEP
                } else {
                    VISUAL_CURVE_STEP
                };
                self.config.settings.visual_curve = (self.config.settings.visual_curve + delta)
                    .clamp(MIN_VISUAL_CURVE, MAX_VISUAL_CURVE);
            }
            "trail" => {
                self.config.settings.trail_enabled = !self.config.settings.trail_enabled;
                if !self.config.settings.trail_enabled {
                    self.spectrum_trail.clone_from(&self.spectrum);
                }
            }
            "trail_decay" => {
                let delta = if direction < 0 {
                    -TRAIL_DECAY_STEP
                } else {
                    TRAIL_DECAY_STEP
                };
                self.config.settings.trail_decay = (self.config.settings.trail_decay + delta)
                    .clamp(MIN_TRAIL_DECAY, MAX_TRAIL_DECAY);
            }
            "accent_display" => self.cycle_accent_display_mode(direction),
            "accent_threshold" => {
                let delta = if direction < 0 {
                    -ACCENT_TRACE_THRESHOLD_STEP
                } else {
                    ACCENT_TRACE_THRESHOLD_STEP
                };
                self.config.settings.accent_trace_threshold =
                    (self.config.settings.accent_trace_threshold + delta)
                        .clamp(MIN_ACCENT_TRACE_THRESHOLD, MAX_ACCENT_TRACE_THRESHOLD);
            }
            "ceiling" => {
                let delta = if direction < 0 {
                    -CEILING_STEP
                } else {
                    CEILING_STEP
                };
                self.config.settings.ceiling =
                    (self.config.settings.ceiling + delta).clamp(MIN_CEILING, MAX_CEILING);
            }
            _ => {}
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

    fn cycle_theme(&mut self, direction: i32) {
        let index = THEMES
            .iter()
            .position(|theme| *theme == self.theme_id)
            .unwrap_or(0);
        let len = THEMES.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.theme_id = THEMES[next];
        self.config.settings.theme = THEMES[next];
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
        self.clear_bpm_state();
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

    fn cycle_accent_display_mode(&mut self, direction: i32) {
        let index = ACCENT_DISPLAY_MODES
            .iter()
            .position(|mode| *mode == self.config.settings.accent_display_mode)
            .unwrap_or(0);
        let len = ACCENT_DISPLAY_MODES.len() as i32;
        let next = (index as i32 + direction).rem_euclid(len) as usize;
        self.config.settings.accent_display_mode = ACCENT_DISPLAY_MODES[next];
        self.pending_accent_trace = None;
        self.accent_traces.clear();
        self.accent_note_bursts.clear();
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
                let mut left_square_sum = 0.0_f32;
                let mut right_square_sum = 0.0_f32;

                for chunk in pending[..bytes_to_read].chunks_exact(8) {
                    let left_sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let right_sample = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    left_square_sum += left_sample * left_sample;
                    right_square_sum += right_sample * right_sample;
                    mono.push((left_sample + right_sample) * 0.5);
                }

                pending.drain(0..bytes_to_read);
                let samples = AudioSamples {
                    left_level: audio_level_from_square_sum(left_square_sum, frame_count),
                    right_level: audio_level_from_square_sum(right_square_sum, frame_count),
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
    adaptive_floor: Vec<f32>,
    adaptive_gain: f32,
    samples_since_analysis: usize,
    has_analysis: bool,
}

#[derive(Clone, Copy)]
struct SpectrumPipeline {
    high_shelf_enabled: bool,
    high_shelf_db: f32,
    auto_sensitivity_enabled: bool,
    noise_reduction: f32,
    ceiling: f32,
}

impl SpectrumPipeline {
    fn from_settings(settings: &Settings) -> Self {
        Self {
            high_shelf_enabled: settings.high_shelf_enabled,
            high_shelf_db: settings.high_shelf_db,
            auto_sensitivity_enabled: settings.auto_sensitivity_enabled,
            noise_reduction: settings.noise_reduction,
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
            adaptive_floor: vec![0.0; bar_count],
            adaptive_gain: 1.0,
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
        self.adaptive_floor = display_bars(&self.adaptive_floor, bar_count);
        if self.adaptive_floor.len() != bar_count {
            self.adaptive_floor.resize(bar_count, 0.0);
        }
    }

    fn reset_adaptive_state(&mut self) {
        self.adaptive_floor.fill(0.0);
        self.adaptive_gain = 1.0;
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

        let mut bars = self.make_bars(&magnitudes, pipeline);
        self.apply_adaptive_processing(&mut bars, source_rms, pipeline);
        let attack = attack.clamp(MIN_ATTACK, MAX_ATTACK);
        let release = release.clamp(MIN_RELEASE, MAX_RELEASE);

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

    fn apply_adaptive_processing(
        &mut self,
        bars: &mut [f32],
        source_rms: f32,
        pipeline: SpectrumPipeline,
    ) {
        if bars.is_empty() {
            return;
        }

        if self.adaptive_floor.len() != bars.len() {
            self.adaptive_floor = vec![0.0; bars.len()];
        }

        let noise = pipeline
            .noise_reduction
            .clamp(MIN_NOISE_REDUCTION, MAX_NOISE_REDUCTION);
        for (index, value) in bars.iter_mut().enumerate() {
            let floor = &mut self.adaptive_floor[index];
            *floor = if *floor <= 0.0 {
                *value * (0.30 + noise * 0.25)
            } else if *value < *floor {
                lerp(*floor, *value, 0.14)
            } else {
                lerp(*floor, *value, 0.006 + noise * 0.010)
            };

            let gate = (*floor * (0.72 + noise * 1.15)).max(VISUAL_NOISE_FLOOR * noise);
            *value = ((*value - gate).max(0.0) / (1.0 - gate).max(0.10)).clamp(0.0, 1.0);
        }

        if pipeline.auto_sensitivity_enabled {
            let peak = bars.iter().copied().fold(0.0_f32, f32::max);
            let rms =
                (bars.iter().map(|value| value * value).sum::<f32>() / bars.len() as f32).sqrt();
            let target_rms = (pipeline.ceiling * ADAPTIVE_TARGET_RMS).clamp(0.24, 0.78);
            let target_peak = (pipeline.ceiling * ADAPTIVE_TARGET_PEAK).clamp(0.40, 0.98);
            let target_gain = if rms <= 0.001 && peak <= 0.001 && source_rms <= SILENCE_GATE * 4.0 {
                1.0
            } else {
                let rms_gain = if rms <= 0.001 {
                    ADAPTIVE_GAIN_MAX
                } else {
                    target_rms / rms
                };
                let peak_gain = if peak <= 0.001 {
                    ADAPTIVE_GAIN_MAX
                } else {
                    target_peak / peak
                };
                rms_gain
                    .min(peak_gain * 1.08)
                    .clamp(ADAPTIVE_GAIN_MIN, ADAPTIVE_GAIN_MAX)
            };
            let follow = if target_gain < self.adaptive_gain {
                0.18
            } else {
                0.060
            };
            self.adaptive_gain = lerp(self.adaptive_gain, target_gain, follow);
        } else {
            self.adaptive_gain = 1.0;
        }

        for value in bars {
            *value = (*value * self.adaptive_gain).clamp(0.0, pipeline.ceiling);
        }
    }
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
    audio_level_from_square_sum(square_sum, count)
}

fn audio_level_from_square_sum(square_sum: f32, count: usize) -> f32 {
    if count == 0 {
        return 0.0;
    }
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
    peak: Color,
}

fn theme(id: ThemeId) -> Theme {
    match id {
        ThemeId::Spring
        | ThemeId::System
        | ThemeId::Graphite
        | ThemeId::Ocean
        | ThemeId::Aurora
        | ThemeId::PitchClass
        | ThemeId::ChromaBands
        | ThemeId::PitchMemory
        | ThemeId::HarmonicComb
        | ThemeId::SonicTexture
        | ThemeId::NoiseWarp
        | ThemeId::Miku
        | ThemeId::AlbumArt
        | ThemeId::AlbumArtCircle
        | ThemeId::Amber => Theme {
            title_key: "theme_spring",
            accent: Color::Rgb(255, 164, 164),
            text: Color::Rgb(252, 249, 234),
            muted: Color::Rgb(186, 223, 219),
            border: Color::Rgb(186, 223, 219),
            low: Color::Rgb(255, 164, 164),
            mid: Color::Rgb(255, 189, 189),
            high: Color::Rgb(252, 249, 234),
            peak: Color::Rgb(186, 223, 219),
        },
        ThemeId::Vintage => Theme {
            title_key: "theme_vintage",
            accent: Color::Rgb(186, 106, 76),
            text: Color::Rgb(238, 224, 204),
            muted: Color::Rgb(96, 116, 86),
            border: Color::Rgb(96, 116, 86),
            low: Color::Rgb(123, 37, 37),
            mid: Color::Rgb(186, 106, 76),
            high: Color::Rgb(238, 224, 204),
            peak: Color::Rgb(96, 116, 86),
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
            peak: Color::White,
        },
    }
}

fn bar_color(theme: Theme, height_ratio: f32, value: f32) -> Color {
    let height_ratio = height_ratio.clamp(0.0, 1.0);
    let value = value.clamp(0.0, 1.0);
    let lifted = (height_ratio * 0.82 + value * 0.18).clamp(0.0, 1.0);
    vertical_palette_color(theme, lifted)
}

fn vertical_palette_color(theme: Theme, level: f32) -> Color {
    let level = level.clamp(0.0, 1.0);
    if level < 0.42 {
        blend_color(theme.low, theme.mid, level / 0.42)
    } else if level < 0.72 {
        blend_color(theme.mid, theme.high, (level - 0.42) / 0.30)
    } else {
        blend_color(theme.high, theme.peak, (level - 0.72) / 0.28)
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
    let _ = (index, len);
    if trail {
        theme.border
    } else {
        bar_color(theme, height_ratio, value)
    }
}

fn waveform_color(app: &App, value: f32) -> Color {
    meter_color(app.theme(), value)
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
        SpectrumRenderer::Cava => inner_width,
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
            .constraints([Constraint::Min(24), Constraint::Length(MASTER_METER_WIDTH)])
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
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(24), Constraint::Length(MASTER_METER_WIDTH)])
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
        height += 9;
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
        constraints.push(Constraint::Length(9));
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
            Span::raw("  "),
            Span::styled(app.t("bpm"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(bpm_label(app), Style::default().fg(theme.text)),
            Span::raw(" "),
            beat_indicator_span(app),
        ]),
        Line::from(vec![
            Span::styled(app.t("refresh_rate"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(
                format!("{}Hz", app.config.settings.refresh_hz),
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

fn beat_indicator_span(app: &App) -> Span<'static> {
    let theme = app.theme();
    let symbol = if app.bpm_pulse > 0.66 {
        "●"
    } else if app.bpm_pulse > 0.24 {
        "◉"
    } else {
        "○"
    };
    let style = if app.bpm_pulse > 0.0 {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.muted)
    };
    Span::styled(symbol, style)
}

fn beat_phase_bar(app: &App, width: usize) -> String {
    if width == 0 || !app.config.settings.bpm_enabled || app.bpm_estimate.is_none() {
        return String::new();
    }

    let position = (app.bpm_phase * width as f32).floor() as usize;
    (0..width)
        .map(|index| {
            if index == position.min(width - 1) {
                if app.bpm_pulse > 0.0 {
                    '●'
                } else {
                    '◆'
                }
            } else {
                '─'
            }
        })
        .collect()
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
    let left_width = master_meter_channel_width(available_width);
    let right_width = left_width;
    let virtual_height = chart_height * 4;
    let left = app.master_left.clamp(0.0, 1.0);
    let right = app.master_right.clamp(0.0, 1.0);
    let mut lines = Vec::with_capacity(chart_height + 1);

    for row in 0..chart_height {
        let mut spans = Vec::with_capacity(left_width + right_width);
        for col in 0..left_width {
            let (mask, value) = master_braille_cell(left, col, row, virtual_height);
            let mask = master_meter_channel_mask(mask, MasterMeterSide::Left, col, left_width);
            spans.push(Span::styled(
                master_braille_symbol(mask),
                Style::default().fg(if mask == 0 {
                    Color::Reset
                } else {
                    master_meter_color(app, value)
                }),
            ));
        }
        for col in 0..right_width {
            let (mask, value) = master_braille_cell(right, col, row, virtual_height);
            let mask = master_meter_channel_mask(mask, MasterMeterSide::Right, col, right_width);
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

    let footer_spans = vec![
        Span::styled(
            master_meter_label('L', left, left_width),
            Style::default().fg(theme.muted),
        ),
        Span::styled(
            master_meter_label('R', right, right_width),
            Style::default().fg(theme.muted),
        ),
    ];
    lines.push(Line::from(footer_spans));

    frame.render_widget(Paragraph::new(lines), inner);
}

fn master_meter_channel_width(available_width: usize) -> usize {
    MASTER_METER_CHANNEL_WIDTH.min(available_width / 2).max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MasterMeterSide {
    Left,
    Right,
}

fn master_meter_channel_mask(
    mask: u8,
    side: MasterMeterSide,
    cell_col: usize,
    channel_width: usize,
) -> u8 {
    match side {
        MasterMeterSide::Left if cell_col + 1 == channel_width => mask & BRAILLE_LEFT_COLUMN_MASK,
        MasterMeterSide::Right if cell_col == 0 => mask & BRAILLE_RIGHT_COLUMN_MASK,
        _ => mask,
    }
}

fn master_meter_label(prefix: char, value: f32, width: usize) -> String {
    let label = format!("{}{:>3}", prefix, (value.clamp(0.0, 1.0) * 100.0) as u16);
    if label.len() >= width {
        label.chars().take(width).collect()
    } else {
        format!("{label:<width$}")
    }
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

    for (_, dot_row, bit) in braille_dot_bits() {
        let virtual_row = cell_row * 4 + dot_row;
        let threshold = 1.0 - (virtual_row as f32 + 0.5) / virtual_height.max(1) as f32;
        if value < threshold {
            continue;
        }

        mask |= bit;
        cell_value = cell_value.max(threshold);
    }

    (mask, cell_value)
}

fn master_meter_color(app: &App, height_ratio: f32) -> Color {
    let theme = app.theme();
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

fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount.clamp(0.0, 1.0)
}

fn normalize_unit(value: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        0.0
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
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

fn braille_dot_bits() -> impl Iterator<Item = (usize, usize, u8)> {
    BRAILLE_DOT_BITS
        .iter()
        .enumerate()
        .flat_map(|(dot_col, rows)| {
            rows.iter()
                .copied()
                .enumerate()
                .map(move |(dot_row, bit)| (dot_col, dot_row, bit))
        })
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
        SpectrumRenderer::Cava => draw_cava_spectrum(frame, app, inner),
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
    let accent_traces = display_accent_traces(app, virtual_width, virtual_height);
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
            } else if let Some(note) = accent_note_overlay_cell(
                app,
                index,
                row,
                bars.len(),
                chart_height,
                !filled && !trail_filled,
            ) {
                (note.label.to_string(), note.color)
            } else {
                (base_symbol, base_color.unwrap_or(Color::Reset))
            };
            spans.push(Span::styled(symbol, Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_cava_spectrum(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let settings = &app.config.settings;
    let bars: Vec<f32> = display_bars(&app.spectrum, area.width as usize)
        .into_iter()
        .map(|value| render_bar_value(value, settings))
        .collect();
    let trail: Vec<f32> = if settings.trail_enabled {
        display_bars(&app.spectrum_trail, area.width as usize)
            .into_iter()
            .map(|value| render_bar_value(value, settings))
            .collect()
    } else {
        Vec::new()
    };
    let chart_height = area.height as usize;
    let virtual_height = chart_height * 8;
    let accent_virtual_height = chart_height * 4;
    let accent_traces = display_accent_traces(app, area.width as usize * 2, accent_virtual_height);
    let mut lines = Vec::with_capacity(chart_height);

    for row in 0..chart_height {
        let mut spans = Vec::with_capacity(bars.len());
        for (index, value) in bars.iter().copied().enumerate() {
            let cell = cava_bar_cell(value, row, chart_height);
            let trail_cell = trail
                .get(index)
                .copied()
                .map(|value| cava_bar_cell(value, row, chart_height))
                .unwrap_or_default();
            let visible_level = cell.level.max(trail_cell.level);
            let trail_only = cell.level == 0 && trail_cell.level > 0;
            let height_ratio = if virtual_height == 0 {
                0.0
            } else {
                1.0 - ((row * 8 + 4) as f32 / virtual_height as f32)
            };
            let accent_trace = accent_trace_overlay_cell(
                app,
                &accent_traces,
                index,
                row,
                area.width as usize * 2,
                accent_virtual_height,
            );
            let accent_note = accent_note_overlay_cell(
                app,
                index,
                row,
                bars.len(),
                chart_height,
                visible_level == 0 && accent_trace.is_none(),
            );

            let symbol = if visible_level > 0 {
                cava_block_symbol(visible_level).to_string()
            } else if accent_trace.is_some() {
                "⠂".to_string()
            } else if let Some(note) = accent_note {
                note.label.to_string()
            } else {
                " ".to_string()
            };
            let intensity = if visible_level > 0 {
                value.max(trail_cell.value)
            } else {
                0.0
            };
            let base_color = if visible_level > 0 {
                Some(spectrum_bar_color_at(
                    app,
                    index,
                    bars.len(),
                    intensity,
                    height_ratio,
                    trail_only,
                ))
            } else {
                accent_note.map(|note| note.color)
            };
            let color = if let Some(overlay) = accent_trace {
                accent_trace_overlay_color(theme, base_color, overlay)
            } else {
                base_color.unwrap_or(Color::Reset)
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
    let accent_traces = display_accent_traces(app, virtual_width, virtual_height);
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
                    let color = accent_trace_overlay_color(theme, Some(base_color), overlay);
                    spans.push(Span::styled(
                        braille_pattern(combined_mask | overlay.mask).to_string(),
                        Style::default().fg(color),
                    ));
                }
            } else if combined_mask == 0 {
                if let Some(note) =
                    accent_note_overlay_cell(app, col, row, cell_width, chart_height, true)
                {
                    spans.push(Span::styled(
                        note.label.to_string(),
                        Style::default().fg(note.color),
                    ));
                } else {
                    spans.push(Span::raw(" "));
                }
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

    let decay = settings.trail_decay.clamp(MIN_TRAIL_DECAY, MAX_TRAIL_DECAY);
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

fn accent_note_pitch_class(bars: &[f32]) -> usize {
    if bars.is_empty() {
        return 0;
    }

    let mut chroma = [0.0_f32; 12];
    let max_index = (bars.len().saturating_sub(1)).max(1) as f32;
    for (index, value) in bars.iter().copied().enumerate() {
        let value = value.clamp(0.0, 1.0);
        if value <= 0.0 {
            continue;
        }
        let position = index as f32 / max_index;
        let frequency = frequency_for_position(position);
        let pitch_class = pitch_class_for_frequency(frequency);
        chroma[pitch_class] += value.powf(1.25) * melody_frequency_weight(frequency);
    }

    strongest_chroma(&chroma).0
}

fn accent_note_label(pitch_class: usize) -> &'static str {
    const LABELS: &[&str; 12] = &["c", "C", "d", "D", "e", "f", "F", "g", "G", "a", "A", "b"];
    LABELS[pitch_class % 12]
}

fn accent_note_seed(pitch_class: usize) -> u64 {
    0x9e37_79b9_7f4a_7c15 ^ ((pitch_class as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
}

fn next_random_unit(seed: &mut u64) -> f32 {
    *seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    let value = seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
    ((value >> 40) as f32 / 0x01_00_00_00_u32 as f32).clamp(0.0, 1.0)
}

fn accent_trigger_thresholds(threshold: f32) -> AccentTriggerThresholds {
    let threshold = normalize_unit(
        threshold.clamp(MIN_ACCENT_TRACE_THRESHOLD, MAX_ACCENT_TRACE_THRESHOLD),
        MIN_ACCENT_TRACE_THRESHOLD,
        MAX_ACCENT_TRACE_THRESHOLD,
    );
    AccentTriggerThresholds {
        peak: lerp(0.10, 0.26, threshold),
        initial_energy: lerp(0.03, 0.09, threshold),
        energy: lerp(0.025, 0.065, threshold),
        flux: lerp(0.015, 0.055, threshold),
        rise: lerp(0.010, 0.060, threshold),
        ratio: lerp(1.15, 1.75, threshold),
    }
}

fn display_accent_traces(
    app: &App,
    virtual_width: usize,
    virtual_height: usize,
) -> Vec<AccentTraceRender> {
    if app.config.settings.accent_display_mode != AccentDisplayMode::Trace
        || virtual_width == 0
        || virtual_height == 0
    {
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

            Some(AccentTraceRender {
                envelope,
                fade,
                vertical_offset_rows: trace.vertical_offset_rows(virtual_height),
            })
        })
        .collect()
}

fn accent_note_overlay_cell(
    app: &App,
    cell_col: usize,
    cell_row: usize,
    cell_width: usize,
    cell_height: usize,
    blank: bool,
) -> Option<AccentNoteOverlay> {
    if !blank
        || app.config.settings.accent_display_mode != AccentDisplayMode::NoteNames
        || cell_width == 0
        || cell_height == 0
    {
        return None;
    }

    let theme = app.theme();
    let mut best: Option<AccentNoteOverlay> = None;
    for burst in &app.accent_note_bursts {
        let opacity = burst.opacity();
        if opacity <= 0.0 {
            continue;
        }

        for note in &burst.notes {
            let note_col = (note.x * cell_width.saturating_sub(1) as f32).round() as usize;
            let note_row = (note.y * cell_height.saturating_sub(1) as f32).round() as usize;
            if note_col != cell_col || note_row != cell_row {
                continue;
            }

            let strength = (opacity * note.intensity).clamp(0.0, 1.0);
            if best
                .map(|overlay| overlay.opacity >= strength)
                .unwrap_or(false)
            {
                continue;
            }

            let base = vertical_palette_color(theme, note.pitch_class as f32 / 11.0);
            best = Some(AccentNoteOverlay {
                label: note.label,
                color: blend_color(
                    accent_trace_background_color(theme),
                    base,
                    0.22 + strength * 0.68,
                ),
                opacity: strength,
            });
        }
    }

    best
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
        let (mask, value) = accent_trace_braille_cell(
            &trace.envelope,
            cell_col,
            cell_row,
            virtual_height,
            trace.vertical_offset_rows,
        );
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
    vertical_offset_rows: f32,
) -> (u8, f32) {
    let mut mask = 0;
    let mut cell_value = 0.0_f32;

    for (dot_col, _) in BRAILLE_DOT_BITS.iter().enumerate() {
        let bar_index = cell_col * 2 + dot_col;
        let Some(value) = envelope.get(bar_index).copied() else {
            continue;
        };
        let value = value.clamp(0.0, 1.0);
        if value <= 0.0 {
            continue;
        }

        let virtual_row = accent_trace_virtual_row(value, virtual_height, vertical_offset_rows);
        let mut start_row = virtual_row;
        let mut end_row = virtual_row;
        let mut segment_value = value;

        if bar_index > 0 {
            if let Some(previous) = envelope.get(bar_index - 1).copied() {
                let previous = previous.clamp(0.0, 1.0);
                if previous > 0.0 {
                    let previous_row =
                        accent_trace_virtual_row(previous, virtual_height, vertical_offset_rows);
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

fn accent_trace_virtual_row(value: f32, virtual_height: usize, vertical_offset_rows: f32) -> usize {
    if virtual_height == 0 {
        return 0;
    }

    let max_row = virtual_height.saturating_sub(1) as i32;
    let row =
        ((1.0 - value.clamp(0.0, 1.0)) * max_row as f32 - vertical_offset_rows).round() as i32;
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

    for (dot_col, _) in BRAILLE_DOT_BITS.iter().enumerate() {
        let column_index = cell_col * 2 + dot_col;
        let Some(column) = columns.get(column_index).copied() else {
            continue;
        };
        let min = (column.min * gain).clamp(-1.0, 1.0);
        let max = (column.max * gain).clamp(-1.0, 1.0);
        let active = max.abs().max(min.abs()) > 0.002;
        cell_value = cell_value.max(max.abs()).max(min.abs());

        for (dot_row, bit) in BRAILLE_DOT_BITS[dot_col].iter().copied().enumerate() {
            let virtual_row = cell_row * 4 + dot_row;
            let position = if center <= 0.0 {
                0.0
            } else {
                ((center - virtual_row as f32) / center).clamp(-1.0, 1.0)
            };
            if active && position >= min && position <= max {
                mask |= bit;
            }
        }
    }

    (mask, cell_value)
}

fn waveform_centerline_mask(cell_row: usize, virtual_height: usize) -> u8 {
    let center_row = ((virtual_height.saturating_sub(1)) as f32 * 0.5).round() as usize;
    let mut mask = 0;

    for (_, dot_row, bit) in braille_dot_bits() {
        let virtual_row = cell_row * 4 + dot_row;
        if virtual_row == center_row {
            mask |= bit;
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CavaCell {
    level: usize,
    value: f32,
}

fn cava_bar_cell(value: f32, row: usize, chart_height: usize) -> CavaCell {
    if chart_height == 0 {
        return CavaCell::default();
    }

    let virtual_height = chart_height * 8;
    let filled = (value.clamp(0.0, 1.0) * virtual_height as f32).round() as usize;
    let bottom = chart_height.saturating_sub(row + 1) * 8;
    let level = filled.saturating_sub(bottom).min(8);
    CavaCell { level, value }
}

fn cava_block_symbol(level: usize) -> &'static str {
    CAVA_BLOCKS[level.min(CAVA_BLOCKS.len() - 1)]
}

fn braille_bar_cell(
    bars: &[f32],
    cell_col: usize,
    cell_row: usize,
    virtual_height: usize,
) -> (u8, f32) {
    let mut mask = 0;
    let mut cell_value = 0.0_f32;

    for (dot_col, _) in BRAILLE_DOT_BITS.iter().enumerate() {
        let bar_index = cell_col * 2 + dot_col;
        let Some(value) = bars.get(bar_index).copied() else {
            continue;
        };
        cell_value = cell_value.max(value);

        for (dot_row, bit) in BRAILLE_DOT_BITS[dot_col].iter().copied().enumerate() {
            let virtual_row = cell_row * 4 + dot_row;
            let threshold = 1.0 - (virtual_row as f32 + 0.5) / virtual_height.max(1) as f32;
            if value >= threshold {
                mask |= bit;
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
    if inner.width >= 58 && inner.height >= 10 {
        draw_settings_board(frame, app, inner);
    } else {
        draw_settings_compact_list(frame, app, inner);
    }
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

fn draw_settings_board(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let rows = settings_rows(app);
    let selected_row = selected_setting_row(app.setting_index, &rows);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(42),
            Constraint::Length(1),
            Constraint::Min(24),
        ])
        .split(sections[0]);
    draw_settings_rows(frame, app, columns[0], &rows);
    draw_vertical_divider(frame, theme, columns[1]);
    draw_settings_description(frame, app, columns[2], selected_row.as_ref());

    let footer = Line::from(vec![
        Span::styled("←", Style::default().fg(theme.accent)),
        Span::styled(" / ", Style::default().fg(theme.muted)),
        Span::styled("→", Style::default().fg(theme.accent)),
        Span::styled(
            format!(" {}  ", app.t("setting_adjust")),
            Style::default().fg(theme.muted),
        ),
        Span::styled("↑", Style::default().fg(theme.accent)),
        Span::styled(" / ", Style::default().fg(theme.muted)),
        Span::styled("↓", Style::default().fg(theme.accent)),
        Span::styled(
            format!(" {}", app.t("setting_select")),
            Style::default().fg(theme.muted),
        ),
    ]);
    frame.render_widget(Paragraph::new(footer), sections[1]);
}

fn draw_vertical_divider(frame: &mut Frame, theme: Theme, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let lines = (0..area.height)
        .map(|_| Line::from(Span::styled("│", Style::default().fg(theme.border))))
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_settings_compact_list(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let rows = settings_rows(app);
    let visible_height = area.height as usize;
    let selected_position = rows
        .iter()
        .position(|row| row.index == app.setting_index)
        .unwrap_or(0);
    let start = visible_list_start(selected_position, rows.len(), visible_height);
    let items: Vec<ListItem> = rows
        .iter()
        .skip(start)
        .take(visible_height)
        .map(|row| ListItem::new(setting_list_line(app, row, area.width as usize)))
        .collect();
    let mut state = ListState::default();
    state.select(Some(selected_position.saturating_sub(start)));
    let list = List::new(items).highlight_symbol("").highlight_style(
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_settings_rows(frame: &mut Frame, app: &App, area: Rect, rows: &[SettingRow]) {
    let theme = app.theme();
    let visible_height = area.height as usize;
    let selected_position = rows
        .iter()
        .position(|row| row.index == app.setting_index)
        .unwrap_or(0);
    let start = visible_list_start(selected_position, rows.len(), visible_height);
    let items: Vec<ListItem> = rows
        .iter()
        .skip(start)
        .take(visible_height)
        .map(|row| ListItem::new(setting_board_line(app, row, area.width as usize)))
        .collect();
    let mut state = ListState::default();
    state.select(Some(selected_position.saturating_sub(start)));
    let list = List::new(items).highlight_symbol("").highlight_style(
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_settings_description(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    selected: Option<&SettingRow>,
) {
    let theme = app.theme();
    let Some(row) = selected else {
        return;
    };
    let area = inset_rect(area, 2, 0);
    let lines = vec![
        Line::from(Span::styled(
            app.t(row.key),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            app.t("current_value"),
            Style::default().fg(theme.muted),
        )),
        Line::from(Span::styled(
            row.value.clone(),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            app.t("setting_range"),
            Style::default().fg(theme.muted),
        )),
        Line::from(Span::styled(
            setting_range(app, row.key),
            Style::default().fg(theme.text),
        )),
        Line::from(""),
        Line::from(Span::styled(
            app.t(setting_help_key(row.key)),
            Style::default().fg(theme.muted),
        )),
    ];
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

fn setting_board_line(app: &App, row: &SettingRow, width: usize) -> Line<'static> {
    let theme = app.theme();
    let selected = row.index == app.setting_index;
    let label_width = ((width.saturating_sub(4)) * 58 / 100).clamp(8, 24);
    let value_width = width.saturating_sub(3 + label_width);
    let label = truncate_to_width(app.t(row.key), label_width);
    let value = truncate_to_width(&row.value, value_width);
    let marker = if selected { "›" } else { " " };
    let base = if selected { theme.text } else { theme.muted };
    Line::from(vec![
        Span::styled(marker.to_string(), Style::default().fg(theme.accent)),
        Span::raw(" "),
        Span::styled(
            pad_right_to_width(&label, label_width),
            Style::default().fg(base).add_modifier(if selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
        ),
        Span::raw(" "),
        Span::styled(
            pad_left_to_width(&value, value_width),
            Style::default().fg(if selected { theme.accent } else { theme.text }),
        ),
    ])
}

fn setting_list_line(app: &App, row: &SettingRow, width: usize) -> Line<'static> {
    let theme = app.theme();
    let selected = row.index == app.setting_index;
    let label_width = width.saturating_sub(10).clamp(8, 18);
    let value_width = width.saturating_sub(3 + label_width);
    let label = truncate_to_width(app.t(row.key), label_width);
    let value = truncate_to_width(&row.value, value_width);
    let style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.muted)
    };
    Line::from(vec![
        Span::styled(if selected { "› " } else { "  " }, style),
        Span::styled(pad_right_to_width(&label, label_width), style),
        Span::raw(" "),
        Span::styled(
            pad_left_to_width(&value, value_width),
            Style::default().fg(if selected { theme.text } else { theme.muted }),
        ),
    ])
}

fn truncate_to_width(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut used = 0;
    for ch in value.chars() {
        let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + char_width > width {
            break;
        }
        output.push(ch);
        used += char_width;
    }
    output
}

fn pad_right_to_width(value: &str, width: usize) -> String {
    let value = truncate_to_width(value, width);
    let padding = width.saturating_sub(UnicodeWidthStr::width(value.as_str()));
    format!("{}{}", value, " ".repeat(padding))
}

fn pad_left_to_width(value: &str, width: usize) -> String {
    let value = truncate_to_width(value, width);
    let padding = width.saturating_sub(UnicodeWidthStr::width(value.as_str()));
    format!("{}{}", " ".repeat(padding), value)
}

fn selected_setting_category(setting_index: usize, rows: &[SettingRow]) -> SettingCategory {
    rows.iter()
        .find(|row| row.index == setting_index)
        .map(|row| row.category)
        .unwrap_or(SettingCategory::General)
}

fn selected_setting_row(setting_index: usize, rows: &[SettingRow]) -> Option<SettingRow> {
    rows.iter().find(|row| row.index == setting_index).cloned()
}

fn previous_setting_index(current: usize, rows: &[SettingRow]) -> usize {
    let position = rows
        .iter()
        .position(|row| row.index == current)
        .unwrap_or(0);
    rows.get(position.saturating_sub(1))
        .map(|row| row.index)
        .unwrap_or(current)
}

fn next_setting_index(current: usize, rows: &[SettingRow]) -> usize {
    let position = rows
        .iter()
        .position(|row| row.index == current)
        .unwrap_or(0);
    rows.get((position + 1).min(rows.len().saturating_sub(1)))
        .map(|row| row.index)
        .unwrap_or(current)
}

fn adjacent_setting_category_index(current: usize, rows: &[SettingRow], direction: i32) -> usize {
    let category = selected_setting_category(current, rows);
    let index = SETTING_CATEGORIES
        .iter()
        .position(|item| *item == category)
        .unwrap_or(0);
    let len = SETTING_CATEGORIES.len() as i32;
    let next = (index as i32 + direction).rem_euclid(len) as usize;
    let next_category = SETTING_CATEGORIES[next];
    rows.iter()
        .find(|row| row.category == next_category)
        .map(|row| row.index)
        .unwrap_or(current)
}

fn settings_rows(app: &App) -> Vec<SettingRow> {
    vec![
        setting_row(
            0,
            "language",
            SettingCategory::General,
            language_label(app.lang),
        ),
        setting_row(1, "theme", SettingCategory::General, theme_label(app)),
        setting_row(
            2,
            "analysis_preset",
            SettingCategory::Analysis,
            preset_label(app, app.config.settings.analysis_preset),
        ),
        setting_row(
            3,
            "attack",
            SettingCategory::Analysis,
            format!("{:>3}%", (app.config.settings.attack * 100.0) as u16),
        ),
        setting_row(
            4,
            "release",
            SettingCategory::Analysis,
            format!("{:>3}%", (app.config.settings.release * 100.0) as u16),
        ),
        setting_row(5, "bars", SettingCategory::Analysis, bar_count_label(app)),
        setting_row(6, "renderer", SettingCategory::Visual, renderer_label(app)),
        setting_row(
            7,
            "fft_size",
            SettingCategory::Analysis,
            app.config.settings.fft_size.to_string(),
        ),
        setting_row(
            8,
            "analysis_hop",
            SettingCategory::Analysis,
            app.config.settings.analysis_hop.to_string(),
        ),
        setting_row(
            9,
            "refresh_rate",
            SettingCategory::General,
            format!("{}Hz", app.config.settings.refresh_hz),
        ),
        setting_row(
            10,
            "high_shelf",
            SettingCategory::Processing,
            on_off_label(app, app.config.settings.high_shelf_enabled),
        ),
        setting_row(
            11,
            "high_shelf_db",
            SettingCategory::Processing,
            format!("{:.0}dB", app.config.settings.high_shelf_db),
        ),
        setting_row(
            12,
            "auto_sensitivity",
            SettingCategory::Processing,
            on_off_label(app, app.config.settings.auto_sensitivity_enabled),
        ),
        setting_row(
            13,
            "noise_reduction",
            SettingCategory::Processing,
            format!(
                "{:>3}%",
                (app.config.settings.noise_reduction * 100.0).round() as u16
            ),
        ),
        setting_row(
            14,
            "bpm_analysis",
            SettingCategory::Analysis,
            on_off_label(app, app.config.settings.bpm_enabled),
        ),
        setting_row(
            15,
            "visual_curve",
            SettingCategory::Visual,
            on_off_label(app, app.config.settings.visual_curve_enabled),
        ),
        setting_row(
            16,
            "curve_power",
            SettingCategory::Visual,
            format!("{:.2}", app.config.settings.visual_curve),
        ),
        setting_row(
            17,
            "trail",
            SettingCategory::Visual,
            on_off_label(app, app.config.settings.trail_enabled),
        ),
        setting_row(
            18,
            "trail_decay",
            SettingCategory::Visual,
            format!("{:>3}%", (app.config.settings.trail_decay * 100.0) as u16),
        ),
        setting_row(
            19,
            "accent_display",
            SettingCategory::Visual,
            accent_display_label(app, app.config.settings.accent_display_mode),
        ),
        setting_row(
            20,
            "accent_threshold",
            SettingCategory::Visual,
            format!(
                "{:>3}%",
                (app.config.settings.accent_trace_threshold * 100.0).round() as u16
            ),
        ),
        setting_row(
            21,
            "ceiling",
            SettingCategory::Processing,
            format!("{:>3}%", (app.config.settings.ceiling * 100.0) as u16),
        ),
    ]
}

fn setting_row(
    index: usize,
    key: &'static str,
    category: SettingCategory,
    value: impl Into<String>,
) -> SettingRow {
    SettingRow {
        index,
        key,
        category,
        value: value.into(),
    }
}

fn setting_range(app: &App, key: &'static str) -> String {
    match key {
        "language" => LANGUAGES
            .iter()
            .map(|(_, label)| *label)
            .collect::<Vec<_>>()
            .join(" / "),
        "theme" => THEMES
            .iter()
            .map(|theme_id| app.t(theme(*theme_id).title_key))
            .collect::<Vec<_>>()
            .join(" / "),
        "analysis_preset" => ANALYSIS_PRESETS
            .iter()
            .map(|preset| app.t(preset.title_key()))
            .collect::<Vec<_>>()
            .join(" / "),
        "attack" => percent_range(app, MIN_ATTACK, MAX_ATTACK, ATTACK_STEP),
        "release" => percent_range(app, MIN_RELEASE, MAX_RELEASE, RELEASE_STEP),
        "bars" => range_with_step(
            app,
            format!("{}-{}", MIN_CONFIG_BARS, MAX_CONFIG_BARS),
            "8".to_string(),
        ),
        "renderer" => SPECTRUM_RENDERERS
            .iter()
            .map(|renderer| app.t(renderer_title_key(*renderer)))
            .collect::<Vec<_>>()
            .join(" / "),
        "fft_size" => FFT_SIZES
            .iter()
            .map(|size| size.to_string())
            .collect::<Vec<_>>()
            .join(" / "),
        "analysis_hop" => ANALYSIS_HOPS
            .iter()
            .map(|size| size.to_string())
            .collect::<Vec<_>>()
            .join(" / "),
        "refresh_rate" => REFRESH_RATES
            .iter()
            .map(|rate| format!("{}Hz", rate))
            .collect::<Vec<_>>()
            .join(" / "),
        "high_shelf" | "auto_sensitivity" | "bpm_analysis" | "visual_curve" | "trail" => {
            format!("{} / {}", app.t("on"), app.t("off"))
        }
        "accent_display" => ACCENT_DISPLAY_MODES
            .iter()
            .map(|mode| accent_display_label(app, *mode))
            .collect::<Vec<_>>()
            .join(" / "),
        "high_shelf_db" => range_with_step(
            app,
            format!("{:.0}-{:.0}dB", MIN_HIGH_SHELF_DB, MAX_HIGH_SHELF_DB),
            format!("{:.0}dB", HIGH_SHELF_DB_STEP),
        ),
        "noise_reduction" => percent_range(
            app,
            MIN_NOISE_REDUCTION,
            MAX_NOISE_REDUCTION,
            NOISE_REDUCTION_STEP,
        ),
        "curve_power" => range_with_step(
            app,
            format!("{:.2}-{:.2}", MIN_VISUAL_CURVE, MAX_VISUAL_CURVE),
            format!("{:.2}", VISUAL_CURVE_STEP),
        ),
        "trail_decay" => percent_range(app, MIN_TRAIL_DECAY, MAX_TRAIL_DECAY, TRAIL_DECAY_STEP),
        "accent_threshold" => percent_range(
            app,
            MIN_ACCENT_TRACE_THRESHOLD,
            MAX_ACCENT_TRACE_THRESHOLD,
            ACCENT_TRACE_THRESHOLD_STEP,
        ),
        "ceiling" => percent_range(app, MIN_CEILING, MAX_CEILING, CEILING_STEP),
        _ => app.t("help_setting_default").to_string(),
    }
}

fn renderer_title_key(renderer: SpectrumRenderer) -> &'static str {
    match renderer {
        SpectrumRenderer::Blocks => "renderer_blocks",
        SpectrumRenderer::Braille => "renderer_braille",
        SpectrumRenderer::Cava => "renderer_cava",
    }
}

fn accent_display_label(app: &App, mode: AccentDisplayMode) -> &'static str {
    app.t(match mode {
        AccentDisplayMode::Trace => "accent_display_trace",
        AccentDisplayMode::NoteNames => "accent_display_note_names",
        AccentDisplayMode::Off => "accent_display_off",
    })
}

fn range_with_step(app: &App, range: String, step: String) -> String {
    format!("{} · {} {}", range, app.t("setting_step"), step)
}

fn percent_range(app: &App, min: f32, max: f32, step: f32) -> String {
    range_with_step(
        app,
        format!("{}-{}", percent_label(min), percent_label(max)),
        percent_label(step),
    )
}

fn percent_label(value: f32) -> String {
    let percent = value * 100.0;
    if (percent - percent.round()).abs() < 0.01 {
        format!("{:.0}%", percent)
    } else {
        format!("{:.1}%", percent)
    }
}

fn setting_help_key(key: &'static str) -> &'static str {
    match key {
        "language" => "help_setting_language",
        "theme" => "help_setting_theme",
        "analysis_preset" => "help_setting_analysis_preset",
        "attack" => "help_setting_attack",
        "release" => "help_setting_release",
        "bars" => "help_setting_bars",
        "renderer" => "help_setting_renderer",
        "fft_size" => "help_setting_fft_size",
        "analysis_hop" => "help_setting_analysis_hop",
        "refresh_rate" => "help_setting_refresh_rate",
        "high_shelf" => "help_setting_high_shelf",
        "high_shelf_db" => "help_setting_high_shelf_db",
        "auto_sensitivity" => "help_setting_auto_sensitivity",
        "noise_reduction" => "help_setting_noise_reduction",
        "bpm_analysis" => "help_setting_bpm_analysis",
        "visual_curve" => "help_setting_visual_curve",
        "curve_power" => "help_setting_curve_power",
        "trail" => "help_setting_trail",
        "trail_decay" => "help_setting_trail_decay",
        "accent_display" => "help_setting_accent_display",
        "accent_threshold" => "help_setting_accent_threshold",
        "ceiling" => "help_setting_ceiling",
        _ => "help_setting_default",
    }
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
    let accent_trace = match settings.accent_display_mode {
        AccentDisplayMode::Trace => format!(
            "accent trace {:>2}%",
            (settings.accent_trace_threshold * 100.0).round() as u16
        ),
        AccentDisplayMode::NoteNames => format!(
            "accent notes {:>2}%",
            (settings.accent_trace_threshold * 100.0).round() as u16
        ),
        AccentDisplayMode::Off => "accent off".to_string(),
    };
    let sensitivity = if settings.auto_sensitivity_enabled {
        "autosens on"
    } else {
        "autosens off"
    };
    let bpm = if settings.bpm_enabled {
        format!("bpm {}", bpm_label(app))
    } else {
        "bpm off".to_string()
    };
    let beat = beat_phase_bar(app, 8);
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
            pipeline_stage(theme, "PRE"),
            Span::styled(
                "stereo -> mono -> DC trim -> Hann",
                Style::default().fg(theme.text),
            ),
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
                    "RMS+peak atk {:>2} rel {:>2} | {}",
                    (settings.attack * 100.0) as u16,
                    (settings.release * 100.0) as u16,
                    sensitivity
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "PROC"),
            Span::styled(
                format!(
                    "{} | NR {:>2}% | lim {:>2}% | {} | {} | {}",
                    shelf,
                    (settings.noise_reduction * 100.0).round() as u16,
                    (settings.ceiling * 100.0) as u16,
                    meter,
                    trail,
                    accent_trace
                ),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            pipeline_stage(theme, "TEMPO"),
            Span::styled(bpm, Style::default().fg(theme.text)),
            Span::raw(" "),
            beat_indicator_span(app),
            Span::raw(" "),
            Span::styled(beat, Style::default().fg(theme.muted)),
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

fn inset_rect(area: Rect, horizontal: u16, vertical: u16) -> Rect {
    let inset_x = horizontal.min(area.width / 2);
    let inset_y = vertical.min(area.height / 2);
    Rect {
        x: area.x + inset_x,
        y: area.y + inset_y,
        width: area.width.saturating_sub(inset_x * 2),
        height: area.height.saturating_sub(inset_y * 2),
    }
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
        SpectrumRenderer::Cava => app.t("renderer_cava"),
    }
}

fn bpm_label(app: &App) -> String {
    if !app.config.settings.bpm_enabled {
        return app.t("off").to_string();
    }

    app.bpm_estimate
        .map(|bpm| {
            format!(
                "{:>3} {:.0}%",
                bpm.round() as u16,
                (app.bpm_confidence * 100.0).round()
            )
        })
        .unwrap_or_else(|| "--".to_string())
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
        (Lang::Zh, "settings_general") => "通用",
        (Lang::Zh, "settings_analysis") => "分析",
        (Lang::Zh, "settings_processing") => "处理",
        (Lang::Zh, "settings_visual") => "视觉",
        (Lang::Zh, "setting_adjust") => "调整",
        (Lang::Zh, "setting_select") => "选择",
        (Lang::Zh, "current_value") => "当前值",
        (Lang::Zh, "setting_range") => "范围",
        (Lang::Zh, "setting_step") => "步进",
        (Lang::Zh, "beat") => "节拍",
        (Lang::Zh, "language") => "语言",
        (Lang::Zh, "theme") => "主题",
        (Lang::Zh, "analysis_preset") => "分析预设",
        (Lang::Zh, "attack") => "Attack",
        (Lang::Zh, "release") => "Release",
        (Lang::Zh, "analysis_hop") => "Hop",
        (Lang::Zh, "bars") => "频段",
        (Lang::Zh, "renderer") => "渲染",
        (Lang::Zh, "fft_size") => "FFT",
        (Lang::Zh, "refresh_rate") => "刷新率",
        (Lang::Zh, "high_shelf") => "高频补偿",
        (Lang::Zh, "high_shelf_db") => "补偿强度",
        (Lang::Zh, "auto_sensitivity") => "自动灵敏度",
        (Lang::Zh, "noise_reduction") => "降噪",
        (Lang::Zh, "bpm_analysis") => "BPM 分析",
        (Lang::Zh, "bpm") => "BPM",
        (Lang::Zh, "visual_curve") => "高度曲线",
        (Lang::Zh, "curve_power") => "曲线指数",
        (Lang::Zh, "trail") => "残影",
        (Lang::Zh, "trail_decay") => "残影衰减",
        (Lang::Zh, "accent_trace") => "重音轮廓",
        (Lang::Zh, "accent_display") => "重音显示",
        (Lang::Zh, "accent_display_trace") => "轮廓",
        (Lang::Zh, "accent_display_note_names") => "音名",
        (Lang::Zh, "accent_display_off") => "关",
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
        (Lang::Zh, "help_4") => "↑/↓ 选择设置，Tab 切分类，←/→ 调整；S 打开全屏设置。",
        (Lang::Zh, "help_5") => "窗口较小时模块会自动隐藏，仍可用快捷键操作；主菜单 q/Esc 退出。",
        (Lang::Zh, "permission_note") => "首次捕获会触发 macOS 屏幕与系统音频录制授权；Terb 只实时分析，不保存音频。",
        (Lang::Zh, "menu_hint") => "↑/↓ 选择 · Enter 确认 · Space 捕获 · ? 帮助 · q 退出",
        (Lang::Zh, "spectrum_hint") => "Space 捕获 · s/p/t/m/w 模块 · S 设置 · q 菜单",
        (Lang::Zh, "sidebar_hint") => "Space 开关捕获\ns/p/t/m/w 模块\nTab 分类\n↑/↓ 选择设置\n←/→ 调整\nS 设置\nq 菜单\n? 帮助",
        (Lang::Zh, "compact_hint") => "s/p/t/m/w 模块 · S 设置 · q 菜单",
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
        (Lang::Zh, "theme_spring") => "Spring",
        (Lang::Zh, "theme_vintage") => "vintage",
        (Lang::Zh, "theme_mono") => "单色",
        (Lang::Zh, "renderer_blocks") => "方块",
        (Lang::Zh, "renderer_braille") => "盲文",
        (Lang::Zh, "renderer_cava") => "CAVA 字符",
        (Lang::Zh, "preset_low_latency") => "低延迟",
        (Lang::Zh, "preset_balanced") => "均衡",
        (Lang::Zh, "preset_precision") => "精细",
        (Lang::Zh, "preset_custom") => "自定义",
        (Lang::Zh, "help_setting_language") => "切换界面语言。不会影响配置结构或音频处理。",
        (Lang::Zh, "help_setting_theme") => "切换终端配色主题。主题只影响显示，不改变分析结果。",
        (Lang::Zh, "help_setting_analysis_preset") => "选择分析延迟、稳定度和刷新速度的预设组合。",
        (Lang::Zh, "help_setting_attack") => "控制频谱上升速度。越高越跟手，越低越稳。",
        (Lang::Zh, "help_setting_release") => "控制频谱回落速度。越高残留越长，越低回落越快。",
        (Lang::Zh, "help_setting_bars") => "设置基础频段数量；大窗口会自动扩展到可用宽度。",
        (Lang::Zh, "help_setting_renderer") => "选择主频谱渲染方式：实心方块、盲文子像素或 CAVA 字符。",
        (Lang::Zh, "help_setting_fft_size") => "FFT 窗口长度。越大低频越稳，延迟和惯性也越高。",
        (Lang::Zh, "help_setting_analysis_hop") => "分析跳步。越小刷新越密，CPU 使用会略升。",
        (Lang::Zh, "help_setting_refresh_rate") => "终端绘制刷新率。高刷新更顺滑，也更吃终端性能。",
        (Lang::Zh, "help_setting_high_shelf") => "启用高频补偿，让高频段不被低频能量长期压住。",
        (Lang::Zh, "help_setting_high_shelf_db") => "高频补偿强度。过高会让齿音和噪声偏亮。",
        (Lang::Zh, "help_setting_auto_sensitivity") => "自动调整显示增益，兼顾安静和响亮片段。",
        (Lang::Zh, "help_setting_noise_reduction") => "降低底噪和稳态背景对频谱高度的影响。",
        (Lang::Zh, "help_setting_bpm_analysis") => "启用宽频谱通量节拍分析，并显示 BPM 与 beat 指示。",
        (Lang::Zh, "help_setting_visual_curve") => "启用高度曲线，用非线性方式重映射柱高。",
        (Lang::Zh, "help_setting_curve_power") => "高度曲线指数。越高越压低小信号，越低越抬高细节。",
        (Lang::Zh, "help_setting_trail") => "显示峰值残影，方便观察瞬态和频段运动。",
        (Lang::Zh, "help_setting_trail_decay") => "残影衰减速度。数值越高拖尾越长。",
        (Lang::Zh, "help_setting_accent_trace") => "显示重音轮廓线，用于突出突然抬升的能量形状。",
        (Lang::Zh, "help_setting_accent_display") => {
            "选择重音显示方式：轮廓线，或在空白区域淡入淡出当前重音音名。"
        }
        (Lang::Zh, "help_setting_accent_threshold") => "重音触发阈值。越高越克制，越低越敏感。",
        (Lang::Zh, "help_setting_ceiling") => "分析显示上限。降低后更少触顶，升高后动态空间更大。",
        (Lang::Zh, "help_setting_default") => "使用左右方向键调整该设置。",

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
        (Lang::En, "settings_general") => "general",
        (Lang::En, "settings_processing") => "processing",
        (Lang::En, "settings_visual") => "visual",
        (Lang::En, "setting_adjust") => "adjust",
        (Lang::En, "setting_select") => "select",
        (Lang::En, "current_value") => "Current",
        (Lang::En, "setting_range") => "Range",
        (Lang::En, "setting_step") => "step",
        (Lang::En, "beat") => "Beat",
        (Lang::En, "language") => "Language",
        (Lang::En, "theme") => "Theme",
        (Lang::En, "analysis_preset") => "Preset",
        (Lang::En, "attack") => "Attack",
        (Lang::En, "release") => "Release",
        (Lang::En, "analysis_hop") => "Hop",
        (Lang::En, "bars") => "Bands",
        (Lang::En, "renderer") => "Render",
        (Lang::En, "fft_size") => "FFT",
        (Lang::En, "refresh_rate") => "Refresh",
        (Lang::En, "high_shelf") => "High-shelf",
        (Lang::En, "high_shelf_db") => "Shelf Gain",
        (Lang::En, "auto_sensitivity") => "Autosens",
        (Lang::En, "noise_reduction") => "Noise Reduce",
        (Lang::En, "bpm_analysis") => "BPM Analyze",
        (Lang::En, "bpm") => "BPM",
        (Lang::En, "visual_curve") => "Height Curve",
        (Lang::En, "curve_power") => "Curve Power",
        (Lang::En, "trail") => "Trail",
        (Lang::En, "trail_decay") => "Trail Decay",
        (Lang::En, "accent_trace") => "Accent Trace",
        (Lang::En, "accent_display") => "Accent Display",
        (Lang::En, "accent_display_trace") => "Trace",
        (Lang::En, "accent_display_note_names") => "Note Names",
        (Lang::En, "accent_display_off") => "Off",
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
        (Lang::En, "help_4") => "Use ↑/↓ to select settings, Tab for groups, and ←/→ to adjust. S opens full-screen settings.",
        (Lang::En, "help_5") => "Small terminals hide modules automatically, but shortcuts still work. q/Esc quits from the main menu.",
        (Lang::En, "permission_note") => "First capture may trigger macOS Screen & System Audio Recording permission. Terb analyzes live audio only and does not save it.",
        (Lang::En, "menu_hint") => "↑/↓ select · Enter confirm · Space capture · ? help · q quit",
        (Lang::En, "spectrum_hint") => "Space capture · s/p/t/m/w modules · S settings · q menu",
        (Lang::En, "sidebar_hint") => "Space toggle capture\ns/p/t/m/w modules\nTab groups\n↑/↓ select setting\n←/→ adjust\nS settings\nq menu\n? help",
        (Lang::En, "compact_hint") => "s/p/t/m/w modules · S settings · q menu",
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
        (Lang::En, "theme_spring") => "Spring",
        (Lang::En, "theme_vintage") => "Vintage",
        (Lang::En, "theme_mono") => "Mono",
        (Lang::En, "renderer_blocks") => "Blocks",
        (Lang::En, "renderer_braille") => "Braille",
        (Lang::En, "renderer_cava") => "CAVA",
        (Lang::En, "preset_low_latency") => "Low Latency",
        (Lang::En, "preset_balanced") => "Balanced",
        (Lang::En, "preset_precision") => "Precision",
        (Lang::En, "preset_custom") => "Custom",
        (Lang::En, "help_setting_language") => "Switches the interface language without changing the audio pipeline.",
        (Lang::En, "help_setting_theme") => "Changes terminal colors only; analysis output stays untouched.",
        (Lang::En, "help_setting_analysis_preset") => "Chooses a preset balance between latency, stability, and refresh density.",
        (Lang::En, "help_setting_attack") => "Controls how quickly spectrum bars rise toward new peaks.",
        (Lang::En, "help_setting_release") => "Controls how quickly spectrum bars fall after peaks.",
        (Lang::En, "help_setting_bars") => "Sets the base band count; wide terminals can expand it automatically.",
        (Lang::En, "help_setting_renderer") => "Chooses blocks, Braille subpixels, or CAVA-style stepped characters.",
        (Lang::En, "help_setting_fft_size") => "FFT window size. Larger windows improve low-end stability but add inertia.",
        (Lang::En, "help_setting_analysis_hop") => "Analysis hop size. Smaller hops update more often and cost slightly more CPU.",
        (Lang::En, "help_setting_refresh_rate") => "Terminal draw rate. Higher values feel smoother if the terminal keeps up.",
        (Lang::En, "help_setting_high_shelf") => "Enables high-frequency compensation so treble is not buried by bass energy.",
        (Lang::En, "help_setting_high_shelf_db") => "High-shelf gain. Too much can make hiss and sibilance too prominent.",
        (Lang::En, "help_setting_auto_sensitivity") => "Adapts display gain across quiet and loud passages.",
        (Lang::En, "help_setting_noise_reduction") => "Reduces floor noise and steady background energy in the visual output.",
        (Lang::En, "help_setting_bpm_analysis") => "Enables wideband spectral-flux tempo tracking and beat indicators.",
        (Lang::En, "help_setting_visual_curve") => "Applies nonlinear height mapping to the spectrum.",
        (Lang::En, "help_setting_curve_power") => "Height curve exponent. Higher values suppress small signals more.",
        (Lang::En, "help_setting_trail") => "Shows a peak trail for transients and band movement.",
        (Lang::En, "help_setting_trail_decay") => "Trail decay amount. Higher values keep the trail longer.",
        (Lang::En, "help_setting_accent_trace") => "Draws accent envelopes when spectrum energy rises suddenly.",
        (Lang::En, "help_setting_accent_display") => {
            "Chooses accent display: envelope trace, or fading note names in empty spectrum space."
        }
        (Lang::En, "help_setting_accent_threshold") => "Accent trigger threshold. Higher is more restrained.",
        (Lang::En, "help_setting_ceiling") => "Display ceiling for analysis headroom and peak mapping.",
        (Lang::En, "help_setting_default") => "Use left and right to adjust this setting.",

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
        (Lang::Ja, "settings_general") => "一般",
        (Lang::Ja, "settings_analysis") => "解析",
        (Lang::Ja, "settings_visual") => "表示",
        (Lang::Ja, "setting_adjust") => "調整",
        (Lang::Ja, "setting_select") => "選択",
        (Lang::Ja, "current_value") => "現在値",
        (Lang::Ja, "setting_range") => "範囲",
        (Lang::Ja, "setting_step") => "刻み",
        (Lang::Ja, "beat") => "拍",
        (Lang::Ja, "language") => "言語",
        (Lang::Ja, "theme") => "テーマ",
        (Lang::Ja, "analysis_preset") => "解析プリセット",
        (Lang::Ja, "attack") => "Attack",
        (Lang::Ja, "release") => "Release",
        (Lang::Ja, "analysis_hop") => "Hop",
        (Lang::Ja, "bars") => "バンド",
        (Lang::Ja, "renderer") => "描画",
        (Lang::Ja, "fft_size") => "FFT",
        (Lang::Ja, "refresh_rate") => "更新率",
        (Lang::Ja, "high_shelf") => "高域補正",
        (Lang::Ja, "high_shelf_db") => "補正量",
        (Lang::Ja, "auto_sensitivity") => "自動感度",
        (Lang::Ja, "noise_reduction") => "ノイズ低減",
        (Lang::Ja, "bpm_analysis") => "BPM 解析",
        (Lang::Ja, "bpm") => "BPM",
        (Lang::Ja, "visual_curve") => "高さ曲線",
        (Lang::Ja, "curve_power") => "曲線指数",
        (Lang::Ja, "trail") => "残像",
        (Lang::Ja, "trail_decay") => "残像減衰",
        (Lang::Ja, "accent_trace") => "アクセント輪郭",
        (Lang::Ja, "accent_display") => "アクセント表示",
        (Lang::Ja, "accent_display_trace") => "輪郭",
        (Lang::Ja, "accent_display_note_names") => "音名",
        (Lang::Ja, "accent_display_off") => "オフ",
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
        (Lang::Ja, "help_4") => "↑/↓ で設定選択、Tab で分類切替、←/→ で変更。S で全画面設定。",
        (Lang::Ja, "help_5") => "小さいウィンドウではモジュールを自動で隠しますが、ショートカットは使えます。メインメニューでは q/Esc で終了します。",
        (Lang::Ja, "permission_note") => "初回キャプチャでは macOS の画面とシステム音声録音権限が必要です。Terb はリアルタイム解析のみ行い、音声を保存しません。",
        (Lang::Ja, "menu_hint") => "↑/↓ 選択 · Enter 決定 · Space キャプチャ · ? ヘルプ · q 終了",
        (Lang::Ja, "spectrum_hint") => "Space キャプチャ · s/p/t/m/w モジュール · S 設定 · q メニュー",
        (Lang::Ja, "sidebar_hint") => "Space キャプチャ切替\ns/p/t/m/w モジュール\nTab 分類\n↑/↓ 設定選択\n←/→ 変更\nS 設定\nq メニュー\n? ヘルプ",
        (Lang::Ja, "compact_hint") => "s/p/t/m/w モジュール · S 設定 · q メニュー",
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
        (Lang::Ja, "theme_spring") => "Spring",
        (Lang::Ja, "theme_vintage") => "ヴィンテージ",
        (Lang::Ja, "theme_mono") => "モノ",
        (Lang::Ja, "renderer_blocks") => "ブロック",
        (Lang::Ja, "renderer_braille") => "点字",
        (Lang::Ja, "renderer_cava") => "CAVA 文字",
        (Lang::Ja, "preset_low_latency") => "低遅延",
        (Lang::Ja, "preset_balanced") => "バランス",
        (Lang::Ja, "preset_precision") => "精密",
        (Lang::Ja, "preset_custom") => "カスタム",
        (Lang::Ja, "help_setting_language") => "表示言語を切り替えます。音声処理には影響しません。",
        (Lang::Ja, "help_setting_theme") => "ターミナルの配色を変更します。解析結果は変わりません。",
        (Lang::Ja, "help_setting_analysis_preset") => "遅延、安定性、更新密度のバランスを選びます。",
        (Lang::Ja, "help_setting_attack") => "スペクトラムがピークへ上がる速さを調整します。",
        (Lang::Ja, "help_setting_release") => "ピーク後にスペクトラムが下がる速さを調整します。",
        (Lang::Ja, "help_setting_bars") => "基本バンド数です。広い端末では自動で拡張されます。",
        (Lang::Ja, "help_setting_renderer") => "ブロック、点字サブピクセル、CAVA 風文字を切り替えます。",
        (Lang::Ja, "help_setting_fft_size") => "FFT 窓長です。大きいほど低域は安定し、慣性も増えます。",
        (Lang::Ja, "help_setting_analysis_hop") => "解析ホップです。小さいほど更新が細かくなります。",
        (Lang::Ja, "help_setting_refresh_rate") => "端末描画の更新率です。高いほど滑らかですが負荷も増えます。",
        (Lang::Ja, "help_setting_high_shelf") => "低域に埋もれやすい高域を補正します。",
        (Lang::Ja, "help_setting_high_shelf_db") => "高域補正量です。上げすぎるとノイズが目立ちます。",
        (Lang::Ja, "help_setting_auto_sensitivity") => "静かな部分と大きい部分に合わせて表示ゲインを調整します。",
        (Lang::Ja, "help_setting_noise_reduction") => "底ノイズや定常成分の表示への影響を抑えます。",
        (Lang::Ja, "help_setting_bpm_analysis") => "広帯域スペクトルフラックスで BPM と拍表示を解析します。",
        (Lang::Ja, "help_setting_visual_curve") => "スペクトラム高さに非線形カーブを適用します。",
        (Lang::Ja, "help_setting_curve_power") => "高さ曲線の指数です。高いほど小信号を抑えます。",
        (Lang::Ja, "help_setting_trail") => "ピーク残像を表示し、瞬間的な動きを見やすくします。",
        (Lang::Ja, "help_setting_trail_decay") => "残像の減衰量です。高いほど長く残ります。",
        (Lang::Ja, "help_setting_accent_trace") => "急なエネルギー上昇を輪郭線として表示します。",
        (Lang::Ja, "help_setting_accent_display") => {
            "アクセント表示を選びます。輪郭線、または空白部にフェードする音名です。"
        }
        (Lang::Ja, "help_setting_accent_threshold") => "アクセント検出の閾値です。高いほど控えめです。",
        (Lang::Ja, "help_setting_ceiling") => "解析表示の上限です。ヘッドルームとピーク表示を調整します。",
        (Lang::Ja, "help_setting_default") => "左右キーでこの設定を調整します。",

        _ => key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    fn buffer_row(buf: &Buffer, y: u16, width: u16) -> String {
        (0..width).map(|x| buf[(x, y)].symbol()).collect()
    }

    fn buffer_text(buf: &Buffer, width: u16, height: u16) -> String {
        (0..height)
            .map(|y| buffer_row(buf, y, width))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn settings_list_fits_full_height_without_scrolling() {
        let rows = settings_rows(&App::new(Config::default()));
        let setting_count = rows.len();
        assert_eq!(visible_list_start(0, setting_count, setting_count), 0);
        assert_eq!(
            visible_list_start(setting_count - 1, setting_count, setting_count),
            0
        );
    }

    #[test]
    fn settings_list_scrolls_selected_row_into_view() {
        let rows = settings_rows(&App::new(Config::default()));
        let setting_count = rows.len();
        assert_eq!(
            visible_list_start(setting_count - 1, setting_count, 18),
            setting_count - 18
        );
    }

    #[test]
    fn tab_setting_category_moves_to_next_group() {
        let rows = settings_rows(&App::new(Config::default()));

        assert_eq!(adjacent_setting_category_index(0, &rows, 1), 2);
        assert_eq!(adjacent_setting_category_index(2, &rows, 1), 10);
        assert_eq!(adjacent_setting_category_index(10, &rows, 1), 6);
        assert_eq!(adjacent_setting_category_index(6, &rows, 1), 0);
    }

    #[test]
    fn setting_navigation_moves_through_full_list() {
        let rows = settings_rows(&App::new(Config::default()));

        assert_eq!(next_setting_index(1, &rows), 2);
        assert_eq!(previous_setting_index(2, &rows), 1);
        let last = rows.last().expect("settings rows").index;
        assert_eq!(next_setting_index(last, &rows), last);
    }

    #[test]
    fn selected_theme_setting_cycles_both_directions() {
        let mut app = App::new(Config::default());
        let initial = app.theme_id;

        app.adjust_setting_by_key("theme", 1);
        assert_ne!(app.theme_id, initial);
        app.adjust_setting_by_key("theme", -1);
        assert_eq!(app.theme_id, initial);
    }

    #[test]
    fn compact_settings_sidebar_scrolls_across_all_categories() {
        let mut config = Config::default();
        config.settings.language = "en".to_string();
        let mut app = App::new(config);
        app.setting_index = settings_rows(&app).last().expect("settings row").index;
        let backend = TestBackend::new(35, 22);
        let mut terminal = Terminal::new(backend).expect("test backend");

        terminal
            .draw(|frame| {
                draw_settings_list(
                    frame,
                    &app,
                    Rect::new(0, 0, 35, 22),
                    panel_title(app.t("settings"), app.theme()),
                )
            })
            .expect("render compact settings");

        let text = buffer_text(terminal.backend().buffer(), 35, 22);
        assert!(text.contains("Accent Display"));
        assert!(text.contains("Ceiling"));
    }

    #[test]
    fn compact_settings_sidebar_preserves_wide_label_values() {
        let mut config = Config::default();
        config.settings.language = "zh".to_string();
        config.settings.theme = ThemeId::Vintage;
        let app = App::new(config);
        let rows = settings_rows(&app);
        let theme_row = rows
            .iter()
            .find(|row| row.key == "theme")
            .expect("theme row");
        let line = setting_list_line(&app, theme_row, 33);
        let text = line_text(&line);

        assert!(text.contains("主题"));
        assert!(text.contains("vintage"));
        assert_eq!(UnicodeWidthStr::width(text.as_str()), 33);
    }

    #[test]
    fn bpm_pulse_lights_when_predicted_beat_arrives() {
        let mut app = App::new(Config::default());
        app.set_bpm_estimate(120.0);
        app.bpm_next_beat_at = Some(Instant::now() - Duration::from_millis(1));

        app.advance_bpm_pulse(Duration::from_millis(16));

        assert_eq!(app.bpm_pulse, 1.0);
        assert!(app.bpm_phase < 0.10);
        assert!(beat_phase_bar(&app, 8).contains('●'));
    }

    #[test]
    fn settings_board_renders_list_divider_and_selected_help() {
        let mut config = Config::default();
        config.settings.language = "en".to_string();
        let app = App::new(config);
        let backend = TestBackend::new(74, 18);
        let mut terminal = Terminal::new(backend).expect("test backend");

        terminal
            .draw(|frame| draw_settings_board(frame, &app, Rect::new(0, 0, 74, 18)))
            .expect("render settings");

        let text = buffer_text(terminal.backend().buffer(), 74, 18);
        assert!(!text.contains("[general]"));
        assert!(text.contains("Language"));
        assert!(text.contains("│"));
        assert!(text.contains("Current"));
        assert!(text.contains("Range"));
        assert!(text.contains("Switches the interface language"));
    }

    #[test]
    fn high_shelf_gain_adjusts_across_expanded_range() {
        let mut app = App::new(Config::default());
        app.config.settings.high_shelf_db = MIN_HIGH_SHELF_DB;

        for _ in 0..64 {
            app.adjust_setting_by_key("high_shelf_db", 1);
        }

        assert_eq!(app.config.settings.high_shelf_db, MAX_HIGH_SHELF_DB);

        for _ in 0..64 {
            app.adjust_setting_by_key("high_shelf_db", -1);
        }

        assert_eq!(app.config.settings.high_shelf_db, MIN_HIGH_SHELF_DB);
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
    fn renderer_cycle_includes_blocks_braille_and_cava() {
        let mut app = App::new(Config::default());

        assert_eq!(app.config.settings.renderer, SpectrumRenderer::Blocks);
        app.cycle_renderer(1);
        assert_eq!(app.config.settings.renderer, SpectrumRenderer::Braille);
        app.cycle_renderer(1);
        assert_eq!(app.config.settings.renderer, SpectrumRenderer::Cava);
        app.cycle_renderer(1);
        assert_eq!(app.config.settings.renderer, SpectrumRenderer::Blocks);
    }

    #[test]
    fn cava_bar_cell_uses_eighth_height_symbols() {
        assert_eq!(cava_bar_cell(0.0, 1, 2).level, 0);
        assert_eq!(cava_bar_cell(0.50, 1, 2).level, 8);
        assert_eq!(cava_bar_cell(0.56, 0, 2).level, 1);
        assert_eq!(cava_block_symbol(1), "▁");
        assert_eq!(cava_block_symbol(8), "█");
    }

    #[test]
    fn vintage_theme_uses_refined_four_color_palette() {
        let vintage = theme(ThemeId::Vintage);

        assert!(THEMES.contains(&ThemeId::Vintage));
        assert_eq!(vintage.border, Color::Rgb(96, 116, 86));
        assert_eq!(vintage.text, Color::Rgb(238, 224, 204));
        assert_eq!(vintage.mid, Color::Rgb(186, 106, 76));
        assert_eq!(vintage.low, Color::Rgb(123, 37, 37));
        assert_eq!(vintage.peak, Color::Rgb(96, 116, 86));
    }

    #[test]
    fn spring_theme_uses_soft_vertical_palette() {
        let spring = theme(ThemeId::Spring);

        assert_eq!(Config::default().settings.theme, ThemeId::Spring);
        assert!(THEMES.contains(&ThemeId::Spring));
        assert_eq!(spring.high, Color::Rgb(252, 249, 234));
        assert_eq!(spring.peak, Color::Rgb(186, 223, 219));
        assert_eq!(spring.low, Color::Rgb(255, 164, 164));
        assert_eq!(spring.mid, Color::Rgb(255, 189, 189));
    }

    #[test]
    fn removed_themes_are_not_selectable() {
        assert!(!THEMES.contains(&ThemeId::System));
        assert!(!THEMES.contains(&ThemeId::Graphite));
        assert!(!THEMES.contains(&ThemeId::Ocean));
        assert!(!THEMES.contains(&ThemeId::NoiseWarp));
        assert!(!THEMES.contains(&ThemeId::Amber));
    }

    #[test]
    fn static_spectrum_color_is_vertical_not_frequency_split() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::Spring;

        let left = spectrum_bar_color_at(&app, 2, 100, 0.80, 0.54, false);
        let right = spectrum_bar_color_at(&app, 92, 100, 0.80, 0.54, false);
        let lower = spectrum_bar_color_at(&app, 40, 100, 0.80, 0.22, false);
        let upper = spectrum_bar_color_at(&app, 40, 100, 0.80, 0.86, false);

        assert_eq!(left, right);
        assert_ne!(lower, upper);
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
    fn fixed_master_meter_width_fills_equal_channels() {
        let inner = Panel::inner(Rect::new(0, 0, MASTER_METER_WIDTH, 10));
        let channel_width = master_meter_channel_width(inner.width as usize);

        assert_eq!(inner.width as usize, MASTER_METER_CHANNEL_WIDTH * 2);
        assert_eq!(channel_width, MASTER_METER_CHANNEL_WIDTH);
        assert_eq!(channel_width * 2, inner.width as usize);
    }

    #[test]
    fn master_meter_center_cells_leave_subpixel_gap() {
        let full = 0xff;
        let left_edge = master_meter_channel_mask(
            full,
            MasterMeterSide::Left,
            MASTER_METER_CHANNEL_WIDTH - 1,
            MASTER_METER_CHANNEL_WIDTH,
        );
        let right_edge =
            master_meter_channel_mask(full, MasterMeterSide::Right, 0, MASTER_METER_CHANNEL_WIDTH);

        assert_eq!(left_edge, BRAILLE_LEFT_COLUMN_MASK);
        assert_eq!(right_edge, BRAILLE_RIGHT_COLUMN_MASK);
        assert_eq!(
            master_meter_channel_mask(full, MasterMeterSide::Left, 0, MASTER_METER_CHANNEL_WIDTH),
            full
        );
        assert_eq!(
            master_meter_channel_mask(
                full,
                MasterMeterSide::Right,
                MASTER_METER_CHANNEL_WIDTH - 1,
                MASTER_METER_CHANNEL_WIDTH
            ),
            full
        );
    }

    #[test]
    fn master_meter_color_blends_border_toward_theme_accent() {
        let mut app = App::new(Config::default());
        app.theme_id = ThemeId::Spring;

        assert_eq!(master_meter_color(&app, 0.0), Color::Rgb(186, 223, 219));
        assert_eq!(master_meter_color(&app, 1.0), Color::Rgb(255, 164, 164));
        assert_ne!(master_meter_color(&app, 0.5), master_meter_color(&app, 0.0));
        assert_ne!(master_meter_color(&app, 0.5), master_meter_color(&app, 1.0));
    }

    #[test]
    fn retired_theme_config_migrates_to_spring() {
        let mut config = Config::default();
        config.settings.theme = ThemeId::HarmonicComb;

        let app = App::new(config);

        assert_eq!(app.theme_id, ThemeId::Spring);
        assert_eq!(app.config.settings.theme, ThemeId::Spring);
    }

    #[test]
    fn removed_theme_config_migrates_to_spring() {
        for removed in [
            ThemeId::System,
            ThemeId::Graphite,
            ThemeId::Ocean,
            ThemeId::NoiseWarp,
            ThemeId::Amber,
        ] {
            let mut config = Config::default();
            config.settings.theme = removed;

            let app = App::new(config);

            assert_eq!(app.theme_id, ThemeId::Spring);
            assert_eq!(app.config.settings.theme, ThemeId::Spring);
        }
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
    fn spectrum_trail_uses_expanded_decay_range() {
        let settings = Settings {
            trail_decay: MIN_TRAIL_DECAY,
            ..Config::default().settings
        };
        let mut trail = vec![1.0];

        update_spectrum_trail(&mut trail, &[0.0], &settings);

        assert!((trail[0] - MIN_TRAIL_DECAY).abs() < f32::EPSILON);
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
        let final_offset = ACCENT_TRACE_END_OFFSET_CELLS * 4.0;

        let (shifted_mask, value) = accent_trace_braille_cell(&envelope, 0, 1, 48, final_offset);
        let (lower_cell_mask, _) = accent_trace_braille_cell(&envelope, 0, 6, 48, final_offset);

        assert_eq!(shifted_mask, 0x09);
        assert_eq!(value, 0.50);
        assert_eq!(lower_cell_mask, 0);
    }

    #[test]
    fn accent_trace_line_animates_from_one_to_five_cells_up() {
        let envelope = vec![0.50, 0.50];
        let mut trace = AccentTrace {
            envelope: envelope.clone(),
            age: Duration::from_millis(0),
        };

        let start_offset = trace.vertical_offset_rows(48);
        let (start_mask, _) = accent_trace_braille_cell(&envelope, 0, 5, 48, start_offset);
        trace.age = Duration::from_millis(ACCENT_TRACE_OFFSET_ANIMATION_MS / 2);
        let mid_offset = trace.vertical_offset_rows(48);
        trace.age = Duration::from_millis(ACCENT_TRACE_OFFSET_ANIMATION_MS);
        let end_offset = trace.vertical_offset_rows(48);

        assert_eq!(start_mask, 0x09);
        assert!((start_offset - ACCENT_TRACE_START_OFFSET_CELLS * 4.0).abs() < 0.001);
        assert!(mid_offset > start_offset);
        assert!(mid_offset < end_offset);
        assert!((end_offset - ACCENT_TRACE_END_OFFSET_CELLS * 4.0).abs() < 0.001);
    }

    #[test]
    fn accent_trace_offset_scales_down_for_short_windows() {
        let trace = AccentTrace {
            envelope: vec![0.50, 0.50],
            age: Duration::from_millis(ACCENT_TRACE_OFFSET_ANIMATION_MS),
        };

        let reference_offset = trace.vertical_offset_rows(48);
        let short_offset = trace.vertical_offset_rows(16);

        assert!((reference_offset - ACCENT_TRACE_END_OFFSET_CELLS * 4.0).abs() < 0.001);
        assert!(short_offset < reference_offset);
        assert!(short_offset <= 16.0 * reference_offset / 48.0 + 0.001);
    }

    #[test]
    fn accent_trace_line_interpolates_large_vertical_jumps() {
        let envelope = vec![0.90, 0.10];
        let final_offset = ACCENT_TRACE_END_OFFSET_CELLS * 4.0;

        let (middle_mask, value) = accent_trace_braille_cell(&envelope, 0, 3, 40, final_offset);

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
    fn accent_note_burst_uses_stable_distribution_for_same_note() {
        let mut app = App::new(Config::default());
        app.config.settings.accent_display_mode = AccentDisplayMode::NoteNames;
        let bars = vec![0.20, 0.92, 0.62, 0.18, 0.05, 0.01, 0.0, 0.0];

        app.push_accent_note_burst(&bars);
        let first = app.accent_note_bursts[0].notes.clone();
        app.accent_note_bursts.clear();
        app.push_accent_note_burst(&bars);
        let second = app.accent_note_bursts[0].notes.clone();

        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.label, right.label);
            assert!((left.x - right.x).abs() < f32::EPSILON);
            assert!((left.y - right.y).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn accent_note_burst_fades_in_then_out() {
        let mut burst = AccentNoteBurst {
            notes: Vec::new(),
            age: Duration::from_millis(50),
        };

        assert!((burst.opacity() - 0.50).abs() < 0.001);
        burst.age = Duration::from_millis(100);
        assert!((burst.opacity() - 1.0).abs() < 0.001);
        burst.age = Duration::from_millis(600);
        assert!(burst.opacity() <= 0.001);
    }

    #[test]
    fn accent_note_overlay_only_renders_on_blank_cells() {
        let mut app = App::new(Config::default());
        app.config.settings.accent_display_mode = AccentDisplayMode::NoteNames;
        app.accent_note_bursts.push(AccentNoteBurst {
            notes: vec![AccentNote {
                label: "c",
                pitch_class: 0,
                x: 0.0,
                y: 0.0,
                intensity: 1.0,
            }],
            age: Duration::from_millis(100),
        });

        assert!(accent_note_overlay_cell(&app, 0, 0, 8, 4, false).is_none());
        assert!(accent_note_overlay_cell(&app, 0, 0, 8, 4, true).is_some());
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

        app.update_accent_trace_detector(&[0.80; 32]);

        assert!(app.pending_accent_trace.is_some());
        assert!(app.accent_traces.is_empty());

        app.spectrum = vec![0.80; 32];
        app.update_accent_trace_detector(&[0.55; 32]);

        assert_eq!(app.accent_traces.len(), 1);
        assert!(!app.accent_trace_cooldown.is_zero());
    }

    #[test]
    fn disabled_accent_trace_detector_skips_trace() {
        let mut app = App::new(Config::default());
        app.config.settings.accent_display_mode = AccentDisplayMode::Off;
        app.spectrum = vec![0.0; 32];

        app.update_accent_trace_detector(&[0.80; 32]);

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
        assert_eq!(App::new(config).theme_id, ThemeId::Spring);
    }

    #[test]
    fn visual_bar_count_tracks_renderer_subpixels() {
        let mut app = App::new(Config::default());
        let area = Rect::new(0, 0, 100, 30);
        app.screen = Screen::Spectrum;

        app.config.settings.renderer = SpectrumRenderer::Blocks;
        assert_eq!(visual_bar_count(&app, area), Some(82));

        app.config.settings.renderer = SpectrumRenderer::Cava;
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
    fn configured_minimum_bar_count_reaches_analyzer() {
        let mut config = Config::default();
        config.settings.bar_count = MIN_CONFIG_BARS;
        let app = App::new(config);

        assert_eq!(app.analysis_bar_count(), MIN_CONFIG_BARS);
        assert_eq!(app.analyzer.bar_count, MIN_CONFIG_BARS);
    }

    #[test]
    fn sample_magnitude_interpolates_fractional_bins() {
        let value = sample_magnitude(&[0.0, 1.0, 3.0, 4.0], 1.5);

        assert!((value - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn adaptive_processing_lifts_quiet_signal_without_exceeding_ceiling() {
        let settings = Settings {
            auto_sensitivity_enabled: true,
            noise_reduction: 0.20,
            ceiling: 0.88,
            ..Config::default().settings
        };
        let pipeline = SpectrumPipeline::from_settings(&settings);
        let mut analyzer = SpectrumAnalyzer::new(1024, 48_000.0, 8, 256);
        let mut bars = vec![0.01; 8];

        for _ in 0..80 {
            bars.fill(0.01);
            analyzer.apply_adaptive_processing(&mut bars, 0.001, pipeline);
        }
        bars = vec![0.05; 8];
        analyzer.apply_adaptive_processing(&mut bars, 0.003, pipeline);

        assert!(bars.iter().any(|value| *value > 0.05));
        assert!(bars.iter().all(|value| *value <= settings.ceiling));
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
    fn low_latency_preset_is_default_and_applies_realtime_settings() {
        let defaults = Config::default().settings;
        assert_eq!(defaults.analysis_preset, AnalysisPreset::LowLatency);

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
