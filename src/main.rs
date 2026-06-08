use std::{
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
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    prelude::{CrosstermBackend, Frame, Terminal},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const LANGUAGES: &[(&str, &str)] = &[("zh", "中文"), ("en", "English"), ("ja", "日本語")];
const THEMES: &[ThemeId] = &[
    ThemeId::System,
    ThemeId::Graphite,
    ThemeId::Ocean,
    ThemeId::Amber,
    ThemeId::Mono,
];
const MENU_ITEMS: &[&str] = &[
    "menu_spectrum",
    "menu_toggle",
    "menu_settings",
    "menu_help",
    "menu_quit",
];
const REFRESH_RATES: &[u16] = &[24, 30, 45, 60, 90, 120];
const MIN_FREQUENCY: f32 = 35.0;
const MAX_FREQUENCY: f32 = 18_000.0;
const DEFAULT_HIGH_SHELF_DB: f32 = 6.0;
const DEFAULT_VISUAL_CURVE: f32 = 0.88;
const DEFAULT_CEILING: f32 = 0.88;
const FFT_SIZES: &[usize] = &[1024, 2048, 4096, 8192];
const SILENCE_GATE: f32 = 0.000_12;
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
        app.drain_audio_events();
        terminal.draw(|frame| draw(frame, app))?;

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

        if last_tick.elapsed() >= tick_rate {
            app.tick();
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
        KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('m') => app.screen = Screen::Menu,
        KeyCode::Char(' ') => app.toggle_capture(),
        KeyCode::Char('s') => app.screen = Screen::Settings,
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
    Amber,
    Mono,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    version: u8,
    settings: Settings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Settings {
    language: String,
    theme: ThemeId,
    smoothing: f32,
    bar_count: usize,
    #[serde(default = "default_fft_size")]
    fft_size: usize,
    #[serde(default = "default_refresh_hz")]
    refresh_hz: u16,
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
}

fn default_refresh_hz() -> u16 {
    45
}

fn default_fft_size() -> usize {
    2048
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

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            settings: Settings {
                language: "zh".to_string(),
                theme: ThemeId::System,
                smoothing: 0.72,
                bar_count: 72,
                fft_size: default_fft_size(),
                refresh_hz: default_refresh_hz(),
                high_shelf_enabled: default_high_shelf_enabled(),
                high_shelf_db: default_high_shelf_db(),
                visual_curve_enabled: default_visual_curve_enabled(),
                visual_curve: default_visual_curve(),
                ceiling: default_ceiling(),
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

fn nearest_fft_size(value: usize) -> usize {
    FFT_SIZES
        .iter()
        .copied()
        .min_by_key(|size| size.abs_diff(value))
        .unwrap_or(default_fft_size())
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
    level: f32,
    analyzer: SpectrumAnalyzer,
    audio: Option<AudioProcess>,
    rx: Receiver<AudioEvent>,
    tx: Sender<AudioEvent>,
    last_samples_at: Option<Instant>,
}

impl App {
    fn new(mut config: Config) -> Self {
        let lang = Lang::from_code(&config.settings.language);
        let theme_id = config.settings.theme;
        let bar_count = config.settings.bar_count.clamp(32, 120);
        let fft_size = nearest_fft_size(config.settings.fft_size);
        config.settings.bar_count = bar_count;
        config.settings.fft_size = fft_size;
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
            level: 0.0,
            analyzer: SpectrumAnalyzer::new(fft_size, 48_000.0, bar_count),
            audio: None,
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
        let refresh_hz = self.config.settings.refresh_hz.clamp(12, 120);
        Duration::from_millis((1000 / refresh_hz as u64).max(1))
    }

    fn tick(&mut self) {
        if self.capture_state == CaptureState::Running {
            if let Some(last) = self.last_samples_at {
                if last.elapsed() > Duration::from_secs(3) {
                    self.status = self.t("waiting_audio").to_string();
                }
            }
        }
    }

    fn drain_audio_events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AudioEvent::Samples(samples) => {
                    self.capture_state = CaptureState::Running;
                    self.last_samples_at = Some(Instant::now());
                    self.level = audio_level(&samples);
                    let pipeline = SpectrumPipeline::from_settings(&self.config.settings);
                    if let Some(bars) =
                        self.analyzer
                            .consume(&samples, self.config.settings.smoothing, pipeline)
                    {
                        self.spectrum = bars;
                    }
                    self.status = self.t("running").to_string();
                }
                AudioEvent::Status(message) => {
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
                AudioEvent::Exit(code) => {
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
                AudioEvent::Error(message) => {
                    self.capture_state = CaptureState::Failed;
                    self.status = message;
                }
            }
        }
    }

    fn start_capture(&mut self) {
        if self.audio.is_some() {
            return;
        }

        self.capture_state = CaptureState::Starting;
        self.status = self.t("starting").to_string();

        match AudioProcess::spawn(self.tx.clone()) {
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
        if let Some(mut audio) = self.audio.take() {
            audio.stop();
        }
        self.capture_state = CaptureState::Idle;
        self.status = self.t("stopped").to_string();
        self.level = 0.0;
        self.spectrum.fill(0.0);
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
        self.setting_index = (self.setting_index + 1).min(10);
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
            2 => {
                let delta = if direction < 0 { -0.04 } else { 0.04 };
                self.config.settings.smoothing =
                    (self.config.settings.smoothing + delta).clamp(0.20, 0.92);
            }
            3 => {
                let current = self.config.settings.bar_count as i32;
                let next = (current + direction * 8).clamp(32, 120) as usize;
                if next != self.config.settings.bar_count {
                    self.config.settings.bar_count = next;
                    self.spectrum = vec![0.0; next];
                    self.rebuild_analyzer();
                }
            }
            4 => {
                self.cycle_fft_size(direction);
                self.rebuild_analyzer();
            }
            5 => self.cycle_refresh_rate(direction),
            6 => self.config.settings.high_shelf_enabled = !self.config.settings.high_shelf_enabled,
            7 => {
                self.config.settings.high_shelf_db =
                    (self.config.settings.high_shelf_db + direction as f32).clamp(0.0, 18.0);
            }
            8 => {
                self.config.settings.visual_curve_enabled =
                    !self.config.settings.visual_curve_enabled;
            }
            9 => {
                let delta = if direction < 0 { -0.04 } else { 0.04 };
                self.config.settings.visual_curve =
                    (self.config.settings.visual_curve + delta).clamp(0.55, 1.35);
            }
            10 => {
                let delta = if direction < 0 { -0.02 } else { 0.02 };
                self.config.settings.ceiling =
                    (self.config.settings.ceiling + delta).clamp(0.70, 0.98);
            }
            _ => {}
        }
        self.save_config();
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
    }

    fn rebuild_analyzer(&mut self) {
        let fft_size = nearest_fft_size(self.config.settings.fft_size);
        let bar_count = self.config.settings.bar_count.clamp(32, 120);
        self.config.settings.fft_size = fft_size;
        self.config.settings.bar_count = bar_count;
        self.analyzer = SpectrumAnalyzer::new(fft_size, 48_000.0, bar_count);
        self.spectrum = vec![0.0; bar_count];
    }

    fn save_config(&self) {
        self.config.save();
    }
}

enum AudioEvent {
    Samples(Vec<f32>),
    Status(String),
    Exit(Option<i32>),
    Error(String),
}

struct AudioProcess {
    child: Child,
}

impl AudioProcess {
    fn spawn(tx: Sender<AudioEvent>) -> io::Result<Self> {
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
        thread::spawn(move || read_audio_stdout(stdout, stdout_tx));

        let stderr_tx = tx.clone();
        thread::spawn(move || read_helper_stderr(stderr, stderr_tx));

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

fn read_audio_stdout(mut stdout: impl Read, tx: Sender<AudioEvent>) {
    let mut buffer = vec![0_u8; 4096 * 4];
    loop {
        match stdout.read(&mut buffer) {
            Ok(0) => break,
            Ok(size) => {
                let sample_count = size / 4;
                if sample_count == 0 {
                    continue;
                }

                let mut samples = Vec::with_capacity(sample_count);
                for chunk in buffer[..sample_count * 4].chunks_exact(4) {
                    samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }

                if tx.send(AudioEvent::Samples(samples)).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = tx.send(AudioEvent::Error(error.to_string()));
                break;
            }
        }
    }
    let _ = tx.send(AudioEvent::Exit(None));
}

fn read_helper_stderr(stderr: impl Read, tx: Sender<AudioEvent>) {
    let reader = BufReader::new(stderr);
    for line in reader.lines().map_while(Result::ok) {
        let _ = tx.send(AudioEvent::Status(line));
    }
}

struct SpectrumAnalyzer {
    fft_size: usize,
    sample_rate: f32,
    bar_count: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    sample_buffer: Vec<f32>,
    smoothed: Vec<f32>,
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
    fn new(fft_size: usize, sample_rate: f32, bar_count: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window: Vec<f32> = (0..fft_size)
            .map(|index| {
                let position = index as f32 / (fft_size - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * position).cos()
            })
            .collect();
        let window_sum = window.iter().sum::<f32>().max(1.0);

        Self {
            fft_size,
            sample_rate,
            bar_count,
            fft,
            window,
            window_sum,
            sample_buffer: Vec::new(),
            smoothed: vec![0.0; bar_count],
        }
    }

    fn consume(
        &mut self,
        samples: &[f32],
        smoothing: f32,
        pipeline: SpectrumPipeline,
    ) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }

        self.sample_buffer.extend_from_slice(samples);
        let max_samples = self.fft_size * 3;
        if self.sample_buffer.len() > max_samples {
            let excess = self.sample_buffer.len() - max_samples;
            self.sample_buffer.drain(0..excess);
        }

        if self.sample_buffer.len() < self.fft_size {
            return None;
        }

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
        let decay = smoothing.clamp(0.20, 0.92);
        let attack = 0.62;

        for (smoothed, target) in self.smoothed.iter_mut().zip(bars.into_iter()) {
            if target > *smoothed {
                *smoothed = *smoothed * (1.0 - attack) + target * attack;
            } else {
                *smoothed = *smoothed * decay + target * (1.0 - decay);
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
            let lower_bin = ((lower_frequency / self.sample_rate) * self.fft_size as f32)
                .round()
                .max(1.0) as usize;
            let upper_bin = ((upper_frequency / self.sample_rate) * self.fft_size as f32)
                .round()
                .max((lower_bin + 1) as f32) as usize;
            let lower_bin = lower_bin.min(magnitudes.len().saturating_sub(1));
            let upper_bin = upper_bin.min(magnitudes.len().saturating_sub(1));

            let mut total = 0.0_f32;
            let mut squared_total = 0.0_f32;
            let mut weight_total = 0.0_f32;
            let mut peak = 0.0_f32;
            for (bin, magnitude) in magnitudes
                .iter()
                .enumerate()
                .take(upper_bin + 1)
                .skip(lower_bin)
            {
                let weight = 1.0 + (bin - lower_bin) as f32 / (upper_bin - lower_bin).max(1) as f32;
                total += *magnitude * weight;
                squared_total += magnitude * magnitude * weight;
                weight_total += weight;
                peak = peak.max(*magnitude);
            }

            let average = total / weight_total.max(1.0);
            let rms = (squared_total / weight_total.max(1.0)).sqrt();
            let energy = (rms * 0.72 + peak * 0.28).max(average);
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
        },
    }
}

fn bar_color(theme: Theme, index: usize, len: usize, value: f32) -> Color {
    if value > 0.72 {
        return theme.high;
    }

    let ratio = index as f32 / len.max(1) as f32;
    if ratio < 0.45 {
        theme.low
    } else if ratio < 0.78 {
        theme.mid
    } else {
        theme.high
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
    frame.render_widget(
        Paragraph::new(art_lines)
            .alignment(Alignment::Center)
            .block(panel_block("terb", theme)),
        rows[0],
    );

    draw_main_menu(frame, app, rows[1]);

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
    .wrap(Wrap { trim: true })
    .block(panel_block(app.t("overview"), theme));
    frame.render_widget(status, rows[2]);

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

    frame.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(panel_block(app.t("main_menu"), theme)),
        panel,
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

    let list = List::new(items)
        .block(panel_block(app.t("main_menu"), theme))
        .highlight_symbol("  ")
        .highlight_style(
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_spectrum_screen(frame: &mut Frame, app: &App, area: Rect) {
    let show_sidebar = area.width >= 90 && area.height >= 22;
    if show_sidebar {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(35), Constraint::Min(24)])
            .split(area);
        draw_sidebar(frame, app, chunks[0]);
        draw_spectrum(frame, app, chunks[1], app.t("spectrum"));
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(3)])
            .split(area);
        draw_spectrum(frame, app, rows[0], "terb");
        draw_compact_footer(frame, app, rows[1]);
    }
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(13),
            Constraint::Length(6),
            Constraint::Min(7),
        ])
        .split(area);

    let data = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(app.t("status"), Style::default().fg(theme.muted)),
            Span::raw(" "),
            Span::styled(state_label(app), Style::default().fg(theme.accent)),
            Span::raw("  "),
            Span::styled(
                format!("{} {:>3}%", app.t("level"), (app.level * 100.0) as u16),
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
        ]),
        Line::from(""),
        Line::from(Span::styled(&app.status, Style::default().fg(theme.text))),
    ])
    .wrap(Wrap { trim: true })
    .block(panel_block("terb", theme));
    frame.render_widget(data, rows[0]);

    draw_settings_list(frame, app, rows[1], app.t("settings"));

    draw_pipeline(frame, app, rows[2]);

    let hint = Paragraph::new(app.t("sidebar_hint"))
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(theme.muted))
        .block(panel_block(app.t("controls"), theme));
    frame.render_widget(hint, rows[3]);
}

fn draw_spectrum(frame: &mut Frame, app: &App, area: Rect, title: &'static str) {
    let theme = app.theme();
    let block = panel_block(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 10 || inner.height < 4 {
        return;
    }

    let bars = display_bars(&app.spectrum, inner.width as usize);
    let chart_height = inner.height as usize;
    let mut lines = Vec::with_capacity(chart_height);

    for row in 0..chart_height {
        let threshold = 1.0 - (row as f32 + 0.5) / chart_height as f32;
        let mut spans = Vec::with_capacity(bars.len());
        for (index, value) in bars.iter().enumerate() {
            let value = render_bar_value(*value, &app.config.settings);
            let filled = value >= threshold;
            let symbol = if filled { "█" } else { " " };
            let color = if filled {
                bar_color(theme, index, bars.len(), value)
            } else {
                Color::Reset
            };
            spans.push(Span::styled(symbol, Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), inner);
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
            format!("{} {:>3}%", app.t("level"), (app.level * 100.0) as u16),
            Style::default().fg(theme.text),
        ),
        Span::raw("  "),
        Span::styled(app.t("compact_hint"), Style::default().fg(theme.muted)),
    ]))
    .wrap(Wrap { trim: true })
    .block(panel_block("", theme));
    frame.render_widget(text, area);
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

fn render_bar_value(value: f32, settings: &Settings) -> f32 {
    let ceiling = settings.ceiling.max(0.01);
    let value = (value.clamp(0.0, ceiling) / ceiling).clamp(0.0, 1.0);
    if settings.visual_curve_enabled {
        value.powf(settings.visual_curve)
    } else {
        value
    }
}

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = if area.width < 70 || area.height < 20 {
        area
    } else {
        centered_rect(54, 52, area)
    };
    frame.render_widget(Clear, chunks);
    draw_settings_list(frame, app, chunks, app.t("settings"));
}

fn draw_settings_list(frame: &mut Frame, app: &App, area: Rect, title: &'static str) {
    let theme = app.theme();
    let rows = vec![
        setting_line(app, "language", language_label(app.lang)),
        setting_line(app, "theme", theme_label(app)),
        setting_line(
            app,
            "smoothing",
            format!("{:>3}%", (app.config.settings.smoothing * 100.0) as u16),
        ),
        setting_line(app, "bars", app.config.settings.bar_count.to_string()),
        setting_line(app, "fft_size", app.config.settings.fft_size.to_string()),
        setting_line(
            app,
            "refresh_rate",
            format!("{}Hz", app.config.settings.refresh_hz),
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
            "ceiling",
            format!("{:>3}%", (app.config.settings.ceiling * 100.0) as u16),
        ),
    ];

    let items: Vec<ListItem> = rows.into_iter().map(ListItem::new).collect();
    let mut state = ListState::default();
    state.select(Some(app.setting_index));

    let list = List::new(items)
        .block(panel_block(title, theme))
        .highlight_symbol("  ")
        .highlight_style(
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_pipeline(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme();
    let settings = &app.config.settings;
    let lines = vec![
        Line::from(vec![
            Span::styled("IN", Style::default().fg(theme.text)),
            Span::styled(" > ", Style::default().fg(theme.muted)),
            Span::styled(
                format!("GATE {:.0}e-5", SILENCE_GATE * 100_000.0),
                Style::default().fg(theme.text),
            ),
            Span::styled(" > ", Style::default().fg(theme.muted)),
            Span::styled("WIN", Style::default().fg(theme.text)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("FFT{}", settings.fft_size),
                Style::default().fg(theme.text),
            ),
            Span::styled(" > ", Style::default().fg(theme.muted)),
            Span::styled(
                format!("BAND{}", settings.bar_count),
                Style::default().fg(theme.text),
            ),
            Span::styled(" > ", Style::default().fg(theme.muted)),
            Span::styled("DET", Style::default().fg(theme.text)),
        ]),
        Line::from(vec![
            Span::styled(
                if settings.high_shelf_enabled {
                    format!("EQ +{:.0}dB", settings.high_shelf_db)
                } else {
                    "EQ bypass".to_string()
                },
                Style::default().fg(if settings.high_shelf_enabled {
                    theme.accent
                } else {
                    theme.muted
                }),
            ),
            Span::styled(" > ", Style::default().fg(theme.muted)),
            Span::styled(
                format!("LIM {}%", (settings.ceiling * 100.0) as u16),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                if settings.visual_curve_enabled {
                    format!("MTR pow {:.2}", settings.visual_curve)
                } else {
                    "MTR linear".to_string()
                },
                Style::default().fg(if settings.visual_curve_enabled {
                    theme.accent
                } else {
                    theme.muted
                }),
            ),
            Span::styled(" > OUT", Style::default().fg(theme.muted)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel_block(app.t("pipeline"), theme)),
        area,
    );
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

    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: true })
            .block(panel_block(app.t("help"), theme)),
        chunks,
    );
}

fn panel_block(title: &'static str, theme: Theme) -> Block<'static> {
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border))
        .style(Style::default().fg(theme.text))
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

fn state_label(app: &App) -> &'static str {
    match app.capture_state {
        CaptureState::Idle => app.t("state_idle"),
        CaptureState::Starting => app.t("state_starting"),
        CaptureState::Running => app.t("state_running"),
        CaptureState::PermissionNeeded => app.t("state_permission"),
        CaptureState::Failed => app.t("state_failed"),
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
        (Lang::Zh, "bars") => "频段",
        (Lang::Zh, "fft_size") => "FFT",
        (Lang::Zh, "refresh_rate") => "刷新率",
        (Lang::Zh, "high_shelf") => "高频补偿",
        (Lang::Zh, "high_shelf_db") => "补偿强度",
        (Lang::Zh, "visual_curve") => "高度曲线",
        (Lang::Zh, "curve_power") => "曲线指数",
        (Lang::Zh, "ceiling") => "上限",
        (Lang::Zh, "pipeline") => "音频链路",
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
        (Lang::Zh, "help_3") => "频谱页左侧栏显示状态和设置；↑/↓ 选择设置，←/→ 调整。",
        (Lang::Zh, "help_4") => "窗口较小时侧栏会隐藏；按 s 打开全屏设置，按 m 或 q 返回主菜单。",
        (Lang::Zh, "help_5") => "q/Esc 返回；在主菜单再次按 q/Esc 退出。",
        (Lang::Zh, "permission_note") => "首次捕获会触发 macOS 屏幕与系统音频录制授权；Terb 只实时分析，不保存音频。",
        (Lang::Zh, "menu_hint") => "↑/↓ 选择 · Enter 确认 · Space 捕获 · ? 帮助 · q 退出",
        (Lang::Zh, "spectrum_hint") => "Space 捕获 · ↑/↓ 选择设置 · ←/→ 调整 · s 设置 · m 菜单",
        (Lang::Zh, "sidebar_hint") => "Space 开关捕获\n↑/↓ 选择设置\n←/→ 调整\ns 全屏设置\nm/q 主菜单\n? 帮助",
        (Lang::Zh, "compact_hint") => "Space 捕获 · s 设置 · m/q 菜单",
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
        (Lang::Zh, "theme_amber") => "琥珀",
        (Lang::Zh, "theme_mono") => "单色",

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
        (Lang::En, "bars") => "Bands",
        (Lang::En, "fft_size") => "FFT",
        (Lang::En, "refresh_rate") => "Refresh",
        (Lang::En, "high_shelf") => "High-shelf",
        (Lang::En, "high_shelf_db") => "Shelf Gain",
        (Lang::En, "visual_curve") => "Height Curve",
        (Lang::En, "curve_power") => "Curve Power",
        (Lang::En, "ceiling") => "Ceiling",
        (Lang::En, "pipeline") => "Pipeline",
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
        (Lang::En, "help_3") => "In Spectrum, the sidebar shows status and settings. Use ↑/↓ to select and ←/→ to adjust.",
        (Lang::En, "help_4") => "When the window is small, the sidebar hides. Press s for full-screen settings, m or q for menu.",
        (Lang::En, "help_5") => "q/Esc goes back; on the main menu it quits.",
        (Lang::En, "permission_note") => "First capture may trigger macOS Screen & System Audio Recording permission. Terb analyzes live audio only and does not save it.",
        (Lang::En, "menu_hint") => "↑/↓ select · Enter confirm · Space capture · ? help · q quit",
        (Lang::En, "spectrum_hint") => "Space capture · ↑/↓ select setting · ←/→ adjust · s settings · m menu",
        (Lang::En, "sidebar_hint") => "Space toggle capture\n↑/↓ select setting\n←/→ adjust\ns full settings\nm/q main menu\n? help",
        (Lang::En, "compact_hint") => "Space capture · s settings · m/q menu",
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
        (Lang::En, "theme_amber") => "Amber",
        (Lang::En, "theme_mono") => "Mono",

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
        (Lang::Ja, "bars") => "バンド",
        (Lang::Ja, "fft_size") => "FFT",
        (Lang::Ja, "refresh_rate") => "更新率",
        (Lang::Ja, "high_shelf") => "高域補正",
        (Lang::Ja, "high_shelf_db") => "補正量",
        (Lang::Ja, "visual_curve") => "高さ曲線",
        (Lang::Ja, "curve_power") => "曲線指数",
        (Lang::Ja, "ceiling") => "上限",
        (Lang::Ja, "pipeline") => "音声チェーン",
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
        (Lang::Ja, "help_3") => "スペクトラム画面のサイドバーで状態と設定を表示します。↑/↓ で選択、←/→ で変更。",
        (Lang::Ja, "help_4") => "小さいウィンドウではサイドバーを隠します。s で全画面設定、m/q でメニュー。",
        (Lang::Ja, "help_5") => "q/Esc で戻る。メインメニューでは終了します。",
        (Lang::Ja, "permission_note") => "初回キャプチャでは macOS の画面とシステム音声録音権限が必要です。Terb はリアルタイム解析のみ行い、音声を保存しません。",
        (Lang::Ja, "menu_hint") => "↑/↓ 選択 · Enter 決定 · Space キャプチャ · ? ヘルプ · q 終了",
        (Lang::Ja, "spectrum_hint") => "Space キャプチャ · ↑/↓ 設定選択 · ←/→ 変更 · s 設定 · m メニュー",
        (Lang::Ja, "sidebar_hint") => "Space キャプチャ切替\n↑/↓ 設定選択\n←/→ 変更\ns 全画面設定\nm/q メニュー\n? ヘルプ",
        (Lang::Ja, "compact_hint") => "Space キャプチャ · s 設定 · m/q メニュー",
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
        (Lang::Ja, "theme_amber") => "アンバー",
        (Lang::Ja, "theme_mono") => "モノ",

        _ => key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn render_bar_value_maps_ceiling_to_full_height() {
        let settings = Settings {
            ceiling: 0.88,
            ..Config::default().settings
        };

        assert!((render_bar_value(0.88, &settings) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn analyzer_gates_silence_to_zero() {
        let settings = Config::default().settings;
        let pipeline = SpectrumPipeline::from_settings(&settings);
        let mut analyzer = SpectrumAnalyzer::new(1024, 48_000.0, 32);
        let bars = analyzer
            .consume(&vec![0.0; 2048], settings.smoothing, pipeline)
            .expect("enough samples");

        assert!(bars.iter().all(|bar| *bar == 0.0));
    }
}
