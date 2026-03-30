//! Terminal-based world sim trace visualizer.
//!
//! Loads a WorldSimTrace JSON file and replays it in the terminal using crossterm.
//! Settlements render as colored nodes on a 2D map, events scroll below,
//! and a status bar shows tick/year/season/speed.

use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    style::{Color, Print, SetForegroundColor, SetBackgroundColor, ResetColor},
    terminal,
};

use bevy_game::world_sim::trace::WorldSimTrace;
use bevy_game::world_sim::visualizer::{
    PlaybackCommand, PlaybackController, TraceFrame, VisualizerBackend,
};

use super::cli::VisualizeArgs;

// ---------------------------------------------------------------------------
// Terminal backend
// ---------------------------------------------------------------------------

struct TerminalBackend {
    running: bool,
    cols: u16,
    rows: u16,
    speed: f32,
    paused: bool,
    /// Zoom level: 1.0 = fit everything, 2.0 = 2x zoom, etc.
    zoom: f32,
    /// Camera center in world coordinates (pan target).
    camera: (f32, f32),
    /// If true, camera is auto-fit to show everything.
    auto_fit: bool,
}

impl TerminalBackend {
    fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;
        let (cols, rows) = terminal::size()?;
        Ok(Self {
            running: true, cols, rows, speed: 100.0, paused: true,
            zoom: 1.0, camera: (0.0, 0.0), auto_fit: true,
        })
    }
}

impl VisualizerBackend for TerminalBackend {
    fn init(&mut self, total_ticks: u64, seed: u64) {
        let _ = (total_ticks, seed);
    }

    fn render_frame(&mut self, frame: &TraceFrame) {
        // Update terminal size.
        if let Ok((c, r)) = terminal::size() {
            self.cols = c;
            self.rows = r;
        }

        // Use a BufWriter to batch all writes — the terminal receives one
        // large write instead of thousands of small ones, eliminating flicker.
        let mut stdout = io::BufWriter::with_capacity(
            self.cols as usize * self.rows as usize * 8,
            io::stdout().lock(),
        );
        let _ = execute!(stdout, cursor::MoveTo(0, 0));

        let map_height = (self.rows as usize).saturating_sub(14).max(5);
        let event_height = 8usize;

        // --- MAP AREA ---
        render_map(&mut stdout, frame, self.cols as usize, map_height,
                   self.zoom, self.camera, self.auto_fit);

        // --- EVENTS AREA ---
        let event_y = map_height as u16 + 1;
        let _ = execute!(stdout, cursor::MoveTo(0, event_y));
        let separator: String = "─".repeat(self.cols as usize);
        let _ = execute!(stdout, SetForegroundColor(Color::DarkGrey), Print(&separator), ResetColor);

        let events_to_show = frame.events.len().min(event_height);
        let start = frame.events.len().saturating_sub(events_to_show);
        for (i, evt) in frame.events[start..].iter().enumerate() {
            let y = event_y + 1 + i as u16;
            if y >= self.rows - 2 { break; }
            let _ = execute!(stdout, cursor::MoveTo(0, y));

            let cat_color = category_color(&evt.category);
            let cat_label = format!("[{}]", evt.category);
            let text_width = (self.cols as usize).saturating_sub(cat_label.len() + 2);
            let text: String = evt.text.chars().take(text_width).collect();
            // Pad to full width to overwrite stale text.
            let padding = self.cols as usize - cat_label.len() - text.len() - 1;

            let _ = execute!(
                stdout,
                SetForegroundColor(cat_color),
                Print(&cat_label),
                SetForegroundColor(Color::White),
                Print(format!(" {}{}", text, " ".repeat(padding))),
                ResetColor,
            );
        }

        // --- STATUS BAR ---
        let status_y = self.rows - 2;
        let _ = execute!(stdout, cursor::MoveTo(0, status_y));
        let _ = execute!(stdout, SetForegroundColor(Color::DarkGrey), Print(&separator), ResetColor);

        let _ = execute!(stdout, cursor::MoveTo(0, status_y + 1));
        let pause_indicator = if self.paused { "▐▐" } else { "▶" };
        let zoom_label = if self.auto_fit { "fit".to_string() } else { format!("{:.1}x", self.zoom) };
        let status = format!(
            " {} tick {}/{} | Y{} {} | {}A {}M pop:{} | spd:x{:.0} zoom:{} | [Space] [←→]seek [+/-]spd [Z/X]zoom [WASD]pan [F]fit [Q]uit",
            pause_indicator,
            frame.tick,
            frame.total_ticks,
            frame.summary.year,
            frame.summary.season,
            frame.summary.alive_npcs,
            frame.summary.alive_monsters,
            frame.summary.total_population,
            self.speed,
            zoom_label,
        );
        // Pad to full width.
        let mut status_padded: String = status.chars().take(self.cols as usize).collect();
        while status_padded.len() < self.cols as usize { status_padded.push(' '); }
        let _ = execute!(
            stdout,
            SetForegroundColor(Color::Cyan),
            Print(status_padded),
            ResetColor,
        );

        let _ = stdout.flush();
    }

    fn handle_input(&mut self) -> PlaybackCommand {
        if event::poll(Duration::from_millis(10)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => return PlaybackCommand::Quit,
                    KeyCode::Char(' ') => return PlaybackCommand::TogglePause,
                    KeyCode::Right => {
                        let amount = if key.modifiers.contains(KeyModifiers::SHIFT) { 1000 } else { 100 };
                        return PlaybackCommand::SeekForward(amount);
                    }
                    KeyCode::Left => {
                        let amount = if key.modifiers.contains(KeyModifiers::SHIFT) { 1000 } else { 100 };
                        return PlaybackCommand::SeekBackward(amount);
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            // Ctrl+= zoom in
                            self.zoom = (self.zoom * 1.5).min(10.0);
                            self.auto_fit = false;
                        } else {
                            return PlaybackCommand::SpeedUp;
                        }
                    }
                    KeyCode::Char('-') | KeyCode::Char('_') => {
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            // Ctrl+- zoom out
                            self.zoom = (self.zoom / 1.5).max(0.5);
                            if self.zoom <= 1.01 { self.auto_fit = true; }
                        } else {
                            return PlaybackCommand::SlowDown;
                        }
                    }
                    // WASD for camera pan
                    KeyCode::Char('w') | KeyCode::Char('W') => {
                        self.camera.1 -= 50.0 / self.zoom;
                        self.auto_fit = false;
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        self.camera.1 += 50.0 / self.zoom;
                        self.auto_fit = false;
                    }
                    KeyCode::Char('a') | KeyCode::Char('A') => {
                        self.camera.0 -= 50.0 / self.zoom;
                        self.auto_fit = false;
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        self.camera.0 += 50.0 / self.zoom;
                        self.auto_fit = false;
                    }
                    // Z to zoom in, X to zoom out (simpler than Ctrl+)
                    KeyCode::Char('z') | KeyCode::Char('Z') => {
                        self.zoom = (self.zoom * 1.5).min(10.0);
                        self.auto_fit = false;
                    }
                    KeyCode::Char('x') | KeyCode::Char('X') => {
                        self.zoom = (self.zoom / 1.5).max(0.5);
                        if self.zoom <= 1.01 { self.auto_fit = true; }
                    }
                    // F to fit/reset view
                    KeyCode::Char('f') | KeyCode::Char('F') => {
                        self.zoom = 1.0;
                        self.auto_fit = true;
                    }
                    _ => {}
                }
            }
        }
        PlaybackCommand::None
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn cleanup(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

impl Drop for TerminalBackend {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ---------------------------------------------------------------------------
// Map renderer — continuous terrain with settlement markers
// ---------------------------------------------------------------------------

fn terrain_style(terrain: &str) -> (char, Color, Color) {
    // (glyph, foreground, background)
    match terrain {
        "Plains"    => ('.', Color::Rgb { r: 80, g: 120, b: 50 }, Color::Rgb { r: 25, g: 40, b: 15 }),
        "Forest"    => ('♣', Color::Rgb { r: 40, g: 100, b: 30 }, Color::Rgb { r: 12, g: 35, b: 10 }),
        "Mountains" => ('^', Color::Rgb { r: 130, g: 110, b: 90 }, Color::Rgb { r: 35, g: 30, b: 25 }),
        "Coast"     => ('~', Color::Rgb { r: 60, g: 120, b: 180 }, Color::Rgb { r: 10, g: 25, b: 45 }),
        "Swamp"     => ('~', Color::Rgb { r: 60, g: 90, b: 50 }, Color::Rgb { r: 20, g: 30, b: 15 }),
        "Desert"    => (':', Color::Rgb { r: 190, g: 160, b: 90 }, Color::Rgb { r: 45, g: 35, b: 18 }),
        "Tundra"    => ('*', Color::Rgb { r: 160, g: 180, b: 200 }, Color::Rgb { r: 30, g: 35, b: 45 }),
        _           => (' ', Color::Grey, Color::Rgb { r: 15, g: 15, b: 15 }),
    }
}

fn render_map(stdout: &mut impl Write, frame: &TraceFrame, cols: usize, rows: usize,
              zoom: f32, camera: (f32, f32), auto_fit: bool) {
    if frame.settlements.is_empty() && frame.regions.is_empty() { return; }

    // Map area: left 70% for map, right 30% for faction panel.
    let map_w = (cols * 7 / 10).max(30);
    let map_h = rows;

    // --- Compute world bounds ---
    let positions: Vec<(f32, f32)> = frame.settlements.iter().map(|s| s.pos)
        .chain(frame.regions.iter().map(|r| r.pos)).collect();
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(px, py) in &positions {
        min_x = min_x.min(px); max_x = max_x.max(px);
        min_y = min_y.min(py); max_y = max_y.max(py);
    }
    let pad = ((max_x - min_x).max(max_y - min_y)) * 0.2;
    min_x -= pad; min_y -= pad; max_x += pad; max_y += pad;

    let (cx, cy) = if auto_fit {
        ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    } else { camera };
    let range_x = (max_x - min_x).max(1.0) / zoom;
    let range_y = (max_y - min_y).max(1.0) / zoom;

    // World-to-screen projection.
    let project = |wx: f32, wy: f32| -> (i32, i32) {
        let sx = ((wx - cx) / range_x * map_w as f32 + map_w as f32 / 2.0) as i32;
        let sy = ((wy - cy) / range_y * map_h as f32 + map_h as f32 / 2.0) as i32;
        (sx, sy)
    };
    let visible = |sx: i32, sy: i32| -> bool {
        sx >= 0 && (sx as usize) < map_w && sy >= 0 && (sy as usize) < map_h
    };

    // --- 1. Paint terrain: for each screen pixel, find nearest region ---
    // Build region lookup: screen coords → nearest region.
    // For efficiency, only paint every other column (terminals are ~2:1 aspect).
    if !frame.regions.is_empty() {
        for sy in 0..map_h {
            let _ = execute!(stdout, cursor::MoveTo(0, sy as u16));
            for sx in 0..map_w {
                // Inverse project to world coords.
                let wx = (sx as f32 - map_w as f32 / 2.0) / map_w as f32 * range_x + cx;
                let wy = (sy as f32 - map_h as f32 / 2.0) / map_h as f32 * range_y + cy;

                // Find nearest region.
                let nearest = frame.regions.iter().min_by(|a, b| {
                    let da = (a.pos.0 - wx).powi(2) + (a.pos.1 - wy).powi(2);
                    let db = (b.pos.0 - wx).powi(2) + (b.pos.1 - wy).powi(2);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some(region) = nearest {
                    let (glyph, fg, bg) = terrain_style(&region.terrain);
                    // Mix faction tint into background.
                    let bg_final = if let Some(fc) = region.faction_color {
                        let (br, bg_g, bb) = match bg { Color::Rgb{r,g,b} => (r,g,b), _ => (20,20,20) };
                        Color::Rgb {
                            r: br.saturating_add(fc[0] / 8),
                            g: bg_g.saturating_add(fc[1] / 8),
                            b: bb.saturating_add(fc[2] / 8),
                        }
                    } else { bg };

                    // Sparse glyphs for texture (every 3rd char).
                    let show = (sx + sy * 3) % 4 == 0;
                    if show {
                        let _ = execute!(stdout, SetBackgroundColor(bg_final), SetForegroundColor(fg), Print(glyph));
                    } else {
                        let _ = execute!(stdout, SetBackgroundColor(bg_final), Print(' '));
                    }
                } else {
                    let _ = execute!(stdout, Print(' '));
                }
            }
            let _ = execute!(stdout, ResetColor);
        }
    }

    // --- 2. Trade routes ---
    for route in &frame.trade_routes {
        let (x1, y1) = project(route.from_pos.0, route.from_pos.1);
        let (x2, y2) = project(route.to_pos.0, route.to_pos.1);
        let dx = x2 - x1; let dy = y2 - y1;
        let steps = dx.abs().max(dy.abs()).max(1);
        for i in (0..=steps).step_by(2) {
            let x = x1 + dx * i / steps;
            let y = y1 + dy * i / steps;
            if visible(x, y) {
                let _ = execute!(stdout, cursor::MoveTo(x as u16, y as u16),
                    SetForegroundColor(Color::Rgb { r: 100, g: 90, b: 60 }), Print('·'), ResetColor);
            }
        }
    }

    // --- 3. Count entities per settlement (they cluster at settlement positions) ---
    let mut npc_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut mon_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for entity in &frame.entities {
        if !entity.alive { continue; }
        if let Some(nearest) = frame.settlements.iter().min_by(|a, b| {
            let da = (a.pos.0 - entity.pos.0).powi(2) + (a.pos.1 - entity.pos.1).powi(2);
            let db = (b.pos.0 - entity.pos.0).powi(2) + (b.pos.1 - entity.pos.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            match entity.kind {
                0 => { *npc_counts.entry(nearest.id).or_default() += 1; }
                1 => { *mon_counts.entry(nearest.id).or_default() += 1; }
                _ => {}
            }
        }
    }

    // --- 4. Settlement markers with entity counts ---
    for s in &frame.settlements {
        let (sx, sy) = project(s.pos.0, s.pos.1);
        if !visible(sx, sy) { continue; }

        let fc = Color::Rgb { r: s.faction_color[0], g: s.faction_color[1], b: s.faction_color[2] };
        let npcs = npc_counts.get(&s.id).copied().unwrap_or(0);
        let mons = mon_counts.get(&s.id).copied().unwrap_or(0);

        // Row 0: Settlement icon.
        let _ = execute!(stdout, cursor::MoveTo(sx as u16, sy as u16),
            SetForegroundColor(Color::White), SetBackgroundColor(fc), Print(" ⌂ "), ResetColor);

        // Row 1: Name + pop.
        let label = format!("{} (p:{})", s.name, s.population);
        let lx = (sx - label.len() as i32 / 2).max(0);
        if visible(lx, sy + 1) {
            let trunc: String = label.chars().take((map_w as i32 - lx).max(0) as usize).collect();
            let _ = execute!(stdout, cursor::MoveTo(lx as u16, (sy + 1) as u16),
                SetForegroundColor(fc), Print(&trunc), ResetColor);
        }

        // Row 2: Entity counts — green NPCs, red monsters.
        if visible(sx - 2, sy + 2) {
            let _ = execute!(stdout, cursor::MoveTo((sx - 2).max(0) as u16, (sy + 2) as u16),
                SetForegroundColor(Color::Green), Print(format!("●{}", npcs)),
                Print(" "),
                SetForegroundColor(Color::Red), Print(format!("◆{}", mons)),
                ResetColor);
        }
    }

    // --- 5. Faction panel (right side) ---
    let panel_x = map_w + 1;
    let panel_w = cols.saturating_sub(panel_x + 1);
    if panel_w >= 15 {
        for (i, f) in frame.factions.iter().enumerate() {
            if i >= 6 { break; }
            let y = 1 + i * 3;
            if y + 2 >= rows { break; }
            let fc = Color::Rgb { r: f.color[0], g: f.color[1], b: f.color[2] };
            let _ = execute!(stdout, cursor::MoveTo(panel_x as u16, y as u16),
                SetForegroundColor(fc),
                Print(&f.name[..f.name.len().min(panel_w)]),
                ResetColor);
            let info = format!("mil:{:.0} ${:.0} t:{}", f.military, f.treasury, f.territory_count);
            let _ = execute!(stdout, cursor::MoveTo(panel_x as u16, (y+1) as u16),
                SetForegroundColor(Color::DarkGrey),
                Print(&info[..info.len().min(panel_w)]),
                ResetColor);
        }
    }
}

fn category_color(cat: &str) -> Color {
    match cat {
        "Battle" => Color::Red,
        "Death" => Color::DarkGrey,
        "Quest" => Color::Green,
        "Diplomacy" => Color::Magenta,
        "Economy" => Color::Yellow,
        "Achievement" => Color::Cyan,
        "Discovery" => Color::Blue,
        "Crisis" => Color::DarkRed,
        "Narrative" => Color::White,
        _ => Color::Grey,
    }
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

pub fn run_visualize(args: VisualizeArgs) -> ExitCode {
    // Load trace.
    eprintln!("Loading trace from {}...", args.trace);
    let trace = match WorldSimTrace::load_from_file(&args.trace) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load trace: {}", e);
            return ExitCode::FAILURE;
        }
    };
    eprintln!("Loaded: {} ticks, {} snapshots, {} chronicle entries",
        trace.total_ticks, trace.snapshots.len(), trace.chronicle_log.len());

    // Create backend and controller.
    let mut backend = match TerminalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to initialize terminal: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let mut controller = PlaybackController::new(trace, 500);
    // Override initial speed from args.
    for _ in 0..(args.speed.log2() as u32) {
        controller.handle_command(PlaybackCommand::SpeedUp);
    }

    backend.init(controller.total_ticks(), controller.seed());

    let frame_interval = Duration::from_millis(1000 / args.fps as u64);

    // Main loop.
    while backend.is_running() {
        let loop_start = Instant::now();

        let cmd = backend.handle_input();
        if matches!(cmd, PlaybackCommand::Quit) {
            break;
        }
        controller.handle_command(cmd);

        if !controller.is_paused() {
            controller.advance(frame_interval.as_secs_f32());
        }

        // Sync backend display state from controller.
        backend.speed = controller.speed();
        backend.paused = controller.is_paused();

        let frame = controller.current_frame();
        backend.render_frame(&frame);

        let elapsed = loop_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    backend.cleanup();
    ExitCode::SUCCESS
}
