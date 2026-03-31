//! Backend-agnostic visualizer types and playback controller.
//!
//! Defines the `TraceFrame` intermediate representation that renderers consume,
//! the `VisualizerBackend` trait for pluggable rendering, and the
//! `PlaybackController` that manages tick progression and frame generation.

use serde::{Deserialize, Serialize};

use super::state::{ChronicleEntry, ChronicleCategory, EntityKind, WorldState};
use super::systems::seasons::{current_season, Season, TICKS_PER_SEASON};
use super::trace::WorldSimTrace;

// ---------------------------------------------------------------------------
// View types — what renderers see
// ---------------------------------------------------------------------------

/// Settlement as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SettlementView {
    pub id: u32,
    pub name: String,
    pub pos: (f32, f32),
    pub faction_color: [u8; 3],
    pub population: u32,
    pub treasury: f32,
    pub specialty: String,
    pub threat_level: f32,
}

/// Entity as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityView {
    pub id: u32,
    pub pos: (f32, f32),
    pub kind: u8,
    pub team: u8,
    pub alive: bool,
    pub level: u32,
    pub name: Option<String>,
}

/// Faction as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionView {
    pub id: u32,
    pub name: String,
    pub color: [u8; 3],
    pub territory_count: u32,
    pub military: f32,
    pub stance: String,
    pub treasury: f32,
}

/// Chronicle/event entry as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventView {
    pub tick: u64,
    pub category: String,
    pub text: String,
}

/// Region as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionView {
    pub id: u32,
    pub name: String,
    pub terrain: String,
    pub pos: (f32, f32),
    pub faction_color: Option<[u8; 3]>,
    pub threat_level: f32,
    pub monster_density: f32,
    pub unrest: f32,
    pub control: f32,
}

/// Trade route between two settlements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeRouteView {
    pub from_pos: (f32, f32),
    pub to_pos: (f32, f32),
}

/// Aggregate stats for a single frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameSummary {
    pub alive_npcs: usize,
    pub alive_monsters: usize,
    pub total_population: u32,
    pub avg_threat: f32,
    pub season: String,
    pub year: u32,
}

/// Complete frame data for a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceFrame {
    pub tick: u64,
    pub total_ticks: u64,
    pub settlements: Vec<SettlementView>,
    pub entities: Vec<EntityView>,
    pub factions: Vec<FactionView>,
    pub regions: Vec<RegionView>,
    pub trade_routes: Vec<TradeRouteView>,
    pub events: Vec<EventView>,
    pub summary: FrameSummary,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a faction ID to a display color. Six preset colors, gray fallback.
fn faction_color(faction_id: u32) -> [u8; 3] {
    match faction_id {
        0 => [66, 133, 244],   // blue
        1 => [219, 68, 55],    // red
        2 => [244, 180, 0],    // yellow
        3 => [15, 157, 88],    // green
        4 => [171, 71, 188],   // purple
        5 => [255, 112, 67],   // orange
        _ => [158, 158, 158],  // gray
    }
}

/// Format a `ChronicleCategory` as a human-readable string.
fn category_name(cat: ChronicleCategory) -> &'static str {
    match cat {
        ChronicleCategory::Battle => "Battle",
        ChronicleCategory::Quest => "Quest",
        ChronicleCategory::Diplomacy => "Diplomacy",
        ChronicleCategory::Economy => "Economy",
        ChronicleCategory::Death => "Death",
        ChronicleCategory::Discovery => "Discovery",
        ChronicleCategory::Crisis => "Crisis",
        ChronicleCategory::Achievement => "Achievement",
        ChronicleCategory::Narrative => "Narrative",
    }
}

/// Format a `Season` as a human-readable string.
fn season_name(season: Season) -> &'static str {
    match season {
        Season::Spring => "Spring",
        Season::Summer => "Summer",
        Season::Autumn => "Autumn",
        Season::Winter => "Winter",
    }
}

// ---------------------------------------------------------------------------
// generate_frame — pure projection from WorldState
// ---------------------------------------------------------------------------

/// Build a `TraceFrame` from the current world state and chronicle log.
///
/// `event_window` controls how many ticks back to include chronicle entries
/// (e.g. 200 means show events from `[tick - 200, tick]`).
pub fn generate_frame(
    state: &WorldState,
    chronicle: &[ChronicleEntry],
    event_window: u64,
    total_ticks: u64,
) -> TraceFrame {
    let tick = state.tick;

    // --- Settlements ---
    let settlements: Vec<SettlementView> = state.settlements.iter().map(|s| {
        let color = s.faction_id
            .map(|fid| faction_color(fid))
            .unwrap_or([158, 158, 158]);
        SettlementView {
            id: s.id,
            name: s.name.clone(),
            pos: s.pos,
            faction_color: color,
            population: s.population,
            treasury: s.treasury,
            specialty: s.specialty.name().to_string(),
            threat_level: s.threat_level,
        }
    }).collect();

    // --- Entities ---
    let entities: Vec<EntityView> = state.entities.iter().map(|e| {
        let name = if e.alive {
            Some(super::naming::entity_display_name(e))
        } else {
            None
        };
        EntityView {
            id: e.id,
            pos: e.pos,
            kind: e.kind as u8,
            team: e.team as u8,
            alive: e.alive,
            level: e.level,
            name,
        }
    }).collect();

    // --- Factions ---
    let factions: Vec<FactionView> = state.factions.iter().map(|f| {
        let stance = format!("{:?}", f.diplomatic_stance);
        FactionView {
            id: f.id,
            name: f.name.clone(),
            color: faction_color(f.id),
            territory_count: f.territory_size,
            military: f.military_strength,
            stance,
            treasury: f.treasury,
        }
    }).collect();

    // --- Events (filtered by window) ---
    let window_start = tick.saturating_sub(event_window);
    let events: Vec<EventView> = chronicle.iter()
        .filter(|e| e.tick >= window_start && e.tick <= tick)
        .map(|e| EventView {
            tick: e.tick,
            category: category_name(e.category).to_string(),
            text: e.text.clone(),
        })
        .collect();

    // --- Summary ---
    let alive_npcs = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();
    let alive_monsters = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Monster)
        .count();
    let total_population: u32 = state.settlements.iter().map(|s| s.population).sum();

    let avg_threat = if state.regions.is_empty() {
        0.0
    } else {
        let sum: f32 = state.regions.iter().map(|r| r.threat_level).sum();
        sum / state.regions.len() as f32
    };

    let season = current_season(tick);
    let year = (tick / (TICKS_PER_SEASON * 4)) as u32 + 1;

    let summary = FrameSummary {
        alive_npcs,
        alive_monsters,
        total_population,
        avg_threat,
        season: season_name(season).to_string(),
        year,
    };

    // --- Regions ---
    // Regions don't store positions — derive from settlement positions.
    // Each settlement maps to its region (settlement_idx → region_idx in worldgen).
    let regions: Vec<RegionView> = state.regions.iter().map(|r| {
        RegionView {
            id: r.id,
            name: r.name.clone(),
            terrain: r.terrain.name().to_string(),
            pos: r.pos,
            faction_color: r.faction_id.map(|fid| faction_color(fid)),
            threat_level: r.threat_level,
            monster_density: r.monster_density,
            unrest: r.unrest,
            control: r.control,
        }
    }).collect();

    // --- Trade routes ---
    let trade_routes: Vec<TradeRouteView> = state.trade_routes.iter().filter_map(|route| {
        let sa = state.settlement(route.settlement_a)?;
        let sb = state.settlement(route.settlement_b)?;
        Some(TradeRouteView { from_pos: sa.pos, to_pos: sb.pos })
    }).collect();

    TraceFrame {
        tick,
        total_ticks,
        settlements,
        entities,
        factions,
        regions,
        trade_routes,
        events,
        summary,
    }
}

// ---------------------------------------------------------------------------
// PlaybackCommand + VisualizerBackend trait
// ---------------------------------------------------------------------------

/// Commands the playback controller can process from user input.
#[derive(Clone, Debug)]
pub enum PlaybackCommand {
    None,
    TogglePause,
    SeekForward(u64),
    SeekBackward(u64),
    SpeedUp,
    SlowDown,
    Quit,
}

/// Pluggable rendering backend. Implementations only consume `TraceFrame`
/// and never touch `WorldState` directly.
pub trait VisualizerBackend {
    /// Called once before the first frame with run metadata.
    fn init(&mut self, total_ticks: u64, seed: u64);
    /// Render a single frame.
    fn render_frame(&mut self, frame: &TraceFrame);
    /// Poll for user input and return a command.
    fn handle_input(&mut self) -> PlaybackCommand;
    /// Whether the backend is still running (false = exit).
    fn is_running(&self) -> bool;
    /// Called once when the playback loop ends.
    fn cleanup(&mut self);
}

// ---------------------------------------------------------------------------
// PlaybackController — drives trace playback
// ---------------------------------------------------------------------------

/// Manages playback state over a `WorldSimTrace`, producing `TraceFrame`s
/// on demand without the renderer needing to know about `WorldState`.
pub struct PlaybackController {
    trace: WorldSimTrace,
    current_tick: u64,
    speed: f32,
    paused: bool,
    cached_state: Option<WorldState>,
    event_window: u64,
}

impl PlaybackController {
    /// Create a new controller from a recorded trace.
    pub fn new(trace: WorldSimTrace, event_window: u64) -> Self {
        PlaybackController {
            current_tick: 0,
            speed: 1.0,
            paused: false,
            cached_state: None,
            event_window,
            trace,
        }
    }

    /// Advance playback by `dt_secs` of real time (scaled by speed).
    /// Does nothing while paused.
    pub fn advance(&mut self, dt_secs: f32) {
        if self.paused {
            return;
        }
        let ticks_to_advance = (dt_secs * self.speed * 10.0) as u64; // 10 ticks/sec base
        let new_tick = (self.current_tick + ticks_to_advance).min(self.trace.total_ticks);
        if new_tick != self.current_tick {
            self.current_tick = new_tick;
            self.cached_state = None; // invalidate cache
        }
    }

    /// Jump to a specific tick (clamped to valid range).
    pub fn seek_to(&mut self, tick: u64) {
        let clamped = tick.min(self.trace.total_ticks);
        if clamped != self.current_tick {
            self.current_tick = clamped;
            self.cached_state = None;
        }
    }

    /// Process a `PlaybackCommand`.
    pub fn handle_command(&mut self, cmd: PlaybackCommand) {
        match cmd {
            PlaybackCommand::None => {}
            PlaybackCommand::TogglePause => {
                self.paused = !self.paused;
            }
            PlaybackCommand::SeekForward(ticks) => {
                self.seek_to(self.current_tick.saturating_add(ticks));
            }
            PlaybackCommand::SeekBackward(ticks) => {
                self.seek_to(self.current_tick.saturating_sub(ticks));
            }
            PlaybackCommand::SpeedUp => {
                self.speed = (self.speed * 2.0).min(1024.0);
            }
            PlaybackCommand::SlowDown => {
                self.speed = (self.speed / 2.0).max(0.125);
            }
            PlaybackCommand::Quit => {
                // Backends handle quit via is_running(); nothing to do here.
            }
        }
    }

    /// Generate the current frame, reconstructing state if needed.
    pub fn current_frame(&mut self) -> TraceFrame {
        // Reconstruct cached state if invalidated.
        if self.cached_state.is_none() {
            self.cached_state = self.trace.state_at_tick(self.current_tick);
        }

        if let Some(state) = &self.cached_state {
            generate_frame(
                state,
                &self.trace.chronicle_log,
                self.event_window,
                self.trace.total_ticks,
            )
        } else {
            // Fallback: empty frame (shouldn't happen with valid trace).
            TraceFrame {
                tick: self.current_tick,
                total_ticks: self.trace.total_ticks,
                settlements: vec![],
                entities: vec![],
                factions: vec![],
                regions: vec![],
                trade_routes: vec![],
                events: vec![],
                summary: FrameSummary {
                    alive_npcs: 0,
                    alive_monsters: 0,
                    total_population: 0,
                    avg_threat: 0.0,
                    season: "Unknown".to_string(),
                    year: 0,
                },
            }
        }
    }

    /// Current tick position.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Current playback speed multiplier.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Whether playback is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Total ticks in the trace.
    pub fn total_ticks(&self) -> u64 {
        self.trace.total_ticks
    }

    /// The trace seed.
    pub fn seed(&self) -> u64 {
        self.trace.seed
    }
}
