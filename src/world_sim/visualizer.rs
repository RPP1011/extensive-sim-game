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
    pub elevation: u8,
    pub is_floating: bool,
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

/// Compact city grid data for rendering tiles.
/// Cells are packed as a flat array (row-major, cols × rows).
/// Each cell is encoded as a single u8: high nibble = CellState, low nibble = ZoneType.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CityGridView {
    pub settlement_id: u32,
    /// World-space position of the grid center.
    pub center_pos: (f32, f32),
    pub cols: usize,
    pub rows: usize,
    /// World units per cell.
    pub cell_size: f32,
    /// Packed cells: (state << 4) | zone. Length = cols * rows.
    pub cells: Vec<u8>,
    /// Density per cell (0-3). Length = cols * rows.
    pub density: Vec<u8>,
    /// Road tier per cell (0-4). Length = cols * rows.
    pub road_tier: Vec<u8>,
}

/// Detailed NPC state for a selected/tracked entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpcDetailView {
    pub entity_id: u32,
    pub name: String,
    pub level: u32,
    pub hp: f32,
    pub max_hp: f32,
    pub gold: f32,
    pub archetype: String,
    // Needs
    pub hunger: f32,
    pub shelter: f32,
    pub safety: f32,
    pub social: f32,
    pub purpose: f32,
    pub esteem: f32,
    // Emotions
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub pride: f32,
    pub anxiety: f32,
    // State
    pub morale: f32,
    pub stress: f32,
    pub economic_intent: String,
    pub work_state: String,
    // Goals
    pub goals: Vec<String>,
    // Classes
    pub classes: Vec<String>,
    // Top behavior tags
    pub top_tags: Vec<(String, f32)>,
    // Recent memory events
    pub recent_memories: Vec<String>,
    // Biography (full text)
    pub biography: String,
    // Position
    pub pos: (f32, f32),
    pub home_settlement: Option<String>,
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
    pub city_grids: Vec<CityGridView>,
    pub events: Vec<EventView>,
    pub summary: FrameSummary,
    /// Detailed state for a selected/tracked NPC (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_npc: Option<NpcDetailView>,
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
    generate_frame_with_selection(state, chronicle, event_window, total_ticks, None)
}

/// Build a `TraceFrame` with optional NPC detail for a selected entity.
pub fn generate_frame_with_selection(
    state: &WorldState,
    chronicle: &[ChronicleEntry],
    event_window: u64,
    total_ticks: u64,
    selected_entity_id: Option<u32>,
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
            elevation: r.elevation,
            is_floating: r.is_floating,
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

    // --- City grids ---
    let city_grids: Vec<CityGridView> = state.settlements.iter().filter_map(|s| {
        let grid_idx = s.city_grid_idx?;
        if grid_idx >= state.city_grids.len() { return None; }
        let grid = &state.city_grids[grid_idx];
        let n = grid.cols * grid.rows;
        let mut cells = Vec::with_capacity(n);
        let mut density = Vec::with_capacity(n);
        let mut road_tier = Vec::with_capacity(n);
        for cell in &grid.cells {
            cells.push((cell.state as u8) << 4 | (cell.zone as u8));
            density.push(cell.density);
            road_tier.push(cell.road_tier);
        }
        Some(CityGridView {
            settlement_id: s.id,
            center_pos: s.pos,
            cols: grid.cols,
            rows: grid.rows,
            cell_size: 2.0,
            cells,
            density,
            road_tier,
        })
    }).collect();

    // --- Selected NPC detail ---
    let selected_npc = selected_entity_id.and_then(|eid| {
        let entity = state.entity(eid)?;
        if entity.kind != super::state::EntityKind::Npc { return None; }
        let npc = entity.npc.as_ref()?;
        let display_name = super::naming::entity_display_name(entity);

        let home_settlement = npc.home_settlement_id
            .and_then(|sid| state.settlement(sid))
            .map(|s| s.name.clone());

        let goals: Vec<String> = npc.goal_stack.goals.iter().map(|g| {
            format!("{:?} (prio {:.1}, progress {:.0}%)", g.kind, g.priority, g.progress * 100.0)
        }).collect();

        let classes: Vec<String> = npc.classes.iter().map(|c| {
            let name = if c.display_name.is_empty() {
                format!("Class({})", c.class_name_hash)
            } else {
                c.display_name.clone()
            };
            format!("{} lv{}", name, c.level)
        }).collect();

        let top_tags: Vec<(String, f32)> = {
            let mut tags: Vec<_> = npc.behavior_profile.iter()
                .map(|&(hash, val)| {
                    let name = super::systems::biography::tag_display_name(hash);
                    (name.to_string(), val)
                })
                .collect();
            tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            tags.truncate(8);
            tags
        };

        let recent_memories: Vec<String> = npc.memory.events.iter().rev().take(10).map(|e| {
            let date_tick = e.tick;
            let year = date_tick / 4800;
            let season = ["Spring", "Summer", "Autumn", "Winter"][(date_tick / 1200 % 4) as usize];
            format!("Y{} {} — {:?}", year, season, e.event_type)
        }).collect();

        let biography = super::systems::biography::generate_biography(entity, state);

        let work_state = format!("{:?}", npc.work_state);
        let economic_intent = format!("{:?}", npc.economic_intent);

        Some(NpcDetailView {
            entity_id: eid,
            name: display_name,
            level: entity.level,
            hp: entity.hp,
            max_hp: entity.max_hp,
            gold: npc.gold,
            archetype: npc.archetype.clone(),
            hunger: npc.needs.hunger,
            shelter: npc.needs.shelter,
            safety: npc.needs.safety,
            social: npc.needs.social,
            purpose: npc.needs.purpose,
            esteem: npc.needs.esteem,
            joy: npc.emotions.joy,
            sadness: npc.emotions.grief,
            anger: npc.emotions.anger,
            fear: npc.emotions.fear,
            pride: npc.emotions.pride,
            anxiety: npc.emotions.anxiety,
            morale: npc.morale,
            stress: npc.stress,
            economic_intent,
            work_state,
            goals,
            classes,
            top_tags,
            recent_memories,
            biography,
            pos: entity.pos,
            home_settlement,
        })
    });

    TraceFrame {
        tick,
        total_ticks,
        settlements,
        entities,
        factions,
        regions,
        trade_routes,
        city_grids,
        events,
        summary,
        selected_npc,
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
                city_grids: vec![],
                events: vec![],
                selected_npc: None,
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
