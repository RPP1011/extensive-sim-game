//! Per-fixture runtime for `assets/sim/objective_capture_10v10.sim`
//! — the FIRST competitive objective-control sim. 10 Red vs 10 Blue
//! race to accumulate "hold time" on a control point at the world
//! origin.
//!
//! ## Two-layer execution
//!
//! **GPU layer (compiled .sim).** Per-tick chain:
//!   1. clear_tail + clear mask_0 bitmap + zero scoring_output
//!   2. mask_verb_Strike — PerPair, sets actor's bit when pair
//!      (agent, cand=0) passes the WHEN clause (cand=0 hardcoded
//!      via task-5.7 mask_k=1u limitation)
//!   3. scoring — PerAgent argmax over Strike row × N candidates,
//!      score = `1000 - target.hp` (no team filter — see .sim doc
//!      §"IMPLEMENTATION GAPS")
//!   4. physics_verb_chronicle_Strike — emits Damaged events
//!   5. physics_ApplyDamage — applies HP delta + flips alive
//!   6. seed_indirect_0
//!   7. fold_damage_dealt — per-source view
//!
//! Because of the mask_k=1 + missing team filter combo, the GPU
//! layer's actual gameplay is degenerate: every alive non-slot-0
//! actor strikes slot 0 on tick 0; slot 0 dies; thereafter the
//! `target.alive` (cand=0) gate fails for everyone and DSL combat
//! halts. This proves the .sim compiles + dispatches end-to-end,
//! but is not a faithful 10v10.
//!
//! **HOST layer (authoritative gameplay).** Per-tick:
//!   1. Each alive Red picks the lowest-HP alive Blue; deals 10 dmg.
//!      Each alive Blue picks the lowest-HP alive Red; deals 10 dmg.
//!      (Tied scores broken by lowest slot index — deterministic.)
//!      Cooldown: every other tick, like the GPU layer.
//!   2. Move alive agents toward OBJECTIVE_POS at MOVE_SPEED.
//!   3. Count alive Red within CONTROL_RADIUS of the objective and
//!      alive Blue within CONTROL_RADIUS.
//!   4. If only Red present → red_score += 1. Only Blue → blue_score
//!      += 1. Both present (contested) or empty → no advance.
//!   5. Trace key transitions (control flips, score increments).
//!
//! Win condition: first team to TARGET_SCORE = 100 wins, else
//! stalemate at MAX_TICKS.
//!
//! Both layers run every tick, on disjoint state. The host layer is
//! the source of truth for the harness's reports.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

// --- Objective parameters (host-side authoritative) --------------
/// World position of the control point. Both teams race toward it.
pub const OBJECTIVE_POS: Vec3 = Vec3::ZERO;
/// Radius around the objective for "in control range" checks.
/// Sized larger than the team's Y-spread so the entire team fits
/// in-zone once they all converge — otherwise stragglers count as
/// out-of-zone forever and contests stay too tight.
pub const CONTROL_RADIUS: f32 = 12.0;
/// Per-tick movement speed for alive agents pulling toward objective.
pub const MOVE_SPEED: f32 = 0.5;
/// Target hold-tick count for victory. Sized to be reachable
/// against the asymmetric spawn arrangement: Blue arrives ~30
/// ticks before Red and accumulates uncontested ticks during that
/// window, then both teams brawl on the point. After the brawl
/// resolves, surviving team accrues remaining ticks until target.
pub const TARGET_SCORE: u32 = 50;
/// Initial separation between teams. Red at -RED_SPAWN_X, Blue at
/// +BLUE_SPAWN_X. Asymmetric: Blue spawns closer so Blue arrives
/// first and gets uncontested ticks before Red contests. Without
/// asymmetry, deterministic mirror combat ends in mutual
/// annihilation and the objective is never held uncontested.
pub const RED_SPAWN_X: f32 = 30.0;
pub const BLUE_SPAWN_X: f32 = 15.0;
/// Per-team agent count. Total = 2 * TEAM_SIZE.
pub const TEAM_SIZE: u32 = 10;
/// Per-strike damage on the host-side combat layer. Lower = combat
/// takes longer relative to objective ticks, which gives the
/// objective time to actually advance scores during the brawl.
/// Per-team multipliers below introduce a small combat asymmetry
/// that tips deterministic mirror duels: without asymmetry the
/// teams kill each other on the same tick and no one ever holds
/// the objective uncontested at the end.
pub const HOST_STRIKE_DAMAGE: f32 = 4.0;
/// Red's per-strike multiplier (Red is the under-dog: spawns
/// further from objective AND deals slightly less damage).
pub const RED_DAMAGE_MULT: f32 = 0.95;
/// Blue's per-strike multiplier (Blue is the dominant team).
pub const BLUE_DAMAGE_MULT: f32 = 1.05;
/// Initial HP per combatant.
pub const INITIAL_HP: f32 = 100.0;
/// Maximum range at which a host-side combatant can attack an enemy.
/// Units must be physically close to fight — combat is concentrated
/// around the objective (or wherever the teams collide). Without
/// this, every enemy is always in range and the teams wipe each
/// other before reaching the objective.
pub const ATTACK_RANGE: f32 = 8.0;

/// Snapshot of the objective contest state at a tick.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveState {
    pub red_alive: u32,
    pub blue_alive: u32,
    pub red_in_zone: u32,
    pub blue_in_zone: u32,
    pub red_score: u32,
    pub blue_score: u32,
}

impl ObjectiveState {
    pub fn control_label(&self) -> &'static str {
        match (self.red_in_zone, self.blue_in_zone) {
            (0, 0) => "EMPTY",
            (_, 0) => "RED",
            (0, _) => "BLUE",
            _ => "CONTESTED",
        }
    }
}

/// Per-fixture state for the 10v10 objective capture.
pub struct ObjectiveCapture10v10State {
    gpu: GpuContext,

    // -- GPU Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    /// `mana` repurposed as team tag. The GPU scoring kernel does
    /// not (today) read it because `self.mana` truncates the score
    /// expression's lowering. Kept declared so the binding shape
    /// matches the emitter's expectation if a future fix lands.
    #[allow(dead_code)]
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmap (1 verb: Strike=0) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_strike_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,

    // -- Host-side authoritative world --
    /// Per-agent position. Initialised on opposite sides of the
    /// objective; stepped toward OBJECTIVE_POS each tick.
    positions: Vec<Vec3>,
    /// Per-agent team tag (0=Red, 1=Blue), parallel to `mana_buf`.
    teams: Vec<u8>,
    /// HOST-side authoritative HP. Independent of the GPU `agent_hp`
    /// buffer (which only ever sees slot 0 die). The harness reads
    /// THIS for its trace lines.
    host_hp: Vec<f32>,
    /// HOST-side authoritative alive flag.
    host_alive: Vec<bool>,
    /// HOST-side per-source damage counter (debug / observability).
    host_damage_dealt: Vec<f32>,
    /// HOST-side total kills per team.
    host_kills: [u32; 2],

    // -- Objective scoring (host-side authoritative) --
    red_score: u32,
    blue_score: u32,
    /// Trace history — appended on score increment, control-state
    /// flip, or every 50 ticks.
    trace: Vec<(u64, ObjectiveState)>,
}

impl ObjectiveCapture10v10State {
    pub fn new(seed: u64) -> Self {
        let agent_count: u32 = 2 * TEAM_SIZE;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Per-team initialisation. Red on -X, Blue on +X. Slot order:
        // [Red0..Red9, Blue0..Blue9]. Mana mirrors team for the GPU
        // binding (unused by scoring today).
        let hp_init: Vec<f32> = vec![INITIAL_HP; agent_count as usize];
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let mut mana_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        let mut teams: Vec<u8> = Vec::with_capacity(agent_count as usize);
        let mut positions: Vec<Vec3> = Vec::with_capacity(agent_count as usize);
        for slot in 0..agent_count {
            let team = if slot < TEAM_SIZE { 0u8 } else { 1u8 };
            teams.push(team);
            mana_init.push(team as f32);
            let in_team_idx = if team == 0 { slot } else { slot - TEAM_SIZE };
            let y = (in_team_idx as f32) - (TEAM_SIZE as f32 - 1.0) * 0.5;
            let x = if team == 0 { -RED_SPAWN_X } else { BLUE_SPAWN_X };
            positions.push(Vec3::new(x, y * 1.5, 0.0));
        }
        let host_hp = vec![INITIAL_HP; agent_count as usize];
        let host_alive = vec![true; agent_count as usize];
        let host_damage_dealt = vec![0.0_f32; agent_count as usize];

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("obj_cap::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("obj_cap::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "obj_cap");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "obj_cap::damage_dealt",
            agent_count,
            false,
            false,
        );

        let mask_cfg_init = mask_verb_Strike::MaskVerbStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("obj_cap::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            mask_0_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_strike_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
            positions,
            teams,
            host_hp,
            host_alive,
            host_damage_dealt,
            host_kills: [0, 0],
            red_score: 0,
            blue_score: 0,
            trace: Vec::new(),
        }
    }

    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    /// GPU-side per-agent HP (limited gameplay — slot 0 dies tick 0,
    /// then halts). The HOST-side authoritative HP lives in
    /// `host_hp()`.
    pub fn read_gpu_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }
    pub fn read_gpu_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn positions(&self) -> &[Vec3] { &self.positions }
    pub fn teams(&self) -> &[u8] { &self.teams }
    pub fn host_hp(&self) -> &[f32] { &self.host_hp }
    pub fn host_alive(&self) -> &[bool] { &self.host_alive }
    pub fn host_damage_dealt(&self) -> &[f32] { &self.host_damage_dealt }
    pub fn host_kills(&self) -> [u32; 2] { self.host_kills }
    pub fn red_score(&self) -> u32 { self.red_score }
    pub fn blue_score(&self) -> u32 { self.blue_score }
    pub fn trace(&self) -> &[(u64, ObjectiveState)] { &self.trace }

    /// Read current objective contest state from host-side
    /// authoritative arrays. The harness uses this for per-50-tick
    /// trace lines and the final report.
    pub fn read_objective_state(&self) -> ObjectiveState {
        let mut red_alive = 0u32;
        let mut blue_alive = 0u32;
        let mut red_in_zone = 0u32;
        let mut blue_in_zone = 0u32;
        for slot in 0..self.agent_count as usize {
            if !self.host_alive[slot] { continue; }
            let dist2 = self.positions[slot].distance_squared(OBJECTIVE_POS);
            let in_zone = dist2 <= CONTROL_RADIUS * CONTROL_RADIUS;
            if self.teams[slot] == 0 {
                red_alive += 1;
                if in_zone { red_in_zone += 1; }
            } else {
                blue_alive += 1;
                if in_zone { blue_in_zone += 1; }
            }
        }
        ObjectiveState {
            red_alive, blue_alive, red_in_zone, blue_in_zone,
            red_score: self.red_score, blue_score: self.blue_score,
        }
    }

    /// Returns Some(team_idx) if the team has reached TARGET_SCORE
    /// (0=Red, 1=Blue), None otherwise.
    pub fn winner(&self) -> Option<u8> {
        if self.red_score >= TARGET_SCORE { Some(0) }
        else if self.blue_score >= TARGET_SCORE { Some(1) }
        else { None }
    }

    /// HOST-side combat round. Each alive agent on even ticks picks
    /// the lowest-HP alive ENEMY (different team) and deals
    /// HOST_STRIKE_DAMAGE. Tied scores broken by lowest slot index
    /// (deterministic per P11).
    fn host_combat_step(&mut self) {
        if self.tick % 2 != 0 { return; }
        // Build an ordered intent list FIRST (read-only snapshot of
        // the pre-tick state), then apply all damage at once. This
        // keeps the round symmetric — both teams "see" the same
        // pre-tick HP when picking targets, no first-mover advantage.
        let n = self.agent_count as usize;
        let mut intents: Vec<(usize, usize)> = Vec::with_capacity(n); // (attacker, target)
        let attack_range_sq = ATTACK_RANGE * ATTACK_RANGE;
        for attacker in 0..n {
            if !self.host_alive[attacker] { continue; }
            let attacker_team = self.teams[attacker];
            let attacker_pos = self.positions[attacker];
            let mut best: Option<usize> = None;
            let mut best_hp = f32::INFINITY;
            for cand in 0..n {
                if !self.host_alive[cand] { continue; }
                if self.teams[cand] == attacker_team { continue; }
                // Range check: only enemies within ATTACK_RANGE are
                // candidates. Forces units to physically close before
                // combat — the fight concentrates near the objective.
                let dist2 = self.positions[cand].distance_squared(attacker_pos);
                if dist2 > attack_range_sq { continue; }
                let hp = self.host_hp[cand];
                if hp < best_hp {
                    best_hp = hp;
                    best = Some(cand);
                }
                // Tied scores: keep first-seen (lowest slot index).
            }
            if let Some(target) = best {
                intents.push((attacker, target));
            }
        }
        // Apply damage. Multiple attackers can damage the same target
        // in the same tick — accumulate.
        for &(attacker, target) in &intents {
            // Re-check alive — a defender could have died this same
            // tick from a prior intent. (Strictly equivalent if we
            // committed all damage simultaneously, but this matches
            // the "events apply in slot order" P11 convention.)
            if !self.host_alive[target] { continue; }
            let mult = if self.teams[attacker] == 0 {
                RED_DAMAGE_MULT
            } else {
                BLUE_DAMAGE_MULT
            };
            let dmg = HOST_STRIKE_DAMAGE * mult;
            self.host_damage_dealt[attacker] += dmg;
            self.host_hp[target] -= dmg;
            if self.host_hp[target] <= 0.0 {
                self.host_alive[target] = false;
                self.host_kills[self.teams[attacker] as usize] += 1;
            }
        }
    }

    /// Move alive agents toward the objective. Snap to objective
    /// when within one step (avoids overshoot oscillation).
    fn host_move_step(&mut self) {
        for slot in 0..self.agent_count as usize {
            if !self.host_alive[slot] { continue; }
            let pos = self.positions[slot];
            let to_obj = OBJECTIVE_POS - pos;
            let len = to_obj.length();
            if len > MOVE_SPEED {
                self.positions[slot] = pos + to_obj / len * MOVE_SPEED;
            } else {
                self.positions[slot] = OBJECTIVE_POS;
            }
        }
    }

    /// Count contest, advance scores, append trace if interesting.
    fn host_objective_step(&mut self) {
        let mut red_in_zone = 0u32;
        let mut blue_in_zone = 0u32;
        for slot in 0..self.agent_count as usize {
            if !self.host_alive[slot] { continue; }
            let dist2 = self.positions[slot].distance_squared(OBJECTIVE_POS);
            if dist2 <= CONTROL_RADIUS * CONTROL_RADIUS {
                if self.teams[slot] == 0 { red_in_zone += 1; }
                else { blue_in_zone += 1; }
            }
        }
        let scored = match (red_in_zone, blue_in_zone) {
            (r, 0) if r > 0 => { self.red_score += 1; true }
            (0, b) if b > 0 => { self.blue_score += 1; true }
            _ => false,
        };
        let prev_label = self.trace.last().map(|(_, s)| s.control_label());
        let snap = self.read_objective_state();
        let label = snap.control_label();
        let label_changed = prev_label.map(|p| p != label).unwrap_or(true);
        let _ = scored;
        if label_changed || self.tick % 50 == 0 || self.winner().is_some() {
            self.trace.push((self.tick, snap));
        }
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("obj_cap::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("obj_cap::read_f32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("obj_cap::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("obj_cap::read_u32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
}

impl CompiledSim for ObjectiveCapture10v10State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("obj_cap::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf, 0, &self.mask_0_bitmap_buf, 0, mask_bytes.max(4),
        );
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round.
        let mask_cfg = mask_verb_Strike::MaskVerbStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = mask_verb_Strike::MaskVerbStrikeBindings {
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (3) Scoring.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Strike chronicle.
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&strike_cfg),
        );
        let strike_bindings = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_strike(
            &mut self.cache, &strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) ApplyDamage.
        let event_count_estimate = self.agent_count * 4;
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6) seed_indirect_0.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg),
        );
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) fold_damage_dealt.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache, &damage_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.tick += 1;

        // (8) HOST-side authoritative gameplay.
        self.host_combat_step();
        self.host_move_step();
        self.host_objective_step();
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &self.positions }
}

pub fn make_sim(seed: u64, _agent_count_ignored: u32) -> Box<dyn CompiledSim> {
    Box::new(ObjectiveCapture10v10State::new(seed))
}
