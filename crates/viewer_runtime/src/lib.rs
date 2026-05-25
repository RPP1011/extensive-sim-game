//! `viewer_runtime` — windowed viewer over `sims::dungeon_horde::GeneratedRuntime`.
//!
//! Reads the GPU agent SoA back to host every sim tick, then mirrors
//! agent positions + types into a single world-spanning CPU
//! `VoxelGrid` the renderer (voxel_engine) consumes via Vulkan. Walls
//! come from the runtime's `voxel_terrain` (dense stone-vs-air);
//! floor cells from the seeded floor map; agents from the per-tick
//! readback.
//!
//! # Two GPU contexts (intentional)
//!
//! - `state.gpu` is a wgpu device used by the sim runtime to drive
//!   compute kernels.
//! - The viewer's renderer (voxel_engine) lives on a separate Vulkan
//!   instance/device created in `bin/viewer_app.rs`.
//!
//! The bridge crosses the boundary by reading sim state back to host
//! (CPU `Vec<u8>` voxel grid) and re-uploading it to the renderer's
//! Vulkan texture each frame. Cheap at 96×96×16 cells.

use anyhow::Result;
use glam::Vec3;
use sims::dungeon_horde::GeneratedRuntime;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};

pub mod dungeon;
pub mod mesh_renderer;
pub mod vs;
pub mod vs_ui;

use dungeon::{Dungeon, GRID_X, GRID_Y, GRID_Z, N_HEROES, ROOM_INTERIOR_Z};

// ---------------------------------------------------------------------
// Material palette — distinct color per (creature_type, role) plus
// terrain. Indices are referenced from `voxel_grid` cells, so they
// must fit in a u8.
// ---------------------------------------------------------------------

pub const MAT_AIR: u8 = 0;
pub const MAT_FLOOR: u8 = 1;
pub const MAT_WALL: u8 = 2;
/// Floor tint for the boss room — clear visual cue that "this is the climax".
pub const MAT_BOSS_FLOOR: u8 = 3;
// Heroes: per-role base index (warrior=10, cleric=11, ranger=12, mage=13, rogue=14).
pub const MAT_HERO_BASE: u8 = 10;
// Enemies: archer=20, brute=21, goblin=22.
pub const MAT_ARCHER: u8 = 20;
pub const MAT_BRUTE: u8 = 21;
pub const MAT_GOBLIN: u8 = 22;
// Dead agent ghost color.
pub const MAT_DEAD: u8 = 30;
/// Boss agent — singular Brute in the boss room with 4x HP. Painted
/// in a brighter accent so the climax target is immediately visible.
pub const MAT_BOSS: u8 = 31;
/// Enemy with `alert >= ALERT_TINT_LO` — visible "I think something
/// is wrong" state. Overrides creature-type color while alerted.
pub const MAT_ALERTED: u8 = 32;
/// Enemy with `alert >= ALERT_TINT_HI` — visible "I know exactly
/// what's happening" state. Maximum saturation.
pub const MAT_PANICKED: u8 = 33;
/// Per-hero HP bar — 3 cells stacked above each hero. Each cell
/// lights up while the hero's HP fraction is above its tier threshold.
pub const MAT_HP_GREEN: u8 = 34;
pub const MAT_HP_YELLOW: u8 = 35;
pub const MAT_HP_RED: u8 = 36;
/// Rogue while stealth is active (stealth_until_tick > world.tick).
/// Desaturated purple so the hero is still locatable but visibly "ghosted".
pub const MAT_STEALTHED: u8 = 37;
/// Per-tick damage flash — agents tinted bright white for a few frames
/// after taking meaningful damage so combat hits are individually visible.
pub const MAT_FLASH: u8 = 38;
/// Per-tick heal flash — green-white tint applied for a few frames
/// after an agent's HP rises (Cleric's Heal verb / Mend on the duel
/// roster). Visible "the cleric just healed someone" signal.
pub const MAT_HEAL: u8 = 41;
/// Floor cell tinted darker beneath each living agent — gives ground
/// reference + depth perception even without proper diffuse lighting.
pub const MAT_AGENT_SHADOW: u8 = 42;
/// Floor tint for rooms whose enemies are all dead — visible "we cleared
/// this room" feedback. Updates per-tick from the agent snapshot.
pub const MAT_FLOOR_CLEARED: u8 = 43;
/// Wall material for cells inside the boss room slot — darker, slightly
/// red-tinted so the climax chamber's walls visibly match the
/// MAT_BOSS_FLOOR floor tint instead of staying the default gray.
pub const MAT_BOSS_WALL: u8 = 44;
/// Floor tint for the warrior's currently-targeted room — surfaces the
/// frontier-greedy AI's next-room choice. Updates per-tick from the
/// warrior's `target_room_idx` GPU buffer.
pub const MAT_FLOOR_TARGET: u8 = 45;
/// Fog-of-war floor tint for rooms no hero has discovered yet. Per-agent
/// knowledge bitmaps track which rooms each agent is "aware of"; the
/// floor paints fog wherever the OR of all hero bitmaps doesn't cover.
pub const MAT_FLOOR_FOG: u8 = 46;
/// Projectile VFX cells — short-lived line segments from attacker to
/// target painted in the bridge for ~PROJECTILE_TTL frames after each
/// HP-delta-detected hit. Two flavors so hero-vs-enemy attacks read
/// distinctly even at high density.
pub const MAT_PROJECTILE_HERO: u8 = 47;   // hero firing on enemy
pub const MAT_PROJECTILE_ENEMY: u8 = 48;  // enemy firing on hero
/// All floor cells tint to this when the dungeon is cleared (every
/// enemy dead, ≥1 hero alive). Replaces MAT_FLOOR / MAT_BOSS_FLOOR.
pub const MAT_VICTORY_FLOOR: u8 = 39;
/// All floor cells tint to this on TPK (every hero dead). Replaces
/// MAT_FLOOR / MAT_BOSS_FLOOR.
pub const MAT_DEFEAT_FLOOR: u8 = 40;

fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    p.set(MAT_FLOOR, palette_entry(180, 160, 130)); // light tan
    p.set(MAT_WALL,  palette_entry(95,  92,  88));  // gray stone
    p.set(MAT_BOSS_FLOOR, palette_entry(160, 80, 80));  // ochre-red — boss-room floor
    // Heroes (slot+1 → role)
    p.set(MAT_HERO_BASE + 0, palette_entry(140, 95,  50));  // Warrior — brown
    p.set(MAT_HERO_BASE + 1, palette_entry(240, 240, 240)); // Cleric  — white
    p.set(MAT_HERO_BASE + 2, palette_entry(60,  170, 75));  // Ranger  — green
    p.set(MAT_HERO_BASE + 3, palette_entry(70,  130, 230)); // Mage    — blue
    p.set(MAT_HERO_BASE + 4, palette_entry(170, 80,  220)); // Rogue   — purple
    p.set(MAT_ARCHER, palette_entry(230, 130, 40));  // orange
    p.set(MAT_BRUTE,  palette_entry(150, 35,  35));  // dark red
    p.set(MAT_GOBLIN, palette_entry(95,  140, 70));  // dim green
    p.set(MAT_DEAD,   palette_entry(50,  50,  50));  // gray
    p.set(MAT_BOSS,   palette_entry(255, 60,  20));  // bright orange-red — the boss
    p.set(MAT_ALERTED, palette_entry(240, 200, 30));  // amber — "something's wrong"
    p.set(MAT_PANICKED, palette_entry(255, 50,  120));// hot pink — "they know"
    p.set(MAT_HP_GREEN,  palette_entry(60,  220, 80));  // bright green   — healthy bar
    p.set(MAT_HP_YELLOW, palette_entry(230, 220, 40));  // bright yellow  — middle bar
    p.set(MAT_HP_RED,    palette_entry(230, 50,  50));  // bright red     — critical bar
    p.set(MAT_STEALTHED, palette_entry(110, 80, 130));  // muted purple   — rogue invisible
    p.set(MAT_FLASH,     palette_entry(255, 255, 255)); // pure white     — damage hit flash
    p.set(MAT_HEAL,      palette_entry(180, 255, 200)); // mint           — heal flash
    p.set(MAT_AGENT_SHADOW, palette_entry(120, 105, 85));// dimmed tan    — shadow under agents
    p.set(MAT_FLOOR_CLEARED, palette_entry(140, 130, 110));// dim tan    — cleared room
    p.set(MAT_BOSS_WALL,  palette_entry(75,  55,  55));  // dark red-gray  — boss-room walls
    p.set(MAT_FLOOR_TARGET, palette_entry(150, 170, 200));// faint blue   — warrior's next target room
    p.set(MAT_FLOOR_FOG,    palette_entry(60,  55,  45)); // very dim     — unexplored room (fog of war)
    p.set(MAT_PROJECTILE_HERO,  palette_entry(180, 220, 255));// pale blue   — hero shot
    p.set(MAT_PROJECTILE_ENEMY, palette_entry(255, 180, 100));// pale orange — enemy shot
    p.set(MAT_VICTORY_FLOOR, palette_entry(80,  180, 100));// muted green  — dungeon cleared
    p.set(MAT_DEFEAT_FLOOR,  palette_entry(180, 50,  50)); // muted red    — TPK
    p
}

fn palette_entry(r: u8, g: u8, b: u8) -> PaletteEntry {
    PaletteEntry { r, g, b, roughness: 200, emissive: 0, material_type: MaterialType::Stone }
}

/// Maps a (creature_type, role) tuple to a palette index.
fn material_for(creature_type: u32, role: u32) -> u8 {
    match creature_type {
        dungeon::CT_HERO => {
            let r = role.saturating_sub(1).min(4) as u8;
            MAT_HERO_BASE + r
        }
        dungeon::CT_ARCHER => MAT_ARCHER,
        dungeon::CT_BRUTE  => MAT_BRUTE,
        dungeon::CT_GOBLIN => MAT_GOBLIN,
        _ => MAT_DEAD,
    }
}

/// Per-frame snapshot of a single agent (post-readback from sim GPU).
#[derive(Clone, Copy, Debug)]
pub struct AgentSnapshot {
    pub pos: Vec3, // world-space xyz (sim Z-up); the viewer swaps Z↔Y in the bridge.
    pub alive: bool,
    pub creature_type: u32,
    pub role: u32,
    pub hp: f32,
    /// Per-tick alert level for enemies (`alert` field in dungeon_horde.sim).
    /// 0 = calm; raises via missing-ally suspicion + ally-death broadcast +
    /// sound-on-damage. Heroes always read 0.
    pub alert: u32,
    /// Absolute tick at which the rogue's Stealth ability expires (per
    /// `stealth_until_tick` custom field). > current tick means stealthed.
    pub stealth_until_tick: u32,
    /// Direction the agent is currently moving (sim-coord unit vector in
    /// the XY plane). Computed as a low-pass-filtered delta of position
    /// across ticks so the mesh pass can rotate models to face their
    /// movement direction. Defaults to `(0, 1)` for stationary agents.
    pub facing_xy: [f32; 2],
}

/// The viewer's host-side state. Owns the sim runtime, the CPU voxel
/// grid mirroring its dungeon + agents, and the dungeon roomgen
/// metadata so it can know where the spawn room is for camera framing.
pub struct ViewerApp {
    pub state: GeneratedRuntime,
    pub dungeon: Dungeon,
    pub seed: u64,
    /// Floor cells as 2D (x, y) tuples — painted on top of the
    /// stone fill each refresh.
    pub floor_cells: std::collections::BTreeSet<(u32, u32)>,
    /// Per-agent latest readback. Index = sim slot id.
    agents: Vec<AgentSnapshot>,
    pub agent_count: u32,
    /// Tick at which sim terminated (TPK or dungeon cleared); `None`
    /// while the sim is still running.
    pub terminated_at_tick: Option<u64>,
    /// Per-hero exploration state — frontier-greedy retargeting
    /// advances `target_room_idx` between ticks. Without this,
    /// heroes stop at the first adjacent room they reach.
    hero_state: dungeon::HeroExploreState,
    /// Agent slot of the singular boss enemy (first Brute placed
    /// in the boss room). `None` if no brute landed in boss_room
    /// (very small dungeons or unlucky rolls).
    pub boss_slot: Option<u32>,
    /// Per-agent HP from the previous refresh — diffed against
    /// current HP to detect damage events and trigger a flash.
    prev_hp: Vec<f32>,
    /// Per-agent remaining damage-flash frames. Decremented each tick;
    /// when > 0, the painter overrides the agent's material to MAT_FLASH.
    /// Set to FLASH_FRAMES whenever an HP drop ≥ FLASH_DELTA is observed.
    flash_ticks: Vec<u8>,
    /// Per-agent remaining heal-flash frames — mirror of `flash_ticks`
    /// for HP increases. Damage flash takes precedence if both fire
    /// the same tick (combat hits trump heal feedback).
    pub heal_ticks: Vec<u8>,
    /// `Some(true)` = dungeon cleared (all enemies dead, ≥1 hero alive).
    /// `Some(false)` = TPK (all heroes dead).
    /// `None` = sim still running. Set in `step()` at termination so the
    /// bridge can tint the whole floor green or red.
    pub outcome: Option<bool>,
    /// Tick at which each hero died (None while alive). Indexed by
    /// hero ordinal (0=Warrior, 1=Cleric, 2=Ranger, 3=Mage, 4=Rogue) —
    /// matches `seed_topology`'s hero placement order.
    hero_death_tick: [Option<u64>; dungeon::N_HEROES as usize],
    /// Tick at which the boss died (None until then or if no boss).
    boss_death_tick: Option<u64>,
    /// Highest `alert` value seen on any enemy across the run.
    max_alert_seen: u32,
    /// Total enemy kills observed (alive→dead transitions among non-heroes).
    total_kills: u32,
    /// Whether `print_run_summary` already printed the verdict for the
    /// current run. Reset to false at startup; set to true after print
    /// so we don't spam the log per-frame after termination.
    summary_printed: bool,
    /// Has the boss been wounded past the enrage threshold this run?
    /// On the transition from `false → true` the viewer writes a one-time
    /// alert spike to every enemy still in the boss room, visibly
    /// escalating the climax fight.
    boss_enraged: bool,
    /// Whether the "boss-push" mode has been announced this run. Set true
    /// after the first log so we don't spam the terminal.
    boss_push_logged: bool,
    /// Previous-tick agent positions; diffed against current to compute
    /// facing direction for the mesh pass.
    /// Per-agent position at the start of the previous tick. Used for
    /// two things: facing-direction delta (XY only) and tick-interp
    /// (renderer lerps between prev and current for smooth movement
    /// between sim ticks).
    pub prev_positions: Vec<[f32; 3]>,
    /// Sticky last-observed facing direction. Used when delta-position is
    /// below the movement epsilon (agent stationary) so models don't
    /// snap back to a default facing every frame they pause.
    last_facing: Vec<[f32; 2]>,
    /// Per-agent ticks since death — 0 means alive. While the value is
    /// in 1..DEATH_FADE_TICKS the agent stays rendered with a fade-out
    /// (shrinking scale + dim tint) instead of vanishing instantly.
    pub death_ticks: Vec<u32>,
    /// Set of room slot indices whose enemies are all dead. Recomputed
    /// each tick from the agent snapshot. The voxel bridge uses this to
    /// tint cleared-room floors with `MAT_FLOOR_CLEARED` so dungeon
    /// progress reads at a glance.
    pub cleared_rooms: std::collections::BTreeSet<u32>,
    /// Warrior hero's `target_room_idx` — the next room the party is
    /// heading toward via the role-based AI. None when warrior is
    /// dead or the GPU buffer hasn't been read yet. Surfaced in the
    /// voxel bridge as MAT_FLOOR_TARGET so the user can see where
    /// the warrior is leading the party next.
    pub warrior_target_room: Option<u32>,
    /// Per-hero room-knowledge bitmaps. Each u64 holds 1 bit per room
    /// slot (max 36 = 6×6 grid). A bit is set when the hero has
    /// personally been in that room. Heroes start knowing only the
    /// spawn room; subsequent rooms light up as they advance.
    /// Information-asymmetric AI uses this to constrain pathfinding —
    /// heroes can't target rooms they haven't discovered.
    pub hero_known_rooms: [u64; dungeon::N_HEROES as usize],
    /// OR of all `hero_known_rooms` — the party's collective knowledge
    /// (which the bridge uses to paint fog-of-war on unknown rooms).
    pub party_known_rooms: u64,
    /// Recently-fired projectile beams. Each entry holds source/target
    /// positions in renderer space + TTL frames remaining + color
    /// flag (true = hero shot, false = enemy shot). Painted as a line
    /// of voxel cells in the bridge; decremented per refresh.
    pub projectiles: Vec<Projectile>,
}

/// Visible attack beam for a few frames after an HP-delta hit. Source
/// + target in renderer XYZ; color flag distinguishes hero/enemy
/// attacks.
#[derive(Clone, Copy, Debug)]
pub struct Projectile {
    pub source: Vec3,
    pub target: Vec3,
    pub ttl: u8,
    pub hero_shot: bool,
}

/// Frames a projectile cell stays painted before being cleared. ~5
/// frames at 50ms tick = 250ms — long enough to register as a flash.
pub const PROJECTILE_TTL: u8 = 5;

/// Number of sim ticks an agent stays rendered after death before
/// disappearing from the mesh pass. ~8 ticks ≈ 400ms wall clock at
/// 50ms tick = a visible death animation.
pub const DEATH_FADE_TICKS: u32 = 8;

impl ViewerApp {
    /// Roll a dungeon, build the runtime, seed walls + agents.
    /// Returns `None` if no wgpu adapter is available (headless host).
    pub fn try_new(seed: u64) -> Option<Self> {
        // Phase 1 asset-pipeline smoke check: load one Kenney character
        // mesh at startup. Loads only; rendering is a separate pass.
        // Failure logs but doesn't abort — voxel-only viewer still works
        // until the mesh pass is wired.
        let mesh_path = "assets/models/kenney_mini-characters/Models/GLB format/character-male-a.glb";
        match mesh_renderer::MeshCpu::load_glb(mesh_path) {
            Ok(m) => eprintln!(
                "[viewer_runtime] loaded mesh {} ({} verts, {} tris)",
                m.source, m.vertex_count(), m.triangle_count(),
            ),
            Err(e) => eprintln!("[viewer_runtime] mesh load failed: {e:#}"),
        }
        let dungeon = dungeon::roll_dungeon(seed);
        let agent_count = dungeon.total_agent_count();
        eprintln!(
            "[viewer_runtime] dungeon: {} rooms, spawn=slot{} boss=slot{} \
             agents={} ({} heroes + {} enemies)",
            dungeon.rooms.len(),
            dungeon.spawn_room.idx(),
            dungeon.boss_room.idx(),
            agent_count,
            N_HEROES,
            agent_count - N_HEROES,
        );
        let mut state = GeneratedRuntime::try_new(seed, agent_count)?;
        let floor_cells = dungeon::seed_voxel_dungeon(&mut state, &dungeon, seed);
        let boss_slot = dungeon::seed_topology(&mut state, &dungeon, seed);

        // Per-creature-type HP overrides so combat resolves in a
        // watchable wall-clock time. Defaults (200 HP for everyone in
        // the .sim's `init { hp: 200 }`) make 5v226 take ~3 minutes
        // wall-clock — far too long for a demo reel. With these
        // overrides a typical run resolves in ~30-60s.
        const ARCHER_HP: f32 = 22.0;
        const BRUTE_HP:  f32 = 45.0;
        const GOBLIN_HP: f32 = 15.0;
        // Heroes stay at the .sim default 200 — they need to tank a few
        // hits to survive crossing each room.
        let agent_count = dungeon.total_agent_count();
        let mut hps = vec![200.0f32; agent_count as usize];
        let mut max_hps = vec![200.0f32; agent_count as usize];
        for (slot, &(_, ct)) in dungeon
            .enemy_placements()
            .iter()
            .filter(|(_, c)| *c == dungeon::CT_ARCHER)
            .enumerate()
        {
            let _ = ct;
            hps[slot] = ARCHER_HP;
            max_hps[slot] = ARCHER_HP;
        }
        let n_archers = dungeon.enemy_placements().iter().filter(|(_, c)| *c == dungeon::CT_ARCHER).count();
        let n_brutes  = dungeon.enemy_placements().iter().filter(|(_, c)| *c == dungeon::CT_BRUTE ).count();
        for i in 0..n_brutes {
            let slot = n_archers + i;
            hps[slot] = BRUTE_HP;
            max_hps[slot] = BRUTE_HP;
        }
        let n_goblins = dungeon.enemy_placements().iter().filter(|(_, c)| *c == dungeon::CT_GOBLIN).count();
        for i in 0..n_goblins {
            let slot = n_archers + n_brutes + i;
            hps[slot] = GOBLIN_HP;
            max_hps[slot] = GOBLIN_HP;
        }
        // Hero HP (last N_HEROES slots) stays at 200.
        state.gpu.queue.write_buffer(
            &state.agent_hp_buf, 0, bytemuck::cast_slice(&hps),
        );
        state.gpu.queue.write_buffer(
            &state.agent_max_hp_buf, 0, bytemuck::cast_slice(&max_hps),
        );

        if let Some(slot) = boss_slot {
            // Boss has ~9x normal Brute HP — heroes have to commit to
            // the climax fight. Writes both hp and max_hp so cleric
            // heals don't cap at the type default.
            const BOSS_HP: f32 = 400.0;
            state.gpu.queue.write_buffer(
                &state.agent_hp_buf, (slot as u64) * 4, bytemuck::bytes_of(&BOSS_HP),
            );
            state.gpu.queue.write_buffer(
                &state.agent_max_hp_buf, (slot as u64) * 4, bytemuck::bytes_of(&BOSS_HP),
            );
            eprintln!(
                "[viewer_runtime] HP overrides: archer={ARCHER_HP} brute={BRUTE_HP} goblin={GOBLIN_HP} hero=200 boss={BOSS_HP} (slot={slot}, room slot{})",
                dungeon.boss_room.idx(),
            );
        } else {
            eprintln!(
                "[viewer_runtime] HP overrides: archer={ARCHER_HP} brute={BRUTE_HP} goblin={GOBLIN_HP} hero=200 — no Brute landed in boss room slot{}",
                dungeon.boss_room.idx(),
            );
        }

        let hero_state = dungeon::HeroExploreState::new(&dungeon);
        let spawn_bitmap = 1u64 << dungeon.spawn_room.idx();
        let mut viewer = Self {
            state,
            dungeon,
            seed,
            floor_cells,
            agents: vec![
                AgentSnapshot {
                    pos: Vec3::ZERO,
                    alive: false,
                    creature_type: 0,
                    role: 0,
                    hp: 0.0,
                    alert: 0,
                    stealth_until_tick: 0,
                    facing_xy: [0.0, 1.0],
                };
                agent_count as usize
            ],
            agent_count,
            terminated_at_tick: None,
            hero_state,
            boss_slot,
            prev_hp: vec![0.0; agent_count as usize],
            flash_ticks: vec![0u8; agent_count as usize],
            heal_ticks: vec![0u8; agent_count as usize],
            outcome: None,
            hero_death_tick: [None; dungeon::N_HEROES as usize],
            boss_death_tick: None,
            max_alert_seen: 0,
            total_kills: 0,
            summary_printed: false,
            boss_enraged: false,
            boss_push_logged: false,
            prev_positions: vec![[0.0, 0.0, 0.0]; agent_count as usize],
            last_facing: vec![[0.0, 1.0]; agent_count as usize],
            death_ticks: vec![0u32; agent_count as usize],
            cleared_rooms: std::collections::BTreeSet::new(),
            warrior_target_room: None,
            hero_known_rooms: [spawn_bitmap; dungeon::N_HEROES as usize],
            party_known_rooms: spawn_bitmap,
            projectiles: Vec::new(),
        };
        viewer.refresh_snapshot();
        // Diagnostic: confirm host-seeded hero state. If creature_type
        // != CT_HERO or role != 1..5 here, the mesh-pass bucketing
        // can't find the right slot and heroes fall through to
        // voxel splat only.
        {
            let n = viewer.agent_count as usize;
            let hero_start = n - dungeon::N_HEROES as usize;
            for h in 0..dungeon::N_HEROES as usize {
                let a = &viewer.agents[hero_start + h];
                eprintln!(
                    "[viewer_runtime] hero[{h}] slot={} ct={} role={} alive={} pos=({:.1},{:.1},{:.1})",
                    hero_start + h, a.creature_type, a.role, a.alive,
                    a.pos.x, a.pos.y, a.pos.z,
                );
            }
        }
        Some(viewer)
    }

    pub fn sim_tick(&self) -> u64 {
        self.state.tick
    }

    /// Slice of the latest per-agent snapshot (post-readback).
    pub fn agents(&self) -> &[AgentSnapshot] {
        &self.agents
    }

    /// World-space centroid of the spawn room — the camera looks here.
    pub fn spawn_centroid(&self) -> Vec3 {
        let c = self.dungeon.spawn_room.centroid();
        Vec3::new(c[0], c[1], c[2])
    }

    /// Step sim one tick and refresh per-agent snapshot.
    pub fn step(&mut self) {
        if self.terminated_at_tick.is_some() {
            return;
        }
        self.state.step();
        self.advance_hero_exploration();
        self.refresh_snapshot();
        self.maybe_enrage_boss_room();
        // Termination heuristic: heroes-alive==0 OR enemies-alive==0.
        let mut heroes_alive = 0u32;
        let mut enemies_alive = 0u32;
        for a in &self.agents {
            if !a.alive {
                continue;
            }
            if a.creature_type == dungeon::CT_HERO {
                heroes_alive += 1;
            } else {
                enemies_alive += 1;
            }
        }
        if (heroes_alive == 0 || enemies_alive == 0) && self.terminated_at_tick.is_none() {
            self.terminated_at_tick = Some(self.state.tick);
            // Victory = at least one hero alive at termination. (We get
            // here when one side hits 0, so the other side is the winner.)
            self.outcome = Some(heroes_alive > 0);
        }
        if self.terminated_at_tick.is_some() && !self.summary_printed {
            self.print_run_summary(heroes_alive, enemies_alive);
            self.summary_printed = true;
        }
        // Live progress ping every 100 ticks. Lets the user follow a
        // run from the terminal without watching the window, and turns
        // the auto-restart reel into visible state transitions in the log.
        if self.state.tick > 0 && self.state.tick % 100 == 0 && self.terminated_at_tick.is_none() {
            eprintln!(
                "[viewer_runtime] tick={} heroes={}/5 enemies={} kills={} alert_max={}{}",
                self.state.tick, heroes_alive, enemies_alive,
                self.total_kills, self.max_alert_seen,
                if self.boss_enraged { " BOSS_ENRAGED" } else { "" },
            );
        }
    }

    /// Host-side hero target_room_idx advancement.
    ///
    /// Heroes are placed in role order by `seed_topology` — h=0 is
    /// Warrior, h=1 Cleric, h=2 Ranger, h=3 Mage, h=4 Rogue — so
    /// `h` doubles as the role discriminant minus one.
    ///
    /// Each tick: read positions/HPs, recompute current_room for every
    /// hero, then derive new targets via a per-role policy:
    /// * **Warrior** — leader. Standard frontier-greedy.
    /// * **Cleric / Ranger / Mage** — followers. Target = warrior's
    ///   current room (cluster with the tank).
    /// * **Rogue** — scout. Target = warrior's TARGET room (one step
    ///   ahead of the party, so LoS reveals enemies in advance).
    /// * If the warrior is dead, every surviving hero falls back to
    ///   their own frontier-greedy target.
    ///
    /// Wounded heroes (`hp < RETREAT_HP`) override the role policy
    /// and route to spawn until they heal past `RESUME_HP`. Hysteresis
    /// between the two prevents oscillation while the cleric mid-heals.
    fn advance_hero_exploration(&mut self) {
        /// Below this HP heroes break off and head back to spawn.
        const RETREAT_HP: f32 = 45.0;
        /// Once retreating, must heal above this to resume exploring.
        const RESUME_HP: f32 = 130.0;
        const H_WARRIOR: usize = 0;
        const H_ROGUE: usize = 4;

        let agent_count = self.agent_count;
        let hero_start = (agent_count - dungeon::N_HEROES) as usize;
        let positions = read_positions(&mut self.state, agent_count);
        let target_buf = self.state.agent_target_room_idx_buf.clone();
        let hp_buf = self.state.agent_hp_buf.clone();
        let alive_buf = self.state.agent_alive_buf.clone();
        let mut targets = read_agent_u32(&mut self.state, &target_buf, agent_count);
        let hps = read_agent_f32(&mut self.state, &hp_buf, agent_count);
        let alive = read_agent_u32(&mut self.state, &alive_buf, agent_count);
        let present_set: std::collections::BTreeSet<dungeon::RoomSlot> =
            self.dungeon.rooms.iter().copied().collect();
        let tick = self.state.tick as u32;
        let spawn_idx = self.dungeon.spawn_room.idx();
        let mut any_change = false;
        // Boss-push: once enough enemies are dead, frontier-greedy
        // exhausts and heroes wander cleared rooms aimlessly. Override
        // the warrior's target to the boss room so the party converges
        // for the climax. Tuned at 60% — leaves room for exploration
        // payoff (treasure rooms) before the final push.
        let initial_enemies = agent_count.saturating_sub(dungeon::N_HEROES);
        let kill_frac = if initial_enemies > 0 {
            self.total_kills as f32 / initial_enemies as f32
        } else { 0.0 };
        let boss_push = kill_frac > 0.60;
        let boss_idx = self.dungeon.boss_room.idx();
        if boss_push && !self.boss_push_logged {
            eprintln!(
                "[viewer_runtime] boss-push engaged @ tick {} ({}/{} enemies down) — party converging on boss room",
                self.state.tick, self.total_kills, initial_enemies,
            );
            self.boss_push_logged = true;
        }

        // Pass 1: recompute current_room + rooms_visited for each hero
        // from their current position. None means the hero isn't in any
        // present room (dead, out of bounds, or in a corridor cell).
        let mut current: [Option<dungeon::RoomSlot>; dungeon::N_HEROES as usize] =
            [None; dungeon::N_HEROES as usize];
        for h in 0..dungeon::N_HEROES as usize {
            let p = positions[hero_start + h];
            if !p[0].is_finite() || !p[1].is_finite() {
                continue;
            }
            let rx = (p[0] / dungeon::SLOT_WIDTH as f32).floor() as i32;
            let ry = (p[1] / dungeon::SLOT_WIDTH as f32).floor() as i32;
            if rx < 0 || ry < 0
                || (rx as u32) >= dungeon::SLOTS_PER_ROW
                || (ry as u32) >= dungeon::SLOTS_PER_ROW
            {
                continue;
            }
            let candidate = dungeon::RoomSlot::new(rx as u32, ry as u32);
            if !present_set.contains(&candidate) {
                continue;
            }
            current[h] = Some(candidate);
            self.hero_state.current_room[h] = Some(candidate.idx());
            let remap = self.dungeon.rooms.iter().position(|r| *r == candidate).unwrap();
            self.hero_state.rooms_visited[h] |= 1u32 << remap;
            // Knowledge update: this hero now "knows" the room they're
            // in. Slot indices range 0..36 so they fit in u64.
            self.hero_known_rooms[h] |= 1u64 << candidate.idx();
        }
        // Recompute party-OR after all heroes have updated.
        self.party_known_rooms = self.hero_known_rooms.iter().fold(0u64, |a, b| a | b);

        // Pass 2: compute the warrior's target first — it anchors the
        // party formation. Followers + rogue read off this in pass 3.
        let warrior_alive = alive[hero_start + H_WARRIOR] != 0;
        let warrior_current = current[H_WARRIOR];
        let warrior_target_after = {
            let hp = hps[hero_start + H_WARRIOR];
            let cur_target = targets[hero_start + H_WARRIOR];
            let retreating = cur_target == spawn_idx
                && warrior_current.map_or(true, |c| c.idx() != spawn_idx);
            match warrior_current {
                Some(cand) if warrior_alive => {
                    if retreating && hp >= RESUME_HP {
                        if boss_push { boss_idx } else {
                            dungeon::pick_next_target(
                                &self.dungeon, cand,
                                self.hero_state.rooms_visited[H_WARRIOR],
                                tick, H_WARRIOR as u32, self.seed,
                            )
                        }
                    } else if !retreating && hp < RETREAT_HP {
                        dungeon::pick_retreat_target(
                            &self.dungeon, cand,
                            self.hero_known_rooms[H_WARRIOR],
                            &self.cleared_rooms,
                            spawn_idx,
                        )
                    } else if boss_push && !retreating {
                        // Once 60%+ cleared, march to the climax instead
                        // of wandering frontier-greedy.
                        boss_idx
                    } else if !retreating && cand.idx() == cur_target {
                        dungeon::pick_next_target(
                            &self.dungeon, cand,
                            self.hero_state.rooms_visited[H_WARRIOR],
                            tick, H_WARRIOR as u32, self.seed,
                        )
                    } else {
                        cur_target
                    }
                }
                _ => cur_target,
            }
        };
        if warrior_target_after != targets[hero_start + H_WARRIOR] {
            targets[hero_start + H_WARRIOR] = warrior_target_after;
            any_change = true;
        }

        // Cleric medic logic: prefer the lowest-HP wounded teammate's
        // current room over warrior-following. CLERIC_HEAL_FLOOR mirrors
        // config.dungeon.cleric_heal_floor (130.0) — the threshold above
        // which the cleric's heal verb doesn't fire, so there's no point
        // chasing teammates healthier than that.
        const H_CLERIC: usize = 1;
        const CLERIC_HEAL_FLOOR: f32 = 130.0;
        let cleric_medic_target: Option<u32> = {
            let mut lowest: Option<(usize, f32)> = None;
            for h in 0..dungeon::N_HEROES as usize {
                if h == H_CLERIC { continue; }
                if alive[hero_start + h] == 0 { continue; }
                let hp = hps[hero_start + h];
                if hp >= CLERIC_HEAL_FLOOR { continue; }
                match lowest {
                    Some((_, prev)) if hp >= prev => {}
                    _ => lowest = Some((h, hp)),
                }
            }
            lowest.and_then(|(h, _)| current[h]).map(|c| c.idx())
        };

        // Pass 3: every other hero. Followers cluster with the warrior;
        // the rogue scouts one step ahead; the cleric peels off to heal
        // the lowest-HP wounded teammate. Warrior dead → per-hero
        // frontier-greedy fallback.
        for h in 0..dungeon::N_HEROES as usize {
            if h == H_WARRIOR { continue; }
            let cand = match current[h] {
                Some(c) => c,
                None => continue,
            };
            let hp = hps[hero_start + h];
            let cur_target = targets[hero_start + h];
            let retreating = cur_target == spawn_idx && cand.idx() != spawn_idx;

            let new_target = if retreating && hp >= RESUME_HP {
                dungeon::pick_next_target(
                    &self.dungeon, cand, self.hero_state.rooms_visited[h],
                    tick, h as u32, self.seed,
                )
            } else if !retreating && hp < RETREAT_HP {
                dungeon::pick_retreat_target(
                    &self.dungeon, cand,
                    self.hero_known_rooms[h],
                    &self.cleared_rooms,
                    spawn_idx,
                )
            } else if !warrior_alive {
                // No leader — fall back to per-hero exploration.
                if cand.idx() == cur_target {
                    dungeon::pick_next_target(
                        &self.dungeon, cand, self.hero_state.rooms_visited[h],
                        tick, h as u32, self.seed,
                    )
                } else {
                    cur_target
                }
            } else if h == H_ROGUE {
                // Scout one room ahead of the party.
                warrior_target_after
            } else if h == H_CLERIC {
                // Medic: chase the wounded; default to warrior's room.
                cleric_medic_target.unwrap_or_else(|| {
                    warrior_current.map_or(cur_target, |c| c.idx())
                })
            } else {
                // Ranger / Mage — cluster with the warrior.
                warrior_current.map_or(cur_target, |c| c.idx())
            };
            if new_target != cur_target {
                targets[hero_start + h] = new_target;
                any_change = true;
            }
        }

        if any_change {
            self.state.gpu.queue.write_buffer(
                &self.state.agent_target_room_idx_buf, 0,
                bytemuck::cast_slice(&targets),
            );
        }

        // Per-tick hero waypoint computation. For each hero:
        //   * if current room is None (out of bounds / corridor): keep
        //     the existing waypoint (don't yank the hero around).
        //   * if current room == target room: waypoint = target room
        //     centroid (so heroes settle at the room's center).
        //   * else if current room is slot-adjacent to target room:
        //     waypoint = door cell between them.
        //   * else (non-adjacent — shouldn't happen with frontier-greedy
        //     advancement but possible after retreat): waypoint = current
        //     room centroid (let the hero re-pick a target next tick).
        let target_buf2 = self.state.agent_target_room_idx_buf.clone();
        let targets_post = read_agent_u32(&mut self.state, &target_buf2, agent_count);
        let wp_x_buf = self.state.agent_hero_waypoint_x_buf.clone();
        let wp_y_buf = self.state.agent_hero_waypoint_y_buf.clone();
        let mut wp_x = read_agent_f32(&mut self.state, &wp_x_buf, agent_count);
        let mut wp_y = read_agent_f32(&mut self.state, &wp_y_buf, agent_count);
        for h in 0..dungeon::N_HEROES as usize {
            let cur = match current[h] {
                Some(c) => c,
                None => continue,
            };
            let target_idx = targets_post[hero_start + h];
            let target_rx = (target_idx % dungeon::SLOTS_PER_ROW) as u32;
            let target_ry = (target_idx / dungeon::SLOTS_PER_ROW) as u32;
            let target_room = dungeon::RoomSlot::new(target_rx, target_ry);
            let waypoint = if cur == target_room {
                let c = target_room.centroid();
                (c[0], c[1])
            } else if let Some((dx, dy)) = dungeon::door_position(self.seed, cur, target_room) {
                (dx, dy)
            } else {
                let c = cur.centroid();
                (c[0], c[1])
            };
            wp_x[hero_start + h] = waypoint.0;
            wp_y[hero_start + h] = waypoint.1;
        }
        self.state.gpu.queue.write_buffer(
            &self.state.agent_hero_waypoint_x_buf, 0, bytemuck::cast_slice(&wp_x),
        );
        self.state.gpu.queue.write_buffer(
            &self.state.agent_hero_waypoint_y_buf, 0, bytemuck::cast_slice(&wp_y),
        );

        // Surface the warrior's target for the bridge to highlight.
        // Only meaningful while the warrior is alive AND the target
        // is a real present room (not the spawn-retreat sentinel
        // unless we want to highlight the retreat destination too).
        if warrior_alive {
            let t = targets[hero_start + H_WARRIOR];
            self.warrior_target_room = Some(t);
        } else {
            self.warrior_target_room = None;
        }
    }

    /// Boss-enrage trigger. When the boss agent first drops below 50%
    /// HP, write a one-time `+ALERT_BUMP` to the `alert` field of every
    /// enemy whose position is inside the boss-room slot. The .sim's
    /// existing alert-driven visuals (MAT_PANICKED tint) then kick in
    /// automatically — no rule changes required.
    ///
    /// Threshold is 50% of the boss HP override (`BOSS_HP=800` in
    /// `try_new`), i.e. 400. Guarded by `boss_enraged` so the bump
    /// fires exactly once per run.
    fn maybe_enrage_boss_room(&mut self) {
        if self.boss_enraged {
            return;
        }
        let boss_slot = match self.boss_slot {
            Some(s) => s as usize,
            None => return,
        };
        let boss = &self.agents[boss_slot];
        const BOSS_ENRAGE_HP: f32 = 200.0;
        const ALERT_BUMP: u32 = 10;
        if !(boss.alive && boss.hp < BOSS_ENRAGE_HP && boss.hp > 0.0) {
            return;
        }
        let n = self.agent_count as usize;
        let alert_buf = self.state.agent_alert_buf.clone();
        let mut alert = read_agent_u32(&mut self.state, &alert_buf, self.agent_count);
        let boss_rx = self.dungeon.boss_room.rx;
        let boss_ry = self.dungeon.boss_room.ry;
        let sw = dungeon::SLOT_WIDTH;
        let mut bumped = 0u32;
        for slot in 0..n {
            let a = &self.agents[slot];
            if !a.alive { continue; }
            if a.creature_type == dungeon::CT_HERO { continue; }
            let rx = (a.pos.x / sw as f32).floor() as i32;
            let ry = (a.pos.y / sw as f32).floor() as i32;
            if rx == boss_rx as i32 && ry == boss_ry as i32 {
                alert[slot] = alert[slot].saturating_add(ALERT_BUMP);
                bumped += 1;
            }
        }
        self.state.gpu.queue.write_buffer(
            &self.state.agent_alert_buf, 0, bytemuck::cast_slice(&alert),
        );
        self.boss_enraged = true;
        eprintln!(
            "[viewer_runtime] boss enraged @ tick {} (hp={:.0}) — +{ALERT_BUMP} alert on {bumped} boss-room enemies",
            self.state.tick, boss.hp,
        );
    }

    /// Pretty-print a per-run scorecard: tick budget, verdict, per-hero
    /// fates, kill count + boss-kill tick, max alert observed. Called once
    /// at termination by `step()`.
    fn print_run_summary(&self, heroes_alive: u32, enemies_alive: u32) {
        let role_names = ["Warrior", "Cleric", "Ranger", "Mage", "Rogue"];
        eprintln!("==== run summary (seed=0x{:X}) ====", self.seed);
        eprintln!("  ticks   : {}", self.state.tick);
        eprintln!(
            "  verdict : {}",
            if heroes_alive > 0 { "DUNGEON CLEARED" } else { "TPK" },
        );
        eprintln!(
            "  heroes  : {}/{} alive",
            heroes_alive, dungeon::N_HEROES,
        );
        for (h, role) in role_names.iter().enumerate() {
            match self.hero_death_tick[h] {
                Some(t) => eprintln!("    {role:>7}: died @ tick {t}"),
                None => eprintln!("    {role:>7}: alive"),
            }
        }
        eprintln!(
            "  kills   : {} / {} enemies",
            self.total_kills,
            self.total_kills + enemies_alive,
        );
        match self.boss_death_tick {
            Some(t) => eprintln!("  boss    : killed @ tick {t}"),
            None => match self.boss_slot {
                Some(_) => eprintln!("  boss    : SURVIVED"),
                None => eprintln!("  boss    : (no boss this roll)"),
            },
        }
        eprintln!("  max alert (any enemy): {}", self.max_alert_seen);
        eprintln!("==========================================");
    }

    /// Read agent SoA back from sim GPU into the viewer's host cache.
    fn refresh_snapshot(&mut self) {
        let n = self.agent_count;
        // Clone the buffer handles up front to release the &self.state
        // borrow before re-borrowing &mut self.state in the readback
        // helper. wgpu::Buffer is Arc-backed; clone is cheap.
        let alive_buf = self.state.agent_alive_buf.clone();
        let type_buf = self.state.agent_creature_type_buf.clone();
        let role_buf = self.state.agent_role_buf.clone();
        let hp_buf = self.state.agent_hp_buf.clone();
        let alert_buf = self.state.agent_alert_buf.clone();
        let stealth_buf = self.state.agent_stealth_until_tick_buf.clone();
        let positions = read_positions(&mut self.state, n);
        let alive = read_agent_u32(&mut self.state, &alive_buf, n);
        let types = read_agent_u32(&mut self.state, &type_buf, n);
        let role  = read_agent_u32(&mut self.state, &role_buf, n);
        let hps   = read_agent_f32(&mut self.state, &hp_buf, n);
        let alert = read_agent_u32(&mut self.state, &alert_buf, n);
        let stealth = read_agent_u32(&mut self.state, &stealth_buf, n);
        // Per-tick damage diff: flash agents that lost meaningful HP since
        // the last refresh. FLASH_FRAMES of 4 at the ~100ms sim tick gives
        // a visible ~400ms white pop on each hit.
        // (cleared_rooms recompute happens at the end of this loop, after
        // self.agents is fully refreshed.)
        const FLASH_DELTA: f32 = 3.0;
        const FLASH_FRAMES: u8 = 4;
        let cur_tick = self.state.tick;
        let hero_start = (n - dungeon::N_HEROES) as usize;
        for slot in 0..n as usize {
            // Tick down any in-progress flashes first so this frame consumes
            // one frame of the existing window before we test for new events.
            if self.flash_ticks[slot] > 0 {
                self.flash_ticks[slot] -= 1;
            }
            if self.heal_ticks[slot] > 0 {
                self.heal_ticks[slot] -= 1;
            }
            let prev = self.prev_hp[slot];
            let cur = hps[slot];
            if alive[slot] != 0 && prev - cur >= FLASH_DELTA {
                self.flash_ticks[slot] = FLASH_FRAMES;
                // Find the nearest opposite-team alive agent as the
                // (presumed) source of this hit. ~440 agents total ⇒
                // 440² = 200k checks per tick worst case (one per
                // damaged agent × all candidates) — but damage events
                // are sparse (typically <10 per tick), so the
                // amortized cost is fine.
                let target_team_hero = types[slot] == dungeon::CT_HERO;
                let target_pos = Vec3::new(
                    positions[slot][0], positions[slot][1], positions[slot][2],
                );
                let mut best_d2 = f32::MAX;
                let mut best_src: Option<Vec3> = None;
                for s2 in 0..n as usize {
                    if alive[s2] == 0 { continue; }
                    let other_hero = types[s2] == dungeon::CT_HERO;
                    if other_hero == target_team_hero { continue; }
                    let dx = positions[s2][0] - positions[slot][0];
                    let dy = positions[s2][1] - positions[slot][1];
                    let dz = positions[s2][2] - positions[slot][2];
                    let d2 = dx*dx + dy*dy + dz*dz;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_src = Some(Vec3::new(
                            positions[s2][0], positions[s2][1], positions[s2][2],
                        ));
                    }
                }
                if let Some(src) = best_src {
                    self.projectiles.push(Projectile {
                        source: src,
                        target: target_pos,
                        ttl: PROJECTILE_TTL,
                        hero_shot: !target_team_hero,
                    });
                }
            }
            // Heal flash: same FLASH_DELTA threshold (≥3 HP), but on HP
            // increase. Cleric's Mend / Heal verbs lift HP by 20+ at a
            // time so this easily crosses the threshold.
            if alive[slot] != 0 && cur - prev >= FLASH_DELTA {
                self.heal_ticks[slot] = FLASH_FRAMES;
            }
            self.prev_hp[slot] = cur;

            // Death-event detection: track per-hero death tick + count
            // enemy kills. Compare prev-snapshot alive vs new alive.
            let was_alive = self.agents[slot].alive;
            let now_alive = alive[slot] != 0;
            if was_alive && !now_alive {
                self.death_ticks[slot] = 1;
                if types[slot] == dungeon::CT_HERO {
                    let h_idx = slot - hero_start;
                    if h_idx < dungeon::N_HEROES as usize
                        && self.hero_death_tick[h_idx].is_none()
                    {
                        self.hero_death_tick[h_idx] = Some(cur_tick);
                    }
                } else {
                    self.total_kills += 1;
                    if Some(slot as u32) == self.boss_slot
                        && self.boss_death_tick.is_none()
                    {
                        self.boss_death_tick = Some(cur_tick);
                    }
                }
            } else if !now_alive && self.death_ticks[slot] > 0 && self.death_ticks[slot] < DEATH_FADE_TICKS {
                self.death_ticks[slot] += 1;
            } else if now_alive {
                self.death_ticks[slot] = 0;
            }
            if alert[slot] > self.max_alert_seen {
                self.max_alert_seen = alert[slot];
            }

            // Facing — delta from previous tick. Below FACING_EPS_SQ we
            // keep the last-observed direction so stationary agents don't
            // jitter (and the very first frame defaults to (0, 1)).
            const FACING_EPS_SQ: f32 = 0.001;
            let dx = positions[slot][0] - self.prev_positions[slot][0];
            let dy = positions[slot][1] - self.prev_positions[slot][1];
            // ([0]/[1] are sim XY; [2] is the sim Z plane = vertical for the
            // bridge. Renderer lerps between this and current for smoothing.)
            let len_sq = dx * dx + dy * dy;
            let facing = if len_sq > FACING_EPS_SQ {
                let inv = 1.0 / len_sq.sqrt();
                let f = [dx * inv, dy * inv];
                self.last_facing[slot] = f;
                f
            } else {
                self.last_facing[slot]
            };
            self.prev_positions[slot] = [positions[slot][0], positions[slot][1], positions[slot][2]];

            self.agents[slot] = AgentSnapshot {
                pos: Vec3::new(positions[slot][0], positions[slot][1], positions[slot][2]),
                alive: now_alive,
                creature_type: types[slot],
                role: role[slot],
                hp: hps[slot],
                alert: alert[slot],
                stealth_until_tick: stealth[slot],
                facing_xy: facing,
            };
        }

        // Per-tick cleared-rooms scan: bucket alive non-hero agents by the
        // dungeon room they're in. Rooms with zero alive enemies (and at
        // least one room cell in dungeon.floor_cells) flip to cleared.
        // Once cleared, stays cleared — bodies still on the floor count
        // as cleared, even though the death_fade keeps the mesh visible
        // for a few frames.
        let mut alive_per_room: std::collections::BTreeMap<u32, u32> =
            std::collections::BTreeMap::new();
        for room in &self.dungeon.rooms {
            alive_per_room.insert(room.idx(), 0);
        }
        let sw = dungeon::SLOT_WIDTH as f32;
        for a in &self.agents {
            if !a.alive || a.creature_type == dungeon::CT_HERO { continue; }
            let rx = (a.pos.x / sw).floor() as i32;
            let ry = (a.pos.y / sw).floor() as i32;
            if rx < 0 || ry < 0
                || (rx as u32) >= dungeon::SLOTS_PER_ROW
                || (ry as u32) >= dungeon::SLOTS_PER_ROW
            {
                continue;
            }
            let idx = (ry as u32) * dungeon::SLOTS_PER_ROW + (rx as u32);
            if let Some(c) = alive_per_room.get_mut(&idx) { *c += 1; }
        }
        for (room_idx, count) in &alive_per_room {
            if *count == 0 {
                self.cleared_rooms.insert(*room_idx);
            }
        }

        // Tick down projectile TTLs; retain only live entries.
        for proj in &mut self.projectiles {
            proj.ttl = proj.ttl.saturating_sub(1);
        }
        self.projectiles.retain(|p| p.ttl > 0);
    }
}

// ---------------------------------------------------------------------
// VoxelBridge — CPU-side world grid, painted from terrain + agents,
// re-uploaded to a Vulkan texture each frame for voxel_engine to
// render.
// ---------------------------------------------------------------------

/// Cell extent of the bridge's voxel grid. Matches the dungeon's
/// `GRID_X × GRID_Y × GRID_Z` plus a few cells of vertical headroom
/// for agent splats above the floor.
pub const BRIDGE_DIM_X: u32 = GRID_X;
pub const BRIDGE_DIM_Y: u32 = GRID_Z + 4; // vertical (renderer Y-up); GRID_Z + headroom for agent splats
pub const BRIDGE_DIM_Z: u32 = GRID_Y; // depth (sim's horizontal Y axis)

/// World-space `(0, 0, 0)` corner of the bridge grid. We anchor at
/// world origin; sim positions are in `[0, GRID_X)` so they map
/// directly into cells.
pub const BRIDGE_WORLD_ORIGIN: Vec3 = Vec3::new(0.0, 0.0, 0.0);
pub const BRIDGE_CELL_SIZE: f32 = 1.0;

/// Paints the dungeon walls + floor + agents into a `VoxelGrid` and
/// re-uploads it to a `GpuVoxelTexture` each frame. Single object,
/// per-cell coloring via `palette_tex[voxel_id]` in voxel_engine's
/// fragment shader.
pub struct VoxelBridge {
    cpu_grid: VoxelGrid,
    /// Cells painted by agents last frame — cleared before this
    /// frame's paint to keep the agent layer sparse.
    last_agent_cells: Vec<(u32, u32, u32)>,
    gpu_tex: Option<GpuVoxelTexture>,
    alloc: VulkanAllocator,
    palette_rgba: [[u8; 4]; 256],
}

impl VoxelBridge {
    pub fn new(ctx: &VulkanContext, app: &ViewerApp) -> Result<Self> {
        let palette = build_palette();
        let palette_rgba = palette.to_rgba();
        let mut cpu_grid = VoxelGrid::new(BRIDGE_DIM_X, BRIDGE_DIM_Y, BRIDGE_DIM_Z);

        // Paint the dungeon: floor cells get MAT_FLOOR at z=0; wall
        // cells get MAT_WALL filling z ∈ [0, ROOM_INTERIOR_Z). This
        // matches the runtime's voxel_terrain contents seeded by
        // `seed_voxel_dungeon`.
        for x in 0..GRID_X {
            for y in 0..GRID_Y {
                if app.floor_cells.contains(&(x, y)) {
                    // sim (x, y, 0) → renderer (x, 0, y): floor sits at vertical=0, depth=sim_y
                    let mat = floor_mat_for_cell(app, x, y);
                    cpu_grid.set(x, 0, y, mat);
                } else {
                    // walls extend up the vertical (renderer Y) axis.
                    // Boss-room walls use MAT_BOSS_WALL for visual
                    // distinction; everything else is default MAT_WALL.
                    let wall_mat = wall_mat_for_cell(app, x, y);
                    for vert in 0..ROOM_INTERIOR_Z.min(GRID_Z) {
                        cpu_grid.set(x, vert, y, wall_mat);
                    }
                }
            }
        }

        let mut alloc = VulkanAllocator::new(ctx)?;
        let gpu_tex = upload_grid_to_gpu(ctx, &mut alloc, &cpu_grid, &palette_rgba)?;
        Ok(Self {
            cpu_grid,
            last_agent_cells: Vec::with_capacity(2048),
            gpu_tex: Some(gpu_tex),
            alloc,
            palette_rgba,
        })
    }

    /// Per-frame refresh: clear last frame's agent cells, paint new
    /// agent positions, re-upload texture (regenerates mips).
    pub fn refresh(&mut self, ctx: &VulkanContext, app: &ViewerApp) -> Result<()> {
        // If the sim ended this frame (or any earlier frame and we
        // haven't yet tinted the floor), repaint every floor cell with
        // the outcome color. Run unconditionally per-frame because the
        // cell count is tiny and the alternative (one-shot dirty flag)
        // is more state to manage.
        if let Some(victory) = app.outcome {
            let outcome_mat = if victory { MAT_VICTORY_FLOOR } else { MAT_DEFEAT_FLOOR };
            for &(x, y) in &app.floor_cells {
                self.cpu_grid.set(x, 0, y, outcome_mat);
            }
        }

        // Clear last frame's agents — restore floor or air based on
        // dungeon membership. Walls aren't in the agent layer (they
        // sit at z ∈ [0, ROOM_INTERIOR_Z)) so we don't touch them.
        for &(x, vert, depth) in &self.last_agent_cells {
            // Restore each agent cell. Note: post axis-swap, the bridge
            // grid is (x, vertical, depth). Floor occupancy is keyed on
            // sim coords (x, sim_y) which now equals (x, depth).
            if app.floor_cells.contains(&(x, depth)) {
                if vert == 0 {
                    let mat = if let Some(victory) = app.outcome {
                        if victory { MAT_VICTORY_FLOOR } else { MAT_DEFEAT_FLOOR }
                    } else {
                        floor_mat_for_cell(app, x, depth)
                    };
                    self.cpu_grid.set(x, vert, depth, mat);
                } else {
                    self.cpu_grid.set(x, vert, depth, MAT_AIR);
                }
            } else {
                if vert < ROOM_INTERIOR_Z.min(GRID_Z) {
                    self.cpu_grid.set(x, vert, depth, MAT_WALL);
                } else {
                    self.cpu_grid.set(x, vert, depth, MAT_AIR);
                }
            }
        }
        self.last_agent_cells.clear();

        // Paint each alive agent as a 2x2x2 splat one cell above the
        // floor (z ∈ [1, 3)) so they sit visibly on the floor without
        // sinking. Out-of-bounds cells are skipped.
        // Alert thresholds — config.stealth.alert_step_ally_death = 2 and
        // alert_step_missing = 1, so a few events suffice to cross LO; HI
        // takes sustained engagement.
        const ALERT_TINT_LO: u32 = 4;
        const ALERT_TINT_HI: u32 = 12;
        for (slot_idx, agent) in app.agents().iter().enumerate() {
            if !agent.alive {
                continue;
            }
            let sim_tick = app.state.tick as u32;
            let stealth_active = agent.creature_type == dungeon::CT_HERO
                && agent.role == 5 /* Rogue */
                && agent.stealth_until_tick > sim_tick;
            let flashing = app.flash_ticks[slot_idx] > 0;
            let healing = app.heal_ticks[slot_idx] > 0;
            // Damage flash trumps heal flash — combat hits are the
            // higher-signal event when both fire the same tick.
            let mat = if flashing {
                MAT_FLASH
            } else if healing {
                MAT_HEAL
            } else if app.boss_slot == Some(slot_idx as u32) {
                MAT_BOSS
            } else if stealth_active {
                MAT_STEALTHED
            } else if agent.creature_type != dungeon::CT_HERO
                && agent.alert >= ALERT_TINT_HI
            {
                MAT_PANICKED
            } else if agent.creature_type != dungeon::CT_HERO
                && agent.alert >= ALERT_TINT_LO
            {
                MAT_ALERTED
            } else {
                material_for(agent.creature_type, agent.role)
            };
            let cx = agent.pos.x.floor() as i32;
            let cd = agent.pos.y.floor() as i32; // sim_y → renderer depth
            // Voxel splat painting: meshes are now the primary agent
            // visualization, so we suppress the 2x2x2 cube for ordinary
            // alive frames. Damage flash + heal flash + boss + alert
            // states still paint the splat so combat events remain
            // legible at the splat layer (the mesh-pass tint blends
            // are subtle compared to a full bright cube).
            let paint_splat = flashing
                || healing
                || app.boss_slot == Some(slot_idx as u32)
                || (agent.creature_type != dungeon::CT_HERO
                    && agent.alert >= ALERT_TINT_LO);
            if paint_splat {
                // 2x2x2 splat: span (x, x+1) × vertical (1..3) × depth (cd, cd+1)
                for dx in 0..2 {
                    for d_depth in 0..2 {
                        for d_vert in 0..2 {
                            let x = cx + dx;
                            let depth = cd + d_depth;
                            let vert = 1 + d_vert;
                            if x < 0 || depth < 0 || vert < 0 {
                                continue;
                            }
                            let (x, vert, depth) = (x as u32, vert as u32, depth as u32);
                            if x >= BRIDGE_DIM_X || vert >= BRIDGE_DIM_Y || depth >= BRIDGE_DIM_Z {
                                continue;
                            }
                            self.cpu_grid.set(x, vert, depth, mat);
                            self.last_agent_cells.push((x, vert, depth));
                        }
                    }
                }
            }

            // Floor shadow right under the agent — single cell at vert=0
            // (replaces the floor tile for this frame; restored on next
            // clear pass). Skipped for dying agents so the dim mesh
            // doesn't double-darken the cell beneath.
            if !flashing && app.flash_ticks[slot_idx] == 0 {
                let cx = agent.pos.x.floor() as i32;
                let cd = agent.pos.y.floor() as i32;
                if cx >= 0 && cd >= 0 {
                    let (sx, sd) = (cx as u32, cd as u32);
                    if sx < BRIDGE_DIM_X && sd < BRIDGE_DIM_Z
                        && app.floor_cells.contains(&(sx, sd))
                    {
                        self.cpu_grid.set(sx, 0, sd, MAT_AGENT_SHADOW);
                        self.last_agent_cells.push((sx, 0, sd));
                    }
                }
            }

            // Heal column VFX — vertical line of mint cells rising above
            // the agent for the duration of the heal flash. Reads as a
            // visible "healing magic landed here" beam, distinct from
            // the per-mesh tint blend on the rendered character.
            if healing {
                let cx = agent.pos.x.floor() as i32;
                let cd = agent.pos.y.floor() as i32;
                if cx >= 0 && cd >= 0 {
                    let (sx, sd) = (cx as u32, cd as u32);
                    if sx < BRIDGE_DIM_X && sd < BRIDGE_DIM_Z {
                        for vert in 4u32..8u32 {
                            if vert >= BRIDGE_DIM_Y { break; }
                            self.cpu_grid.set(sx, vert, sd, MAT_HEAL);
                            self.last_agent_cells.push((sx, vert, sd));
                        }
                    }
                }
            }

            // HP bar — 3 stacked cells above heroes only. Green/yellow/red
            // light up while the corresponding HP fraction is above
            // the tier threshold (33% / 67%).
            if agent.creature_type == dungeon::CT_HERO {
                // Heroes' max_hp is 200 (sim init); but the boss-HP override
                // doesn't apply to heroes, so use 200 as the divisor.
                const HERO_MAX_HP: f32 = 200.0;
                let frac = (agent.hp / HERO_MAX_HP).clamp(0.0, 1.0);
                let cx = agent.pos.x.floor() as i32;
                let cd = agent.pos.y.floor() as i32;
                // Place bar one cell to the side of the head so it doesn't
                // get overwritten on the next agent-cell clear (the splat
                // clear only restores cells listed in last_agent_cells).
                let bar_x = cx;
                let bar_depth = cd;
                for (tier, threshold, tier_mat) in [
                    (0u32, 0.0f32, MAT_HP_RED),
                    (1, 0.33, MAT_HP_YELLOW),
                    (2, 0.67, MAT_HP_GREEN),
                ] {
                    if frac > threshold {
                        let vert = (4 + tier) as i32;
                        if bar_x >= 0 && bar_depth >= 0 {
                            let (x, depth) = (bar_x as u32, bar_depth as u32);
                            if x < BRIDGE_DIM_X && (vert as u32) < BRIDGE_DIM_Y
                                && depth < BRIDGE_DIM_Z
                            {
                                self.cpu_grid.set(x, vert as u32, depth, tier_mat);
                                self.last_agent_cells.push((x, vert as u32, depth));
                            }
                        }
                    }
                }
            }
        }

        // Projectile beams — paint a sparse line of cells from each
        // active projectile's source toward its target, at vert=2 (over
        // the floor + agent splat layer). Cells tracked in
        // last_agent_cells so they clear on the next frame.
        for proj in &app.projectiles {
            // sim (x,y,z) -> renderer (x, z, y) Z↔Y swap
            let s_x = proj.source.x;
            let s_z = proj.source.y;
            let t_x = proj.target.x;
            let t_z = proj.target.y;
            let dx = t_x - s_x;
            let dz = t_z - s_z;
            let len = (dx * dx + dz * dz).sqrt();
            if len < 0.5 { continue; }
            // Step every ~1 cell along the line. Limited to 32 cells max
            // to bound cost when projectile spans a long distance.
            let steps = (len.ceil() as i32).min(32);
            let mat = if proj.hero_shot {
                MAT_PROJECTILE_HERO
            } else {
                MAT_PROJECTILE_ENEMY
            };
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let px = (s_x + dx * t).floor() as i32;
                let pz = (s_z + dz * t).floor() as i32;
                if px < 0 || pz < 0 { continue; }
                let (x, depth) = (px as u32, pz as u32);
                if x >= BRIDGE_DIM_X || depth >= BRIDGE_DIM_Z { continue; }
                let vert: u32 = 2;
                if vert < BRIDGE_DIM_Y {
                    self.cpu_grid.set(x, vert, depth, mat);
                    self.last_agent_cells.push((x, vert, depth));
                }
            }
        }

        // Destroy + re-upload to refresh the mip chain. voxel_engine's
        // gbuffer.frag uses mip3 for empty-block jump-skip; without a
        // refresh, agent cells painted after the initial upload would
        // render at one cell's worth of geometry only.
        if let Some(old) = self.gpu_tex.take() {
            old.destroy(ctx, &mut self.alloc);
        }
        let new_tex =
            upload_grid_to_gpu(ctx, &mut self.alloc, &self.cpu_grid, &self.palette_rgba)?;
        self.gpu_tex = Some(new_tex);
        Ok(())
    }

    /// Single render-object tuple for `Renderer::render_frame_gpu`.
    /// Per-cell color comes from the palette LUT (per-cell material
    /// id), not this tuple's `palette_color`.
    pub fn render_object(
        &self,
    ) -> Option<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> {
        let dims = [
            BRIDGE_DIM_X as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Y as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Z as f32 * BRIDGE_CELL_SIZE,
        ];
        let pos: [f32; 3] = BRIDGE_WORLD_ORIGIN.into();
        self.gpu_tex
            .as_ref()
            .map(|t| (t, [1.0, 1.0, 1.0, 0.5], pos, dims))
    }

    /// Drop the GPU texture explicitly. winit's run_app doesn't drop
    /// the app on exit; without this the texture leaks until process
    /// teardown.
    pub fn destroy(mut self, ctx: &VulkanContext) {
        if let Some(tex) = self.gpu_tex.take() {
            tex.destroy(ctx, &mut self.alloc);
        }
    }
}

// ---------------------------------------------------------------------
// GPU readback helpers — copy from sim's wgpu buffers to host Vec.
// Mirrors `dungeon_horde_pin`'s helpers.
// ---------------------------------------------------------------------

/// Decide the floor material for a single cell — boss room takes
/// priority, then cleared-room dim tint, else default tan. Outcome
/// tints (victory/defeat) are checked at the call site since they
/// override everything when set.
fn floor_mat_for_cell(app: &ViewerApp, x: u32, y: u32) -> u8 {
    let sw = dungeon::SLOT_WIDTH;
    let rx = x / sw;
    let ry = y / sw;
    let room_idx = ry * dungeon::SLOTS_PER_ROW + rx;
    // Fog of war: rooms no hero has personally entered render in
    // MAT_FLOOR_FOG. Beats every other tint — even the boss floor's
    // distinctive red is hidden until the party discovers it.
    if (app.party_known_rooms & (1u64 << room_idx)) == 0 {
        return MAT_FLOOR_FOG;
    }
    if rx == app.dungeon.boss_room.rx && ry == app.dungeon.boss_room.ry {
        return MAT_BOSS_FLOOR;
    }
    // Target highlight beats cleared so we can re-highlight a room
    // the warrior chose to revisit.
    if app.warrior_target_room == Some(room_idx) {
        return MAT_FLOOR_TARGET;
    }
    if app.cleared_rooms.contains(&room_idx) {
        return MAT_FLOOR_CLEARED;
    }
    MAT_FLOOR
}

/// Wall material for a single cell — matches the boss-room floor's
/// red-stone aesthetic when inside boss-room bounds, else default gray.
fn wall_mat_for_cell(app: &ViewerApp, x: u32, y: u32) -> u8 {
    let sw = dungeon::SLOT_WIDTH;
    let rx = x / sw;
    let ry = y / sw;
    if rx == app.dungeon.boss_room.rx && ry == app.dungeon.boss_room.ry {
        MAT_BOSS_WALL
    } else {
        MAT_WALL
    }
}

fn read_positions(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<[f32; 4]> {
    let n = agent_count as usize;
    let bytes = (n as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::pos_readback") },
    );
    let buf = state.agent_pos_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[[f32; 4]] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_u32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    agent_count: u32,
) -> Vec<u32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::u32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_f32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    agent_count: u32,
) -> Vec<f32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::f32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
