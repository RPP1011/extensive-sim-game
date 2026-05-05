//! Per-fixture runtime for the disease_spread sim — 22nd REAL fixture
//! and FIRST sim with SIR-style epidemic dynamics.
//!
//! Sibling to crafting_diffusion (information spread through trade)
//! but with **mortality**. All logic is CPU-side (no DSL/WGSL),
//! mirroring the "get it working first" pre-authorization.
//!
//! ## State encoding
//!
//! Per-agent `state` field (also mirrored in `mana` for the
//! visualization snapshot path):
//!
//! - `0 = Susceptible (S)` — never infected, can be infected by
//!   neighbouring infected agents.
//! - `1 = Infected (I)`    — currently sick, can transmit; rolls
//!   mortality each tick; recovers after `INFECTION_DURATION`.
//! - `2 = Recovered (R)`   — immune, no further transmission.
//! - `3 = Dead (D)`        — `alive=false`; no movement, no
//!   transmission, frozen in place as a tombstone.
//!
//! ## Per-tick rules
//!
//! 1. **Movement** — every still-alive agent does a small random
//!    walk in the unit square `[0, WORLD_SIZE)`. Movement is
//!    deterministic: the random kick comes from
//!    `per_agent_u32(seed, agent_id, tick, b"walk_x"/b"walk_y")`.
//! 2. **Transmission** — every Susceptible agent scans its
//!    neighbourhood (O(N²) brute force; 200 agents = 40k pair
//!    checks/tick which is plenty fast on CPU). For each Infected
//!    agent within `INFECTION_RADIUS`, it rolls
//!    `per_agent_u32(seed, agent_id, tick, b"infect_<src>")` and
//!    if the value modulo a denominator is below the per-pair
//!    infection probability threshold, transitions to Infected.
//!    `infection_tick` records the tick of infection so the
//!    recovery timer is per-agent.
//! 3. **Lifecycle** — every Infected agent:
//!    - rolls mortality with `b"mortality"` purpose; on hit, sets
//!      `alive=false` and `state=Dead`.
//!    - otherwise, if `tick - infection_tick >= INFECTION_DURATION`,
//!      transitions to Recovered (immune).
//!
//! ## Constants
//!
//! See module-level `pub const` block below; all tunable from
//! call sites (the bin harness reuses these defaults).

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use glam::Vec3;

/// Side length of the square world the agents wander in (world
/// coordinates, not grid cells). 50×50 with 200 agents gives
/// density ≈ 0.08 agents per unit² — high enough that patient
/// zero finds neighbours within a few dozen ticks instead of
/// wandering in isolation for hundreds.
pub const WORLD_SIZE: f32 = 50.0;

/// Per-tick max step distance for the random-walk physics.
pub const STEP_SIZE: f32 = 1.0;

/// Two agents within this distance count as a contact for
/// transmission purposes. Combined with WORLD_SIZE=50 and
/// 200 agents this puts ~4 neighbours in range on average.
pub const INFECTION_RADIUS: f32 = 3.0;
pub const INFECTION_RADIUS_SQ: f32 = INFECTION_RADIUS * INFECTION_RADIUS;

/// Per-contact per-tick infection probability (numerator / 1_000_000).
/// 50_000 / 1_000_000 = 0.05 = 5% — matches the prompt's
/// suggested infection_prob.
pub const INFECTION_PROB_NUM: u32 = 50_000;
pub const PROB_DENOM: u32 = 1_000_000;

/// Per-tick mortality probability for Infected agents
/// (1_000 / 1_000_000 = 0.001 = 0.1%). Matches the prompt's
/// suggested mortality. Over a 100-tick infection this gives
/// roughly a 10% case fatality rate.
pub const MORTALITY_PROB_NUM: u32 = 1_000;

/// How many ticks an agent stays in the Infected state before
/// recovering. Matches the SIR model's `1/gamma` recovery time.
pub const INFECTION_DURATION: u64 = 100;

/// SIR epidemic states.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum InfectionState {
    Susceptible = 0,
    Infected = 1,
    Recovered = 2,
    Dead = 3,
}

impl InfectionState {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Susceptible,
            1 => Self::Infected,
            2 => Self::Recovered,
            3 => Self::Dead,
            _ => Self::Dead, // shouldn't happen; treat as terminal
        }
    }
}

/// Population-level SIR snapshot: (S, I, R, D) counts.
#[derive(Copy, Clone, Debug, Default)]
pub struct SirCounts {
    pub susceptible: u32,
    pub infected: u32,
    pub recovered: u32,
    pub dead: u32,
}

/// CPU-side SIR sim state. All fields are public to the trait
/// implementation; outside callers go through the `CompiledSim` /
/// fixture-specific accessors.
pub struct DiseaseSpreadState {
    seed: u64,
    tick: u64,
    agent_count: u32,

    // -- Per-agent SoA --
    positions: Vec<Vec3>,
    /// SIR state: 0=S, 1=I, 2=R, 3=D. Mirrored into `mana` field
    /// of the snapshot for the convention described in the prompt.
    state: Vec<u32>,
    /// Tick at which each agent became infected (only meaningful
    /// when `state[i] == Infected`). Used by the recovery timer.
    infection_tick: Vec<u64>,
    /// 1 = alive, 0 = dead. Flips to 0 only on `Dead` transition;
    /// Susceptible/Infected/Recovered are all "alive".
    alive: Vec<u32>,

    /// Reusable scratch buffer for the per-tick state mutation pass
    /// — we read from `state` while iterating but write to this
    /// buffer to keep the per-tick semantics order-independent
    /// (every Susceptible sees the SAME set of Infected agents,
    /// not a partially-mutated set). Same for `alive` and
    /// `infection_tick`.
    next_state: Vec<u32>,
    next_alive: Vec<u32>,
    next_infection_tick: Vec<u64>,

    /// Running max of the Infected count + the tick at which that
    /// peak occurred. Updated lazily by the `sir_counts()` accessor.
    peak_infected: u32,
    peak_tick: u64,
}

impl DiseaseSpreadState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;

        // Seed initial positions deterministically using the same
        // per_agent_u32 function we use for everything else.
        let mut positions = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let id = agent_id(slot);
            let rx = per_agent_u32(seed, id, 0, b"init_x");
            let ry = per_agent_u32(seed, id, 0, b"init_y");
            let x = (rx as f32 / u32::MAX as f32) * WORLD_SIZE;
            let y = (ry as f32 / u32::MAX as f32) * WORLD_SIZE;
            positions.push(Vec3::new(x, y, 0.0));
        }

        // 199 Susceptible + 1 patient zero (slot 0).
        let mut state = vec![InfectionState::Susceptible as u32; n];
        let mut infection_tick = vec![0u64; n];
        if n > 0 {
            state[0] = InfectionState::Infected as u32;
            infection_tick[0] = 0;
        }
        let alive = vec![1u32; n];

        let next_state = state.clone();
        let next_alive = alive.clone();
        let next_infection_tick = infection_tick.clone();

        Self {
            seed,
            tick: 0,
            agent_count,
            positions,
            state,
            infection_tick,
            alive,
            next_state,
            next_alive,
            next_infection_tick,
            peak_infected: 1,
            peak_tick: 0,
        }
    }

    /// Aggregate (S, I, R, D) counts. Called by the harness for the
    /// per-50-tick trace + termination check. Side-effect-free.
    pub fn sir_counts(&self) -> SirCounts {
        let mut c = SirCounts::default();
        for &s in &self.state {
            match InfectionState::from_u32(s) {
                InfectionState::Susceptible => c.susceptible += 1,
                InfectionState::Infected => c.infected += 1,
                InfectionState::Recovered => c.recovered += 1,
                InfectionState::Dead => c.dead += 1,
            }
        }
        c
    }

    pub fn peak_infected(&self) -> u32 {
        self.peak_infected
    }

    pub fn peak_tick(&self) -> u64 {
        self.peak_tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Per-agent state read (mostly for tests / introspection).
    pub fn state(&self) -> &[u32] {
        &self.state
    }

    pub fn alive(&self) -> &[u32] {
        &self.alive
    }
}

/// Helper — slot 0..n maps to AgentId 1..=n.
fn agent_id(slot: u32) -> AgentId {
    AgentId::new(slot.saturating_add(1)).expect("slot+1 fits NonZeroU32")
}

impl CompiledSim for DiseaseSpreadState {
    fn step(&mut self) {
        // Snapshot pre-mutation buffers — the next_* scratch buffers
        // hold the post-tick state so concurrent reads of `state`
        // during the transmission pass see a consistent pre-tick
        // population (no order-of-iteration sensitivity).
        self.next_state.copy_from_slice(&self.state);
        self.next_alive.copy_from_slice(&self.alive);
        self.next_infection_tick.copy_from_slice(&self.infection_tick);

        let n = self.agent_count;
        let tick = self.tick;
        let seed = self.seed;

        // --- 1. Movement: random walk for everyone still alive ---
        for slot in 0..n {
            if self.alive[slot as usize] == 0 {
                continue;
            }
            let id = agent_id(slot);
            let rx = per_agent_u32(seed, id, tick, b"walk_x");
            let ry = per_agent_u32(seed, id, tick, b"walk_y");
            // Map u32 -> [-1, 1] then scale by step size.
            let dx = ((rx as f32 / u32::MAX as f32) * 2.0 - 1.0) * STEP_SIZE;
            let dy = ((ry as f32 / u32::MAX as f32) * 2.0 - 1.0) * STEP_SIZE;
            let p = &mut self.positions[slot as usize];
            p.x = (p.x + dx).clamp(0.0, WORLD_SIZE);
            p.y = (p.y + dy).clamp(0.0, WORLD_SIZE);
        }

        // --- 2. Transmission: each Susceptible scans neighbours ---
        // O(N^2) brute force; n=200 means 40k pair checks/tick, well
        // within budget. A spatial hash would be the obvious upgrade
        // for n>>1000.
        for slot in 0..n {
            let i = slot as usize;
            if InfectionState::from_u32(self.state[i]) != InfectionState::Susceptible {
                continue;
            }
            let id = agent_id(slot);
            let pi = self.positions[i];
            for src_slot in 0..n {
                if src_slot == slot {
                    continue;
                }
                let j = src_slot as usize;
                if InfectionState::from_u32(self.state[j]) != InfectionState::Infected {
                    continue;
                }
                let pj = self.positions[j];
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let d2 = dx * dx + dy * dy;
                if d2 > INFECTION_RADIUS_SQ {
                    continue;
                }
                // Per-(target, source, tick) RNG: separates contacts
                // from different infected sources during the same
                // tick. Purpose tag includes source slot so the same
                // susceptible can roll independently against each.
                let purpose: [u8; 12] = make_infect_purpose(src_slot);
                let r = per_agent_u32(seed, id, tick, &purpose);
                if r % PROB_DENOM < INFECTION_PROB_NUM {
                    self.next_state[i] = InfectionState::Infected as u32;
                    self.next_infection_tick[i] = tick + 1; // recovery timer starts fresh
                    break; // already infected — no need to roll the rest
                }
            }
        }

        // --- 3. Lifecycle: Infected → {Dead, Recovered} ---
        for slot in 0..n {
            let i = slot as usize;
            // Read PRE-tick state — newly infected from step 2 don't
            // get a free mortality / recovery roll on the same tick.
            if InfectionState::from_u32(self.state[i]) != InfectionState::Infected {
                continue;
            }
            let id = agent_id(slot);
            let mortality_roll = per_agent_u32(seed, id, tick, b"mortality");
            if mortality_roll % PROB_DENOM < MORTALITY_PROB_NUM {
                self.next_state[i] = InfectionState::Dead as u32;
                self.next_alive[i] = 0;
                continue;
            }
            // Recovery — `>=` so an agent infected at tick T recovers
            // at tick T + INFECTION_DURATION (i.e. INFECTION_DURATION
            // ticks of being sick).
            if tick.saturating_sub(self.infection_tick[i]) >= INFECTION_DURATION {
                self.next_state[i] = InfectionState::Recovered as u32;
            }
        }

        // Commit scratch → live.
        std::mem::swap(&mut self.state, &mut self.next_state);
        std::mem::swap(&mut self.alive, &mut self.next_alive);
        std::mem::swap(&mut self.infection_tick, &mut self.next_infection_tick);

        // Track peak — count infected after the swap.
        let mut infected = 0u32;
        for &s in &self.state {
            if s == InfectionState::Infected as u32 {
                infected += 1;
            }
        }
        if infected > self.peak_infected {
            self.peak_infected = infected;
            self.peak_tick = tick + 1;
        }

        self.tick += 1;
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn positions(&mut self) -> &[Vec3] {
        &self.positions
    }
}

/// Per-agent SIR state + display attribute hint, surfaced for a
/// future `viz_app` integration once the trait grows the
/// `snapshot()` / `glyph_table()` / `default_viewport()` methods.
/// Today this is just data — no renderer plumbing exists in the
/// worktree yet.
///
/// # Stable contract for callers
///
/// `glyph_for(state)` returns a `(glyph, ANSI 256-color fg)` pair:
///   - `0 (Susceptible)` → `('.', 245)` — dim grey
///   - `1 (Infected)`    → `('@', 196)` — bright red
///   - `2 (Recovered)`   → `('+', 40)`  — green
///   - `3 (Dead)`        → `('x', 238)` — dark grey
///
/// `viewport()` returns the axis-aligned `(min, max)` extents of
/// the world, suitable as the default for an ASCII renderer.
pub fn glyph_for(state: u32) -> (char, u8) {
    match InfectionState::from_u32(state) {
        InfectionState::Susceptible => ('.', 245),
        InfectionState::Infected => ('@', 196),
        InfectionState::Recovered => ('+', 40),
        InfectionState::Dead => ('x', 238),
    }
}

pub fn viewport() -> (Vec3, Vec3) {
    (Vec3::ZERO, Vec3::new(WORLD_SIZE, WORLD_SIZE, 0.0))
}

/// Build a 12-byte purpose tag of the form `b"infect_NNNNN"` so each
/// (target, source) contact gets an independent RNG draw without
/// allocating per call. Source slot serialised as a 5-digit
/// decimal stub, padded with `_`s, fixed-width — different sources
/// produce different byte slices, satisfying the per-purpose stream
/// requirement for P5 keyed PCG.
fn make_infect_purpose(src_slot: u32) -> [u8; 12] {
    // Layout: 7-byte prefix `infect_` + 5-byte zero-padded decimal
    // source slot. Total = 12 bytes. High digits truncate silently
    // when `src_slot >= 100_000` — fine for the 200-agent default.
    let mut out = [0u8; 12];
    out[..7].copy_from_slice(b"infect_");
    let mut s = src_slot;
    for i in (0..5).rev() {
        out[7 + i] = b'0' + (s % 10) as u8;
        s /= 10;
    }
    out
}

/// Per-fixture factory used by the generic sim_app harness.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(DiseaseSpreadState::new(seed, agent_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_one_infected() {
        let sim = DiseaseSpreadState::new(0xDEAD_BEEF, 200);
        let c = sim.sir_counts();
        assert_eq!(c.susceptible, 199);
        assert_eq!(c.infected, 1);
        assert_eq!(c.recovered, 0);
        assert_eq!(c.dead, 0);
        assert_eq!(c.susceptible + c.infected + c.recovered + c.dead, 200);
    }

    #[test]
    fn determinism_same_seed_same_curve() {
        let mut a = DiseaseSpreadState::new(0xABCD_1234_5678_9ABC, 200);
        let mut b = DiseaseSpreadState::new(0xABCD_1234_5678_9ABC, 200);
        for _ in 0..50 {
            a.step();
            b.step();
        }
        assert_eq!(a.state, b.state);
        assert_eq!(a.alive, b.alive);
        assert_eq!(a.infection_tick, b.infection_tick);
        assert_eq!(a.sir_counts().infected, b.sir_counts().infected);
    }

    #[test]
    fn epidemic_progresses_then_extinguishes() {
        let mut sim = DiseaseSpreadState::new(0xCAFE_F00D_42, 200);
        let mut max_i = 0u32;
        for _ in 0..1500 {
            sim.step();
            let i = sim.sir_counts().infected;
            if i > max_i {
                max_i = i;
            }
            if i == 0 && sim.tick() > 100 {
                break;
            }
        }
        let c = sim.sir_counts();
        // Epidemic should have spread beyond patient zero.
        assert!(max_i > 1, "epidemic never spread (peak = {max_i})");
        // And eventually died out (everyone recovered or dead).
        assert_eq!(c.infected, 0, "epidemic still active after 1500 ticks: {c:?}");
        // Total population conserved.
        assert_eq!(c.susceptible + c.infected + c.recovered + c.dead, 200);
    }

    #[test]
    fn glyph_table_has_four_states() {
        // S
        assert_eq!(glyph_for(0).0, '.');
        // I
        assert_eq!(glyph_for(1).0, '@');
        // R
        assert_eq!(glyph_for(2).0, '+');
        // D
        assert_eq!(glyph_for(3).0, 'x');
    }

    #[test]
    fn viewport_matches_world() {
        let (min, max) = viewport();
        assert_eq!(min, Vec3::ZERO);
        assert_eq!(max, Vec3::new(WORLD_SIZE, WORLD_SIZE, 0.0));
    }
}
