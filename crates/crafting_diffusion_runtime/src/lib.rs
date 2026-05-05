//! Per-fixture runtime for `assets/sim/crafting_diffusion.sim` — the
//! twenty-first real sim and the FIRST that propagates **information**
//! through agent interactions.
//!
//! ## What this sim demonstrates
//!
//! - 50 Traders distributed across 5 specialization groups (10 each).
//! - Each group starts knowing 2 of 10 recipes (a 10-bit `mana`
//!   bitmask). Initial masks per group:
//!     - 0 (wood): `0b0000000011 =    3`
//!     - 1 (iron): `0b0000001100 =   12`
//!     - 2 (gem):  `0b0000110000 =   48`
//!     - 3 (herb): `0b0011000000 =  192`
//!     - 4 (bone): `0b1100000000 =  768`
//! - Per tick: every Trader random-walks. Then for each Trader,
//!   if a different-group Trader within `trade_radius` carries any
//!   recipe bits not subsumed by self, both end up with the
//!   union (associative + commutative — order-independent).
//! - Convergence target: every slot's mask = `0b1111111111 = 1023`.
//!
//! ## Why CPU-only
//!
//! The natural DSL chronicle —
//! `agents.set_mana(r, mana | other_mana)` — does NOT lower today
//! because the surface language doesn't parse bitwise `|` (only
//! logical `||`). The full gap chain is documented in the .sim
//! file. Per the compiler-gap policy, the runtime owns the per-tick
//! step while the gap stays open. The .sim file is exercised at
//! parse + resolve through workspace dsl_compiler tests; the runtime
//! is the executable demonstration of the architectural surface.
//!
//! ## Determinism (P5)
//!
//! Every random draw flows through `engine::rng::per_agent_u32(seed,
//! agent_id, tick, purpose)`. The trade pairing is an O(N²) scan in
//! deterministic agent-id order, so no RNG-driven scheduling.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use glam::Vec3;

/// 10 recipes total → 10-bit mask. Full convergence = `(1 << 10) - 1`.
pub const NUM_RECIPES: u32 = 10;
/// 5 specialization groups (wood / iron / gem / herb / bone).
pub const NUM_GROUPS: u32 = 5;
/// Spatial radius inside which two Traders can trade. Sized larger
/// than the inter-group anchor spacing so adjacent group clusters
/// can trade from tick 1, with diffusion across the full ring
/// dependent on multi-hop chains.
pub const TRADE_RADIUS: f32 = 8.0;
/// Trade cooldown — each Trader emits at most one KnowledgeShared
/// every TRADE_COOLDOWN ticks. Cooldown is per-agent, evaluated as
/// `(tick - last_trade_tick) >= TRADE_COOLDOWN`.
pub const TRADE_COOLDOWN: u64 = 5;
/// Random-walk acceleration scale per tick. Velocity is INTEGRATED
/// into the per-agent vel buffer (additive Brownian drift), not
/// replaced per tick — without integration the agents would just
/// vibrate around their spawn point.
pub const WANDER_SPEED: f32 = 0.05;
/// Drag coefficient applied per tick (`vel *= (1 - WANDER_DRAG)`).
/// Bounds the random walk so velocity doesn't diverge.
pub const WANDER_DRAG: f32 = 0.05;
/// Group spawn ring radius. Each group's anchor sits on a circle of
/// this radius around the origin, evenly spaced. With NUM_GROUPS=5,
/// the chord between adjacent anchors is `2 * R * sin(π/5) ≈ 1.176 R`.
/// Pick R so adjacent-anchor chord ≈ 1.5 × TRADE_RADIUS — close enough
/// for adjacent groups to trade once jitter widens the cluster, but
/// far enough that opposite groups need multi-hop chains to mix.
pub const SPAWN_RADIUS: f32 = 10.0;
/// Per-group spawn jitter — Traders spawn within this radius of
/// their group anchor. Sized so adjacent-group clusters initially
/// overlap (jitter ≥ chord / 2) — guarantees first-tick trades.
pub const SPAWN_JITTER: f32 = 6.0;
/// Hard wander bound: agents stay inside this box (reflect-on-edge).
/// Keeps the population physically bounded so all groups stay
/// reachable through chained trades.
pub const WANDER_BOUND: f32 = 25.0;
/// Full-knowledge mask — every recipe bit set.
pub const FULL_MASK: u32 = (1u32 << NUM_RECIPES) - 1;

/// Per-fixture state.
///
/// All hot state lives in flat parallel SoA `Vec`s indexed by agent
/// slot 0..agent_count. The trait `positions()` returns a borrowed
/// slice so cache layout matches the engine's expectation.
pub struct CraftingDiffusionState {
    seed: u64,
    tick: u64,
    agent_count: u32,

    /// Per-agent world position. Mutated by the wander step.
    pos: Vec<Vec3>,
    /// Per-agent random-walk velocity. Re-rolled each tick.
    vel: Vec<Vec3>,
    /// Per-agent group_id (0..NUM_GROUPS). Constant for the run.
    group: Vec<u32>,
    /// Per-agent recipe-knowledge bitmask. Initialised from group_id;
    /// mutated by the trade merge.
    knowledge: Vec<u32>,
    /// Per-agent next-trade tick (for cooldown gating). Initialised
    /// to 0 so all agents are trade-eligible from tick 0. After a
    /// trade at tick `t`, `next_trade_tick[a] = t + TRADE_COOLDOWN`.
    /// The gate is `tick >= next_trade_tick[a]`.
    next_trade_tick: Vec<u64>,

    /// Total trade events fired this run. Equivalent to the value
    /// the .sim file's `trades_received` view-fold WOULD accumulate
    /// if its `+= 1` lowered to atomicAdd (it currently lowers to
    /// atomicOr — see the quest_probe gap doc).
    pub total_trades: u64,
    /// Trade events fired in the current tick (for harness traces).
    pub trades_this_tick: u32,
}

impl CraftingDiffusionState {
    /// Construct a 50-Trader fixture with deterministic init.
    ///
    /// Group anchors lie on a circle of radius `SPAWN_RADIUS` around
    /// the origin at angles `2π * group / NUM_GROUPS`. Per-agent
    /// jitter from each anchor draws from `per_agent_u32(seed,
    /// agent_id, 0, "init_pos_*")` (P5).
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut group = Vec::with_capacity(n);
        let mut knowledge = Vec::with_capacity(n);
        let mut next_trade_tick = Vec::with_capacity(n);

        for slot in 0..agent_count {
            // Round-robin group assignment so every group gets
            // ~equal share. `group_id = slot % NUM_GROUPS` is
            // deterministic and balanced when agent_count is a
            // multiple of NUM_GROUPS (50/5 = 10 each).
            let g = slot % NUM_GROUPS;
            group.push(g);

            // Initial knowledge: bits `2g` and `2g+1`.
            let init_mask = (1u32 << (2 * g)) | (1u32 << (2 * g + 1));
            knowledge.push(init_mask);

            // Deterministic spawn around the per-group anchor.
            let angle = (g as f32) * std::f32::consts::TAU / NUM_GROUPS as f32;
            let anchor = Vec3::new(
                SPAWN_RADIUS * angle.cos(),
                0.0,
                SPAWN_RADIUS * angle.sin(),
            );
            let agent_id = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let jitter_x = unit_jitter(per_agent_u32(seed, agent_id, 0, b"cd_init_pos_x"));
            let jitter_z = unit_jitter(per_agent_u32(seed, agent_id, 0, b"cd_init_pos_z"));
            let p = anchor + Vec3::new(
                jitter_x * SPAWN_JITTER,
                0.0,
                jitter_z * SPAWN_JITTER,
            );
            pos.push(p);
            vel.push(Vec3::ZERO);
            next_trade_tick.push(0);
        }

        Self {
            seed,
            tick: 0,
            agent_count,
            pos,
            vel,
            group,
            knowledge,
            next_trade_tick,
            total_trades: 0,
            trades_this_tick: 0,
        }
    }

    /// Per-agent group id (0..NUM_GROUPS).
    pub fn group(&self, slot: usize) -> u32 { self.group[slot] }
    /// Per-agent recipe bitmask.
    pub fn knowledge(&self, slot: usize) -> u32 { self.knowledge[slot] }
    /// Number of bits set in this slot's knowledge mask.
    pub fn knowledge_bits(&self, slot: usize) -> u32 { self.knowledge[slot].count_ones() }

    /// Slice over per-agent group ids.
    pub fn groups(&self) -> &[u32] { &self.group }
    /// Slice over per-agent knowledge bitmasks.
    pub fn knowledge_slice(&self) -> &[u32] { &self.knowledge }

    /// Total agents whose mask == FULL_MASK.
    pub fn full_knowledge_count(&self) -> u32 {
        self.knowledge.iter().filter(|&&k| k == FULL_MASK).count() as u32
    }

    /// True if every Trader knows every recipe.
    pub fn fully_converged(&self) -> bool {
        self.full_knowledge_count() == self.agent_count
    }

    /// Per-bit-count histogram. Index `i` holds the count of agents
    /// whose mask has exactly `i` bits set (range 0..=NUM_RECIPES).
    pub fn knowledge_histogram(&self) -> Vec<u32> {
        let mut h = vec![0u32; NUM_RECIPES as usize + 1];
        for &k in &self.knowledge {
            h[k.count_ones() as usize] += 1;
        }
        h
    }

    /// Mean bits-set across all agents.
    pub fn mean_bits(&self) -> f32 {
        if self.agent_count == 0 { return 0.0; }
        let sum: u32 = self.knowledge.iter().map(|k| k.count_ones()).sum();
        sum as f32 / self.agent_count as f32
    }

    /// Total trade events fired this run.
    pub fn total_trades(&self) -> u64 { self.total_trades }
    /// Trade events fired in the most recent step.
    pub fn trades_this_tick(&self) -> u32 { self.trades_this_tick }

    /// Per-agent random-walk integration: velocity accumulates
    /// uniform-[-1,1) acceleration jitter scaled by WANDER_SPEED,
    /// damped by WANDER_DRAG, then position integrates. Deterministic
    /// via P5 keyed PCG. Reflect-on-bound at WANDER_BOUND so the
    /// population stays compact enough for chained trades.
    fn wander_step(&mut self) {
        for slot in 0..self.agent_count {
            let agent_id = AgentId::new(slot + 1).expect("non-zero slot+1");
            let dvx = unit_jitter(per_agent_u32(self.seed, agent_id, self.tick, b"cd_wander_x"))
                * WANDER_SPEED;
            let dvz = unit_jitter(per_agent_u32(self.seed, agent_id, self.tick, b"cd_wander_z"))
                * WANDER_SPEED;
            // Acceleration adds, drag dissipates.
            let i = slot as usize;
            let mut v = self.vel[i] + Vec3::new(dvx, 0.0, dvz);
            v *= 1.0 - WANDER_DRAG;
            // Reflect-velocity at the bound so we don't lose drift
            // direction when clamping.
            let mut p = self.pos[i] + v;
            if p.x.abs() > WANDER_BOUND {
                p.x = WANDER_BOUND.copysign(p.x);
                v.x = -v.x;
            }
            if p.z.abs() > WANDER_BOUND {
                p.z = WANDER_BOUND.copysign(p.z);
                v.z = -v.z;
            }
            self.vel[i] = v;
            self.pos[i] = p;
        }
    }

    /// Per-tick trade scan + knowledge merge.
    ///
    /// For each agent A in slot order, find the FIRST other-group
    /// agent B within trade_radius whose knowledge carries bits A
    /// doesn't have. Merge both ways: knowledge[A] |= knowledge[B]
    /// AND knowledge[B] |= knowledge[A]. Set both cooldowns.
    ///
    /// "First found" is deterministic (slot order). The merge is
    /// associative + commutative so the order doesn't change the
    /// asymptote, only intermediate state.
    ///
    /// Why both sides: in the .sim semantics, A emits
    /// KnowledgeShared{recipient: A, source: B} and B emits
    /// KnowledgeShared{recipient: B, source: A} symmetrically — the
    /// chronicle ApplyKnowledge merges in both directions. We compress
    /// both halves into one host-side merge here.
    fn trade_step(&mut self) {
        self.trades_this_tick = 0;
        let r2 = TRADE_RADIUS * TRADE_RADIUS;
        for a in 0..self.agent_count as usize {
            // Cooldown: A is eligible only at or after its
            // next_trade_tick. Init = 0 so all agents eligible at
            // tick 0.
            if self.tick < self.next_trade_tick[a] { continue; }
            for b in 0..self.agent_count as usize {
                if a == b { continue; }
                if self.group[a] == self.group[b] { continue; }
                if self.tick < self.next_trade_tick[b] { continue; }
                // Knowledge gate: B carries bits A doesn't have
                // (`(b.knowledge & ~a.knowledge) != 0`). If A is
                // already a superset of B's knowledge there's no
                // benefit, skip.
                let novel = self.knowledge[b] & !self.knowledge[a];
                if novel == 0 { continue; }
                // Spatial gate.
                let d2 = self.pos[a].distance_squared(self.pos[b]);
                if d2 > r2 { continue; }

                // Trade: merge both ways. The merge is symmetric so
                // both agents end up with the union.
                let union = self.knowledge[a] | self.knowledge[b];
                self.knowledge[a] = union;
                self.knowledge[b] = union;
                self.next_trade_tick[a] = self.tick + TRADE_COOLDOWN;
                self.next_trade_tick[b] = self.tick + TRADE_COOLDOWN;
                // Two events conceptually fire (one per recipient
                // direction). Both contribute to `total_trades`.
                self.total_trades += 2;
                self.trades_this_tick += 2;
                break; // A trades at most once per tick.
            }
        }
    }
}

/// Map a u32 to a uniform `[-1, 1)` float.
fn unit_jitter(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl CompiledSim for CraftingDiffusionState {
    fn step(&mut self) {
        self.wander_step();
        self.trade_step();
        self.tick += 1;
    }

    fn tick(&self) -> u64 { self.tick }
    fn agent_count(&self) -> u32 { self.agent_count }

    fn positions(&mut self) -> &[Vec3] { &self.pos }

    fn snapshot(&mut self) -> AgentSnapshot {
        // Encode per-agent visualization-discriminant by combining
        // group_id with a knowledge-tier hint:
        //   creature_type = group * (NUM_RECIPES + 1) + bits_set
        // The glyph_table below indexes `group` for color and uses
        // bits_set to render a digit glyph (0..9 or '*' for full).
        // Renderer code that's group-only colour ignores the bits
        // dimension; renderer code that wants bits-aware glyphs uses
        // it directly — but we keep the discriminant compact and
        // group-aligned so colour-only renderers work too.
        let positions = self.pos.clone();
        let alive = vec![1u32; self.agent_count as usize];
        let creature_types: Vec<u32> = (0..self.agent_count as usize)
            .map(|s| {
                let g = self.group[s];
                let bits = self.knowledge[s].count_ones();
                g * (NUM_RECIPES + 1) + bits
            })
            .collect();
        AgentSnapshot { positions, creature_types, alive }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        // 5 groups × 11 knowledge tiers (0..10 bits set) = 55 entries.
        // Group colours: distinct ANSI 256-colour foregrounds:
        //   0 wood → 130 (brown)
        //   1 iron → 245 (grey)
        //   2 gem  → 207 (magenta)
        //   3 herb →  46 (green)
        //   4 bone → 230 (cream)
        // Glyph encodes bits-set: '0'..'9' for 0..=9 bits, '*' for 10.
        // Renderers that pre-date the bits-aware encoding fall back to
        // the group base glyph (uppercase initial).
        const GROUP_COLORS: [u8; NUM_GROUPS as usize] = [130, 245, 207, 46, 230];
        let mut table = Vec::with_capacity((NUM_GROUPS * (NUM_RECIPES + 1)) as usize);
        for g in 0..NUM_GROUPS {
            for bits in 0..=NUM_RECIPES {
                let glyph = if bits >= NUM_RECIPES {
                    '*'
                } else {
                    char::from_digit(bits, 10).unwrap_or('?')
                };
                table.push(VizGlyph::new(glyph, GROUP_COLORS[g as usize]));
            }
        }
        table
    }

    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        let lo = Vec3::new(-WANDER_BOUND, 0.0, -WANDER_BOUND);
        let hi = Vec3::new(WANDER_BOUND, 0.0, WANDER_BOUND);
        Some((lo, hi))
    }
}

/// Trait-object factory for the universal harness.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(CraftingDiffusionState::new(seed, agent_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_distributes_agents_across_groups() {
        let sim = CraftingDiffusionState::new(0xC0FFEE, 50);
        let mut counts = [0u32; NUM_GROUPS as usize];
        for slot in 0..50 {
            counts[sim.group(slot) as usize] += 1;
        }
        // 50 / 5 = 10 per group.
        for c in counts {
            assert_eq!(c, 10, "expected 10 traders per group, got {c}");
        }
    }

    #[test]
    fn init_knowledge_two_bits_per_agent() {
        let sim = CraftingDiffusionState::new(0xC0FFEE, 50);
        for slot in 0..50 {
            assert_eq!(
                sim.knowledge_bits(slot),
                2,
                "slot {slot} (group {}) should start with exactly 2 bits, got {} (mask=0b{:010b})",
                sim.group(slot),
                sim.knowledge_bits(slot),
                sim.knowledge(slot),
            );
        }
    }

    #[test]
    fn init_knowledge_matches_group_partition() {
        let sim = CraftingDiffusionState::new(0xC0FFEE, 50);
        for slot in 0..50 {
            let g = sim.group(slot);
            let expected = (1u32 << (2 * g)) | (1u32 << (2 * g + 1));
            assert_eq!(
                sim.knowledge(slot), expected,
                "slot {slot} group {g}: expected mask 0b{:010b}, got 0b{:010b}",
                expected, sim.knowledge(slot),
            );
        }
    }

    #[test]
    fn determinism_step_seq_byte_identical() {
        let mut a = CraftingDiffusionState::new(42, 50);
        let mut b = CraftingDiffusionState::new(42, 50);
        for _ in 0..100 {
            a.step();
            b.step();
        }
        assert_eq!(a.knowledge_slice(), b.knowledge_slice());
        assert_eq!(a.total_trades(), b.total_trades());
        for slot in 0..50 {
            assert_eq!(a.pos[slot as usize], b.pos[slot as usize]);
        }
    }

    #[test]
    fn debug_initial_distances() {
        let sim = CraftingDiffusionState::new(0xC0FFEE_BEEF_CAFE_42, 50);
        let positions = &sim.pos;
        let mut min_cross_dist = f32::INFINITY;
        let mut count_in_range = 0;
        for a in 0..50 {
            for b in (a + 1)..50 {
                if sim.group(a) == sim.group(b) { continue; }
                let d = positions[a].distance(positions[b]);
                if d < min_cross_dist { min_cross_dist = d; }
                if d <= TRADE_RADIUS { count_in_range += 1; }
            }
        }
        eprintln!("min cross-group dist: {:.2} (TRADE_RADIUS={})", min_cross_dist, TRADE_RADIUS);
        eprintln!("cross-group pairs in trade range at tick 0: {}", count_in_range);
        assert!(count_in_range > 0, "expected some cross-group pairs in trade range; min dist was {min_cross_dist}");
    }

    #[test]
    fn knowledge_grows_over_time() {
        let mut sim = CraftingDiffusionState::new(0xC0FFEE, 50);
        let initial_mean = sim.mean_bits();
        for _ in 0..200 {
            sim.step();
        }
        let later_mean = sim.mean_bits();
        assert!(
            later_mean > initial_mean,
            "knowledge should grow: initial mean={initial_mean}, after 200 ticks={later_mean}",
        );
    }

    #[test]
    fn full_mask_value() {
        assert_eq!(FULL_MASK, 0b11_1111_1111);
        assert_eq!(FULL_MASK, 1023);
        assert_eq!(FULL_MASK.count_ones(), NUM_RECIPES);
    }
}
