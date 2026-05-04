//! Abilities probe harness — drives `abilities_runtime` for N ticks
//! and reports the observed per-attacker `damage_total` view.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — all gaps closed
//!
//! With AGENT_COUNT=32, TICKS=100, strike_damage=10.0, every alive
//! agent's `verb_Strike` row wins argmax every tick (Strike's score
//! 10.0 > Heal's 5.0). The Strike chronicle physics rule emits one
//! `DamageDealt{ attacker=self, target=self, amount=10.0 }` per
//! agent per tick. The fold accumulates per slot:
//!
//!   - `damage_total[i]` = T * strike_damage = 100 * 10.0 = 1000.0
//!   - `healing_done[i]` = 0.0  (Heal never wins argmax)
//!
//! ### (b) NO FIRE — Gap #1 active (TODAY)
//!
//! The schedule fuses `fold_healing_done` with the Strike chronicle
//! physics rule into `fused_fold_healing_done_healed`. The fused
//! kernel's WGSL body references an undeclared
//! `view_storage_primary` (only `view_1_primary` is bound). Naga
//! rejects the shader, so the runtime SKIPS that kernel. Strike
//! chronicle never fires; no DamageDealt events; per-slot:
//!
//!   - `damage_total[i]` = 0.0  for every i
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-abilities_probe.md`.

use abilities_runtime::AbilitiesProbeState;
use engine::CompiledSim;

const SEED: u64 = 0xAB1_1771E5_1234_56;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
const STRIKE_DAMAGE: f32 = 10.0;

fn main() {
    let mut sim = AbilitiesProbeState::new(SEED, AGENT_COUNT);
    println!(
        "abilities_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let damage = sim.damage_total().to_vec();
    let healing = sim.healing_done().to_vec();
    println!(
        "abilities_app: finished — final tick={} agents={} damage_total.len()={}",
        sim.tick(),
        sim.agent_count(),
        damage.len(),
    );

    let expected_per_slot = (TICKS as f32) * STRIKE_DAMAGE;

    let (min, mean, max, sum, zero_slots) = stats(&damage);
    let nonzero_slots = damage.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (damage.len().max(1) as f32);

    println!(
        "abilities_app: damage_total readback — min={:.3} mean={:.3} max={:.3} sum={:.3}",
        min, mean, max, sum,
    );
    println!(
        "abilities_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        damage.len(),
        observed_fraction * 100.0,
    );
    println!(
        "abilities_app: expected per-slot value (full cascade): {:.3} \
         (= TICKS={} × strike_damage={:.3})",
        expected_per_slot, TICKS, STRIKE_DAMAGE,
    );
    let (h_min, h_mean, h_max, h_sum, _h_zero) = stats(&healing);
    println!(
        "abilities_app: healing_done readback — min={:.3} mean={:.3} max={:.3} sum={:.3} \
         (negative-control: Heal never wins argmax; expected all-0.0)",
        h_min, h_mean, h_max, h_sum,
    );

    // OUTCOME classification.
    if min >= expected_per_slot * 0.99 {
        println!(
            "abilities_app: OUTCOME = (a) FULL FIRE — every slot ≈ expected value; \
             the verb cascade routes Strike's win through the chronicle into \
             damage_total cleanly.",
        );
    } else if max == 0.0 {
        println!(
            "abilities_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0. \
             Root cause: Gap #1 (compiler emit). The Strike chronicle physics \
             rule was fused into `fused_fold_healing_done_healed`, whose WGSL \
             references an undeclared `view_storage_primary` (only \
             `view_1_primary` is bound). Naga rejects the shader; the runtime \
             skips dispatching it; no DamageDealt events ever land in the \
             ring. See docs/superpowers/notes/2026-05-04-abilities_probe.md \
             for the full gap chain (4 surfaced gaps).",
        );
    } else {
        println!(
            "abilities_app: OUTCOME = (b) PARTIAL FIRE — {:.1}% of slots fired \
             (mean={:.3} vs expected {:.3}). Mid-failure between Gap #1 \
             surfacing and the cascade running clean.",
            observed_fraction * 100.0,
            mean,
            expected_per_slot,
        );
    }
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32, usize) {
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut zero = 0usize;
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x;
        if x == 0.0 {
            zero += 1;
        }
    }
    let mean = if v.is_empty() { 0.0 } else { sum / v.len() as f32 };
    (min, mean, max, sum, zero)
}
