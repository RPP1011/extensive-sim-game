//! Cooldowns Task 5 — local gate end-to-end through the real cast path.
//!
//! Casts A (local cooldown 20) at tick 0 via the DSL-emitted `physics cast`
//! rule. Asserts:
//!   * At tick GCD, the global gate has cleared but A's per-slot local
//!     cursor (20) still blocks A specifically.
//!   * B on a different slot (local cursor still 0) clears at tick GCD.
//!   * At tick 20, A's local clears and A is ready again.

use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: ct,
            pos,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn local_gate_blocks_same_slot_after_global_clears() {
    // A: long local cooldown (20). B: distinct slot, short local cooldown
    // (3) — never cast in this test, so B's local cursor stays at 0 and
    // only the shared GCD can block B.
    let mut b = AbilityRegistryBuilder::new();
    let ability_a = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let ability_b = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 3, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let registry = b.build();

    let mut state = SimState::new(8, 42);
    state.ability_registry = registry;
    state.config.combat.global_cooldown_ticks = 5;
    let gcd = state.config.combat.global_cooldown_ticks;

    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(2.0, 0.0, 0.0));

    let slot_a = ability_a.slot() as u8;
    let slot_b = ability_b.slot() as u8;

    // Cast A at tick 0 through the full cascade.
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();
    let mut events = EventRing::<Event>::with_cap(512);
    events.push(Event::AgentCast {
        actor: caster,
        ability: ability_a,
        target,
        depth: 0,
        tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);

    // Post-cast: global=GCD, A's local=20.
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(gcd));

    // At tick GCD+1 the global gate has elapsed (global_ready = 5 <= 6).
    // A's local (20) still blocks A specifically; B's local (0) + cleared
    // global → B is ready.
    let now = gcd + 1;
    assert!(
        !state.can_cast_ability(caster, slot_a, now),
        "A's local cursor must still block at tick GCD+1={now}"
    );
    assert!(
        state.can_cast_ability(caster, slot_b, now),
        "B must be ready at tick GCD+1 (global cleared, B's local is 0)"
    );

    // Tick 19 is still inside A's local cooldown (20). Tick 20 = boundary;
    // `local_ready = local_slots[slot] <= now` → cleared.
    assert!(
        !state.can_cast_ability(caster, slot_a, 19),
        "A still gated at tick 19 < 20"
    );
    assert!(
        state.can_cast_ability(caster, slot_a, 20),
        "A's local cursor clears at tick == 20"
    );
    // B remains ready throughout.
    assert!(state.can_cast_ability(caster, slot_b, 20));
}
