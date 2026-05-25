//! Subkind seeding (Plan A) — the runtime gate. Constructs the
//! `subkind_seeding_probe` fixture through `make_playable` (so `try_new`'s
//! generated per-subkind seeding actually runs on the GPU) and reads back
//! the seeded agent buffers via `agent_snapshot`. Verifies:
//!   * per-subkind counts (4 Hero + 16 Mob = 20 seeded; slot 0 is the
//!     AgentId sentinel and stays creature_type 0 / dead),
//!   * each seeded agent's `creature_type` == its subkind's ordinal
//!     (Hero = 0, Mob = 1 — declaration order),
//!   * Hero `hp == 100.0` (f32 init value, Task 1) + `alive`; Mob `alive == 0`
//!     (pool override),
//!   * Mob positions are within the scatter radius (40) of the origin (Task 4),
//!   * two `try_new(seed)` runs produce byte-identical positions (P5).
//!
//! Requires a GPU adapter; run with
//! `RUST_MIN_STACK=33554432 cargo test -p sims --test subkind_seeding_exec`.
//! Without an adapter `make_playable` yields `None` and the tests skip.

const SEED: u64 = 0x1234_5678_9ABC;
const AGENTS: u32 = 64;

// Declaration order in subkind_seeding_probe.sim: Hero = 0, Mob = 1.
const CT_HERO: u32 = 0;
const CT_MOB: u32 = 1;
const HERO_COUNT: usize = 4;
const MOB_COUNT: usize = 16;
const SCATTER_R: f32 = 40.0;

#[test]
fn subkind_seeding_is_registered() {
    assert!(
        sims::PLAYABLE_FIXTURES.contains(&"subkind_seeding_probe"),
        "subkind_seeding_probe must be in the registry: {:?}",
        sims::PLAYABLE_FIXTURES
    );
}

#[test]
fn seeded_population_has_correct_counts_types_fields() {
    let Some(mut rt) = sims::make_playable("subkind_seeding_probe", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping subkind_seeding count/type/field checks");
        return;
    };
    let snap = rt.agent_snapshot();
    assert_eq!(snap.len(), AGENTS as usize, "snapshot covers every slot");

    // Slot 0 is the AgentId NonZeroU32 sentinel — never seeded (alive false).
    assert!(!snap[0].alive, "slot 0 (AgentId sentinel) is not seeded");

    // Hero range: slots 1..=4. creature_type 0, alive, hp 100.
    let heroes: Vec<_> = snap
        .iter()
        .filter(|a| a.creature_type == CT_HERO && a.alive)
        .collect();
    assert_eq!(
        heroes.len(),
        HERO_COUNT,
        "expected {HERO_COUNT} live Hero agents (creature_type=0)"
    );
    for h in &heroes {
        assert!(
            (h.hp - 100.0).abs() < 1e-3,
            "Hero hp should be the seeded f32 100.0, got {}",
            h.hp
        );
    }

    // Mob range: slots 5..=20. creature_type 1, alive 0 (dormant pool).
    let mobs: Vec<_> = snap
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.creature_type == CT_MOB)
        .map(|(_, a)| a)
        .collect();
    assert_eq!(
        mobs.len(),
        MOB_COUNT,
        "expected {MOB_COUNT} Mob agents (creature_type=1)"
    );
    for m in &mobs {
        assert!(
            !m.alive,
            "Mob pool agents are seeded alive:0 (overridden default)"
        );
    }

    // Total seeded = 20; the rest (slots 21..64) are untouched (creature_type
    // 0, dead) — they overlap the Hero ct ordinal but are not alive.
    let live = snap.iter().filter(|a| a.alive).count();
    assert_eq!(live, HERO_COUNT, "only Heroes are seeded alive");

    eprintln!(
        "[subkind_seeding] PASS counts/types/fields: {} Hero (hp100, alive), {} Mob (alive0)",
        heroes.len(),
        mobs.len()
    );
}

#[test]
fn scattered_positions_within_radius_and_deterministic() {
    let Some(mut rt1) = sims::make_playable("subkind_seeding_probe", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping subkind_seeding position checks");
        return;
    };
    let snap1 = rt1.agent_snapshot();

    // All Mob positions must lie within the scatter radius (40) of origin.
    // (Heroes are at origin.) Pull Mob slots by creature_type.
    let mob_positions: Vec<[f32; 3]> = snap1
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.creature_type == CT_MOB)
        .map(|(_, a)| a.pos)
        .collect();
    assert_eq!(mob_positions.len(), MOB_COUNT);
    let mut any_nonzero = false;
    for p in &mob_positions {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!(
            r <= SCATTER_R + 1e-3,
            "Mob scattered position {p:?} (r={r}) exceeds scatter radius {SCATTER_R}"
        );
        if r > 1e-4 {
            any_nonzero = true;
        }
    }
    assert!(
        any_nonzero,
        "scatter should place Mobs off the origin (not all at [0,0,0])"
    );

    // Heroes seeded at origin.
    for (i, a) in snap1.iter().enumerate() {
        if i >= 1 && a.creature_type == CT_HERO && a.alive {
            let r = (a.pos[0] * a.pos[0] + a.pos[1] * a.pos[1]).sqrt();
            assert!(r < 1e-4, "Hero seeded at origin, got {:?}", a.pos);
        }
    }

    // P5 determinism: a second try_new with the SAME seed produces byte-equal
    // seeded positions.
    let Some(mut rt2) = sims::make_playable("subkind_seeding_probe", SEED, AGENTS) else {
        return;
    };
    let snap2 = rt2.agent_snapshot();
    let mob_positions2: Vec<[f32; 3]> = snap2
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.creature_type == CT_MOB)
        .map(|(_, a)| a.pos)
        .collect();
    assert_eq!(
        mob_positions, mob_positions2,
        "same-seed reruns must seed byte-identical scatter positions (P5)"
    );

    eprintln!(
        "[subkind_seeding] PASS positions: {} Mobs within r={SCATTER_R}, deterministic across reruns",
        mob_positions.len()
    );
}
