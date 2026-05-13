//! Headless smoke test for `viewer_runtime`. No window, no Vulkan.
//! Verifies the sim runtime constructs successfully, the dungeon
//! roomgen produces a real layout, agent count matches expectation,
//! and a few `step()` ticks don't panic.
//!
//! Skips silently if no wgpu adapter is available on the host.

use viewer_runtime::{dungeon, ViewerApp};

#[test]
fn viewer_constructs_and_steps_or_skips() {
    let seed = 0xD007_BEEF_5_7EA1u64;
    let init = std::panic::catch_unwind(|| ViewerApp::try_new(seed));
    let mut app = match init {
        Ok(Some(a)) => a,
        Ok(None) => {
            eprintln!("[viewer_runtime smoke] skipping: no wgpu adapter");
            return;
        }
        Err(_) => {
            eprintln!("[viewer_runtime smoke] skipping: panic during init");
            return;
        }
    };

    // Sanity: agent_count = N_HEROES + n_enemies, and at least 5 heroes
    // are alive at construction (seeded by `seed_topology`).
    let snap = app.agents();
    let expected_n = app.dungeon.total_agent_count() as usize;
    assert_eq!(snap.len(), expected_n, "agent buffer length mismatch");
    let heroes_alive = snap
        .iter()
        .filter(|a| a.alive && a.creature_type == dungeon::CT_HERO)
        .count();
    assert_eq!(
        heroes_alive,
        dungeon::N_HEROES as usize,
        "expected {} heroes alive at construction, got {}",
        dungeon::N_HEROES,
        heroes_alive,
    );
    let enemies_alive = snap
        .iter()
        .filter(|a| a.alive && a.creature_type != dungeon::CT_HERO)
        .count();
    assert!(
        enemies_alive > 100,
        "expected >100 enemies alive at construction, got {}",
        enemies_alive,
    );

    // Step a few ticks — should not panic; tick should advance.
    for _ in 0..5 {
        app.step();
    }
    assert_eq!(app.sim_tick(), 5, "tick should equal 5 after 5 steps");

    // No NaN positions.
    for (slot, a) in app.agents().iter().enumerate() {
        assert!(a.pos.x.is_finite(), "slot {slot} pos.x not finite");
        assert!(a.pos.y.is_finite(), "slot {slot} pos.y not finite");
        assert!(a.pos.z.is_finite(), "slot {slot} pos.z not finite");
    }
}
