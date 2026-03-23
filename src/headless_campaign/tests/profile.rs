//! Profiling test — measures time spent in each MCTS component.

use std::time::Instant;

use crate::headless_campaign::actions::CampaignAction;
use crate::headless_campaign::mcts::{mcts_search, MctsConfig};
// heuristic_rollout is private — inline the rollout for profiling
use crate::headless_campaign::state::CampaignState;
use crate::headless_campaign::step::step_campaign;

#[test]
fn profile_mcts_components() {
    let mut state = CampaignState::default_test_campaign(42);

    // Init with starting choice
    let choice = state.available_starting_choices[0].clone();
    step_campaign(&mut state, Some(CampaignAction::ChooseStartingPackage { choice }));
    // Advance to a state with quests/parties/choices
    for _ in 0..1000 {
        step_campaign(&mut state, None);
    }
    // Accept a quest if available
    if let Some(req) = state.request_board.first() {
        let id = req.id;
        step_campaign(&mut state, Some(CampaignAction::AcceptQuest { request_id: id }));
    }

    let n_actions = state.valid_actions().len();
    eprintln!("\n=== MCTS Component Profile ===");
    eprintln!("State at tick {}, {} adventurers, {} quests, {} battles, {} choices pending",
        state.tick,
        state.adventurers.len(),
        state.active_quests.len(),
        state.active_battles.len(),
        state.pending_choices.len(),
    );
    eprintln!("Valid actions: {}", n_actions);

    // 1. State clone cost
    let n_clones = 200;
    let t = Instant::now();
    for _ in 0..n_clones {
        let _ = state.clone();
    }
    let clone_us = t.elapsed().as_micros() as f64 / n_clones as f64;
    eprintln!("\n1. State clone: {:.1}us each", clone_us);

    // 2. Single step cost
    let n_steps = 10000;
    let mut s = state.clone();
    let t = Instant::now();
    for _ in 0..n_steps {
        step_campaign(&mut s, None);
    }
    let step_us = t.elapsed().as_micros() as f64 / n_steps as f64;
    eprintln!("2. step_campaign(None): {:.1}us each", step_us);

    // 3. valid_actions cost
    let n_va = 500;
    let t = Instant::now();
    for _ in 0..n_va {
        let _ = state.valid_actions();
    }
    let va_us = t.elapsed().as_micros() as f64 / n_va as f64;
    eprintln!("3. valid_actions(): {:.1}us each ({} actions)", va_us, n_actions);

    // 4. Single rollout cost (5000 ticks) — inline since the module is private
    let rollout_ticks = 5000u64;
    let n_rollouts = 5;
    let t = Instant::now();
    for _ in 0..n_rollouts {
        let mut rs = state.clone();
        for _ in 0..rollout_ticks {
            step_campaign(&mut rs, None);
        }
    }
    let rollout_ms = t.elapsed().as_millis() as f64 / n_rollouts as f64;
    eprintln!("4. Rollout ({} ticks, step only): {:.1}ms each", rollout_ticks, rollout_ms);

    // 5. Full MCTS search at different sim counts
    for sims in [10, 50, 100, 200] {
        let config = MctsConfig {
            simulations_per_move: sims,
            rollout_horizon_ticks: rollout_ticks,
            decision_interval_ticks: 200,
            max_campaign_ticks: 30000,
            ..Default::default()
        };
        let t = Instant::now();
        let (_action, decision) = mcts_search(&state, &config);
        let search_ms = t.elapsed().as_millis();
        let per_sim_ms = search_ms as f64 / sims as f64;
        eprintln!("5. mcts_search({} sims): {}ms total, {:.1}ms/sim, {} child actions",
            sims, search_ms, per_sim_ms, decision.action_visits.len());
    }

    // 6. Cost breakdown estimate
    eprintln!("\n=== Cost Breakdown (50 sims, 5000 rollout) ===");
    let clone_cost = clone_us / 1000.0; // ms
    let step_cost = step_us / 1000.0; // ms
    let selection_steps = 200; // decision_interval ticks during selection
    let rollout_steps = rollout_ticks;
    let total_steps_per_sim = selection_steps as f64 + rollout_steps as f64;
    let step_cost_per_sim = total_steps_per_sim * step_cost;
    let clone_cost_per_sim = clone_cost; // 1 clone per sim
    let va_cost_per_sim = va_us / 1000.0; // 1 valid_actions per expansion
    eprintln!("  Clone:          {:.2}ms/sim", clone_cost_per_sim);
    eprintln!("  Steps:          {:.2}ms/sim ({:.0} steps × {:.3}ms/step)", step_cost_per_sim, total_steps_per_sim, step_cost);
    eprintln!("  valid_actions:  {:.2}ms/sim", va_cost_per_sim);
    eprintln!("  Total estimate: {:.2}ms/sim", clone_cost_per_sim + step_cost_per_sim + va_cost_per_sim);
    eprintln!("  For 50 sims:    {:.0}ms", 50.0 * (clone_cost_per_sim + step_cost_per_sim + va_cost_per_sim));
}
