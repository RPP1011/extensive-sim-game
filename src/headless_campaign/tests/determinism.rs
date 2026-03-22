//! Determinism tests: same seed + same actions = identical state.

use crate::headless_campaign::actions::CampaignAction;
use crate::headless_campaign::state::CampaignState;
use crate::headless_campaign::step::step_campaign;

fn campaign_hash(state: &CampaignState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mix = |h: &mut u64, v: u64| {
        *h ^= v;
        *h = h.wrapping_mul(0x1000_0000_01b3);
    };
    mix(&mut h, state.tick);
    mix(&mut h, state.rng);
    mix(&mut h, (state.guild.gold * 100.0) as u64);
    mix(&mut h, (state.guild.supplies * 100.0) as u64);
    mix(&mut h, (state.guild.reputation * 100.0) as u64);
    mix(&mut h, state.adventurers.len() as u64);
    mix(&mut h, state.parties.len() as u64);
    mix(&mut h, state.request_board.len() as u64);
    mix(&mut h, state.active_quests.len() as u64);
    mix(&mut h, state.active_battles.len() as u64);
    mix(&mut h, state.completed_quests.len() as u64);
    for adv in &state.adventurers {
        mix(&mut h, adv.id as u64);
        mix(&mut h, (adv.stress * 100.0) as u64);
        mix(&mut h, (adv.fatigue * 100.0) as u64);
        mix(&mut h, (adv.injury * 100.0) as u64);
        mix(&mut h, adv.status as u64);
    }
    for party in &state.parties {
        mix(&mut h, party.id as u64);
        mix(&mut h, (party.position.0 * 100.0) as u64);
        mix(&mut h, (party.position.1 * 100.0) as u64);
        mix(&mut h, party.status as u64);
    }
    h
}

#[test]
fn test_determinism_no_actions() {
    let mut state_a = CampaignState::default_test_campaign(42);
    let mut state_b = CampaignState::default_test_campaign(42);

    for _ in 0..500 {
        step_campaign(&mut state_a, None);
        step_campaign(&mut state_b, None);
    }

    assert_eq!(
        campaign_hash(&state_a),
        campaign_hash(&state_b),
        "States diverged after 500 ticks with no actions"
    );
}

#[test]
fn test_determinism_with_actions() {
    let mut state_a = CampaignState::default_test_campaign(42);
    let mut state_b = CampaignState::default_test_campaign(42);

    // Run until a quest appears
    for _ in 0..2000 {
        step_campaign(&mut state_a, None);
        step_campaign(&mut state_b, None);

        // Accept first quest if available
        if let Some(req) = state_a.request_board.first() {
            let id = req.id;
            step_campaign(
                &mut state_a,
                Some(CampaignAction::AcceptQuest { request_id: id }),
            );
        }
        if let Some(req) = state_b.request_board.first() {
            let id = req.id;
            step_campaign(
                &mut state_b,
                Some(CampaignAction::AcceptQuest { request_id: id }),
            );
        }
    }

    assert_eq!(
        campaign_hash(&state_a),
        campaign_hash(&state_b),
        "States diverged after 2000 ticks with same actions"
    );
}

#[test]
fn test_different_seeds_diverge() {
    let mut state_a = CampaignState::default_test_campaign(42);
    let mut state_b = CampaignState::default_test_campaign(99);

    for _ in 0..500 {
        step_campaign(&mut state_a, None);
        step_campaign(&mut state_b, None);
    }

    assert_ne!(
        campaign_hash(&state_a),
        campaign_hash(&state_b),
        "Different seeds should produce different states"
    );
}

#[test]
fn test_no_violations_basic_run() {
    let mut state = CampaignState::default_test_campaign(42);

    for _ in 0..1000 {
        let result = step_campaign(&mut state, None);
        assert!(
            result.violations.is_empty(),
            "Violations at tick {}: {:?}",
            state.tick,
            result.violations
        );
    }
}

#[test]
fn test_quest_lifecycle() {
    let mut state = CampaignState::default_test_campaign(42);

    // Run until a quest appears
    let mut quest_appeared = false;
    for _ in 0..5000 {
        step_campaign(&mut state, None);
        if !state.request_board.is_empty() {
            quest_appeared = true;
            break;
        }
    }
    assert!(quest_appeared, "No quest appeared in 5000 ticks");

    // Accept the quest
    let req_id = state.request_board[0].id;
    let result = step_campaign(
        &mut state,
        Some(CampaignAction::AcceptQuest { request_id: req_id }),
    );
    assert!(matches!(result.action_result, Some(crate::headless_campaign::ActionResult::Success(_))));
    assert_eq!(state.active_quests.len(), 1);
    assert_eq!(state.active_quests[0].status, crate::headless_campaign::state::ActiveQuestStatus::Preparing);

    // Assign an adventurer
    let adv_id = state.adventurers[0].id;
    let quest_id = state.active_quests[0].id;
    step_campaign(
        &mut state,
        Some(CampaignAction::AssignToPool {
            adventurer_id: adv_id,
            quest_id,
        }),
    );
    assert!(state.active_quests[0].assigned_pool.contains(&adv_id));

    // Dispatch
    step_campaign(
        &mut state,
        Some(CampaignAction::DispatchQuest { quest_id }),
    );
    assert_eq!(state.parties.len(), 1);
    assert_eq!(
        state.active_quests[0].status,
        crate::headless_campaign::state::ActiveQuestStatus::Dispatched
    );
}
