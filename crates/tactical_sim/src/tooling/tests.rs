use std::collections::HashMap;

use crate::tooling::*;

#[test]
fn scenario_matrix_is_deterministic_flagged() {
    let rows = run_scenario_matrix();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r.deterministic));
    assert!(rows
        .iter()
        .all(|r| r.hero_deaths <= 3 && r.enemy_deaths <= 3));
}

#[test]
fn debug_output_contains_ranked_candidates() {
    let ticks = build_phase5_debug(31, 2, 3);
    assert_eq!(ticks.len(), 2);
    assert!(ticks
        .iter()
        .flat_map(|t| t.decisions.iter())
        .any(|d| !d.top_k.is_empty()));
}

#[test]
fn tuning_grid_returns_sorted_results() {
    let rows = run_personality_grid_tuning();
    assert!(!rows.is_empty());
    for pair in rows.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn cc_metrics_are_within_expected_bounds() {
    let cc = analyze_phase4_cc_metrics(29, 320);
    assert!(cc.primary_target.is_some());
    assert!(cc.windows >= 3);
    assert!(cc.coverage_ratio >= 0.2 && cc.coverage_ratio <= 1.0);
    assert!(cc.overlap_ratio >= 0.0 && cc.overlap_ratio <= 1.0);
    assert!(cc.avg_gap_ticks >= 0.0);
}

#[test]
fn scenario_matrix_hash_regression_snapshot() {
    let rows = run_scenario_matrix();
    let map = rows
        .into_iter()
        .map(|r| (r.name, (r.event_hash, r.state_hash)))
        .collect::<HashMap<_, _>>();

    assert_eq!(
        map.get("phase2_seed17"),
        Some(&(0xa192_946a_101d_8b9b, 0x688c_7967_a1d2_b856))
    );
    assert_eq!(
        map.get("phase3_seed23"),
        Some(&(0x0c61_dec7_078a_ad8e, 0xafb2_7256_0a79_da9f))
    );
    assert_eq!(
        map.get("phase4_seed29"),
        Some(&(0x21a7_b000_cc46_c8b2, 0x376e_9b1e_458d_c9f1))
    );
    assert_eq!(
        map.get("phase5_seed31"),
        Some(&(0x2df8_c3e7_3dbd_7c93, 0x2fab_b081_1b37_917f))
    );
}

#[test]
fn visualization_html_contains_expected_sections() {
    let html = crate::tooling::custom::build_phase5_event_visualization_html(31, 40);
    assert!(html.contains("AI Event Visualization"));
    assert!(html.contains("event_hash="));
    assert!(html.contains("timeline"));
    assert!(html.contains("DamageApplied"));
}
