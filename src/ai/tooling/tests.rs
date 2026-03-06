use std::collections::HashMap;

use crate::ai::tooling::*;

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
        Some(&(0x0122_c661_c895_0a47, 0xd403_2cbd_c589_780e))
    );
    assert_eq!(
        map.get("phase3_seed23"),
        Some(&(0x08ad_4478_343f_050f, 0x5c66_2925_360a_49a1))
    );
    assert_eq!(
        map.get("phase4_seed29"),
        Some(&(0xa27a_1183_d869_1ddd, 0x86ee_da84_10ec_becd))
    );
    assert_eq!(
        map.get("phase5_seed31"),
        Some(&(0x8d03_5a97_0ee4_d0d0, 0x5f4d_5cfd_8a52_119e))
    );
}

#[test]
fn visualization_html_contains_expected_sections() {
    let html = crate::ai::tooling::custom::build_phase5_event_visualization_html(31, 40);
    assert!(html.contains("AI Event Visualization"));
    assert!(html.contains("event_hash="));
    assert!(html.contains("timeline"));
    assert!(html.contains("DamageApplied"));
}
