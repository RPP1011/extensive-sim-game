use super::*;


// ---------------------------------------------------------------------------
// Regression fixture types and helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
enum RegressionStage {
    StartMenu,
    CharacterCreation,
    Overworld,
    RegionView,
    LocalIntroFrame,
}

impl RegressionStage {
    fn label(self) -> &'static str {
        match self {
            Self::StartMenu => "StartMenu",
            Self::CharacterCreation => "CharacterCreation",
            Self::Overworld => "Overworld",
            Self::RegionView => "RegionView",
            Self::LocalIntroFrame => "LocalIntroFrame",
        }
    }
}

const REGRESSION_STAGES: [RegressionStage; 5] = [
    RegressionStage::StartMenu,
    RegressionStage::CharacterCreation,
    RegressionStage::Overworld,
    RegressionStage::RegionView,
    RegressionStage::LocalIntroFrame,
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegressionCapture {
    stage: RegressionStage,
    signature: u64,
}

fn fixture_seed_from_label(seed_label: &str) -> u64 {
    let trimmed = seed_label.trim();
    if let Some(num) = trimmed.strip_prefix("S-") {
        num.parse::<u64>().unwrap_or(1)
    } else {
        trimmed.parse::<u64>().unwrap_or(1)
    }
}

fn regression_signature(value: serde_json::Value) -> u64 {
    let bytes = serde_json::to_vec(&value).expect("serialize regression fixture");
    bytes
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |acc, b| {
            (acc ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        })
}

fn capture_for_stage(stage: RegressionStage, payload: serde_json::Value) -> RegressionCapture {
    RegressionCapture {
        stage,
        signature: regression_signature(payload),
    }
}

fn build_campaign_regression_fixture(seed_label: &str) -> Vec<RegressionCapture> {
    let seed = fixture_seed_from_label(seed_label);
    let mut overworld = game_core::OverworldMap::from_seed(seed);
    let region_id = 2_usize.min(overworld.regions.len().saturating_sub(1));
    overworld.selected_region = region_id;
    overworld.current_region = region_id;

    let faction_choice = build_faction_selection_choices(&overworld)
        .into_iter()
        .next()
        .expect("at least one faction");
    let mut creation = CharacterCreationState {
        selected_faction_id: Some(faction_choice.id.clone()),
        selected_faction_index: Some(faction_choice.index),
        selected_backstory_id: Some("scout-pathfinder".to_string()),
        stat_modifiers: vec!["Scouting: +2".to_string()],
        recruit_bias_modifiers: vec!["Recruit Bias: Tactician and Vanguard +15%".to_string()],
        is_confirmed: true,
    };
    let mut diplomacy = game_core::DiplomacyState::default();
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationFaction,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation_ui = CharacterCreationUiState::default();
    let _ = confirm_faction_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut diplomacy,
        &overworld,
    );

    let roster = game_core::CampaignRoster::default();
    let parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let player_party = parties
        .parties
        .iter()
        .find(|party| party.is_player_controlled)
        .expect("player party");
    let payload =
        build_region_transition_payload(&overworld, &creation).expect("region payload fixture");
    let transition = RegionLayerTransitionState {
        active_payload: Some(payload.clone()),
        pending_payload: None,
        pending_frames: 0,
        interaction_locked: false,
        status: "Region fixture".to_string(),
    };
    let mut local_intro = LocalEagleEyeIntroState::default();
    let _ = bootstrap_local_eagle_eye_intro(
        &mut hub_ui,
        &mut local_intro,
        &transition,
        &overworld,
    );
    let _ = advance_local_eagle_eye_intro(&mut local_intro);

    vec![
        capture_for_stage(
            RegressionStage::StartMenu,
            serde_json::json!({
                "seed_label": seed_label,
                "screen": "StartMenu",
                "subtitle": StartMenuState::default_subtitle(),
                "status": StartMenuState::default_status()
            }),
        ),
        capture_for_stage(
            RegressionStage::CharacterCreation,
            serde_json::json!({
                "seed_label": seed_label,
                "screen": "CharacterCreationFaction",
                "faction_id": creation.selected_faction_id,
                "faction_index": creation.selected_faction_index,
                "creation_status": creation_ui.status
            }),
        ),
        capture_for_stage(
            RegressionStage::Overworld,
            serde_json::json!({
                "seed_label": seed_label,
                "screen": "OverworldMap",
                "map_seed": overworld.map_seed,
                "selected_region_id": overworld.selected_region,
                "player_party_id": player_party.id,
                "player_party_region_id": player_party.region_id
            }),
        ),
        capture_for_stage(
            RegressionStage::RegionView,
            serde_json::json!({
                "seed_label": seed_label,
                "screen": "RegionView",
                "region_id": payload.region_id,
                "faction_id": payload.faction_id,
                "campaign_seed": payload.campaign_seed,
                "region_seed": payload.region_seed
            }),
        ),
        capture_for_stage(
            RegressionStage::LocalIntroFrame,
            serde_json::json!({
                "seed_label": seed_label,
                "screen": "LocalEagleEyeIntro",
                "source_region_id": local_intro.source_region_id,
                "phase": format!("{:?}", local_intro.phase),
                "phase_frames": local_intro.phase_frames,
                "anchor_prefab": local_intro.anchor.as_ref().map(|anchor| anchor.prefab_id),
                "intro_completed": local_intro.intro_completed,
                "input_handoff_ready": local_intro.input_handoff_ready
            }),
        ),
    ]
}

fn dedupe_regression_captures(captures: &[RegressionCapture]) -> Vec<RegressionCapture> {
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for capture in captures {
        if seen.insert(capture.signature) {
            deduped.push(capture.clone());
        }
    }
    deduped
}

fn verify_regression_baseline(
    captures: &[RegressionCapture],
    baseline: &BTreeMap<RegressionStage, u64>,
) -> Result<(), String> {
    let mut diagnostics = Vec::new();
    for (stage, expected_hash) in baseline {
        match captures.iter().find(|capture| capture.stage == *stage) {
            Some(capture) if capture.signature == *expected_hash => {}
            Some(capture) => diagnostics.push(format!(
                "{} baseline mismatch: expected {:016x}, got {:016x}",
                stage.label(),
                expected_hash,
                capture.signature
            )),
            None => diagnostics.push(format!("{} baseline missing capture", stage.label())),
        }
    }
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(diagnostics.join(" | "))
    }
}

#[test]
fn campaign_regression_fixture_is_deterministic_for_seed_s001() {
    let first = build_campaign_regression_fixture("S-001");
    let second = build_campaign_regression_fixture("S-001");

    assert_eq!(first, second);
    let stages = first
        .iter()
        .map(|capture| capture.stage)
        .collect::<Vec<_>>();
    assert_eq!(stages, REGRESSION_STAGES);
}

#[test]
fn campaign_regression_stage_capture_supports_dedupe_and_baseline_checks() {
    let fixture = build_campaign_regression_fixture("S-001");
    let mut with_duplicates = vec![
        fixture[0].clone(),
        fixture[0].clone(),
        fixture[1].clone(),
        fixture[2].clone(),
        fixture[2].clone(),
        fixture[3].clone(),
        fixture[4].clone(),
    ];
    let deduped = dedupe_regression_captures(&with_duplicates);
    assert_eq!(deduped, fixture);

    let baseline = fixture
        .iter()
        .map(|capture| (capture.stage, capture.signature))
        .collect::<BTreeMap<RegressionStage, u64>>();
    assert!(verify_regression_baseline(&deduped, &baseline).is_ok());

    with_duplicates[5].signature ^= 0x40;
    let deduped_mutated = dedupe_regression_captures(&with_duplicates);
    let error = verify_regression_baseline(&deduped_mutated, &baseline)
        .expect_err("mismatch should fail baseline verification");
    assert!(error.contains("RegionView baseline mismatch"));
}
