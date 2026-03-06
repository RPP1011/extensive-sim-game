use bevy::prelude::*;
use bevy_egui::egui;

use crate::camera::{
    CameraFocusTransitionQueueResult, CameraFocusTransitionState, CameraFocusTrigger,
    SceneViewBounds, camera_focus_for_overworld_region,
};
use crate::game_core::{
    self, CharacterCreationState, HubScreen, HubUiState,
    RegionTransitionPayload, derive_region_transition_seed, validate_region_transition_payload,
};

#[derive(Resource, Debug, Clone)]
pub struct RegionLayerTransitionState {
    pub active_payload: Option<RegionTransitionPayload>,
    pub pending_payload: Option<RegionTransitionPayload>,
    pub pending_frames: u8,
    pub interaction_locked: bool,
    pub status: String,
}

impl Default for RegionLayerTransitionState {
    fn default() -> Self {
        Self {
            active_payload: None,
            pending_payload: None,
            pending_frames: 0,
            interaction_locked: false,
            status: "No active region context.".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartyCommandHandoff {
    pub new_party_id: u32,
    pub previous_party_name: String,
    pub new_party_name: String,
    pub new_region_id: usize,
    pub new_leader_hero_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionTargetPickerMode {
    Inactive,
    Picking {
        party_id: u32,
        previous_target_region_id: Option<usize>,
        selected_region_id: Option<usize>,
    },
}

#[derive(Resource, Debug, Clone, PartialEq, Eq)]
pub struct RegionTargetPickerState {
    pub mode: RegionTargetPickerMode,
}

impl Default for RegionTargetPickerState {
    fn default() -> Self {
        Self {
            mode: RegionTargetPickerMode::Inactive,
        }
    }
}

impl RegionTargetPickerState {
    pub fn active_party_id(&self) -> Option<u32> {
        match self.mode {
            RegionTargetPickerMode::Picking { party_id, .. } => Some(party_id),
            RegionTargetPickerMode::Inactive => None,
        }
    }

    pub fn selected_region_id(&self) -> Option<usize> {
        match self.mode {
            RegionTargetPickerMode::Picking {
                selected_region_id, ..
            } => selected_region_id,
            RegionTargetPickerMode::Inactive => None,
        }
    }

    pub fn is_active_for_party(&self, party_id: u32) -> bool {
        self.active_party_id() == Some(party_id)
    }

    pub fn clear(&mut self) {
        self.mode = RegionTargetPickerMode::Inactive;
    }
}

pub fn begin_region_target_picker(
    picker: &mut RegionTargetPickerState,
    party: &game_core::CampaignParty,
) -> String {
    picker.mode = RegionTargetPickerMode::Picking {
        party_id: party.id,
        previous_target_region_id: party.order_target_region_id,
        selected_region_id: None,
    };
    format!(
        "Target picker active for {}. Click a map region, then confirm or cancel.",
        party.name
    )
}

pub fn update_region_target_picker_selection(
    picker: &mut RegionTargetPickerState,
    party_id: u32,
    region_id: usize,
    overworld: &game_core::OverworldMap,
) -> Result<String, String> {
    if overworld.regions.get(region_id).is_none() {
        return Err("Target picker selection ignored: region no longer exists.".to_string());
    }
    match &mut picker.mode {
        RegionTargetPickerMode::Picking {
            party_id: active_party_id,
            selected_region_id,
            ..
        } => {
            if *active_party_id != party_id {
                return Err("Target picker selection ignored: active party changed.".to_string());
            }
            *selected_region_id = Some(region_id);
            let region_name = overworld
                .regions
                .get(region_id)
                .map(|r| r.name.as_str())
                .unwrap_or("Unknown");
            Ok(format!("Target picker selected {}.", region_name))
        }
        RegionTargetPickerMode::Inactive => Err("Target picker is not active.".to_string()),
    }
}

pub fn confirm_region_target_picker(
    picker: &mut RegionTargetPickerState,
    parties: &mut game_core::CampaignParties,
    party_id: u32,
    overworld: &game_core::OverworldMap,
) -> Result<String, String> {
    let selected_region_id = match &picker.mode {
        RegionTargetPickerMode::Picking {
            party_id: active_party_id,
            selected_region_id,
            ..
        } => {
            if *active_party_id != party_id {
                return Err("Target picker confirm blocked: active party changed.".to_string());
            }
            selected_region_id.ok_or_else(|| {
                "Target picker confirm blocked: select a map region first.".to_string()
            })?
        }
        RegionTargetPickerMode::Inactive => {
            return Err("Target picker confirm blocked: picker is not active.".to_string());
        }
    };
    let Some(party_idx) = parties.parties.iter().position(|p| p.id == party_id) else {
        return Err("Target picker confirm blocked: selected party no longer exists.".to_string());
    };
    if parties.parties[party_idx].is_player_controlled {
        return Err(
            "Target picker confirm blocked: selected party is directly controlled.".to_string(),
        );
    }
    if overworld.regions.get(selected_region_id).is_none() {
        return Err("Target picker confirm blocked: selected region is invalid.".to_string());
    }
    let party_name = parties.parties[party_idx].name.clone();
    parties.parties[party_idx].order_target_region_id = Some(selected_region_id);
    picker.clear();
    let region_name = overworld
        .regions
        .get(selected_region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown");
    Ok(format!("{party_name} target region set to {region_name}."))
}

pub fn cancel_region_target_picker(
    picker: &mut RegionTargetPickerState,
    party_id: u32,
) -> Result<String, String> {
    let previous_target_region_id = match picker.mode {
        RegionTargetPickerMode::Picking {
            party_id: active_party_id,
            previous_target_region_id,
            ..
        } => {
            if active_party_id != party_id {
                return Err("Target picker cancel ignored: active party changed.".to_string());
            }
            previous_target_region_id
        }
        RegionTargetPickerMode::Inactive => {
            return Err("Target picker cancel ignored: picker is not active.".to_string());
        }
    };
    picker.clear();
    Ok(match previous_target_region_id {
        Some(_) => "Target picker canceled. Previous target preserved.".to_string(),
        None => "Target picker canceled. Party remains without a target region.".to_string(),
    })
}

pub fn region_from_map_click(
    points: &[egui::Pos2],
    pointer: egui::Pos2,
    max_distance: f32,
) -> Option<usize> {
    let max_dist_sq = max_distance * max_distance;
    points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            let dist_sq = point.distance_sq(pointer);
            (dist_sq <= max_dist_sq).then_some((idx, dist_sq))
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

pub fn party_target_region_label(
    party: &game_core::CampaignParty,
    overworld: &game_core::OverworldMap,
) -> String {
    party
        .order_target_region_id
        .and_then(|id| overworld.regions.get(id))
        .map(|region| region.name.clone())
        .unwrap_or_else(|| "none".to_string())
}

pub fn party_panel_label(
    party: &game_core::CampaignParty,
    is_selected: bool,
    leader: &str,
    region: &str,
    target_region: &str,
) -> String {
    let selected_marker = if is_selected { "[SELECTED] " } else { "" };
    let control_marker = if party.is_player_controlled {
        "[CONTROLLED]"
    } else {
        "[DELEGATED]"
    };
    format!(
        "{selected_marker}{control_marker} {} | leader={} | region={} | order={:?} -> {} | supply={:.0}",
        party.name, leader, region, party.order, target_region, party.supply
    )
}

pub fn default_camera_focus(bounds: &SceneViewBounds) -> Vec3 {
    Vec3::new(
        (bounds.min_x + bounds.max_x) * 0.5,
        0.0,
        (bounds.min_z + bounds.max_z) * 0.5,
    )
}

pub fn queue_party_camera_focus_transition(
    camera_focus_transition: &mut CameraFocusTransitionState,
    camera_focus_start: Vec3,
    overworld: &game_core::OverworldMap,
    bounds: &SceneViewBounds,
    party_id: u32,
    region_id: usize,
    trigger: CameraFocusTrigger,
) -> Result<CameraFocusTransitionQueueResult, String> {
    let Some(target_focus) = camera_focus_for_overworld_region(overworld, bounds, region_id) else {
        return Err("Camera focus transition blocked: selected region is invalid.".to_string());
    };
    Ok(camera_focus_transition.begin(
        camera_focus_start,
        target_focus,
        party_id,
        region_id,
        trigger,
    ))
}

pub fn transfer_direct_command_to_selected(
    parties: &mut game_core::CampaignParties,
) -> Result<PartyCommandHandoff, String> {
    let Some(selected_party_id) = parties.selected_party_id else {
        return Err("Take Command blocked: select a delegated party first.".to_string());
    };
    let Some(selected_idx) = parties
        .parties
        .iter()
        .position(|p| p.id == selected_party_id)
    else {
        return Err("Take Command blocked: selected party is no longer available.".to_string());
    };
    if parties.parties[selected_idx].is_player_controlled {
        return Err(format!(
            "Take Command blocked: {} is already directly controlled.",
            parties.parties[selected_idx].name
        ));
    }
    let Some(previous_idx) = parties.parties.iter().position(|p| p.is_player_controlled) else {
        return Err(
            "Take Command blocked: no controlled party is available for handoff.".to_string(),
        );
    };

    let previous_party_name = parties.parties[previous_idx].name.clone();
    let new_party_name = parties.parties[selected_idx].name.clone();
    let new_party_id = parties.parties[selected_idx].id;
    let new_region_id = parties.parties[selected_idx].region_id;
    let new_leader_hero_id = parties.parties[selected_idx].leader_hero_id;

    for party in &mut parties.parties {
        party.is_player_controlled = party.id == selected_party_id;
    }

    Ok(PartyCommandHandoff {
        new_party_id,
        previous_party_name,
        new_party_name,
        new_region_id,
        new_leader_hero_id,
    })
}

pub fn build_region_transition_payload(
    overworld: &game_core::OverworldMap,
    character_creation: &CharacterCreationState,
) -> Result<RegionTransitionPayload, String> {
    if overworld.regions.is_empty() {
        return Err("Region entry failed: no regions are available in the overworld.".to_string());
    }
    let region_id = overworld
        .selected_region
        .min(overworld.regions.len().saturating_sub(1));
    if overworld.regions.get(region_id).is_none() {
        return Err("Region entry failed: selected region is invalid.".to_string());
    }
    let Some(faction_id) = character_creation.selected_faction_id.as_ref() else {
        return Err("Region entry failed: missing faction context.".to_string());
    };
    if faction_id.trim().is_empty() {
        return Err("Region entry failed: missing faction context.".to_string());
    }
    let Some(faction_index) = character_creation.selected_faction_index else {
        return Err("Region entry failed: missing faction context.".to_string());
    };
    if faction_index >= overworld.factions.len() {
        return Err("Region entry failed: faction context is no longer valid.".to_string());
    }
    let campaign_seed = overworld.map_seed;
    let region_seed = derive_region_transition_seed(campaign_seed, region_id, faction_index);
    Ok(RegionTransitionPayload {
        region_id,
        faction_id: faction_id.clone(),
        faction_index,
        campaign_seed,
        region_seed,
    })
}

pub fn request_enter_selected_region(
    hub_ui: &mut HubUiState,
    target_picker: &mut RegionTargetPickerState,
    camera_focus_transition: &CameraFocusTransitionState,
    region_transition: &mut RegionLayerTransitionState,
    overworld: &game_core::OverworldMap,
    character_creation: &CharacterCreationState,
) -> String {
    if region_transition.interaction_locked {
        return "Region transition already in progress; wait for completion.".to_string();
    }
    if target_picker.active_party_id().is_some() {
        return "Region entry blocked: finish or cancel the target picker first.".to_string();
    }
    if camera_focus_transition.is_active() {
        return "Region entry blocked: wait for camera focus transition to finish.".to_string();
    }
    match build_region_transition_payload(overworld, character_creation) {
        Ok(payload) => {
            let region_name = overworld
                .regions
                .get(payload.region_id)
                .map(|r| r.name.as_str())
                .unwrap_or("Unknown");
            target_picker.clear();
            region_transition.pending_payload = Some(payload.clone());
            region_transition.pending_frames = 1;
            region_transition.interaction_locked = true;
            region_transition.status = format!(
                "Entering {} (id {}, faction {}, seed {}). Transition lock active.",
                region_name, payload.region_id, payload.faction_id, payload.region_seed
            );
            hub_ui.screen = HubScreen::OverworldMap;
            region_transition.status.clone()
        }
        Err(reason) => {
            region_transition.pending_payload = None;
            region_transition.pending_frames = 0;
            region_transition.interaction_locked = false;
            region_transition.status = format!("{reason} Returned to overworld map.");
            hub_ui.screen = HubScreen::OverworldMap;
            region_transition.status.clone()
        }
    }
}

pub fn advance_region_layer_transition(
    hub_ui: &mut HubUiState,
    region_transition: &mut RegionLayerTransitionState,
    overworld: &game_core::OverworldMap,
) -> Option<String> {
    if region_transition.pending_payload.is_none() {
        return None;
    }
    if region_transition.pending_frames > 0 {
        region_transition.pending_frames -= 1;
        return None;
    }

    let payload = region_transition.pending_payload.take()?;
    match validate_region_transition_payload(&payload, overworld) {
        Ok(()) => {
            let region_name = overworld
                .regions
                .get(payload.region_id)
                .map(|region| region.name.as_str())
                .unwrap_or("Unknown");
            region_transition.active_payload = Some(payload.clone());
            region_transition.interaction_locked = false;
            region_transition.status = format!(
                "Region scene loaded: {} (id {}, faction {}, campaign seed {}, region seed {}).",
                region_name,
                payload.region_id,
                payload.faction_id,
                payload.campaign_seed,
                payload.region_seed
            );
            hub_ui.screen = HubScreen::RegionView;
            Some(region_transition.status.clone())
        }
        Err(reason) => {
            region_transition.active_payload = None;
            region_transition.interaction_locked = false;
            region_transition.status = format!(
                "Region transition failed: {}. Returned to overworld map.",
                reason
            );
            hub_ui.screen = HubScreen::OverworldMap;
            Some(region_transition.status.clone())
        }
    }
}
