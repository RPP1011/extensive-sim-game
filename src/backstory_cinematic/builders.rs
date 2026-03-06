use std::path::PathBuf;

use crate::character_select::build_backstory_selection_choices;
use crate::game_core::{self, CharacterCreationState};
use crate::runtime_assets::{
    RuntimeAssetGenState, RuntimeAssetJob, RuntimeAssetJobKind, RuntimeAssetStyle,
    compose_runtime_backstory_scene_prompt,
};

use super::types::*;

pub fn build_backstory_cinematic_beats(
    character_creation: &CharacterCreationState,
    overworld: &game_core::OverworldMap,
) -> Vec<BackstoryCinematicBeat> {
    let style = RuntimeAssetStyle::LineArt;
    let faction_name = character_creation
        .selected_faction_index
        .and_then(|idx| overworld.factions.get(idx))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "Unknown Faction".to_string());
    let backstory = build_backstory_selection_choices()
        .into_iter()
        .find(|choice| {
            character_creation.selected_backstory_id.as_deref() == Some(choice.id)
        });
    let (backstory_name, backstory_summary) = backstory
        .map(|choice| (choice.name.to_string(), choice.summary.to_string()))
        .unwrap_or_else(|| ("Unknown Backstory".to_string(), "Unwritten origins.".to_string()));

    let beat_specs = vec![
        (
            "Origin",
            format!(
                "Epic fantasy story scene concept art depicting the homeland roots of a {} operative aligned with {}. The same protagonist from the provided reference portrait must be visible in frame.",
                backstory_name, faction_name
            ),
            "Before the guild banners, your roots were forged in hard country.".to_string(),
        ),
        (
            "Crisis",
            format!(
                "Dark fantasy story scene concept art portraying the defining crisis from '{}' backstory: {}. The same protagonist from the provided reference portrait must be actively present in frame.",
                backstory_name, backstory_summary
            ),
            format!("Then came the break: {}.", backstory_summary),
        ),
        (
            "Decision",
            format!(
                "Fantasy story scene concept art of the decisive turning-point location where allegiance to {} was forged. The same protagonist from the provided reference portrait must be shown making the choice.",
                faction_name
            ),
            format!("At the turning point, you chose {} over retreat.", faction_name),
        ),
        (
            "Vow",
            format!(
                "Cinematic fantasy story scene concept art of the oath site representing present-day resolve for the {} path under {} banner. The same protagonist from the provided reference portrait must be centered in frame.",
                backstory_name, faction_name
            ),
            "Now your oath is fixed, and the campaign begins.".to_string(),
        ),
    ];

    beat_specs
        .into_iter()
        .enumerate()
        .map(|(idx, (title, prompt_seed, subtitle))| BackstoryCinematicBeat {
            index: idx,
            title: title.to_string(),
            subtitle,
            source_id: format!("backstory-beat-{}", idx),
            prompt: compose_runtime_backstory_scene_prompt(&prompt_seed, style),
            image_file: None,
            texture_handle: None,
        })
        .collect()
}

pub fn build_backstory_narrative_summary(
    character_creation: &CharacterCreationState,
    overworld: &game_core::OverworldMap,
) -> String {
    let faction_name = character_creation
        .selected_faction_index
        .and_then(|idx| overworld.factions.get(idx))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "Unknown Faction".to_string());
    let (backstory_name, backstory_summary) = build_backstory_selection_choices()
        .into_iter()
        .find(|choice| character_creation.selected_backstory_id.as_deref() == Some(choice.id))
        .map(|choice| (choice.name.to_string(), choice.summary.to_string()))
        .unwrap_or_else(|| ("Unknown Backstory".to_string(), "Unwritten origins.".to_string()));
    format!(
        "{} origin. {} That history drives your oath to {}.",
        backstory_name, backstory_summary, faction_name
    )
}

pub fn build_backstory_narrative_generation_prompt(
    character_creation: &CharacterCreationState,
    overworld: &game_core::OverworldMap,
    beats: &[BackstoryCinematicBeat],
) -> String {
    let faction_name = character_creation
        .selected_faction_index
        .and_then(|idx| overworld.factions.get(idx))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "Unknown Faction".to_string());
    let (backstory_name, backstory_summary) = build_backstory_selection_choices()
        .into_iter()
        .find(|choice| character_creation.selected_backstory_id.as_deref() == Some(choice.id))
        .map(|choice| (choice.name.to_string(), choice.summary.to_string()))
        .unwrap_or_else(|| ("Unknown Backstory".to_string(), "Unwritten origins.".to_string()));
    let beat_labels = beats
        .iter()
        .map(|beat| beat.title.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "You are writing concise cinematic narration for a fantasy game backstory montage.\n\
         Backstory archetype: {backstory_name}\n\
         Backstory facts: {backstory_summary}\n\
         Faction allegiance: {faction_name}\n\
         Beat order: {beat_labels}\n\
         Return STRICT JSON only with this schema:\n\
         {{\"summary\":\"...\",\"subtitles\":[\"...\",\"...\",\"...\",\"...\"]}}\n\
         Constraints:\n\
         - summary: one sentence, 18-32 words, present tense.\n\
         - subtitles: exactly {count} entries, one per beat in order.\n\
         - each subtitle: 7-16 words, direct and visual, no quotation marks.\n\
         - no markdown, no code fences, no extra keys.",
        count = beats.len()
    )
}

pub fn parse_backstory_narrative_payload(
    text: &str,
    beat_count: usize,
) -> Result<(String, Vec<String>), String> {
    let trimmed = text.trim();
    let json_slice = if trimmed.starts_with('{') {
        trimmed
    } else {
        let start = trimmed
            .find('{')
            .ok_or_else(|| "narrative payload missing JSON object start".to_string())?;
        let end = trimmed
            .rfind('}')
            .ok_or_else(|| "narrative payload missing JSON object end".to_string())?;
        &trimmed[start..=end]
    };
    let value: serde_json::Value = serde_json::from_str(json_slice)
        .map_err(|err| format!("narrative json parse failed: {err}"))?;
    let summary = value
        .get("summary")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "narrative summary missing".to_string())?
        .to_string();
    let subtitles = value
        .get("subtitles")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "narrative subtitles missing".to_string())?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if subtitles.len() < beat_count {
        return Err(format!(
            "narrative subtitles insufficient: expected {beat_count}, got {}",
            subtitles.len()
        ));
    }
    Ok((summary, subtitles.into_iter().take(beat_count).collect()))
}

pub fn build_character_portrait_prompt(
    character_creation: &CharacterCreationState,
    overworld: &game_core::OverworldMap,
) -> String {
    let faction_name = character_creation
        .selected_faction_index
        .and_then(|idx| overworld.factions.get(idx))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "Unknown Faction".to_string());
    let backstory_name = build_backstory_selection_choices()
        .into_iter()
        .find(|choice| character_creation.selected_backstory_id.as_deref() == Some(choice.id))
        .map(|choice| choice.name.to_string())
        .unwrap_or_else(|| "Wanderer".to_string());
    format!(
        "Fantasy line art portrait illustration, half-body hero from faction {} with {} archetype. High-contrast clean contour lines, readable face, minimal hatch shading, neutral parchment background for reference usage, no text, single subject.",
        faction_name, backstory_name
    )
}

pub fn enqueue_backstory_cinematic_jobs(
    asset_gen: &mut RuntimeAssetGenState,
    beats: &[BackstoryCinematicBeat],
    portrait_reference: Option<PathBuf>,
) {
    let style = RuntimeAssetStyle::LineArt;
    for beat in beats.iter().rev() {
        let job = RuntimeAssetJob {
            id: asset_gen.next_job_id,
            source_id: beat.source_id.clone(),
            source_title: format!("Backstory {}", beat.title),
            prompt: beat.prompt.clone(),
            style,
            kind: RuntimeAssetJobKind::EnvironmentScene,
            reference_image_path: portrait_reference.clone(),
            scene_tag: Some("backstory_cinematic".to_string()),
            sequence_index: Some(beat.index),
        };
        asset_gen.next_job_id += 1;
        asset_gen.pending.push_front(job);
    }
}

pub fn enqueue_backstory_portrait_job(
    asset_gen: &mut RuntimeAssetGenState,
    portrait_prompt: String,
) {
    let job = RuntimeAssetJob {
        id: asset_gen.next_job_id,
        source_id: "backstory-character-portrait".to_string(),
        source_title: "Backstory Character Portrait".to_string(),
        prompt: portrait_prompt,
        style: RuntimeAssetStyle::LineArt,
        kind: RuntimeAssetJobKind::CharacterPortrait,
        reference_image_path: None,
        scene_tag: Some("backstory_cinematic".to_string()),
        sequence_index: None,
    };
    asset_gen.next_job_id += 1;
    asset_gen.pending.push_front(job);
}
