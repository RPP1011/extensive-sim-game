use crate::content::schema::ThemeContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const THEME_MOODS: &[&str] = &[
    "dark and foreboding",
    "heroic and triumphant",
    "mysterious and ancient",
    "savage and untamed",
    "arcane and otherworldly",
    "grim and war-torn",
    "whimsical and enchanted",
    "desolate and forgotten",
];

const THEME_NAMES: &[&str] = &[
    "The Shattered Realms",
    "Age of Iron and Ash",
    "The Verdant Abyss",
    "Twilight of Empires",
    "The Frozen Concord",
    "Echoes of the First War",
    "The Sunken Dominion",
    "Rise of the Hollow Crown",
];

const THEME_KEYWORDS: &[&[&str]] = &[
    &["ruins", "shadow", "prophecy", "decay"],
    &["valor", "conquest", "glory", "steel"],
    &["nature", "corruption", "ancient", "overgrown"],
    &["politics", "betrayal", "decline", "legacy"],
    &["ice", "pact", "survival", "endurance"],
    &["war", "memory", "artifact", "echo"],
    &["ocean", "lost", "treasure", "depth"],
    &["crown", "void", "ascension", "bone"],
];

pub struct ThemeStage;

impl GenerationStage for ThemeStage {
    fn id(&self) -> StageId {
        StageId::Theme
    }

    fn run(
        &self,
        ctx: &mut WorldContext,
        model: Option<&ModelClient>,
        rng: &mut Lcg,
    ) -> Result<(), PipelineError> {
        if model.is_some() && model.unwrap().is_available() {
            // Model-backed generation would go here.
            // For now, fall through to procedural.
        }

        // Procedural fallback: deterministic theme selection.
        let idx = rng.next_usize_range(0, THEME_NAMES.len() - 1);
        let mood_idx = rng.next_usize_range(0, THEME_MOODS.len() - 1);

        let base_palette: Vec<[u8; 3]> = (0..5)
            .map(|_| {
                [
                    (rng.next_f32() * 200.0 + 40.0) as u8,
                    (rng.next_f32() * 200.0 + 40.0) as u8,
                    (rng.next_f32() * 200.0 + 40.0) as u8,
                ]
            })
            .collect();

        ctx.theme = Some(ThemeContent {
            name: THEME_NAMES[idx].to_string(),
            mood: THEME_MOODS[mood_idx].to_string(),
            color_palette: base_palette,
            keywords: THEME_KEYWORDS[idx].iter().map(|s| s.to_string()).collect(),
        });

        Ok(())
    }
}
