use crate::content::schema::FactionContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const FACTION_PREFIXES: &[&str] = &[
    "The Order of",
    "The Brotherhood of",
    "House",
    "The Free Company of",
    "The Cult of",
    "The Guild of",
    "The Legion of",
    "Clan",
];

const FACTION_NOUNS: &[&str] = &[
    "the Iron Crown",
    "Ashen Wolves",
    "the Emerald Flame",
    "Stormwatch",
    "the Silent Hand",
    "the Golden Veil",
    "Thornwall",
    "the Black Tide",
    "Duskhollow",
    "the Crimson Pact",
];

const FACTION_MOTTOS: &[&str] = &[
    "Strength through unity.",
    "In shadow we thrive.",
    "Honor above all.",
    "The old ways endure.",
    "From ashes, glory.",
    "Knowledge is dominion.",
    "Blood remembers.",
    "By fire, forged.",
];

pub struct FactionStage;

impl GenerationStage for FactionStage {
    fn id(&self) -> StageId {
        StageId::Factions
    }

    fn run(
        &self,
        ctx: &mut WorldContext,
        model: Option<&ModelClient>,
        rng: &mut Lcg,
    ) -> Result<(), PipelineError> {
        if model.is_some() && model.unwrap().is_available() {
            // Model-backed generation placeholder.
        }

        let count = rng.next_usize_range(3, 6);
        let mut used_nouns: Vec<usize> = Vec::new();

        for _ in 0..count {
            let prefix = rng.choose(FACTION_PREFIXES);
            let mut noun_idx = rng.next_usize_range(0, FACTION_NOUNS.len() - 1);
            // Avoid duplicate faction nouns.
            while used_nouns.contains(&noun_idx) {
                noun_idx = (noun_idx + 1) % FACTION_NOUNS.len();
            }
            used_nouns.push(noun_idx);

            let name = format!("{} {}", prefix, FACTION_NOUNS[noun_idx]);
            let motto = rng.choose(FACTION_MOTTOS).to_string();
            let color = [
                (rng.next_f32() * 200.0 + 30.0) as u8,
                (rng.next_f32() * 200.0 + 30.0) as u8,
                (rng.next_f32() * 200.0 + 30.0) as u8,
            ];

            let theme_keywords = ctx
                .theme
                .as_ref()
                .map(|t| t.keywords.join(", "))
                .unwrap_or_default();
            let description = format!(
                "A faction driven by ambition in a world of {}.",
                theme_keywords
            );

            ctx.factions.push(FactionContent {
                name,
                description,
                color_rgb: color,
                motto,
            });
        }

        Ok(())
    }
}
