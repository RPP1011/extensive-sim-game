use crate::content::schema::RegionContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const TERRAIN_TYPES: &[&str] = &[
    "plains", "forest", "mountains", "swamp", "desert", "tundra", "coast", "highlands",
];

const REGION_PREFIXES: &[&str] = &[
    "The", "Northern", "Southern", "Eastern", "Western", "Great", "Lost", "Old",
];

const REGION_NOUNS: &[&str] = &[
    "Reach", "Marches", "Wilds", "Expanse", "Wastes", "Hollows", "Fjords", "Steppe",
    "Heartlands", "Barrens", "Glades", "Peaks", "Drifts", "Mire", "Shores",
];

pub struct GeographyStage;

impl GenerationStage for GeographyStage {
    fn id(&self) -> StageId {
        StageId::Geography
    }

    fn run(
        &self,
        ctx: &mut WorldContext,
        model: Option<&ModelClient>,
        rng: &mut Lcg,
    ) -> Result<(), PipelineError> {
        if model.is_some() && model.unwrap().is_available() {
            // Model-backed placeholder.
        }

        let count = rng.next_usize_range(4, 8);
        for _ in 0..count {
            let prefix = rng.choose(REGION_PREFIXES);
            let noun = rng.choose(REGION_NOUNS);
            let terrain = rng.choose(TERRAIN_TYPES);

            let name = format!("{} {}", prefix, noun);
            let mood = ctx.theme.as_ref().map(|t| t.mood.as_str()).unwrap_or("neutral");
            let description = format!(
                "A {} region of {} terrain, {} in character.",
                name, terrain, mood
            );

            ctx.regions.push(RegionContent {
                name,
                terrain_type: terrain.to_string(),
                description,
                settlement_ids: Vec::new(), // Linked during settlements stage.
            });
        }

        Ok(())
    }
}
