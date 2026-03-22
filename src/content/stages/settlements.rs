use crate::content::schema::SettlementContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const SETTLEMENT_PREFIXES: &[&str] = &[
    "Fort", "Port", "Haven", "New", "Old", "Upper", "Lower", "Iron", "Silver", "Stone",
];

const SETTLEMENT_SUFFIXES: &[&str] = &[
    "wall", "gate", "hold", "ford", "bridge", "keep", "watch", "holm", "crest", "vale",
    "hollow", "march", "fell", "stead", "moor",
];

pub struct SettlementStage;

impl GenerationStage for SettlementStage {
    fn id(&self) -> StageId {
        StageId::Settlements
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

        // Generate 1-3 settlements per region.
        let regions_snapshot: Vec<(String, String)> = ctx
            .regions
            .iter()
            .map(|r| (r.name.clone(), r.terrain_type.clone()))
            .collect();

        let faction_names: Vec<String> = ctx.factions.iter().map(|f| f.name.clone()).collect();

        for (region_idx, (region_name, terrain)) in regions_snapshot.iter().enumerate() {
            let num_settlements = rng.next_usize_range(1, 3);
            for _ in 0..num_settlements {
                let prefix = rng.choose(SETTLEMENT_PREFIXES);
                let suffix = rng.choose(SETTLEMENT_SUFFIXES);
                let name = format!("{}{}", prefix, suffix);

                let population = (rng.next_f32() * 9000.0 + 1000.0) as u32;

                let faction_id = if faction_names.is_empty() {
                    "unaligned".to_string()
                } else {
                    rng.choose(&faction_names).clone()
                };

                let description = format!(
                    "A {} settlement in the {} region, population {}.",
                    terrain, region_name, population
                );

                let settlement_name = name.clone();
                ctx.settlements.push(SettlementContent {
                    name,
                    region: region_name.clone(),
                    population,
                    description,
                    faction_id,
                });

                // Back-link to region.
                if region_idx < ctx.regions.len() {
                    ctx.regions[region_idx]
                        .settlement_ids
                        .push(settlement_name);
                }
            }
        }

        Ok(())
    }
}
