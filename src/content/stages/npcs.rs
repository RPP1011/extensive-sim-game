use crate::content::schema::NpcContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const FIRST_NAMES: &[&str] = &[
    "Aldric", "Brenna", "Cael", "Dara", "Elric", "Freya", "Gareth", "Hild",
    "Ivar", "Jael", "Kara", "Loric", "Mira", "Nolan", "Orin", "Petra",
    "Quinn", "Rowan", "Sera", "Theron", "Una", "Voss", "Wren", "Yara", "Zane",
];

const ROLES: &[&str] = &[
    "merchant", "blacksmith", "healer", "scholar", "guard_captain",
    "tavern_keeper", "scout", "elder", "spy", "courier",
    "priest", "alchemist", "ranger", "diplomat", "refugee",
];

pub struct NpcStage;

impl GenerationStage for NpcStage {
    fn id(&self) -> StageId {
        StageId::Npcs
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

        // Generate 2-4 NPCs per settlement.
        let settlements_snapshot: Vec<(String, String)> = ctx
            .settlements
            .iter()
            .map(|s| (s.name.clone(), s.faction_id.clone()))
            .collect();

        for (settlement_name, faction_id) in &settlements_snapshot {
            let count = rng.next_usize_range(2, 4);
            for _ in 0..count {
                let name = rng.choose(FIRST_NAMES).to_string();
                let role = rng.choose(ROLES).to_string();

                ctx.npcs.push(NpcContent {
                    name,
                    role,
                    faction_id: faction_id.clone(),
                    settlement_id: Some(settlement_name.clone()),
                    dialogue_ids: Vec::new(), // Linked later by narrative system.
                });
            }
        }

        Ok(())
    }
}
