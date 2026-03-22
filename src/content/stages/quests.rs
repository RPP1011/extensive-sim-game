use crate::content::schema::QuestContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const QUEST_TEMPLATES: &[(&str, &[&str])] = &[
    (
        "Retrieve the Lost {artifact}",
        &["Find the artifact's location", "Defeat the guardian", "Return to the quest giver"],
    ),
    (
        "Defend {settlement} from Raiders",
        &["Prepare defenses", "Repel the first wave", "Defeat the raid leader"],
    ),
    (
        "Investigate the {location} Ruins",
        &["Travel to the ruins", "Explore the lower levels", "Recover the ancient text"],
    ),
    (
        "Escort the {npc} to Safety",
        &["Meet the NPC at the rendezvous", "Guard through hostile territory", "Deliver safely"],
    ),
    (
        "Hunt the {creature} of {region}",
        &["Gather intel on the creature", "Track it to its lair", "Slay or capture"],
    ),
    (
        "Broker Peace between {faction_a} and {faction_b}",
        &["Speak with both leaders", "Find common ground", "Present the accord"],
    ),
];

const ARTIFACTS: &[&str] = &["Sunblade", "Moonshard", "Crown of Echoes", "Tome of Binding", "Starforge Hammer"];
const CREATURES: &[&str] = &["Wyrm", "Shade Beast", "Iron Golem", "Blight Spider", "Storm Raptor"];

pub struct QuestStage;

impl GenerationStage for QuestStage {
    fn id(&self) -> StageId {
        StageId::Quests
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

        let num_quests = rng.next_usize_range(5, 10);
        let region_names: Vec<String> = ctx.regions.iter().map(|r| r.name.clone()).collect();
        let settlement_names: Vec<String> = ctx.settlements.iter().map(|s| s.name.clone()).collect();
        let faction_names: Vec<String> = ctx.factions.iter().map(|f| f.name.clone()).collect();
        let npc_names: Vec<String> = ctx.npcs.iter().map(|n| n.name.clone()).collect();

        for i in 0..num_quests {
            let (template_name, template_objectives) = rng.choose(QUEST_TEMPLATES);

            let name = template_name
                .replace("{artifact}", *rng.choose(ARTIFACTS))
                .replace("{creature}", *rng.choose(CREATURES))
                .replace(
                    "{settlement}",
                    if settlement_names.is_empty() { "the town" } else { rng.choose(&settlement_names).as_str() },
                )
                .replace(
                    "{region}",
                    if region_names.is_empty() { "the wilds" } else { rng.choose(&region_names).as_str() },
                )
                .replace(
                    "{npc}",
                    if npc_names.is_empty() { "the stranger" } else { rng.choose(&npc_names).as_str() },
                )
                .replace(
                    "{faction_a}",
                    if faction_names.len() >= 2 { &faction_names[0] } else { "the first faction" },
                )
                .replace(
                    "{faction_b}",
                    if faction_names.len() >= 2 { &faction_names[1] } else { "the second faction" },
                )
                .replace(
                    "{location}",
                    if region_names.is_empty() { "the ancient" } else { rng.choose(&region_names).as_str() },
                );

            let objectives: Vec<String> = template_objectives.iter().map(|s| s.to_string()).collect();
            let reward_description = format!("Gold and renown (reward tier {})", (i % 3) + 1);

            ctx.quests.push(QuestContent {
                name,
                description: format!("An adventure awaits in the lands shaped by {}.",
                    ctx.theme.as_ref().map(|t| t.mood.as_str()).unwrap_or("fate")),
                objectives,
                reward_description,
                prerequisite_quest_ids: Vec::new(),
            });
        }

        Ok(())
    }
}
