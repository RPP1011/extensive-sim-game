use crate::content::schema::EventContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const EVENT_TEMPLATES: &[(&str, &str, &[&str], &[&str])] = &[
    (
        "Bandit Raids",
        "Organised bandits target trade routes and settlements.",
        &["faction_tension_high", "weak_garrison"],
        &["trade_disrupted", "faction_reputation_change"],
    ),
    (
        "Plague Outbreak",
        "A mysterious illness spreads through the population.",
        &["settlement_overcrowded", "season_change"],
        &["population_decline", "healer_demand"],
    ),
    (
        "Festival of the Harvest Moon",
        "A regional celebration that lifts morale and attracts travelers.",
        &["season_autumn", "peace_period"],
        &["morale_boost", "trade_boost"],
    ),
    (
        "Arcane Storm",
        "Magical disturbances erupt across the region.",
        &["leyline_surge", "artifact_activated"],
        &["terrain_change", "creature_spawn"],
    ),
    (
        "Diplomatic Summit",
        "Faction leaders convene to discuss borders and alliances.",
        &["war_ended", "new_leader"],
        &["alliance_formed", "territory_exchange"],
    ),
    (
        "Mine Collapse",
        "A critical resource mine suffers a catastrophic failure.",
        &["overworked_mine", "earthquake"],
        &["resource_shortage", "rescue_quest"],
    ),
    (
        "Dragon Sighting",
        "A great wyrm is spotted near civilization for the first time in ages.",
        &["ancient_seal_broken", "hoard_discovered"],
        &["panic", "bounty_posted", "adventurer_influx"],
    ),
];

pub struct EventStage;

impl GenerationStage for EventStage {
    fn id(&self) -> StageId {
        StageId::Events
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

        let count = rng.next_usize_range(3, 6);
        let mut used: Vec<usize> = Vec::new();

        for _ in 0..count {
            let mut idx = rng.next_usize_range(0, EVENT_TEMPLATES.len() - 1);
            while used.contains(&idx) && used.len() < EVENT_TEMPLATES.len() {
                idx = (idx + 1) % EVENT_TEMPLATES.len();
            }
            used.push(idx);

            let (name, desc, triggers, effects) = EVENT_TEMPLATES[idx];
            ctx.events.push(EventContent {
                name: name.to_string(),
                description: desc.to_string(),
                triggers: triggers.iter().map(|s| s.to_string()).collect(),
                effects: effects.iter().map(|s| s.to_string()).collect(),
            });
        }

        Ok(())
    }
}
