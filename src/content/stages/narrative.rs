use crate::content::schema::NarrativeArcContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const ARC_TEMPLATES: &[(&str, &[&str])] = &[
    (
        "The Rising Threat",
        &[
            "Act I: Omens — strange events signal trouble brewing.",
            "Act II: Escalation — the threat reveals itself and grows.",
            "Act III: Confrontation — heroes face the threat head-on.",
        ],
    ),
    (
        "The Broken Alliance",
        &[
            "Act I: Fractures — old allies begin to disagree.",
            "Act II: Betrayal — a faction breaks trust openly.",
            "Act III: Resolution — war or reconciliation follows.",
        ],
    ),
    (
        "The Lost Heir",
        &[
            "Act I: Discovery — evidence of a lost bloodline surfaces.",
            "Act II: Pursuit — factions compete to find or silence the heir.",
            "Act III: Coronation — the heir claims power or falls.",
        ],
    ),
    (
        "The Ancient Awakening",
        &[
            "Act I: Tremors — the world shifts, seals weaken.",
            "Act II: Emergence — an ancient force returns.",
            "Act III: Reckoning — civilization adapts or crumbles.",
        ],
    ),
    (
        "The Trade War",
        &[
            "Act I: Embargo — resources become scarce.",
            "Act II: Proxy Conflicts — mercenaries fight on behalf of merchants.",
            "Act III: New Order — trade routes are redrawn.",
        ],
    ),
];

pub struct NarrativeStage;

impl GenerationStage for NarrativeStage {
    fn id(&self) -> StageId {
        StageId::NarrativeArcs
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

        // 1-3 narrative arcs, each involving some factions.
        let count = rng.next_usize_range(1, 3);
        let faction_names: Vec<String> = ctx.factions.iter().map(|f| f.name.clone()).collect();
        let mut used: Vec<usize> = Vec::new();

        for _ in 0..count {
            let mut idx = rng.next_usize_range(0, ARC_TEMPLATES.len() - 1);
            while used.contains(&idx) && used.len() < ARC_TEMPLATES.len() {
                idx = (idx + 1) % ARC_TEMPLATES.len();
            }
            used.push(idx);

            let (name, acts) = ARC_TEMPLATES[idx];

            // Assign 1-2 factions to this arc.
            let n_factions = rng.next_usize_range(1, faction_names.len().min(2));
            let mut arc_factions = Vec::new();
            for _ in 0..n_factions {
                if !faction_names.is_empty() {
                    arc_factions.push(rng.choose(&faction_names).clone());
                }
            }

            ctx.narrative_arcs.push(NarrativeArcContent {
                name: name.to_string(),
                acts: acts.iter().map(|s| s.to_string()).collect(),
                faction_ids: arc_factions,
            });
        }

        Ok(())
    }
}
