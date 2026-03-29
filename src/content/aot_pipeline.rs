//! Ahead-of-time content generation pipeline.
//!
//! Runs 9 sequential world-building stages that accumulate a [`WorldContext`],
//! each using an optional [`ModelClient`] for text generation with deterministic
//! procedural fallbacks when no model is available.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::registry::{ContentEntry, ContentId, ContentKind, ContentRegistry, ContentTier};
use super::schema::*;
use crate::model_backend::{ModelClient, ModelConfig};

use super::stages;

// ---------------------------------------------------------------------------
// Pipeline error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum PipelineError {
    /// A stage failed with an error message.
    StageFailed { stage: StageId, message: String },
    /// IO error during disk caching.
    Io(std::io::Error),
    /// JSON serialization error.
    Json(serde_json::Error),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::StageFailed { stage, message } => {
                write!(f, "stage {stage:?} failed: {message}")
            }
            PipelineError::Io(e) => write!(f, "IO error: {e}"),
            PipelineError::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::Io(e)
    }
}

impl From<serde_json::Error> for PipelineError {
    fn from(e: serde_json::Error) -> Self {
        PipelineError::Json(e)
    }
}

// ---------------------------------------------------------------------------
// Stage identifier (ordered)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageId {
    Theme = 0,
    Factions = 1,
    Geography = 2,
    Settlements = 3,
    Npcs = 4,
    Quests = 5,
    Events = 6,
    Items = 7,
    NarrativeArcs = 8,
}

impl StageId {
    /// All stages in pipeline order.
    pub const ALL: [StageId; 9] = [
        StageId::Theme,
        StageId::Factions,
        StageId::Geography,
        StageId::Settlements,
        StageId::Npcs,
        StageId::Quests,
        StageId::Events,
        StageId::Items,
        StageId::NarrativeArcs,
    ];

    pub fn index(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// World context (accumulated across stages)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldContext {
    pub seed: u64,
    pub theme: Option<ThemeContent>,
    pub factions: Vec<FactionContent>,
    pub regions: Vec<RegionContent>,
    pub settlements: Vec<SettlementContent>,
    pub npcs: Vec<NpcContent>,
    pub quests: Vec<QuestContent>,
    pub events: Vec<EventContent>,
    pub items: Vec<ItemContent>,
    pub narrative_arcs: Vec<NarrativeArcContent>,
    /// Tracks which stages have been completed.
    pub completed_stages: Vec<StageId>,
}

// ---------------------------------------------------------------------------
// Generation stage trait
// ---------------------------------------------------------------------------

/// A single stage in the AOT pipeline.
pub trait GenerationStage {
    fn id(&self) -> StageId;
    fn run(
        &self,
        ctx: &mut WorldContext,
        model: Option<&ModelClient>,
        rng: &mut Lcg,
    ) -> Result<(), PipelineError>;
}

// ---------------------------------------------------------------------------
// LCG (copied from mission/room_gen/lcg.rs for independence)
// ---------------------------------------------------------------------------

pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        let s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut lcg = Self(s);
        for _ in 0..8 {
            lcg.next_u64();
        }
        lcg
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
    }

    pub fn next_usize_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        let range = (hi - lo + 1) as u64;
        lo + (self.next_u64() % range) as usize
    }

    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let idx = self.next_u64() as usize % items.len();
        &items[idx]
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_u64() as usize % (i + 1);
            items.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline config and orchestrator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AotPipelineConfig {
    pub campaign_seed: u64,
    pub model_config: Option<ModelConfig>,
    pub output_dir: PathBuf,
    /// Which stages to run. Empty = all stages.
    pub stages_to_run: Vec<StageId>,
}

impl Default for AotPipelineConfig {
    fn default() -> Self {
        Self {
            campaign_seed: 42,
            model_config: None,
            output_dir: PathBuf::from("generated/campaigns"),
            stages_to_run: Vec::new(),
        }
    }
}

/// The AOT content generation pipeline.
pub struct AotPipeline {
    stages: Vec<Box<dyn GenerationStage>>,
}

impl AotPipeline {
    /// Build the default 9-stage pipeline.
    pub fn new() -> Self {
        Self {
            stages: vec![
                Box::new(stages::ThemeStage),
                Box::new(stages::FactionStage),
                Box::new(stages::GeographyStage),
                Box::new(stages::SettlementStage),
                Box::new(stages::NpcStage),
                Box::new(stages::QuestStage),
                Box::new(stages::EventStage),
                Box::new(stages::ItemStage),
                Box::new(stages::NarrativeStage),
            ],
        }
    }

    /// Run the full pipeline (or a subset of stages).
    pub fn run(
        &self,
        config: &AotPipelineConfig,
        registry: &mut ContentRegistry,
    ) -> Result<WorldContext, PipelineError> {
        let model = config.model_config.as_ref().map(|c| ModelClient::new(c.clone()));
        let mut ctx = WorldContext {
            seed: config.campaign_seed,
            ..Default::default()
        };

        // Try to load existing context from disk for partial regeneration.
        let ctx_path = config
            .output_dir
            .join(format!("{}", config.campaign_seed))
            .join("world_context.json");
        if ctx_path.exists() {
            if let Ok(data) = std::fs::read_to_string(&ctx_path) {
                if let Ok(loaded) = serde_json::from_str::<WorldContext>(&data) {
                    ctx = loaded;
                }
            }
        }

        let stages_filter: Vec<StageId> = if config.stages_to_run.is_empty() {
            StageId::ALL.to_vec()
        } else {
            config.stages_to_run.clone()
        };

        for stage in &self.stages {
            if !stages_filter.contains(&stage.id()) {
                continue;
            }
            // Per-stage RNG: campaign_seed mixed with stage index for independence.
            let stage_seed = config
                .campaign_seed
                .wrapping_add((stage.id().index() as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
            let mut rng = Lcg::new(stage_seed);

            stage.run(&mut ctx, model.as_ref(), &mut rng)?;
            ctx.completed_stages.push(stage.id());

            // Cache to disk after each stage.
            let out_dir = config
                .output_dir
                .join(format!("{}", config.campaign_seed));
            std::fs::create_dir_all(&out_dir)?;
            let json = serde_json::to_string_pretty(&ctx)?;
            std::fs::write(out_dir.join("world_context.json"), json)?;
        }

        // Register generated content into the registry.
        register_world_context(&ctx, registry);

        Ok(ctx)
    }
}

/// Push all generated content from a `WorldContext` into the `ContentRegistry`.
fn register_world_context(ctx: &WorldContext, registry: &mut ContentRegistry) {
    if let Some(ref theme) = ctx.theme {
        let id = ContentId::gen(ContentKind::Theme, &theme.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Theme(theme.clone()),
        });
    }

    for f in &ctx.factions {
        let id = ContentId::gen(ContentKind::Faction, &f.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Faction(f.clone()),
        });
    }

    for r in &ctx.regions {
        let id = ContentId::gen(ContentKind::Region, &r.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Region(r.clone()),
        });
    }

    for s in &ctx.settlements {
        let id = ContentId::gen(ContentKind::Settlement, &s.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Settlement(s.clone()),
        });
    }

    for n in &ctx.npcs {
        let id = ContentId::gen(ContentKind::Npc, &n.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Npc(n.clone()),
        });
    }

    for q in &ctx.quests {
        let id = ContentId::gen(ContentKind::Quest, &q.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Quest(q.clone()),
        });
    }

    for e in &ctx.events {
        let id = ContentId::gen(ContentKind::Event, &e.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Event(e.clone()),
        });
    }

    for it in &ctx.items {
        let id = ContentId::gen(ContentKind::Item, &it.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::Item(it.clone()),
        });
    }

    for na in &ctx.narrative_arcs {
        let id = ContentId::gen(ContentKind::NarrativeArc, &na.name);
        registry.insert_unchecked(ContentEntry {
            id,
            tier: ContentTier::AotGenerated,
            data: ContentData::NarrativeArc(na.clone()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!("/tmp/test_aot_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn pipeline_runs_with_no_model() {
        let config = AotPipelineConfig {
            campaign_seed: 12345,
            model_config: None,
            output_dir: fresh_dir("full_run"),
            stages_to_run: vec![],
        };
        let mut registry = ContentRegistry::default();
        let pipeline = AotPipeline::new();
        let ctx = pipeline.run(&config, &mut registry).unwrap();
        assert_eq!(ctx.completed_stages.len(), 9);
        assert!(ctx.theme.is_some());
        assert!(!ctx.factions.is_empty());
        assert!(!ctx.regions.is_empty());
        assert!(!ctx.settlements.is_empty());
    }

    #[test]
    fn pipeline_partial_regeneration() {
        let config = AotPipelineConfig {
            campaign_seed: 99,
            model_config: None,
            output_dir: fresh_dir("partial"),
            stages_to_run: vec![StageId::Theme, StageId::Factions],
        };
        let mut registry = ContentRegistry::default();
        let pipeline = AotPipeline::new();
        let ctx = pipeline.run(&config, &mut registry).unwrap();
        assert_eq!(ctx.completed_stages.len(), 2);
        assert!(ctx.theme.is_some());
        assert!(!ctx.factions.is_empty());
        // Later stages not run
        assert!(ctx.regions.is_empty());
    }

    #[test]
    fn pipeline_deterministic_with_same_seed() {
        let run = |seed: u64, suffix: &str| {
            let config = AotPipelineConfig {
                campaign_seed: seed,
                model_config: None,
                output_dir: fresh_dir(suffix),
                stages_to_run: vec![StageId::Theme, StageId::Factions],
            };
            let mut registry = ContentRegistry::default();
            let pipeline = AotPipeline::new();
            pipeline.run(&config, &mut registry).unwrap()
        };
        let a = run(777, "det_a");
        let b = run(777, "det_b");
        assert_eq!(a.theme.as_ref().unwrap().name, b.theme.as_ref().unwrap().name);
        assert_eq!(a.factions.len(), b.factions.len());
        for (fa, fb) in a.factions.iter().zip(b.factions.iter()) {
            assert_eq!(fa.name, fb.name);
        }
    }
}
