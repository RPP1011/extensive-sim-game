use std::process::ExitCode;

use bevy_game::content::aot_pipeline::{AotPipeline, AotPipelineConfig, StageId};
use bevy_game::content::ContentRegistry;

use super::cli::{ContentGenCommand, ContentGenSubcommand};

pub fn run_content_gen_cmd(cmd: ContentGenCommand) -> ExitCode {
    match cmd.sub {
        ContentGenSubcommand::Generate(args) => run_generate(args),
        ContentGenSubcommand::Inspect(args) => run_inspect(args),
    }
}

fn run_generate(args: super::cli::ContentGenGenerateArgs) -> ExitCode {
    println!(
        "=== AOT Content Generation (seed: {}) ===\n",
        args.seed
    );

    let stages = if let Some(ref from) = args.from_stage {
        match parse_stage(from) {
            Some(stage) => {
                let idx = stage.index();
                StageId::ALL[idx..].to_vec()
            }
            None => {
                eprintln!("Unknown stage: {from}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        Vec::new() // Empty = all stages.
    };

    let config = AotPipelineConfig {
        campaign_seed: args.seed,
        model_config: None, // Procedural fallback (model configured separately).
        output_dir: args.output_dir.clone(),
        stages_to_run: stages,
    };

    let mut registry = ContentRegistry::default();
    let pipeline = AotPipeline::new();

    let start = std::time::Instant::now();
    match pipeline.run(&config, &mut registry) {
        Ok(ctx) => {
            let elapsed = start.elapsed();
            println!("Pipeline completed in {:.2}s\n", elapsed.as_secs_f64());
            print_context_summary(&ctx);
            println!("\nContent cached to: {}/{}",
                args.output_dir.display(), args.seed);
            println!("Registry entries: {}", registry.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Pipeline failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_inspect(args: super::cli::ContentGenInspectArgs) -> ExitCode {
    let ctx_path = args
        .output_dir
        .join(format!("{}", args.seed))
        .join("world_context.json");

    if !ctx_path.exists() {
        println!("No cached context found for seed {} at {}", args.seed, ctx_path.display());
        println!("Run `content-gen generate --seed {}` first.", args.seed);
        return ExitCode::SUCCESS;
    }

    let data = match std::fs::read_to_string(&ctx_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read {}: {e}", ctx_path.display());
            return ExitCode::FAILURE;
        }
    };

    let ctx: bevy_game::content::aot_pipeline::WorldContext = match serde_json::from_str(&data) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to parse context: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("=== World Context (seed: {}) ===\n", args.seed);
    print_context_summary(&ctx);

    ExitCode::SUCCESS
}

fn print_context_summary(ctx: &bevy_game::content::aot_pipeline::WorldContext) {
    if let Some(ref theme) = ctx.theme {
        println!("Theme: \"{}\" (mood: {})", theme.name, theme.mood);
        println!("  Keywords: {}", theme.keywords.join(", "));
    }
    println!("Factions: {}", ctx.factions.len());
    for f in &ctx.factions {
        println!("  - {} ({})", f.name, f.motto);
    }
    println!("Regions: {}", ctx.regions.len());
    for r in &ctx.regions {
        println!("  - {} [{}]", r.name, r.terrain_type);
    }
    println!("Settlements: {}", ctx.settlements.len());
    println!("NPCs: {}", ctx.npcs.len());
    println!("Quests: {}", ctx.quests.len());
    for q in &ctx.quests {
        println!("  - {}", q.name);
    }
    println!("Events: {}", ctx.events.len());
    println!("Items: {}", ctx.items.len());
    println!("Narrative arcs: {}", ctx.narrative_arcs.len());
    for n in &ctx.narrative_arcs {
        println!("  - {} ({} acts)", n.name, n.acts.len());
    }
    println!("Completed stages: {}", ctx.completed_stages.len());
}

fn parse_stage(s: &str) -> Option<StageId> {
    match s.to_lowercase().as_str() {
        "theme" => Some(StageId::Theme),
        "factions" => Some(StageId::Factions),
        "geography" => Some(StageId::Geography),
        "settlements" => Some(StageId::Settlements),
        "npcs" => Some(StageId::Npcs),
        "quests" => Some(StageId::Quests),
        "events" => Some(StageId::Events),
        "items" => Some(StageId::Items),
        "narrative" | "narrative_arcs" => Some(StageId::NarrativeArcs),
        _ => None,
    }
}
