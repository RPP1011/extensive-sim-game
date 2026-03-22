use std::process::ExitCode;

use bevy_game::model_backend::{
    detect_best_tier, detect_gpu_available, ModelClient, ModelConfig, ModelTier, ProviderConfig,
};

use super::cli::{ModelCommand, ModelSubcommand};

pub fn run_model_cmd(cmd: ModelCommand) -> ExitCode {
    match cmd.sub {
        ModelSubcommand::Status => run_status(),
        ModelSubcommand::Test(args) => run_test(args),
    }
}

fn run_status() -> ExitCode {
    println!("=== Model Backend Status ===\n");

    let gpu = detect_gpu_available();
    println!("GPU available: {}", if gpu { "yes" } else { "no" });
    println!("Recommended tier: {}", detect_best_tier());

    let client = ModelClient::new(ModelConfig::default());
    println!("Default backend available: {}", client.is_available());
    println!(
        "\nNote: Configure a provider (Subprocess) in your model config"
    );
    println!("to enable model-backed generation. Without a model, all");
    println!("pipelines use deterministic procedural fallbacks.");

    ExitCode::SUCCESS
}

fn run_test(args: super::cli::ModelTestArgs) -> ExitCode {
    let config = ModelConfig {
        tier: ModelTier::Micro,
        provider: if let (Some(model), Some(script)) = (&args.model_path, &args.script_path) {
            ProviderConfig::Subprocess {
                model_path: model.clone(),
                script_path: script.clone(),
                python: args.python.clone().unwrap_or_else(|| "python3".to_string()),
            }
        } else {
            ProviderConfig::None
        },
        seed: Some(args.seed),
        ..Default::default()
    };

    let client = ModelClient::new(config);

    if !client.is_available() {
        println!("No model backend available.");
        println!("Pass --model-path and --script-path to test subprocess backend.");
        return ExitCode::SUCCESS;
    }

    println!("Testing model with prompt: \"{}\"", args.prompt);
    match client.generate(&args.prompt, args.seed) {
        Ok(text) => {
            println!("\n--- Generated Output ---\n{}\n--- End ---", text);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            ExitCode::FAILURE
        }
    }
}
