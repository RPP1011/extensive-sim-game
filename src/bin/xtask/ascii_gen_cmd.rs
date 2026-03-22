use std::process::ExitCode;

use bevy_game::ascii_gen::{AsciiArtGenerator, AsciiArtRequest, AsciiArtStyle};

use super::cli::{AsciiGenCommand, AsciiGenSubcommand};

pub fn run_ascii_gen_cmd(cmd: AsciiGenCommand) -> ExitCode {
    match cmd.sub {
        AsciiGenSubcommand::Generate(args) => run_generate(args),
        AsciiGenSubcommand::Export(args) => run_export(args),
    }
}

fn run_generate(args: super::cli::AsciiGenGenerateArgs) -> ExitCode {
    let style = match args.style.to_lowercase().as_str() {
        "environment" | "env" => AsciiArtStyle::Environment,
        "portrait" | "character" => AsciiArtStyle::CharacterPortrait,
        "item" | "icon" => AsciiArtStyle::ItemIcon,
        "ui" | "decoration" => AsciiArtStyle::UiDecoration,
        _ => {
            eprintln!(
                "Unknown style: '{}'. Use: environment, portrait, item, ui",
                args.style
            );
            return ExitCode::FAILURE;
        }
    };

    let generator = AsciiArtGenerator::new(None); // Procedural only for now.
    let request = AsciiArtRequest {
        prompt: args.prompt.clone(),
        style,
        width: args.width,
        height: args.height,
        seed: args.seed,
        palette_constraint: None,
    };

    match generator.generate(&request) {
        Ok(grid) => {
            println!(
                "=== ASCII Art ({}x{}, seed={}, style={}) ===\n",
                args.width, args.height, args.seed, args.style
            );
            print!("{}", grid.to_plain_text());
            println!("\n=== End ===");

            if let Some(ref path) = args.output {
                let json = serde_json::to_string_pretty(&grid).unwrap();
                if let Err(e) = std::fs::write(path, json) {
                    eprintln!("Failed to write {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
                println!("Grid saved to: {}", path.display());
            }

            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_export(args: super::cli::AsciiGenExportArgs) -> ExitCode {
    use std::io::Write;

    let generator = AsciiArtGenerator::new(None);
    let styles = [
        AsciiArtStyle::Environment,
        AsciiArtStyle::CharacterPortrait,
        AsciiArtStyle::ItemIcon,
    ];

    let output_path = &args.output;
    if let Some(parent) = output_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let file = match std::fs::File::create(output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create {}: {e}", output_path.display());
            return ExitCode::FAILURE;
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    let mut count = 0u64;
    for style in &styles {
        for seed_offset in 0..args.count_per_style {
            let seed = args.seed + seed_offset as u64;
            let request = AsciiArtRequest {
                prompt: format!("{:?} scene", style),
                style: *style,
                width: args.width,
                height: args.height,
                seed,
                palette_constraint: None,
            };

            match generator.generate(&request) {
                Ok(grid) => {
                    let json = serde_json::to_string(&grid).unwrap();
                    writeln!(writer, "{}", json).unwrap();
                    count += 1;
                }
                Err(e) => {
                    eprintln!("Seed {seed} failed: {e}");
                }
            }
        }
    }

    println!("Exported {count} ASCII art grids to {}", output_path.display());
    ExitCode::SUCCESS
}
