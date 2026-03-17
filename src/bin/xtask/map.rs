use std::fs;
use std::process::ExitCode;

use bevy_game::mapgen_voronoi::{build_prompt, build_spec, load_overworld};

use super::cli::MapVoronoiArgs;

pub fn run_map_voronoi(args: MapVoronoiArgs) -> ExitCode {
    let save_data = match fs::read_to_string(&args.save) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to load save {}: {err}", args.save.display());
            return ExitCode::from(2);
        }
    };

    let save_json: serde_json::Value = match serde_json::from_str(&save_data) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to parse save {}: {err}", args.save.display());
            return ExitCode::from(2);
        }
    };

    let overworld = match load_overworld(&save_json) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to load overworld data: {err}");
            return ExitCode::from(2);
        }
    };

    let spec = build_spec(
        &overworld,
        args.grid_w,
        args.grid_h,
        args.strength_scale,
        args.organic_jitter,
    );
    let prompt = build_prompt(&spec);

    if let Some(parent) = args.out_prompt.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("Failed to create {}: {err}", parent.display());
            return ExitCode::from(1);
        }
    }
    if let Err(err) = fs::write(&args.out_prompt, format!("{prompt}\n")) {
        eprintln!("Failed to write {}: {err}", args.out_prompt.display());
        return ExitCode::from(1);
    }
    println!("Wrote prompt: {}", args.out_prompt.display());

    if let Some(parent) = args.out_spec.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("Failed to create {}: {err}", parent.display());
            return ExitCode::from(1);
        }
    }
    let spec_json = match serde_json::to_string_pretty(&spec) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to serialize spec: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = fs::write(&args.out_spec, format!("{spec_json}\n")) {
        eprintln!("Failed to write {}: {err}", args.out_spec.display());
        return ExitCode::from(1);
    }
    println!("Wrote spec:   {}", args.out_spec.display());

    ExitCode::SUCCESS
}
