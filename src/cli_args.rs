/// Command-line argument parsing for the game binary.

pub struct CliArgs {
    pub headless_mode: bool,
    pub simulation_steps: Option<u32>,
    pub run_phase0_sim: bool,
    pub run_phase1_sim: bool,
    pub run_phase2_sim: bool,
    pub run_phase3_sim: bool,
    pub run_phase4_sim: bool,
    pub run_phase5_sim: bool,
    pub run_phase6_report_flag: bool,
    pub run_phase6_viz_flag: bool,
    pub run_pathing_viz_flag: bool,
    pub run_pathing_hero_win_viz_flag: bool,
    pub run_phase7_sim: bool,
    pub run_phase8_sim: bool,
    pub run_phase9_sim: bool,
    pub run_pathing_sim: bool,
    pub run_pathing_hero_win_sim: bool,
    pub run_pathing_hero_hp_ablation: bool,
    pub run_viz_index_flag: bool,
    pub scenario_template_path: Option<String>,
    pub scenario_viz_path: Option<String>,
    pub scenario_viz_out_path: Option<String>,
    pub scenario_3d_path: Option<String>,
    pub map_seed: Option<u64>,
    pub campaign_load_path: Option<String>,
    pub horde_3d_flag: bool,
    pub horde_3d_hero_win_flag: bool,
    pub run_hub_flag: bool,
    pub run_dev_mode: bool,
    pub screenshot_dir: Option<String>,
    pub screenshot_sequence_dir: Option<String>,
    pub screenshot_hub_stages_dir: Option<String>,
    pub screenshot_every: u32,
    pub screenshot_warmup_frames: u32,
}

fn parse_seed_arg(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

/// Parse command-line arguments. Returns `None` if the program should exit
/// (due to an error message already printed to stderr).
pub fn parse_cli_args(args: &[String]) -> Option<CliArgs> {
    let mut cli = CliArgs {
        headless_mode: false,
        simulation_steps: None,
        run_phase0_sim: false,
        run_phase1_sim: false,
        run_phase2_sim: false,
        run_phase3_sim: false,
        run_phase4_sim: false,
        run_phase5_sim: false,
        run_phase6_report_flag: false,
        run_phase6_viz_flag: false,
        run_pathing_viz_flag: false,
        run_pathing_hero_win_viz_flag: false,
        run_phase7_sim: false,
        run_phase8_sim: false,
        run_phase9_sim: false,
        run_pathing_sim: false,
        run_pathing_hero_win_sim: false,
        run_pathing_hero_hp_ablation: false,
        run_viz_index_flag: false,
        scenario_template_path: None,
        scenario_viz_path: None,
        scenario_viz_out_path: None,
        scenario_3d_path: None,
        map_seed: None,
        campaign_load_path: None,
        horde_3d_flag: false,
        horde_3d_hero_win_flag: false,
        run_hub_flag: false,
        run_dev_mode: false,
        screenshot_dir: None,
        screenshot_sequence_dir: None,
        screenshot_hub_stages_dir: None,
        screenshot_every: 1,
        screenshot_warmup_frames: 3,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--headless" => {
                cli.headless_mode = true;
                i += 1;
            }
            "--steps" => {
                if let Some(steps_str) = args.get(i + 1) {
                    if let Ok(steps) = steps_str.parse::<u32>() {
                        cli.simulation_steps = Some(steps);
                        i += 2;
                    } else {
                        eprintln!("Error: --steps requires a valid number.");
                        return None;
                    }
                } else {
                    eprintln!("Error: --steps requires a value.");
                    return None;
                }
            }
            "--phase0-sim" => { cli.run_phase0_sim = true; i += 1; }
            "--phase1-sim" => { cli.run_phase1_sim = true; i += 1; }
            "--phase2-sim" => { cli.run_phase2_sim = true; i += 1; }
            "--phase3-sim" => { cli.run_phase3_sim = true; i += 1; }
            "--phase4-sim" => { cli.run_phase4_sim = true; i += 1; }
            "--phase5-sim" => { cli.run_phase5_sim = true; i += 1; }
            "--phase6-report" => { cli.run_phase6_report_flag = true; i += 1; }
            "--phase6-viz" => { cli.run_phase6_viz_flag = true; i += 1; }
            "--pathing-viz" => { cli.run_pathing_viz_flag = true; i += 1; }
            "--pathing-viz-hero-win" => { cli.run_pathing_hero_win_viz_flag = true; i += 1; }
            "--phase7-sim" => { cli.run_phase7_sim = true; i += 1; }
            "--phase8-sim" => { cli.run_phase8_sim = true; i += 1; }
            "--phase9-sim" => { cli.run_phase9_sim = true; i += 1; }
            "--pathing-sim" => { cli.run_pathing_sim = true; i += 1; }
            "--pathing-sim-hero-win" => { cli.run_pathing_hero_win_sim = true; i += 1; }
            "--pathing-sim-hero-hp-ablation" => { cli.run_pathing_hero_hp_ablation = true; i += 1; }
            "--viz-index" => { cli.run_viz_index_flag = true; i += 1; }
            "--scenario-template" => {
                if let Some(path) = args.get(i + 1) {
                    cli.scenario_template_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-template requires a file path.");
                    return None;
                }
            }
            "--scenario-viz" => {
                if let Some(path) = args.get(i + 1) {
                    cli.scenario_viz_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-viz requires a scenario json path.");
                    return None;
                }
            }
            "--scenario-out" => {
                if let Some(path) = args.get(i + 1) {
                    cli.scenario_viz_out_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-out requires an output html path.");
                    return None;
                }
            }
            "--scenario-3d" => {
                if let Some(path) = args.get(i + 1) {
                    cli.scenario_3d_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-3d requires a scenario json path.");
                    return None;
                }
            }
            "--load-campaign" => {
                if let Some(path) = args.get(i + 1) {
                    cli.campaign_load_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --load-campaign requires a file path.");
                    return None;
                }
            }
            "--map-seed" => {
                if let Some(seed_str) = args.get(i + 1) {
                    if let Some(seed) = parse_seed_arg(seed_str) {
                        cli.map_seed = Some(seed);
                        i += 2;
                    } else {
                        eprintln!(
                            "Error: --map-seed requires a valid u64 (decimal or 0x-prefixed hex)."
                        );
                        return None;
                    }
                } else {
                    eprintln!("Error: --map-seed requires a value.");
                    return None;
                }
            }
            "--horde-3d" => { cli.horde_3d_flag = true; i += 1; }
            "--horde-3d-hero-win" => { cli.horde_3d_hero_win_flag = true; i += 1; }
            "--hub" => { cli.run_hub_flag = true; i += 1; }
            "--dev" => { cli.run_dev_mode = true; i += 1; }
            "--screenshot" => {
                if let Some(dir) = args.get(i + 1) {
                    cli.screenshot_dir = Some(dir.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --screenshot requires a directory path.");
                    return None;
                }
            }
            "--screenshot-sequence" => {
                if let Some(dir) = args.get(i + 1) {
                    cli.screenshot_sequence_dir = Some(dir.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --screenshot-sequence requires a directory path.");
                    return None;
                }
            }
            "--screenshot-hub-stages" => {
                if let Some(dir) = args.get(i + 1) {
                    cli.screenshot_hub_stages_dir = Some(dir.clone());
                    cli.run_hub_flag = true;
                    i += 2;
                } else {
                    eprintln!("Error: --screenshot-hub-stages requires a directory path.");
                    return None;
                }
            }
            "--screenshot-every" => {
                if let Some(value) = args.get(i + 1) {
                    if let Ok(every) = value.parse::<u32>() {
                        if every == 0 {
                            eprintln!("Error: --screenshot-every must be >= 1.");
                            return None;
                        }
                        cli.screenshot_every = every;
                        i += 2;
                    } else {
                        eprintln!("Error: --screenshot-every requires a valid number.");
                        return None;
                    }
                } else {
                    eprintln!("Error: --screenshot-every requires a value.");
                    return None;
                }
            }
            "--screenshot-warmup-frames" => {
                if let Some(value) = args.get(i + 1) {
                    if let Ok(frames) = value.parse::<u32>() {
                        cli.screenshot_warmup_frames = frames;
                        i += 2;
                    } else {
                        eprintln!("Error: --screenshot-warmup-frames requires a valid number.");
                        return None;
                    }
                } else {
                    eprintln!("Error: --screenshot-warmup-frames requires a value.");
                    return None;
                }
            }
            _ => {
                eprintln!("Warning: Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    Some(cli)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seed_arg_supports_decimal_and_hex() {
        assert_eq!(parse_seed_arg("42"), Some(42));
        assert_eq!(parse_seed_arg("0x2a"), Some(42));
        assert_eq!(parse_seed_arg("0X2A"), Some(42));
        assert_eq!(parse_seed_arg("nope"), None);
    }

    #[test]
    fn parse_cli_args_defaults() {
        let args = vec!["game".to_string()];
        let cli = parse_cli_args(&args).unwrap();
        assert!(!cli.headless_mode);
        assert!(cli.simulation_steps.is_none());
        assert!(!cli.run_hub_flag);
    }

    #[test]
    fn parse_cli_args_hub_and_steps() {
        let args = vec![
            "game".to_string(),
            "--hub".to_string(),
            "--steps".to_string(),
            "10".to_string(),
        ];
        let cli = parse_cli_args(&args).unwrap();
        assert!(cli.run_hub_flag);
        assert_eq!(cli.simulation_steps, Some(10));
    }
}
