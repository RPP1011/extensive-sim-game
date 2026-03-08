mod eval;
mod dataset;
mod training;
mod selfplay;
mod transformer_play;
mod transformer_rl;

use std::path::PathBuf;
use std::process::ExitCode;

use super::cli::{OracleArgs, OracleSubcommand};

pub fn run_oracle_cmd(args: OracleArgs) -> ExitCode {
    match args.sub {
        OracleSubcommand::Eval(eval_args) => eval::run_oracle_eval(eval_args),
        OracleSubcommand::Play(play_args) => eval::run_oracle_play(play_args),
        OracleSubcommand::Dataset(dataset_args) => dataset::run_oracle_dataset(dataset_args),
        OracleSubcommand::CombatDataset(args) => dataset::run_combat_dataset(args),
        OracleSubcommand::Student(student_args) => training::run_oracle_student(student_args),
        OracleSubcommand::AbilityDataset(args) => dataset::run_ability_dataset(args),
        OracleSubcommand::AbilityEncoderExport(args) => training::run_ability_encoder_export(args),
        OracleSubcommand::SelfPlay(args) => selfplay::run_self_play(args),
        OracleSubcommand::RawDataset(args) => selfplay::run_raw_dataset(args),
        OracleSubcommand::OutcomeDataset(args) => dataset::run_outcome_dataset(args),
        OracleSubcommand::NextstateDataset(args) => dataset::run_nextstate_dataset(args),
        OracleSubcommand::TransformerPlay(args) => transformer_play::run_transformer_play(args),
        OracleSubcommand::TransformerRl(args) => transformer_rl::run_transformer_rl(args),
    }
}

pub(crate) fn collect_toml_paths(path: &std::path::Path) -> Vec<PathBuf> {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to read directory {}: {e}", path.display());
                std::process::exit(1);
            })
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
            .collect();
        entries.sort();
        entries
    } else {
        vec![path.to_path_buf()]
    }
}
