mod transformer_rl;
mod rl_episode;
mod rl_policies;
mod rl_eval;
mod rl_generate;
mod impala_train;
mod playtester;
#[cfg(feature = "stream-monitor")]
mod monitor_traces;

use std::path::PathBuf;
use std::process::ExitCode;

use super::cli::{OracleArgs, OracleSubcommand};

pub fn run_oracle_cmd(args: OracleArgs) -> ExitCode {
    match args.sub {
        OracleSubcommand::TransformerRl(args) => transformer_rl::run_transformer_rl(args),
        OracleSubcommand::Playtester(args) => playtester::run_playtester(args),
        #[cfg(feature = "stream-monitor")]
        OracleSubcommand::MonitorTraces(args) => {
            monitor_traces::run_monitor_traces(&args.path, args.sample)
        }
        #[cfg(not(feature = "stream-monitor"))]
        OracleSubcommand::MonitorTraces(_) => {
            eprintln!("Error: monitor-traces requires --features stream-monitor");
            ExitCode::FAILURE
        }
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
