//! `xtask train-v6` — auto-configure libtorch, build with burn-gpu, run IMPALA training.
//!
//! Finds CUDA-enabled PyTorch in uv cache, sets up all env vars, invokes
//! `cargo run --features burn-gpu` with the impala-train subcommand.
//!
//! Usage:
//!   cargo run --bin xtask -- train-v6 dataset/scenarios/hvh --iters 50
//!   cargo run --bin xtask -- train-v6 --checkpoint generated/impala_v6/v6_iter0050.bin

use std::process::{Command, ExitCode};
use crate::cli::TrainV6Args;

pub fn run_train_v6(args: TrainV6Args) -> ExitCode {
    // --- Find CUDA-enabled PyTorch in uv cache ---
    let torch_site = find_cuda_torch();
    let torch_site = match torch_site {
        Some(p) => p,
        None => {
            eprintln!("ERROR: No CUDA-enabled PyTorch found in ~/.cache/uv/");
            eprintln!("Install with: uv pip install torch");
            return ExitCode::from(1);
        }
    };
    let torch_lib = format!("{}/torch/lib", torch_site);
    eprintln!("Using PyTorch: {}/torch", torch_site);

    // --- Ensure c++ is reachable ---
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ricky".to_string());
    let bin_dir = format!("{}/bin", home);
    std::fs::create_dir_all(&bin_dir).ok();
    let cxx_link = format!("{}/c++", bin_dir);
    if !std::path::Path::new(&cxx_link).exists() {
        if std::path::Path::new("/usr/bin/g++-12").exists() {
            std::os::unix::fs::symlink("/usr/bin/g++-12", &cxx_link).ok();
        } else if std::path::Path::new("/usr/bin/g++-13").exists() {
            std::os::unix::fs::symlink("/usr/bin/g++-13", &cxx_link).ok();
        }
    }

    // --- Build args for the inner impala-train command ---
    let mut inner_args: Vec<String> = vec![
        "scenario".into(), "oracle".into(), "transformer-rl".into(), "impala-train".into(),
    ];
    for p in &args.path {
        inner_args.push(p.display().to_string());
    }
    inner_args.extend(["--output-dir".into(), args.output_dir.display().to_string()]);
    inner_args.extend(["--iters".into(), args.iters.to_string()]);
    inner_args.extend(["--episodes".into(), args.episodes.to_string()]);
    inner_args.extend(["--threads".into(), args.threads.to_string()]);
    inner_args.extend(["--sims-per-thread".into(), args.sims_per_thread.to_string()]);
    inner_args.extend(["--batch-size".into(), args.batch_size.to_string()]);
    inner_args.extend(["--train-steps".into(), args.train_steps.to_string()]);
    inner_args.extend(["--lr".into(), args.lr.to_string()]);
    inner_args.extend(["--temperature".into(), args.temperature.to_string()]);
    inner_args.extend(["--step-interval".into(), args.step_interval.to_string()]);
    inner_args.extend(["--entropy-coef".into(), args.entropy_coef.to_string()]);
    inner_args.extend(["--value-coef".into(), args.value_coef.to_string()]);
    if let Some(ref ckpt) = args.checkpoint {
        inner_args.extend(["--checkpoint".into(), ckpt.display().to_string()]);
    }
    if let Some(ref reg) = args.embedding_registry {
        inner_args.extend(["--embedding-registry".into(), reg.display().to_string()]);
    }
    if args.grokfast { inner_args.push("--grokfast".into()); }
    if args.self_play { inner_args.push("--self-play".into()); }
    if args.bc { inner_args.push("--bc".into()); }

    // --- PATH with ~/bin for c++ ---
    let path_env = format!("{}:{}", bin_dir,
        std::env::var("PATH").unwrap_or_default());

    // --- Library path for linker (libstdc++) ---
    let library_path = {
        let gcc_lib = "/usr/lib/gcc/x86_64-linux-gnu/12";
        let existing = std::env::var("LIBRARY_PATH").unwrap_or_default();
        if existing.is_empty() { gcc_lib.to_string() } else { format!("{gcc_lib}:{existing}") }
    };

    // --- LD_LIBRARY_PATH for runtime ---
    let ld_path = {
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if existing.is_empty() { torch_lib.clone() } else { format!("{torch_lib}:{existing}") }
    };

    if !args.no_build {
        eprintln!("Building with --features burn-gpu (release)...");
        let build_status = Command::new("cargo")
            .args(["build", "--release", "--features", "burn-gpu", "--bin", "xtask"])
            .env("PATH", &path_env)
            .env("CXX", "/usr/bin/g++-12")
            .env("LIBTORCH_USE_PYTORCH", "1")
            .env("LIBTORCH_BYPASS_VERSION_CHECK", "1")
            .env("PYTHONPATH", &torch_site)
            .env("LD_LIBRARY_PATH", &ld_path)
            .env("LIBRARY_PATH", &library_path)
            .status();

        match build_status {
            Ok(s) if s.success() => eprintln!("Build succeeded."),
            Ok(s) => {
                eprintln!("Build failed (exit {})", s.code().unwrap_or(-1));
                return ExitCode::from(1);
            }
            Err(e) => {
                eprintln!("Failed to run cargo: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // --- Run the built binary directly (it's already compiled with burn-gpu) ---
    let bin_path = std::path::Path::new("target/release/xtask");
    if !bin_path.exists() {
        eprintln!("ERROR: target/release/xtask not found. Build may have failed.");
        return ExitCode::from(1);
    }

    eprintln!("Running IMPALA V6 training...");
    let status = Command::new(bin_path)
        .args(&inner_args)
        .env("LD_LIBRARY_PATH", &ld_path)
        .status();

    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("Failed to run training: {e}");
            ExitCode::from(1)
        }
    }
}

/// Search uv cache for a CUDA-enabled PyTorch site-packages directory.
fn find_cuda_torch() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ricky".to_string());
    let cache_dir = format!("{}/.cache/uv/archive-v0", home);

    let Ok(entries) = std::fs::read_dir(&cache_dir) else { return None };

    for entry in entries.flatten() {
        let path = entry.path();
        // Look for torch/lib/libtorch_cuda.so inside site-packages
        let site_packages = path.join("lib").join("python3.12").join("site-packages");
        let cuda_marker = site_packages.join("torch").join("lib").join("libtorch_cuda.so");
        if cuda_marker.exists() {
            return Some(site_packages.display().to_string());
        }
    }

    // Also check environments-v2
    let env_dir = format!("{}/.cache/uv/environments-v2", home);
    if let Ok(entries) = std::fs::read_dir(&env_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let site_packages = path.join("lib").join("python3.12").join("site-packages");
            let cuda_marker = site_packages.join("torch").join("lib").join("libtorch_cuda.so");
            if cuda_marker.exists() {
                return Some(site_packages.display().to_string());
            }
        }
    }

    None
}
