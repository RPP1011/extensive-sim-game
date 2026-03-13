//! CLI entry point for ability operator training.

use std::path::PathBuf;

use burn::backend::Autodiff;
use clap::Parser;

use ability_operator::data::OperatorDataset;
use ability_operator::train::{self, TrainConfig};

#[derive(Parser, Debug)]
#[command(name = "train_operator", about = "Train ability latent operator model")]
struct Args {
    /// Path to npz dataset.
    #[arg(long)]
    data: PathBuf,

    /// Output path for checkpoint.
    #[arg(long, default_value = "generated/ability_operator.json")]
    output: PathBuf,

    /// Maximum training steps.
    #[arg(long, default_value_t = 200_000)]
    max_steps: usize,

    /// Evaluate every N steps.
    #[arg(long, default_value_t = 5_000)]
    eval_every: usize,

    /// Batch size.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Path to checkpoint to resume from.
    #[arg(long)]
    warm_start: Option<PathBuf>,

    /// Backend: "tch" for libtorch/CUDA, "wgpu" for wgpu (default).
    #[arg(long, default_value = "tch")]
    backend: String,
}

fn main() {
    let args = Args::parse();

    let config = TrainConfig {
        max_steps: args.max_steps,
        eval_every: args.eval_every,
        batch_size: args.batch_size,
        lr: args.lr,
        seed: args.seed,
        ..TrainConfig::default()
    };

    match args.backend.as_str() {
        "tch" => {
            use burn::backend::libtorch::{LibTorch, LibTorchDevice};
            type B = Autodiff<LibTorch>;
            let device = if tch::Cuda::is_available() {
                eprintln!("Using libtorch CUDA backend");
                LibTorchDevice::Cuda(0)
            } else {
                eprintln!("CUDA not available, using libtorch CPU backend");
                LibTorchDevice::Cpu
            };
            eprintln!("Loading dataset from {}...", args.data.display());
            let dataset = OperatorDataset::<B>::load(&args.data, &device);
            eprintln!("Loaded {} samples", dataset.n_samples);
            train::train(&dataset, &config, &args.output, &device);
        }
        "wgpu" => {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            type B = Autodiff<Wgpu>;
            let device = WgpuDevice::default();
            eprintln!("Using wgpu backend");
            eprintln!("Loading dataset from {}...", args.data.display());
            let dataset = OperatorDataset::<B>::load(&args.data, &device);
            eprintln!("Loaded {} samples", dataset.n_samples);
            train::train(&dataset, &config, &args.output, &device);
        }
        other => {
            eprintln!("Unknown backend: {other}. Use 'tch' or 'wgpu'.");
            std::process::exit(1);
        }
    }

    eprintln!("Training complete. Best checkpoint at {}", args.output.display());
}
