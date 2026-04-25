use std::process::Command as ProcessCommand;
use std::process::ExitCode;

use super::cli::{CaptureDedupeArgs, CaptureWindowsArgs};

pub fn run_capture_windows(args: CaptureWindowsArgs) -> ExitCode {
    let steps = args.steps.to_string();
    let every = args.every.to_string();
    let warmup_frames = args.warmup_frames.to_string();

    let mut command = ProcessCommand::new("powershell.exe");
    command.args([
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/capture_windows.ps1",
        "-Mode",
        args.mode.as_ps_value(),
        "-OutDir",
    ]);
    command.arg(args.out_dir);
    command.args([
        "-Steps",
        &steps,
        "-Every",
        &every,
        "-WarmupFrames",
        &warmup_frames,
    ]);
    if args.hub {
        command.arg("-Hub");
    }
    if args.persist {
        command.arg("-Persist");
    }

    println!(
        "Running Windows capture wrapper: scripts/capture_windows.ps1 (mode={})",
        args.mode.as_ps_value()
    );
    match command.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(err) => {
            eprintln!(
                "Failed to execute powershell.exe: {err}. Run this command from Windows/WSL with PowerShell available."
            );
            ExitCode::from(1)
        }
    }
}

pub fn run_capture_dedupe(args: CaptureDedupeArgs) -> ExitCode {
    let mut command = ProcessCommand::new("powershell.exe");
    command.args([
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/dedupe_capture.ps1",
        "-OutDir",
    ]);
    command.arg(args.out_dir);

    println!("Running screenshot dedupe wrapper: scripts/dedupe_capture.ps1");
    match command.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(err) => {
            eprintln!(
                "Failed to execute powershell.exe: {err}. Run this command from Windows/WSL with PowerShell available."
            );
            ExitCode::from(1)
        }
    }
}
