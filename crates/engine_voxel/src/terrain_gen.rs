//! Worker-thread harness for terrain generation. Generator is a `fn`
//! pointer (not a closure) — keeps the Send-bound story simple.
//! Panics are caught and surfaced as `TerrainGenError::Panicked` so
//! the app does not abort. See spec section "Async execution model".

use crate::VoxelTerrain;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;

#[derive(Debug)]
pub enum TerrainGenError {
    Panicked(String),
}

impl std::fmt::Display for TerrainGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerrainGenError::Panicked(msg) => write!(f, "terrain generation panicked: {msg}"),
        }
    }
}

impl std::error::Error for TerrainGenError {}

pub struct TerrainGenHandle {
    rx: mpsc::Receiver<Result<VoxelTerrain, TerrainGenError>>,
    join: Option<std::thread::JoinHandle<()>>,
    pub seed: u64,
}

impl TerrainGenHandle {
    pub fn spawn(seed: u64, gen_fn: fn(u64) -> VoxelTerrain) -> Self {
        let (tx, rx) = mpsc::channel();
        let join = std::thread::Builder::new()
            .name(format!("terrain-gen-{seed}"))
            .spawn(move || {
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| gen_fn(seed)))
                    .map_err(|payload| {
                        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                            (*s).to_string()
                        } else if let Some(s) = payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "<unknown panic payload>".to_string()
                        };
                        TerrainGenError::Panicked(msg)
                    });
                let _ = tx.send(result);
            })
            .expect("spawn terrain-gen worker thread");
        Self { rx, join: Some(join), seed }
    }

    /// Non-blocking poll. Returns `None` while the worker is still
    /// running; `Some(Ok)` on completion; `Some(Err)` on panic.
    pub fn try_take(&mut self) -> Option<Result<VoxelTerrain, TerrainGenError>> {
        match self.rx.try_recv() {
            Ok(r) => {
                if let Some(j) = self.join.take() { let _ = j.join(); }
                Some(r)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                TerrainGenError::Panicked("worker disconnected without result".into())
            )),
        }
    }

    /// Block until ready. Used by tests + headless tools. Production
    /// app loops should poll `try_take` so the UI thread stays
    /// responsive.
    pub fn block_until_ready(self) -> Result<VoxelTerrain, TerrainGenError> {
        let result = self.rx.recv().unwrap_or(Err(TerrainGenError::Panicked(
            "worker disconnected without result".into()
        )));
        if let Some(j) = self.join.into_iter().next() { let _ = j.join(); }
        result
    }
}
