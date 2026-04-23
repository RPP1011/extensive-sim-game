//! Sync-path state on the GPU backend — kernels and buffers used
//! exclusively by `SimBackend::step()`.

#![cfg(feature = "gpu")]

use crate::cascade::CascadeCtx;
use crate::mask::FusedMaskKernel;
use crate::scoring::{ScoreOutput, ScoringKernel};
use crate::view_storage::ViewStorage;
use crate::PhaseTimings;

pub struct SyncPathContext {
    pub mask_kernel:             FusedMaskKernel,
    pub scoring_kernel:          ScoringKernel,
    pub view_storage:            ViewStorage,
    pub cascade_ctx:             Option<CascadeCtx>,
    pub backend_label:           String,
    pub last_mask_bitmaps:       Vec<Vec<u32>>,
    pub last_scoring_outputs:    Vec<ScoreOutput>,
    pub last_cascade_iterations: Option<u32>,
    pub last_cascade_error:      Option<String>,
    pub skip_scoring_sidecar:    bool,
    pub last_phase_us:           PhaseTimings,
}

impl SyncPathContext {
    pub fn new(
        mask_kernel:    FusedMaskKernel,
        scoring_kernel: ScoringKernel,
        view_storage:   ViewStorage,
        backend_label:  String,
    ) -> Self {
        Self {
            mask_kernel,
            scoring_kernel,
            view_storage,
            cascade_ctx:             None,
            backend_label,
            last_mask_bitmaps:       Vec::new(),
            last_scoring_outputs:    Vec::new(),
            last_cascade_iterations: None,
            last_cascade_error:      None,
            skip_scoring_sidecar:    true,
            last_phase_us:           PhaseTimings::default(),
        }
    }
}
