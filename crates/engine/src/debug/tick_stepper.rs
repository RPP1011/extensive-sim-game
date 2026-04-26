//! Pause-and-inspect harness for the per-tick step pipeline.
//!
//! Per spec/runtime.md §24: "tick_stepper — stops between phases; can request
//! phase-specific downloads."

use std::sync::mpsc::{channel, Receiver, Sender};

/// Each phase in the serial tick pipeline at which the stepper can pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    BeforeViewFold,
    AfterViewFold,
    AfterMaskFill,
    AfterScoring,
    AfterActionSelect,
    AfterCascadeDispatch,
    TickEnd,
}

/// What the controller wants to happen after a checkpoint.
#[derive(Debug, Clone, Copy)]
pub enum Step {
    /// Let the tick continue to the next phase.
    Continue,
    /// Hold at this phase (currently treated identically to Continue in the
    /// emitted loop — the controller has already received the phase signal and
    /// can inspect state before sending Continue or Abort).
    Pause,
    /// Abort the current tick immediately (step function returns).
    Abort,
}

/// Handle held by the tick driver (the `step` function). Sends reached phases
/// to the controller; receives `Step` instructions back.
///
/// `StepperHandle` wraps two channels:
/// - `tx`: driver → controller (sends `Phase` reached)
/// - `rx`: controller → driver (receives `Step` instruction)
///
/// The driver blocks at each `checkpoint` call until the controller responds.
#[derive(Clone)]
pub struct StepperHandle {
    tx: Sender<Phase>,
    rx: std::sync::Arc<std::sync::Mutex<Receiver<Step>>>,
}

impl std::fmt::Debug for StepperHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepperHandle").finish_non_exhaustive()
    }
}

impl StepperHandle {
    /// Construct a new stepper and return:
    /// - `StepperHandle` — for embedding in `DebugConfig`
    /// - `Sender<Step>` — for the controller to send continue/abort
    /// - `Receiver<Phase>` — for the controller to receive phase notifications
    pub fn new() -> (Self, Sender<Step>, Receiver<Phase>) {
        let (tx_phase, rx_phase) = channel::<Phase>();
        let (tx_step, rx_step) = channel::<Step>();
        let handle = Self {
            tx: tx_phase,
            rx: std::sync::Arc::new(std::sync::Mutex::new(rx_step)),
        };
        (handle, tx_step, rx_phase)
    }

    /// Called by the tick driver between phases. Sends the phase to the
    /// controller and blocks until the controller responds.
    ///
    /// Returns `Step::Abort` if the controller has disconnected.
    pub fn checkpoint(&self, phase: Phase) -> Step {
        let _ = self.tx.send(phase);
        self.rx
            .lock()
            .unwrap()
            .recv()
            .unwrap_or(Step::Abort)
    }
}
