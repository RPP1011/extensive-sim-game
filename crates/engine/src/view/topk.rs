//! Stub — Task 2 fills this in.
use crate::event::Event;

pub trait TopKView: Send + Sync {
    fn k(&self) -> usize;
    fn update(&mut self, event: &Event);
}

/// Placeholder — Task 2 implementation.
pub struct MostHostileTopK;
