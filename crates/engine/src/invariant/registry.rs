use super::Invariant;

pub struct InvariantRegistry { _inner: Vec<Box<dyn Invariant>> }

impl InvariantRegistry {
    pub fn new() -> Self { Self { _inner: Vec::new() } }
}

impl Default for InvariantRegistry {
    fn default() -> Self { Self::new() }
}
