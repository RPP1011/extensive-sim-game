//! `AbilityRegistry` — stub for Combat Foundation Task 8. Task 6 scaffold only.

use super::{AbilityId, AbilityProgram};

/// Append-only table of compiled `AbilityProgram`s. Slot-stable: the Nth
/// registration hands back `AbilityId::new(N as u32).unwrap()`. Real API in
/// Task 8.
pub struct AbilityRegistry {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistry {
    pub fn new() -> Self { Self { programs: Vec::new() } }

    pub fn get(&self, id: AbilityId) -> Option<&AbilityProgram> {
        self.programs.get(id.slot())
    }

    pub fn len(&self) -> usize { self.programs.len() }
    pub fn is_empty(&self) -> bool { self.programs.is_empty() }
}

impl Default for AbilityRegistry {
    fn default() -> Self { Self::new() }
}

pub struct AbilityRegistryBuilder {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistryBuilder {
    pub fn new() -> Self { Self { programs: Vec::new() } }

    pub fn register(&mut self, program: AbilityProgram) -> AbilityId {
        self.programs.push(program);
        AbilityId::new(self.programs.len() as u32)
            .expect("programs.len() > 0 after push")
    }

    pub fn build(self) -> AbilityRegistry { AbilityRegistry { programs: self.programs } }
}

impl Default for AbilityRegistryBuilder {
    fn default() -> Self { Self::new() }
}
