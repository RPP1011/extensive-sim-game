//! Top-K views: fixed-size "most-X" list per entity. Bounded memory at large
//! N. Example: `MostHostileTopK` — the K attackers with highest cumulative
//! damage dealt to each agent.

use crate::event::EventLike;
use crate::ids::AgentId;

pub trait TopKView<E: EventLike>: crate::cascade::handler::__sealed::Sealed + Send + Sync {
    fn k(&self) -> usize;
    fn update(&mut self, event: &E);
}

const DEFAULT_K: usize = 8;

pub struct MostHostileTopK {
    per_target: Vec<Vec<(AgentId, f32)>>,
    k: usize,
}

impl MostHostileTopK {
    pub fn new(cap: usize) -> Self { Self::with_k(cap, DEFAULT_K) }
    pub fn with_k(cap: usize, k: usize) -> Self {
        Self { per_target: (0..cap).map(|_| Vec::with_capacity(k)).collect(), k }
    }

    pub fn topk(&self, target: AgentId) -> &[(AgentId, f32)] {
        let slot = (target.raw() - 1) as usize;
        self.per_target.get(slot).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn accumulate(&mut self, attacker: AgentId, target: AgentId, damage: f32) {
        let slot = (target.raw() - 1) as usize;
        let Some(list) = self.per_target.get_mut(slot) else { return };
        if let Some(entry) = list.iter_mut().find(|(a, _)| *a == attacker) {
            entry.1 += damage;
        } else {
            list.push((attacker, damage));
        }
        list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if list.len() > self.k { list.truncate(self.k); }
    }
}

// The Sealed + TopKView impls for MostHostileTopK use the concrete
// engine_data::Event type. engine_data is a regular dep (chronicle.rs
// retains the dep until Plan B2).
impl crate::cascade::handler::__sealed::Sealed for MostHostileTopK {}

impl TopKView<engine_data::events::Event> for MostHostileTopK {
    fn k(&self) -> usize { self.k }
    fn update(&mut self, event: &engine_data::events::Event) {
        if let engine_data::events::Event::AgentAttacked { actor, target, damage, .. } = event {
            self.accumulate(*actor, *target, *damage);
        }
    }
}
