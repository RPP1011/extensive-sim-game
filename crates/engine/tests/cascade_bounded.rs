use engine::cascade::{CascadeHandler, CascadeRegistry, EventKindId, Lane};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Amplifier(Arc<AtomicUsize>);
impl CascadeHandler for Amplifier {
    fn trigger(&self) -> EventKindId { EventKindId::AgentAttacked }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, _: &mut SimState, events: &mut EventRing) {
        self.0.fetch_add(1, Ordering::Relaxed);
        if let Event::AgentAttacked { attacker, target, damage, tick } = event {
            events.push(Event::AgentAttacked {
                attacker: *attacker, target: *target,
                damage: *damage, tick: tick.saturating_add(1),
            });
        }
    }
}

#[test]
#[cfg(not(debug_assertions))]  // in debug, this panics (see next test)
fn release_dispatch_truncates_at_max_cascade_iterations() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Amplifier(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(1024);

    ring.push(Event::AgentAttacked { attacker: a, target: a, damage: 1.0, tick: 0 });
    reg.run_fixed_point(&mut state, &mut ring);

    let n = hits.load(Ordering::Relaxed);
    // Primary + 8 cascade iterations = 9 fires.
    assert!(n <= 9, "handler fired {} times — expected <= 9", n);
    assert!(n >= 2, "cascade didn't amplify at all?");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cascade did not converge")]
fn debug_dispatch_panics_on_non_convergence() {
    let mut reg = CascadeRegistry::new();
    reg.register(Amplifier(Arc::new(AtomicUsize::new(0))));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(1024);
    ring.push(Event::AgentAttacked { attacker: a, target: a, damage: 1.0, tick: 0 });
    reg.run_fixed_point(&mut state, &mut ring);
}

#[test]
fn converging_cascade_terminates_early() {
    // Handler that doesn't re-emit. Should fire exactly once per triggering event.
    struct Once(Arc<AtomicUsize>);
    impl CascadeHandler for Once {
        fn trigger(&self) -> EventKindId { EventKindId::AgentDied }
        fn handle(&self, _: &Event, _: &mut SimState, _: &mut EventRing) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Once(hits.clone()));

    let mut state = SimState::new(2, 42);
    let a = AgentId::new(1).unwrap();
    let mut ring = EventRing::with_cap(16);

    ring.push(Event::AgentDied { agent_id: a, tick: 0 });
    reg.run_fixed_point(&mut state, &mut ring);
    assert_eq!(hits.load(Ordering::Relaxed), 1);
}
