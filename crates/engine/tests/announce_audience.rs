use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, AnnounceAudience, MacroAction, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct OneAnnounce(AgentId, AnnounceAudience);
impl PolicyBackend for OneAnnounce {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.0,
            kind: ActionKind::Macro(MacroAction::Announce {
                speaker: self.0,
                audience: self.1,
                fact_payload: 0xDEADBEEF_CAFEF00D,
            }),
        });
        for id in state.agents_alive() {
            if id != self.0 {
                out.push(Action::hold(id));
            }
        }
    }
}

#[test]
fn announce_area_emits_recordmemory_for_each_agent_within_radius() {
    // Boundary-pinning test: recipients placed at 0.0 (center), 9.999 (just
    // inside), and 10.0 (exactly on boundary — must be included if impl uses
    // `<=`). Outliers at 10.001 (just outside) and 20.0 (far outside). Radius
    // is 10. Expect exactly 3 recipients. Pins both `<=` vs `<` and the exact
    // radius passed to the kernel.
    let mut state = SimState::new(32, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(2048);
    let cascade = CascadeRegistry::new();

    let center = Vec3::new(0.0, 0.0, 10.0);
    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center,
            hp: 100.0,
        })
        .unwrap();
    // 3 within/on radius 10: distances [0.0, 9.999, 10.0].
    // Distance 0.0 is co-located with the speaker (speaker-exclusion is
    // by id, not position). 10.0 pins the `<=` side of the comparison —
    // an impl using `<` would drop it and the test flips to 2.
    let inside_distances = [0.0f32, 9.999, 10.0];
    for &d in &inside_distances {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(d, 0.0, 0.0),
            hp: 100.0,
        });
    }
    // 2 outside radius 10: distances [10.001, 20.0].
    for &d in &[10.001f32, 20.0] {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(d, 0.0, 0.0),
            hp: 100.0,
        });
    }

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(center, 10.0)),
        &cascade,
    );

    // Primary recipients (confidence 0.8) are the audience-in-range set;
    // the overhear scan (0.6 confidence) can pick up outliers within
    // OVERHEAR_RANGE of the speaker, so we filter on the primary channel
    // to isolate the Area-radius gate being tested here.
    let primary: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { confidence, .. } if (*confidence - 0.8).abs() < 1e-6))
        .count();
    assert_eq!(primary, 3, "exactly 3 primary recipients at distances [0.0, 9.999, 10.0] within/on 10m");

    assert!(events.iter().any(|e| matches!(e,
        Event::AnnounceEmitted { speaker: s, audience_tag, .. }
            if *s == speaker && *audience_tag == 1 /* Area */)));
}

#[test]
fn speaker_excluded_from_recipients() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 100.0)),
        &cascade,
    );

    let speaker_recipient = events.iter().any(|e| match e {
        Event::RecordMemory { observer, .. } => *observer == speaker,
        _ => false,
    });
    assert!(!speaker_recipient, "speaker should not receive their own announce");
}

#[test]
fn announce_bounded_by_max_recipients() {
    let mut state = SimState::new(128, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(8192);
    let cascade = CascadeRegistry::new();

    let center = Vec3::new(0.0, 0.0, 10.0);
    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center,
            hp: 100.0,
        })
        .unwrap();
    // 64 agents, all within radius — primary audience caps at
    // MAX_ANNOUNCE_RECIPIENTS=32 (0.8 confidence). The remaining 32 are still
    // within OVERHEAR_RANGE=30 of the speaker so they receive overhear memories
    // at 0.6 confidence (Task 15).
    for i in 0..64 {
        let angle = (i as f32 / 64.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.0),
            hp: 100.0,
        });
    }
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(center, 50.0)),
        &cascade,
    );

    // Collect per-observer sets to verify identity behavior, not just counts.
    // Each of the 64 non-speaker agents must appear in exactly ONE of the two
    // sets (primary 0.8 OR overhear 0.6), never both, never neither.
    use std::collections::HashSet;
    let mut primary_ids: HashSet<u32> = HashSet::new();
    let mut overhear_ids: HashSet<u32> = HashSet::new();
    for e in events.iter() {
        if let Event::RecordMemory { observer, confidence, .. } = e {
            if (*confidence - 0.8).abs() < 1e-6 {
                primary_ids.insert(observer.raw());
            } else if (*confidence - 0.6).abs() < 1e-6 {
                overhear_ids.insert(observer.raw());
            }
        }
    }
    assert_eq!(primary_ids.len(), 32, "primary bounded by MAX_ANNOUNCE_RECIPIENTS");
    assert_eq!(overhear_ids.len(), 32, "remaining 32 bystanders overhear at 0.6");

    // Disjointness: no agent in both sets (dedup against primary-cap overflow).
    let intersection: HashSet<_> = primary_ids.intersection(&overhear_ids).collect();
    assert!(intersection.is_empty(),
        "primary and overhear sets must be disjoint, overlap: {:?}", intersection);

    // Union covers all 64 spawned non-speaker agents (speaker excluded from both).
    let union: HashSet<u32> = primary_ids.union(&overhear_ids).copied().collect();
    let expected_non_speaker: HashSet<u32> = (1u32..=65)
        .filter(|raw| *raw != speaker.raw())
        .collect();
    assert_eq!(union, expected_non_speaker,
        "primary ∪ overhear must equal all 64 non-speaker agents");
}

#[test]
fn announce_anyone_uses_max_announce_radius_around_speaker() {
    // Boundary-pinning test: MAX_ANNOUNCE_RADIUS = 80m. Agents placed at
    // distances [50.0, 79.9, 80.1, 200.0]; expect 79.9 included and 80.1
    // excluded. That pins the constant to 80m ± 0.2m and rules out a
    // silent halving to 40 or doubling to 150.
    let mut state = SimState::new(16, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 10.0),
            hp: 100.0,
        })
        .unwrap();
    // Close agent — deep inside MAX_ANNOUNCE_RADIUS=80:
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(50.0, 0.0, 10.0),
        hp: 100.0,
    });
    // Just inside (79.9m):
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(79.9, 0.0, 10.0),
        hp: 100.0,
    });
    // Just outside (80.1m):
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(80.1, 0.0, 10.0),
        hp: 100.0,
    });
    // Far agent — beyond 80:
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(200.0, 0.0, 10.0),
        hp: 100.0,
    });

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Anyone),
        &cascade,
    );

    let recipients: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(
        recipients, 2,
        "agents at 50m and 79.9m are within MAX_ANNOUNCE_RADIUS=80; 80.1m and 200m excluded"
    );
}
