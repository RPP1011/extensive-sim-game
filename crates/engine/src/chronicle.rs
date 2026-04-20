//! Chronicle — prose side channel for `ChronicleEntry` events.
//!
//! The DSL-owned physics rules in `assets/sim/physics.sim` push
//! `Event::ChronicleEntry { template_id, agent, target, tick }` onto the
//! event ring whenever a narrative-worthy transition fires (a death, a
//! strike, an engagement). `ChronicleEntry` is `@non_replayable` — it
//! never folds into `replayable_sha256`, so emitting it cannot perturb
//! the wolves+humans parity baseline.
//!
//! This module owns the template id catalogue and a deterministic prose
//! renderer. The template ids are stable `u32` constants — the DSL
//! emit-sites use the literal values (1, 2, 3) and this module resolves
//! them back to a formatted string via [`render_entry`].
//!
//! Output is stable: no timestamps, no randomness, no Debug formatting.
//! A name like "Human #3" is derived from `SimState::agent_creature_type`
//! + the raw agent id. If the agent has been killed + its slot recycled
//! by the time rendering runs, we fall back to "Agent #<id>".

use crate::event::Event;
use crate::ids::AgentId;
use crate::state::SimState;

/// Stable template id catalogue. Keep in sync with the emit-sites in
/// `assets/sim/physics.sim` (`chronicle_*` rules). Adding a template
/// appends; reordering shifts the ids of existing templates and
/// invalidates any log that referenced the old ids.
pub mod templates {
    /// An agent died. Payload uses `agent` (the dead agent) — `target`
    /// is redundantly set to the same id by the emit-site.
    pub const AGENT_DIED: u32 = 1;
    /// One agent struck another. Payload: `agent` = attacker, `target`
    /// = victim.
    pub const FIRST_ATTACK: u32 = 2;
    /// Engagement formed between two agents. Payload: `agent` = the
    /// agent whose engagement slot transitioned to the target, `target`
    /// = the new engagement partner.
    pub const ENGAGEMENT: u32 = 3;
}

/// Render a single `ChronicleEntry` event as a one-line human-readable
/// string. Returns a generic placeholder when the event isn't a
/// `ChronicleEntry` — that's a caller-programming error, not a recoverable
/// condition, but `debug_assert!` keeps it visible during development
/// while production builds still produce a stable string.
pub fn render_entry(state: &SimState, event: &Event) -> String {
    let (template_id, agent, target, tick) = match event {
        Event::ChronicleEntry {
            template_id,
            agent,
            target,
            tick,
        } => (*template_id, *agent, *target, *tick),
        other => {
            debug_assert!(
                false,
                "chronicle::render_entry called on non-ChronicleEntry event: {other:?}"
            );
            return format!("(non-chronicle event passed to renderer)");
        }
    };

    match template_id {
        templates::AGENT_DIED => {
            format!("Tick {tick}: {} fell.", name_of(state, agent))
        }
        templates::FIRST_ATTACK => {
            format!(
                "Tick {tick}: {} struck {}.",
                name_of(state, agent),
                name_of(state, target),
            )
        }
        templates::ENGAGEMENT => {
            format!(
                "Tick {tick}: {} engaged {}.",
                name_of(state, agent),
                name_of(state, target),
            )
        }
        _ => format!(
            "Tick {tick}: (unknown chronicle template {template_id} for agent {}, target {})",
            agent.raw(),
            target.raw(),
        ),
    }
}

/// Render every `ChronicleEntry` in the slice as one line each, in the
/// order encountered. Non-chronicle events are skipped silently — this
/// is the caller's bulk convenience path when walking the event ring.
pub fn render_entries<'a>(
    state: &SimState,
    events: impl IntoIterator<Item = &'a Event>,
) -> Vec<String> {
    let mut out = Vec::new();
    for ev in events {
        if matches!(ev, Event::ChronicleEntry { .. }) {
            out.push(render_entry(state, ev));
        }
    }
    out
}

/// Resolve an agent id to a readable display name. Uses the creature type
/// ("Human", "Wolf", "Deer", "Dragon") + the raw id so the chronicle
/// reader can distinguish "Human #2" from "Human #3". If the agent's slot
/// has been recycled / the type lookup fails, falls back to "Agent #<id>"
/// so the output stays deterministic and non-panicking.
pub fn name_of(state: &SimState, id: AgentId) -> String {
    match state.agent_creature_type(id) {
        Some(ct) => format!("{} #{}", ct.name(), id.raw()),
        None => format!("Agent #{}", id.raw()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creature::CreatureType;
    use crate::state::{AgentSpawn, SimState};
    use glam::Vec3;

    fn make_state() -> SimState {
        let mut state = SimState::new(4, 0xC0FFEE);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(0.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(1.0, 0.0, 0.0),
                hp: 80.0,
                ..Default::default()
            })
            .expect("wolf spawn");
        state
    }

    #[test]
    fn renders_death_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::AGENT_DIED,
            agent: AgentId::new(1).unwrap(),
            target: AgentId::new(1).unwrap(),
            tick: 47,
        };
        assert_eq!(render_entry(&state, &ev), "Tick 47: Human #1 fell.");
    }

    #[test]
    fn renders_attack_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::FIRST_ATTACK,
            agent: AgentId::new(2).unwrap(),
            target: AgentId::new(1).unwrap(),
            tick: 3,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 3: Wolf #2 struck Human #1.",
        );
    }

    #[test]
    fn renders_engagement_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::ENGAGEMENT,
            agent: AgentId::new(1).unwrap(),
            target: AgentId::new(2).unwrap(),
            tick: 0,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 0: Human #1 engaged Wolf #2.",
        );
    }

    #[test]
    fn renders_unknown_template_stably() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: 999,
            agent: AgentId::new(1).unwrap(),
            target: AgentId::new(2).unwrap(),
            tick: 10,
        };
        let out = render_entry(&state, &ev);
        assert!(out.starts_with("Tick 10: (unknown chronicle template 999"));
    }

    #[test]
    fn name_of_falls_back_on_unknown_id() {
        let state = make_state();
        // Id 7 was never spawned (cap=4, only 1+2 used).
        let id = AgentId::new(7).unwrap();
        assert_eq!(name_of(&state, id), "Agent #7");
    }

    #[test]
    fn render_entries_filters_non_chronicle() {
        let state = make_state();
        let events = [
            Event::AgentAttacked {
                actor: AgentId::new(2).unwrap(),
                target: AgentId::new(1).unwrap(),
                damage: 10.0,
                tick: 0,
            },
            Event::ChronicleEntry {
                template_id: templates::AGENT_DIED,
                agent: AgentId::new(1).unwrap(),
                target: AgentId::new(1).unwrap(),
                tick: 5,
            },
        ];
        let lines = render_entries(&state, events.iter());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "Tick 5: Human #1 fell.");
    }
}
