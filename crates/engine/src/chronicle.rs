//! Chronicle — prose side channel for `ChronicleEntry` events.
//!
//! The DSL-owned physics rules in `assets/sim/physics.sim` push
//! `Event::ChronicleEntry { template_id, agent, target, tick }` onto the
//! event ring whenever a narrative-worthy transition fires (a death, a
//! strike, an engagement, a wound, or an engagement break).
//! `ChronicleEntry` is `@non_replayable` — it never folds into
//! `replayable_sha256`, so emitting it cannot perturb the wolves+humans
//! parity baseline.
//!
//! This module owns the template id catalogue and a deterministic prose
//! renderer. The template ids are stable `u32` constants — the DSL
//! emit-sites use the literal values (1..=8) and this module resolves
//! them back to a formatted string via [`render_entry`].
//!
//! Output is stable: no timestamps, no randomness, no Debug formatting.
//! A name like "Human #3" is derived from `SimState::agent_creature_type`
//! + the raw agent id. If the agent has been killed + its slot recycled
//! by the time rendering runs, we fall back to "Agent #<id>".

// chronicle.rs retains a direct engine_data dep until Plan B2 migrates
// chronicle to compiler-emitted DSL (Spec B' §4.2, deferred).
use engine_data::events::Event;
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
    /// A strike left the victim below half hp (hp_pct < 0.5) while still
    /// alive. Emitted once per in-band `AgentAttacked` — the threshold-
    /// transition variant ("pre ≥ 0.5 → post < 0.5") wasn't expressible
    /// against the event's `damage` field alone (shields absorb part of
    /// the requested damage), so the renderer is comfortable with multi-
    /// wound chronicle noise. Payload: `agent` = attacker, `target` =
    /// victim.
    pub const WOUND: u32 = 4;
    /// An existing engagement pair dissolved. Payload: `agent` = the
    /// actor whose slot transitioned, `target` = the former partner.
    /// Emitted for every reason (switch / out-of-range / partner-died)
    /// with the same prose — `reason` is intentionally not carried on
    /// `ChronicleEntry` to keep the payload shape fixed.
    pub const BREAK: u32 = 5;
    /// A kin's death spooked a same-species neighbour within the fear
    /// radius (see `fear_spread_on_death` in `assets/sim/physics.sim`).
    /// Payload: `agent` = the shaken observer, `target` = the dead
    /// kin whose death triggered the reaction. One line per
    /// (observer, dead_kin) pair — if the radius catches multiple
    /// neighbours, each surfaces its own line so the rout reads as a
    /// collective reaction.
    pub const ROUT: u32 = 6;
    /// An agent retreated (single-subject flee — `AgentFled` carries
    /// `agent_id` + movement delta but no explicit "from whom"
    /// target). Payload: the emit-site sets `agent` = `target` =
    /// the fleer; the renderer uses the `agent` slot and ignores the
    /// redundant `target`. Fires on every `AgentFled` so repeated
    /// retreats from the same agent each surface their own line —
    /// matches the single-subject pattern of `AGENT_DIED`.
    pub const FLEE: u32 = 7;
    /// A kin's wound rallied a same-species neighbour within the rally
    /// radius (see `rally_on_wound` in `assets/sim/physics.sim`).
    /// Payload: `agent` = the rallying observer, `target` = the
    /// wounded kin whose injury triggered the reaction. One line per
    /// (observer, wounded_kin) pair — the positive mirror of `ROUT`:
    /// if the radius catches multiple neighbours, each surfaces its
    /// own line so the rally reads as a collective reaction.
    pub const RALLY: u32 = 8;
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
        templates::WOUND => {
            format!(
                "Tick {tick}: {} wounded {}.",
                name_of(state, agent),
                name_of(state, target),
            )
        }
        templates::BREAK => {
            format!(
                "Tick {tick}: {} disengaged from {}.",
                name_of(state, agent),
                name_of(state, target),
            )
        }
        templates::ROUT => {
            format!(
                "Tick {tick}: {} was shaken by {}'s death.",
                name_of(state, agent),
                name_of(state, target),
            )
        }
        templates::FLEE => {
            format!("Tick {tick}: {} retreated.", name_of(state, agent))
        }
        templates::RALLY => {
            format!(
                "Tick {tick}: {} rallied around {}.",
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
    fn renders_wound_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::WOUND,
            agent: AgentId::new(1).unwrap(),
            target: AgentId::new(2).unwrap(),
            tick: 12,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 12: Human #1 wounded Wolf #2.",
        );
    }

    #[test]
    fn renders_break_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::BREAK,
            agent: AgentId::new(2).unwrap(),
            target: AgentId::new(1).unwrap(),
            tick: 21,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 21: Wolf #2 disengaged from Human #1.",
        );
    }

    #[test]
    fn renders_rout_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::ROUT,
            agent: AgentId::new(2).unwrap(),
            target: AgentId::new(1).unwrap(),
            tick: 33,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 33: Wolf #2 was shaken by Human #1's death.",
        );
    }

    #[test]
    fn renders_flee_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::FLEE,
            agent: AgentId::new(2).unwrap(),
            target: AgentId::new(2).unwrap(),
            tick: 44,
        };
        assert_eq!(render_entry(&state, &ev), "Tick 44: Wolf #2 retreated.");
    }

    #[test]
    fn renders_rally_line() {
        let state = make_state();
        let ev = Event::ChronicleEntry {
            template_id: templates::RALLY,
            agent: AgentId::new(1).unwrap(),
            target: AgentId::new(2).unwrap(),
            tick: 55,
        };
        assert_eq!(
            render_entry(&state, &ev),
            "Tick 55: Human #1 rallied around Wolf #2.",
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
