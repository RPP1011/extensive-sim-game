use engine_data::entities::CreatureType;
use engine::probe::{run_probe, Probe};
use engine::state::AgentSpawn;
use glam::Vec3;

fn two_agents_hold() -> Probe {
    Probe {
        name: "two_agents_spawn_and_hold",
        seed: 42,
        spawn: |state| {
            state.spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::ZERO,
                hp: 100.0,
                max_hp: 100.0,
            });
            state.spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                // 30 m away so UtilityBackend doesn't decide to attack.
                pos: Vec3::new(30.0, 0.0, 0.0),
                hp: 100.0,
                max_hp: 100.0,
            });
        },
        ticks: 10,
        assert: |state, _events| {
            if state.tick != 10 {
                return Err(format!("expected tick=10, got {}", state.tick));
            }
            if state.agents_alive().count() != 2 {
                return Err(format!(
                    "expected 2 agents alive, got {}",
                    state.agents_alive().count()
                ));
            }
            Ok(())
        },
    }
}

#[test]
fn run_probe_returns_ok_when_assertions_pass() {
    run_probe(&two_agents_hold()).unwrap();
}

#[test]
fn run_probe_returns_err_when_assertion_fails() {
    let bad = Probe {
        name: "expect_wrong_tick",
        seed: 42,
        spawn: |_| {},
        ticks: 3,
        assert: |state, _| {
            if state.tick == 3 {
                Err("intentional".into())
            } else {
                Ok(())
            }
        },
    };
    let r = run_probe(&bad);
    assert!(r.is_err(), "expected error, got {:?}", r);
    let msg = r.unwrap_err();
    assert!(
        msg.contains("expect_wrong_tick"),
        "error message should carry probe name, got {}",
        msg
    );
    assert!(msg.contains("intentional"), "err body missing: {}", msg);
}

#[test]
fn probe_advances_tick_counter() {
    let p = Probe {
        name: "zero_tick",
        seed: 0,
        spawn: |_| {},
        ticks: 0,
        assert: |state, _| {
            if state.tick == 0 {
                Ok(())
            } else {
                Err(format!("expected tick=0, got {}", state.tick))
            }
        },
    };
    run_probe(&p).unwrap();

    let p = Probe {
        name: "fifty_ticks",
        seed: 0,
        spawn: |_| {},
        ticks: 50,
        assert: |state, _| {
            if state.tick == 50 {
                Ok(())
            } else {
                Err(format!("expected tick=50, got {}", state.tick))
            }
        },
    };
    run_probe(&p).unwrap();
}
