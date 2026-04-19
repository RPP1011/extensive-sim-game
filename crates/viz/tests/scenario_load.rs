use viz::scenario;

#[test]
fn viz_basic_parses_into_4_humans_and_1_wolf() {
    let s = scenario::load("scenarios/viz_basic.toml").expect("viz_basic parses");
    assert_eq!(s.agent.len(), 5);
    assert_eq!(s.world.seed, 42);
    let humans = s.agent.iter().filter(|a| a.creature_type == "Human").count();
    let wolves = s.agent.iter().filter(|a| a.creature_type == "Wolf").count();
    assert_eq!(humans, 4);
    assert_eq!(wolves, 1);
}

#[test]
fn unknown_creature_type_errors() {
    let spec = scenario::AgentSpec {
        creature_type: "Goblin".into(),
        pos: [0.0, 0.0, 0.0],
        hp: 100.0,
    };
    let err = spec.creature().unwrap_err().to_string();
    assert!(err.contains("Goblin"), "error mentions bad type: {}", err);
}
