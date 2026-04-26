use engine_data::events::Event;
use engine::ids::AgentId;
use engine::view::{MostHostileTopK, TopKView};

#[test]
fn empty_view_returns_empty_topk() {
    let view = MostHostileTopK::new(8);
    let a = AgentId::new(1).unwrap();
    assert_eq!(view.topk(a).len(), 0);
}

#[test]
fn one_attacker_populates_topk_for_victim() {
    let mut view = MostHostileTopK::new(8);
    let attacker = AgentId::new(1).unwrap();
    let victim   = AgentId::new(2).unwrap();
    view.update(&Event::AgentAttacked {
        actor: attacker, target: victim, damage: 20.0, tick: 0,
    });
    let topk = view.topk(victim);
    assert_eq!(topk.len(), 1);
    assert_eq!(topk[0].0, attacker);
    assert!((topk[0].1 - 20.0).abs() < 1e-6);
}

#[test]
fn repeated_attacks_accumulate_hostility_score() {
    let mut view = MostHostileTopK::new(8);
    let attacker = AgentId::new(1).unwrap();
    let victim   = AgentId::new(2).unwrap();
    view.update(&Event::AgentAttacked { actor: attacker, target: victim, damage: 20.0, tick: 0 });
    view.update(&Event::AgentAttacked { actor: attacker, target: victim, damage: 30.0, tick: 1 });
    let topk = view.topk(victim);
    assert_eq!(topk.len(), 1);
    assert!((topk[0].1 - 50.0).abs() < 1e-6);
}

#[test]
fn topk_bounded_keeps_highest_scoring_attackers() {
    const K: usize = 4;
    let mut view = MostHostileTopK::with_k(16, K);
    let victim = AgentId::new(1).unwrap();
    for i in 0..6 {
        let attacker = AgentId::new(i + 2).unwrap();
        view.update(&Event::AgentAttacked {
            actor: attacker, target: victim,
            damage: 10.0 * (i + 1) as f32, tick: 0,
        });
    }
    let topk = view.topk(victim);
    assert_eq!(topk.len(), K);
    assert!(topk[0].1 >= topk[1].1);
    assert!(topk[1].1 >= topk[2].1);
    assert!(topk[2].1 >= topk[3].1);
    assert!((topk[0].1 - 60.0).abs() < 1e-6);
    assert!((topk[3].1 - 30.0).abs() < 1e-6);
}
