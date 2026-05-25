//! Regression: an Agent×Agent `@materialized(storage = pair_map)` view read
//! with 2 args (`view(self, other)`) must emit a 2-arg accessor whose call
//! site arity matches, and the emitted WGSL must pass naga VALIDATION (not
//! just parsing — `parse_str` does not check call arity).
//!
//! Before the fix (`cg/emit/program.rs`), a dynamic-K pair view (no static
//! `@key_pop`) emitted a 1-arg `view_N_get(idx)` definition while the call
//! site passed 2 args → naga `ArgumentCount { required: 1, seen: 2 }`. This
//! was the latent bug predator_prey's `predator_focus(self, target)` surfaced.
use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| panic!("lower: {:?}", o.diagnostics));
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn naga_validate(art: &EmittedArtifacts) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    for (name, body) in &art.wgsl_files {
        let module = naga::front::wgsl::parse_str(body)
            .unwrap_or_else(|e| panic!("naga parse [{name}]: {e:?}"));
        let mut v = Validator::new(ValidationFlags::all(), Capabilities::all());
        v.validate(&module)
            .unwrap_or_else(|e| panic!("naga VALIDATE [{name}]: {e:?}\n--- body ---\n{body}"));
    }
}

const PAIR_VIEW_READ: &str = r#"
event Tick { }
event Damaged { source: AgentId, target: AgentId, amount: f32 }
event Killed { by: AgentId, prey: AgentId }
entity Hare : Agent { pos: vec3 }
entity Wolf : Agent { pos: vec3 }
config hunt { strike_radius: f32 = 1.5 }
@materialized(on_event = [Killed], storage = pair_map)
@decay(rate = 0.98, per = tick)
view pair_focus(a: Agent, b: Agent) -> f32 {
  initial: 0.0,
  on Killed { by: p, prey: q } where p == a && q == b { self += 1.0 }
  clamp: [0.0, 100.0],
}
@spatial(radius = config.hunt.strike_radius, kind = [Agent])
@top_k(1)
query closest_prey(self: Agent) -> [Agent]
sort_by distance(self, _) limit 1 { candidate != self }
@phase(per_agent)
physics ReadPair {
  on Tick {} where self.creature_type == Wolf {
    for prey in spatial.closest_prey(self) {
      emit Damaged { source: self, target: prey, amount: 1.0 + pair_focus(self, prey) }
    }
  }
}
"#;

#[test]
fn dynamic_k_pair_view_read_emits_valid_2arg_accessor() {
    let art = compile(PAIR_VIEW_READ);
    // The accessor def must be 2-arg f32 (the view is f32-typed) using the
    // runtime cfg.agent_cap stride — matching the 2-arg call site.
    let body = art
        .wgsl_files
        .iter()
        .find(|(n, _)| n.contains("ReadPair"))
        .map(|(_, b)| b.as_str())
        .expect("ReadPair kernel");
    assert!(
        body.contains("fn view_0_get(observer: u32, key: u32) -> f32"),
        "expected a 2-arg f32 pair accessor; got:\n{body}"
    );
    assert!(
        body.contains("observer * cfg.agent_cap + key"),
        "expected cfg.agent_cap stride; got:\n{body}"
    );
    // The whole point: naga VALIDATION (call arity) must pass.
    naga_validate(&art);
}
