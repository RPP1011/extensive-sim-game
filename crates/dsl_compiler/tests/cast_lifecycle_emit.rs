//! Plan G — cast lifecycle emit pin (#284 sub-item #7).
//!
//! Proves that `emit CastResolved { actor: self }` in a `.sim`
//! lowers to a WGSL chronicle write tagged with engine kind id 79
//! (the `EventKindId::CastResolved` discriminant — see
//! `crates/engine/src/cascade/handler.rs`). Without this end-to-end
//! wiring, GPU-side cast resolution can't surface a chronicle row
//! the host-side replay path will recognise as a CastResolved
//! lifecycle event, breaking cross-backend parity for replay.
//!
//! The companion alias entry landed at `843f033e`
//! (`engine_events.rs::ENGINE_EVENT_KIND_IDS`); this test exercises
//! the full parse → resolve → lower → emit pipeline that depends on
//! that alias.
//!
//! Same shape as `crates/dsl_compiler/tests/memory_ordering_cas_emit.rs`:
//! inline a tiny `.sim`, run through `parse → resolve →
//! lower_compilation_to_cg → synthesize_schedule → emit_cg_program`,
//! and inspect the emitted WGSL.

#[test]
fn cast_resolved_emit_writes_chronicle_kind_79() {
    let src = r#"
        event Tick { }
        event CastResolved { actor: AgentId, ability_id: u32 }

        entity Caster : Agent { }

        config probe {
            firebolt_id: u32 = 1,
        }

        @phase(per_agent)
        physics ResolveCast {
            on Tick {} where (self.alive) {
                emit CastResolved { actor: self, ability_id: config.probe.firebolt_id }
            }
        }
    "#;
    let program = dsl_compiler::parse(src).expect("parse inline sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    // Belt-and-suspenders sanity: the resolved event MUST carry the
    // engine kind id 79. If this trips, the alias table regressed and
    // the WGSL emit-tag assertion below would point at the wrong
    // failure layer.
    let cast_resolved = comp
        .events
        .iter()
        .find(|e| e.name == "CastResolved")
        .expect("CastResolved declared");
    assert_eq!(
        cast_resolved.engine_kind_id,
        Some(79),
        "CastResolved must alias to engine kind 79; got {:?}",
        cast_resolved.engine_kind_id,
    );

    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");

    // Find the kernel that emits CastResolved — its WGSL must contain
    // an atomicStore on event_ring slot 0 with the literal `79u`.
    let resolve_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ResolveCast"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> =
                artifacts.wgsl_files.keys().map(|s| s.as_str()).collect();
            panic!("no kernel containing 'ResolveCast' in emitted WGSL; got: {names:?}")
        });

    assert!(
        resolve_wgsl.contains("atomicStore(&event_ring[slot * 11u + 0u], 79u);"),
        "expected chronicle kind tag = 79u (EventKindId::CastResolved); WGSL was:\n{resolve_wgsl}"
    );

    // The actor field (slot 2) MUST be the running agent_id — no
    // sentinel, no zero. Catches a future regression where the emit
    // path stops binding the `self` ident.
    assert!(
        resolve_wgsl.contains("atomicStore(&event_ring[slot * 11u + 2u], (agent_id));"),
        "expected actor (slot 2) = agent_id; WGSL was:\n{resolve_wgsl}"
    );
}
