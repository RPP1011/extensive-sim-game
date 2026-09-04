//! Pin: Gap T2 (2026-05-12) — multi-Item entity declarations allocate
//! ONE field-keyed buffer per unique field name across all Items, sized
//! to the count of Item-rooted entities, indexed by Item-type
//! discriminant (position in declaration order among Items).
//!
//! ## Why this matters
//!
//! Pre-T2 the build_helper emitted one buffer per `(entity, field)`
//! pair using the name `<entity_snake>_<field>` (e.g. `grain_base_price`,
//! `spice_base_price`, `silk_base_price`). The unique-binding collector
//! was a `BTreeMap<binding_name, wgsl_ty>` keyed by name — DIFFERENT
//! entities with overlapping field names produced DIFFERENT keys, so
//! all three SHOULD have been emitted. But the LOWERING's
//! `resolve_item_by_name` always returned the FIRST Item entity
//! declaring the field, so every `items.<field>(N)` call lowered to a
//! handle pointing at the same first-declaring entity. The resulting
//! WGSL binding name was always `grain_base_price` regardless of N;
//! only that one buffer ever got allocated; reads with N=1, 2 aliased
//! Grain's buffer at the user's index expression.
//!
//! Post-T2 the binding is FIELD-keyed across all Items (`item_<field>`).
//! The buffer is sized to one slot per declared Item-rooted entity, and
//! the user's index `N` directly selects the right Item — Grain=0,
//! Spice=1, Silk=2 in declaration order among Items.
//!
//! ## What this test pins
//!
//! 1. A `.sim` declaring 3 Item entities each with the same field name
//!    (`base_price: f32`) emits exactly ONE `item_base_price_buf` field
//!    on the synthesized `GeneratedRuntime` struct — not three
//!    entity-prefixed ones.
//! 2. The buffer is sized to 3 slots (one per Item-rooted entity), not
//!    `agent_count` slots (the pre-T2 default).
//! 3. Cross-Item-type distinguishing: emitted WGSL uses the user's
//!    index expression directly against the shared binding — so
//!    `items.base_price(0)` reads slot 0 of `item_base_price`,
//!    `items.base_price(1)` reads slot 1, etc.

use dsl_compiler::cg::emit::EmittedArtifacts;

const MULTI_ITEM_SNIPPET: &str = r#"
entity Hero : Agent {
  pos: vec3,
  vel: vec3,
}

entity Grain : Item { base_price: f32 }
entity Spice : Item { base_price: f32 }
entity Silk  : Item { base_price: f32 }

event Tick { }

@phase(per_agent)
physics ReadAllPrices {
  on Tick {} where self.alive {
    // Read each Item-type's base_price by discriminant.
    let grain_p = items.base_price(0u);
    let spice_p = items.base_price(1u);
    let silk_p  = items.base_price(2u);
    let total = grain_p + spice_p + silk_p;
    let new_pos = self.pos + self.vel * total;
    agents.set_pos(self, new_pos);
  }
}
"#;

fn compile_snippet(src: &str) -> (dsl_ast::ir::Compilation, EmittedArtifacts) {
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts =
        dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg).expect("emit");
    (comp, artifacts)
}

#[test]
fn multi_item_shared_field_emits_single_field_keyed_binding_in_wgsl() {
    let (_comp, artifacts) = compile_snippet(MULTI_ITEM_SNIPPET);

    // Find the ReadAllPrices physics kernel and inspect its WGSL.
    let body = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ReadAllPrices"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            panic!(
                "no ReadAllPrices kernel in artifacts; available: {:?}",
                artifacts.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // Field-keyed shared binding: ONE `item_base_price` declaration
    // across all reads, NOT three entity-prefixed ones.
    let item_base_price_count = body.matches("var<storage").filter(|_| true).count();
    assert!(
        body.contains("item_base_price"),
        "expected `item_base_price` binding (field-keyed across all \
         Items declaring `base_price`); got:\n{body}",
    );
    let _ = item_base_price_count;

    // Pre-T2 (negative): the per-entity bindings MUST NOT exist —
    // they'd indicate the entity-prefixed naming regressed.
    assert!(
        !body.contains("grain_base_price"),
        "post-T2 binding name must be field-keyed `item_base_price`, \
         NOT entity-prefixed `grain_base_price`; got:\n{body}",
    );
    assert!(
        !body.contains("spice_base_price"),
        "post-T2 binding name must be field-keyed `item_base_price`, \
         NOT entity-prefixed `spice_base_price`; got:\n{body}",
    );
    assert!(
        !body.contains("silk_base_price"),
        "post-T2 binding name must be field-keyed `item_base_price`, \
         NOT entity-prefixed `silk_base_price`; got:\n{body}",
    );

    // All three reads route through the SAME binding, each with its
    // own index expression. The exact target_expr ids drift with
    // lowering; assert by counting indexed accesses.
    let n_indexed_reads = body.matches("item_base_price[").count();
    assert!(
        n_indexed_reads >= 3,
        "expected at least 3 indexed reads against `item_base_price[…]` \
         (one per `items.base_price(N)` call for N=0,1,2); got {n_indexed_reads} \
         in body:\n{body}",
    );
}

#[test]
fn multi_item_shared_field_runtime_struct_has_one_buffer_per_field() {
    // Drive the full build-helper synthesis path to confirm the
    // `GeneratedRuntime` struct + try_new emit only ONE buffer per
    // unique field name (not one per (entity, field)).
    let (comp, artifacts) = compile_snippet(MULTI_ITEM_SNIPPET);

    let item_entity_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, dsl_ast::ast::EntityRoot::Item))
        .count() as u32;
    assert_eq!(item_entity_count, 3, "fixture declares 3 Item entities");

    let group_entity_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, dsl_ast::ast::EntityRoot::Group))
        .count() as u32;

    let pair_keyed_second_key = dsl_compiler::build_helper::detect_pair_keyed_second_key(&comp);
    let materialized_views = dsl_compiler::build_helper::collect_materialized_views(&comp);
    let runtime_core = dsl_compiler::build_helper::synthesize_runtime_core_a2(
        "multi_item_smoke",
        &artifacts,
        &[],
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &comp.events,
        pair_keyed_second_key,
        &materialized_views,
        false,
        false, // binds_navgrid
        &[],
        &[],
        item_entity_count,
        group_entity_count,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
        dsl_compiler::cg::lower::DebugDepth::Off,
    );

    // Exactly ONE `item_base_price_buf` field on the struct (field-
    // keyed, shared across Grain / Spice / Silk).
    let n_struct_fields = runtime_core.matches("pub item_base_price_buf: wgpu::Buffer,").count();
    assert_eq!(
        n_struct_fields, 1,
        "expected exactly 1 `item_base_price_buf` field on GeneratedRuntime; \
         got {n_struct_fields} in:\n{runtime_core}",
    );

    // No entity-prefixed buffer fields (pre-T2 naming).
    for legacy_name in &["grain_base_price_buf", "spice_base_price_buf", "silk_base_price_buf"] {
        assert!(
            !runtime_core.contains(legacy_name),
            "post-T2 runtime_core must not emit entity-prefixed buffer \
             `{legacy_name}` — all Item fields collapse onto field-keyed names; \
             got source:\n{runtime_core}",
        );
    }

    // Sized to 3 slots (one per Item entity), not `agent_count`. The
    // emitted try_new line has the literal shape
    // `size: (3u64 * 4u64).max(16)` — assert on the 3-slot count
    // (independent of the elem-bytes constant).
    assert!(
        runtime_core.contains("item_base_price"),
        "expected the alloc to reference the item_base_price name in \
         try_new; got:\n{runtime_core}",
    );
    // Strong assertion: the literal 3-slot sizing landed (not the
    // pre-T2 `agent_count as u64` shape).
    let alloc_line = runtime_core
        .lines()
        .find(|l| l.contains("item_base_price") && l.contains("size:"))
        .or_else(|| {
            // The size: line is on the line after the BufferDescriptor
            // line; scan ±5 lines around the create_buffer call.
            let i = runtime_core
                .lines()
                .position(|l| l.contains("let item_base_price_buf"))?;
            runtime_core
                .lines()
                .skip(i)
                .take(8)
                .find(|l| l.contains("size:"))
        })
        .unwrap_or_else(|| panic!("no `size:` line for item_base_price_buf in:\n{runtime_core}"));
    assert!(
        alloc_line.contains("3u64"),
        "expected `item_base_price_buf` to be sized to 3 slots (one per \
         Item-rooted entity declared in this fixture); got line:\n{alloc_line}\n\
         in source:\n{runtime_core}",
    );
    assert!(
        !alloc_line.contains("agent_count"),
        "post-T2 the `item_<field>` buffer must NOT be sized to \
         agent_count (pre-T2 default); the storage is per-Item-discriminant. \
         got line:\n{alloc_line}",
    );
}

#[test]
fn multi_group_shared_field_emits_field_keyed_binding() {
    // Symmetric coverage for Group-rooted entities — the fix is shared
    // between Item and Group, so a single Group test is sufficient to
    // pin the parallel surface.
    const MULTI_GROUP_SNIPPET: &str = r#"
entity Hero : Agent {
  pos: vec3,
  vel: vec3,
}

entity Guild   : Group { reputation: f32 }
entity Faction : Group { reputation: f32 }

event Tick { }

@phase(per_agent)
physics ReadReputations {
  on Tick {} where self.alive {
    let guild_r   = groups.reputation(0u);
    let faction_r = groups.reputation(1u);
    let total = guild_r + faction_r;
    let new_pos = self.pos + self.vel * total;
    agents.set_pos(self, new_pos);
  }
}
"#;
    let (_comp, artifacts) = compile_snippet(MULTI_GROUP_SNIPPET);
    let body = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ReadReputations"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            panic!(
                "no ReadReputations kernel; available: {:?}",
                artifacts.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    assert!(
        body.contains("group_reputation"),
        "expected field-keyed `group_reputation` binding; got:\n{body}",
    );
    assert!(
        !body.contains("guild_reputation"),
        "post-T2 must NOT emit entity-prefixed `guild_reputation`; got:\n{body}",
    );
    assert!(
        !body.contains("faction_reputation"),
        "post-T2 must NOT emit entity-prefixed `faction_reputation`; got:\n{body}",
    );
}
