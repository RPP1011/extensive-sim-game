//! World Sim DSL compiler — Rust/Python/WGSL emission.
//!
//! Frontend (parser, AST, IR, name resolution) lives in the `dsl_ast`
//! crate and is re-exported here for backward compatibility. This
//! crate owns schema hashing + Rust/Python/WGSL emitters; emitted
//! files land on disk via the xtask `compile-dsl` subcommand.

// Frontend re-exports
pub use dsl_ast::ast;
pub use dsl_ast::error;
pub use dsl_ast::ir;
pub use dsl_ast::parser;
pub use dsl_ast::resolve;
pub use dsl_ast::resolve_error;
pub use dsl_ast::tokens;

// Emitter modules
pub mod emit_backend;
pub mod emit_binding_sources;
pub mod emit_cascade_register;
pub mod emit_config;
pub mod emit_entity;
pub mod emit_enum;
pub mod emit_external_buffers;
pub mod emit_kernel_index;
pub mod emit_mask;
pub mod emit_mask_fill;
pub mod emit_mask_kernel;
pub mod emit_mask_wgsl;
pub mod emit_megakernel;
pub mod emit_movement_kernel;
pub mod emit_physics;
pub mod emit_physics_wgsl;
pub mod emit_pick_ability_kernel;
pub mod emit_pingpong_context;
pub mod emit_pool;
pub mod emit_python;
pub mod emit_resident_context;
pub mod emit_rust;
pub mod emit_schedule;
pub mod emit_scoring;
pub mod emit_scoring_kernel;
pub mod emit_scoring_wgsl;
pub mod emit_sim_cfg;
pub mod emit_spatial_kernel;
pub mod emit_step;
pub mod emit_transient_handles;
pub mod emit_view;
pub mod emit_view_fold_kernel;
pub mod emit_view_wgsl;
pub mod schema_hash;

// Top-level symbol re-exports
pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;
pub use ir::Compilation;
pub use resolve_error::ResolveError;
pub use dsl_ast::{compile, compile_ast, parse, CompileError};

// ---------------------------------------------------------------------------
// Emission bundle
// ---------------------------------------------------------------------------

/// Rust + Python source strings, ready to write to disk. Filenames are
/// snake-cased declaration names (minus extension); the caller decides what
/// extension and parent directory to use.
#[derive(Debug, Clone)]
pub struct EmittedArtifacts {
    /// Content of the events-mod file (`crates/engine_data/src/events/mod.rs`).
    pub rust_events_mod: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub rust_event_structs: Vec<(String, String)>,
    /// Physics-rule modules. `(filename_without_dir, content)` pairs, one
    /// per `physics` declaration. Path target is `crates/engine/src/generated/physics/`
    /// — see `docs/game/feature_flow.md` for the rationale (engine_rules
    /// can't import `engine::cascade::*` without breaking the dep graph).
    pub rust_physics_modules: Vec<(String, String)>,
    /// Content of `crates/engine/src/generated/physics/mod.rs` — the
    /// aggregator that exposes a single `pub fn register(registry)` for the
    /// engine's builtin registration to call.
    pub rust_physics_mod: String,
    /// Mask-predicate modules (milestone 4). One file per `mask`
    /// declaration; target `crates/engine/src/generated/mask/`. Empty Vec
    /// when no masks are in scope; the aggregator still emits an empty
    /// `register()` stub.
    pub rust_mask_modules: Vec<(String, String)>,
    /// Content of `crates/engine/src/generated/mask/mod.rs`.
    pub rust_mask_mod: String,
    /// Scoring-table modules (milestone 4). One file per `scoring`
    /// declaration; target `crates/engine_data/src/scoring/`. Scoring rows
    /// are POD `#[repr(C)]` so CPU + GPU backends read the same layout.
    pub rust_scoring_modules: Vec<(String, String)>,
    /// Content of `crates/engine_data/src/scoring/mod.rs`.
    pub rust_scoring_mod: String,
    /// Entity modules (milestone 5). One file per `entity` declaration;
    /// target `crates/engine_data/src/entities/`.
    pub rust_entity_modules: Vec<(String, String)>,
    /// Content of `crates/engine_data/src/entities/mod.rs`.
    pub rust_entity_mod: String,
    /// Config modules. One file per `config` declaration; target
    /// `crates/engine_data/src/config/`. Pure data, no engine dependency.
    pub rust_config_modules: Vec<(String, String)>,
    /// Content of `crates/engine_data/src/config/mod.rs` — aggregate
    /// `Config` struct, per-block re-exports, TOML loader.
    pub rust_config_mod: String,
    /// TOML-encoded defaults — written to `assets/config/default.toml`.
    pub config_default_toml: String,
    /// Enum modules. One file per `enum` declaration; target
    /// `crates/engine_data/src/enums/`.
    pub rust_enum_modules: Vec<(String, String)>,
    /// Content of `crates/engine_data/src/enums/mod.rs`.
    pub rust_enum_mod: String,
    /// View modules. One file per `view` declaration; target
    /// `crates/engine/src/generated/views/`. `@lazy` views become inline
    /// `pub fn`s; `@materialized` views become fold-storage structs.
    pub rust_view_modules: Vec<(String, String)>,
    /// Content of `crates/engine/src/generated/views/mod.rs` — the
    /// aggregator that exposes `pub struct ViewRegistry` + `fold_all`.
    pub rust_view_mod: String,
    /// Python enum modules. One file per `enum`; target
    /// `generated/python/enums/<name>.py`.
    pub python_enum_modules: Vec<(String, String)>,
    /// Content of `generated/python/enums/__init__.py`.
    pub python_enum_init: String,
    /// Content of `generated/python/events/__init__.py`.
    pub python_events_init: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub python_event_modules: Vec<(String, String)>,
    /// Raw 32-byte schema hash covering the event taxonomy.
    pub event_hash: [u8; 32],
    /// Raw 32-byte schema hash covering the physics-rule subset of the rules
    /// taxonomy. Will grow to also cover masks (milestone 4) and verbs
    /// (milestone 7) without changing the API surface here.
    pub rules_hash: [u8; 32],
    /// Raw 32-byte schema hash covering every `config` block's name + field
    /// names + field types. Default *values* are intentionally excluded so
    /// balance tuning via TOML doesn't perturb the schema fingerprint.
    pub config_hash: [u8; 32],
    /// Raw 32-byte schema hash covering every standalone `enum`
    /// declaration (name + variants in source order). Variant order is
    /// part of the hash since variant ordinals are load-bearing.
    pub enums_hash: [u8; 32],
    /// Raw 32-byte schema hash covering every `view` declaration (name +
    /// params + kind + storage hint + decay params + fold-body structure).
    /// Spec §2.3.
    pub views_hash: [u8; 32],
    /// Combined hash per `docs/compiler/spec.md` §2 — rolls every
    /// sub-hash together with a stable canonical ordering.
    pub combined_hash: [u8; 32],
    /// Content of `crates/engine_data/src/schema.rs`.
    pub schema_rs: String,
    /// Content of `crates/engine/src/event/event_like_impl.rs` — the
    /// machine-generated `impl engine::event::EventLike for Event { ... }`.
    /// Lives in `engine` (not `engine_data`) to avoid a dep cycle while
    /// `engine` retains its `engine_data` regular dep (Plan B2 deferred).
    /// Included from `engine/src/event/mod.rs` via `mod event_like_impl;`.
    pub engine_event_like_impl: String,
}

/// Emit the full artefact bundle for a resolved `Compilation`. Covers
/// events (milestone 2) and physics rules (milestone 3); other declaration
/// kinds produce no output until their milestones land.
pub fn emit(comp: &Compilation) -> EmittedArtifacts {
    emit_with_source(comp, None)
}

/// Like [`emit`], but stamp a single `source_file` into every per-decl
/// header — used when the compilation is sourced from one `.sim` file.
pub fn emit_with_source(comp: &Compilation, source_file: Option<&str>) -> EmittedArtifacts {
    emit_with_per_kind_sources(
        comp,
        EmissionSources {
            events: source_file,
            physics: source_file,
            masks: source_file,
            scoring: source_file,
            entities: source_file,
            configs: source_file,
            enums: source_file,
            views: source_file,
        },
    )
}

/// Per-decl-kind source paths. Each is the path to stamp into that kind's
/// emitted-file headers. `None` for a kind means "emit a generic header"
/// (no path stamp). The xtask uses this when events live in `events.sim`
/// and physics in `physics.sim` — each kind's emission gets the right
/// source-file annotation.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmissionSources<'a> {
    pub events: Option<&'a str>,
    pub physics: Option<&'a str>,
    pub masks: Option<&'a str>,
    pub scoring: Option<&'a str>,
    pub entities: Option<&'a str>,
    pub configs: Option<&'a str>,
    pub enums: Option<&'a str>,
    pub views: Option<&'a str>,
}

/// Like [`emit`], but stamp the appropriate source file into each kind's
/// per-decl headers. Used by the xtask which tracks per-kind source files.
pub fn emit_with_per_kind_sources(
    comp: &Compilation,
    sources: EmissionSources<'_>,
) -> EmittedArtifacts {
    let mut rust_event_structs = Vec::with_capacity(comp.events.len());
    let mut python_event_modules = Vec::with_capacity(comp.events.len());
    for event in &comp.events {
        let stem = snake_case(&event.name);
        let rs = emit_rust::emit_event(event, sources.events);
        let py = emit_python::emit_event_dataclass(event, sources.events);
        rust_event_structs.push((format!("{stem}.rs"), rs));
        python_event_modules.push((format!("{stem}.py"), py));
    }
    rust_event_structs.sort_by(|a, b| a.0.cmp(&b.0));
    python_event_modules.sort_by(|a, b| a.0.cmp(&b.0));

    // Physics emission is fallible (unsupported IR shape, unresolved event
    // ref). On error we panic with a diagnostic — the xtask catches user
    // errors via `compile()` returning IR errors, so a resolved IR shape we
    // can't emit is a compiler bug.
    let physics_ctx = emit_physics::EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut rust_physics_modules: Vec<(String, String)> = Vec::with_capacity(comp.physics.len());
    for physics in &comp.physics {
        let stem = snake_case(&physics.name);
        match emit_physics::emit_physics(physics, sources.physics, &physics_ctx) {
            Ok(rs) => rust_physics_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => {
                panic!("physics emission failed for `{}`: {e}", physics.name);
            }
        }
    }
    rust_physics_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_physics_mod = emit_physics::emit_physics_mod(&comp.physics, &physics_ctx);

    // Mask / scoring / entity emission — scaffolded at milestone-3-completion.
    // The per-decl emitters panic via `Err` until their milestones land;
    // the aggregators produce valid empty output so the scaffold compiles
    // without any masks / scoring / entities in scope.
    let mask_ctx = emit_mask::EmitContext { views: &comp.views };
    let mut rust_mask_modules: Vec<(String, String)> = Vec::with_capacity(comp.masks.len());
    for mask in &comp.masks {
        let stem = snake_case(&mask.head.name);
        match emit_mask::emit_mask_with_ctx(mask, sources.masks, mask_ctx) {
            Ok(rs) => rust_mask_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => panic!("mask emission failed for `{}`: {e}", mask.head.name),
        }
    }
    rust_mask_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_mask_mod = emit_mask::emit_mask_mod(&comp.masks);

    let mut rust_scoring_modules: Vec<(String, String)> = Vec::with_capacity(comp.scoring.len());
    for (i, scoring) in comp.scoring.iter().enumerate() {
        // Scoring decls are anonymous — index them for a stable filename.
        let stem = format!("scoring_{i:03}");
        match emit_scoring::emit_scoring(scoring, &comp.views, sources.scoring) {
            Ok(rs) => rust_scoring_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => panic!("scoring emission failed for entry {i}: {e}"),
        }
    }
    rust_scoring_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_scoring_mod = emit_scoring::emit_scoring_mod(&comp.scoring, &comp.views);

    let mut rust_entity_modules: Vec<(String, String)> = Vec::with_capacity(comp.entities.len());
    for entity in &comp.entities {
        let stem = snake_case(&entity.name);
        match emit_entity::emit_entity(entity, sources.entities) {
            Ok(rs) => rust_entity_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => panic!("entity emission failed for `{}`: {e}", entity.name),
        }
    }
    rust_entity_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_entity_mod = emit_entity::emit_entity_mod(&comp.entities);

    // Config emission — one Rust struct + one TOML table per `config` block.
    let mut rust_config_modules: Vec<(String, String)> = Vec::with_capacity(comp.configs.len());
    for block in &comp.configs {
        let stem = config_snake_case(&block.name);
        match emit_config::emit_config(block, sources.configs) {
            Ok(rs) => rust_config_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => panic!("config emission failed for `{}`: {e}", block.name),
        }
    }
    rust_config_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_config_mod = emit_config::emit_config_mod(&comp.configs);
    let config_default_toml = emit_config::emit_config_toml(&comp.configs)
        .unwrap_or_else(|e| panic!("config TOML emission failed: {e}"));

    // Enum emission — per-block Rust + Python files + aggregate mod/init.
    let mut rust_enum_modules: Vec<(String, String)> = Vec::with_capacity(comp.enums.len());
    let mut python_enum_modules: Vec<(String, String)> = Vec::with_capacity(comp.enums.len());
    for e in &comp.enums {
        let stem = snake_case(&e.name);
        rust_enum_modules.push((format!("{stem}.rs"), emit_enum::emit_enum(e, sources.enums)));
        python_enum_modules.push((
            format!("{stem}.py"),
            emit_enum::emit_enum_py(e, sources.enums),
        ));
    }
    rust_enum_modules.sort_by(|a, b| a.0.cmp(&b.0));
    python_enum_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_enum_mod = emit_enum::emit_enum_mod(&comp.enums);
    let python_enum_init = emit_enum::emit_enum_init(&comp.enums);

    // View emission — one module per view + an aggregator with
    // `ViewRegistry`. Each `@lazy` view lowers to a fn, each
    // `@materialized` view to a struct + `fold_event`.
    let mut rust_view_modules: Vec<(String, String)> = Vec::with_capacity(comp.views.len());
    for v in &comp.views {
        let stem = snake_case(&v.name);
        match emit_view::emit_view(v, &comp.events, sources.views) {
            Ok(rs) => rust_view_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => panic!("view emission failed for `{}`: {e}", v.name),
        }
    }
    rust_view_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_view_mod = emit_view::emit_view_mod(&comp.views);

    let event_hash = schema_hash::event_hash(&comp.events);
    let rules_hash = schema_hash::rules_hash(&comp.physics, &comp.event_tags, &comp.events);
    let state_hash = schema_hash::state_hash(&comp.entities);
    let scoring_hash = schema_hash::scoring_hash(&comp.scoring);
    let config_hash = schema_hash::config_hash(&comp.configs);
    let enums_hash = schema_hash::enums_hash(&comp.enums);
    let views_hash = schema_hash::views_hash(&comp.views);
    // gpu_rules_hash is bootstrapped here as the empty input (no
    // engine_gpu_rules generated bytes are visible from this crate). The
    // xtask's compile-dsl flow recomputes the real value once the
    // engine_gpu_rules/src/ tree has been emitted, and writes the result
    // to `crates/engine_gpu_rules/.schema_hash`. Threading the real
    // bytes through the EmittedArtifacts pipeline isn't required for
    // any current consumer.
    let gpu_rules_hash = schema_hash::gpu_rules_hash(&[]);
    let combined_hash = schema_hash::combined_hash(
        &state_hash,
        &event_hash,
        &rules_hash,
        &scoring_hash,
        &config_hash,
        &enums_hash,
        &views_hash,
        &gpu_rules_hash,
    );
    // Emit the `impl EventLike for Event` block. Lives in engine (not
    // engine_data) to avoid a dep cycle while engine retains its
    // engine_data regular dep (chronicle.rs, Plan B2 deferred).
    let engine_event_like_impl = {
        // Reuse the same sorted+hydrated list that emit_events_mod uses.
        let hydrated: Vec<ir::EventIR> = comp.events
            .iter()
            .map(|e| {
                let mut c = e.clone();
                let mut out = e.fields.clone();
                if !out.iter().any(|f| f.name == "tick") {
                    out.push(ir::EventField {
                        name: "tick".into(),
                        ty: ir::IrType::U32,
                        span: ast::Span::dummy(),
                    });
                }
                c.fields = out;
                c
            })
            .collect();
        let mut sorted: Vec<&ir::EventIR> = hydrated.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut buf = String::new();
        emit_rust::emit_event_like_impl(&mut buf, &sorted);
        buf
    };

    EmittedArtifacts {
        rust_events_mod: emit_rust::emit_events_mod(&comp.events),
        rust_event_structs,
        rust_physics_modules,
        rust_physics_mod,
        rust_mask_modules,
        rust_mask_mod,
        rust_scoring_modules,
        rust_scoring_mod,
        rust_entity_modules,
        rust_entity_mod,
        rust_config_modules,
        rust_config_mod,
        config_default_toml,
        rust_enum_modules,
        rust_enum_mod,
        rust_view_modules,
        rust_view_mod,
        python_enum_modules,
        python_enum_init,
        python_events_init: emit_python::emit_events_init(&comp.events),
        python_event_modules,
        event_hash,
        rules_hash,
        config_hash,
        enums_hash,
        views_hash,
        combined_hash,
        schema_rs: schema_hash::emit_schema_rs(
            &state_hash,
            &event_hash,
            &rules_hash,
            &scoring_hash,
            &config_hash,
            &enums_hash,
            &views_hash,
            &gpu_rules_hash,
        ),
        engine_event_like_impl,
    }
}

/// Config block names use DSL-author lowercase (`combat`, `movement`); keep
/// the snake-case transform consistent with the rest of the emitter.
fn config_snake_case(name: &str) -> String {
    snake_case(name)
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}
