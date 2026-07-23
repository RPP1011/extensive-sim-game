//! Plan A — lower a parsed `render {}` block to a JSON descriptor string
//! matching `engine_play_api::RenderDescriptor`'s serde shape. The enums are
//! serde-default externally tagged:
//!
//! - `CameraSpec`: `{"Follow":{"field":"mana","lo":0.5,"hi":1.5}}` / `"Observer"`
//! - `VfxKind`: `"Ring"` / `{"BeamToNearest":{"target":{...}}}`
//!
//! ```json
//! {"arena_radius":120.0,
//!  "camera":{"Follow":{"field":"mana","lo":0.5,"hi":1.5}},
//!  "agents":[{"when":{"field":"mana","lo":0.5,"hi":1.5},"color":[0,220,220]}],
//!  "vfx":[{"on_rule":"NovaFire","period":40,"kind":"Ring","radius":6.0,"color":[255,255,120]}]}
//! ```

use std::collections::BTreeMap;

use dsl_ast::ast::{CameraDecl, FieldRangeDecl, RenderDecl, VfxKindDecl};

use super::json::{json_escape, json_f32};

/// Lower a `FieldRangeDecl` to the `RenderDescriptor` `FieldRange` JSON.
///
/// Subkind selectors (`when creature_type is <Subkind>`) resolve the
/// subkind name to its declaration-order `creature_type` ordinal via
/// `entity_ordinals` and emit `field:"creature_type", lo = ordinal,
/// hi = ordinal + 1` — the bridge's in_range test is half-open [lo, hi),
/// so the +1 selects exactly this ordinal (lo == hi would match nothing;
/// that latent defect hid every subkind-keyed render block until the
/// webband port's S8-prep caught it).
fn field_range_json(r: &FieldRangeDecl, entity_ordinals: &BTreeMap<String, u32>) -> String {
    if let Some(subkind) = &r.subkind {
        let ord = entity_ordinals.get(subkind).copied().unwrap_or_else(|| {
            panic!(
                "render `creature_type is {subkind}`: unknown entity subkind \
                 (declared entities: {:?})",
                entity_ordinals.keys().collect::<Vec<_>>(),
            )
        });
        // The bridge's in_range is half-open [lo, hi): lo == hi matches
        // NOTHING, so a subkind selector must emit hi = ordinal + 1 to
        // select exactly its ordinal (webband-port S8-prep finding).
        let ord_f = ord as f32;
        return format!(
            "{{\"field\":\"creature_type\",\"lo\":{lo},\"hi\":{hi}}}",
            lo = json_f32(ord_f),
            hi = json_f32(ord_f + 1.0),
        );
    }
    format!(
        "{{\"field\":\"{field}\",\"lo\":{lo},\"hi\":{hi}}}",
        field = json_escape(&r.field),
        lo = json_f32(r.lo as f32),
        hi = json_f32(r.hi as f32),
    )
}

fn color_json(c: &[u8; 3]) -> String {
    format!("[{},{},{}]", c[0], c[1], c[2])
}

/// Serialize a `RenderDecl` into the `RenderDescriptor` JSON string.
///
/// `entity_ordinals` maps each `entity X : Agent` subkind name to its
/// declaration-order `creature_type` ordinal, used to lower the
/// `when creature_type is <Subkind>` selector (see [`field_range_json`]).
pub fn render_decl_to_json(
    d: &RenderDecl,
    entity_ordinals: &BTreeMap<String, u32>,
) -> String {
    let camera = match &d.camera {
        CameraDecl::Follow(r) => {
            format!("{{\"Follow\":{}}}", field_range_json(r, entity_ordinals))
        }
        CameraDecl::Observer => "\"Observer\"".to_string(),
    };
    let mut agents = String::from("[");
    for (i, a) in d.agents.iter().enumerate() {
        if i > 0 {
            agents.push(',');
        }
        agents.push_str(&format!(
            "{{\"when\":{when},\"color\":{color}}}",
            when = field_range_json(&a.when, entity_ordinals),
            color = color_json(&a.color),
        ));
    }
    agents.push(']');
    let mut vfx = String::from("[");
    for (i, v) in d.vfx.iter().enumerate() {
        if i > 0 {
            vfx.push(',');
        }
        let kind = match &v.kind {
            VfxKindDecl::Ring => "\"Ring\"".to_string(),
            VfxKindDecl::BeamToNearest { target } => {
                format!(
                    "{{\"BeamToNearest\":{{\"target\":{}}}}}",
                    field_range_json(target, entity_ordinals)
                )
            }
        };
        vfx.push_str(&format!(
            "{{\"on_rule\":\"{on_rule}\",\"period\":{period},\"kind\":{kind},\"radius\":{radius},\"color\":{color}}}",
            on_rule = json_escape(&v.on_rule),
            period = v.period,
            radius = json_f32(v.radius as f32),
            color = color_json(&v.color),
        ));
    }
    vfx.push(']');
    format!(
        "{{\"arena_radius\":{arena},\"camera\":{camera},\"agents\":{agents},\"vfx\":{vfx}}}",
        arena = json_f32(d.arena_radius as f32),
    )
}

/// The empty `render {}` default — observer camera, no agents/vfx, radius 0.
/// (Valid `RenderDescriptor` JSON so the player never panics on a fixture
/// that declares no `render` block.)
pub fn empty_render_json() -> String {
    "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}".to_string()
}
