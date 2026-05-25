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

use dsl_ast::ast::{CameraDecl, FieldRangeDecl, RenderDecl, VfxKindDecl};

use super::json::{json_escape, json_f32};

fn field_range_json(r: &FieldRangeDecl) -> String {
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
pub fn render_decl_to_json(d: &RenderDecl) -> String {
    let camera = match &d.camera {
        CameraDecl::Follow(r) => format!("{{\"Follow\":{}}}", field_range_json(r)),
        CameraDecl::Observer => "\"Observer\"".to_string(),
    };
    let mut agents = String::from("[");
    for (i, a) in d.agents.iter().enumerate() {
        if i > 0 {
            agents.push(',');
        }
        agents.push_str(&format!(
            "{{\"when\":{when},\"color\":{color}}}",
            when = field_range_json(&a.when),
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
                format!("{{\"BeamToNearest\":{{\"target\":{}}}}}", field_range_json(target))
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
