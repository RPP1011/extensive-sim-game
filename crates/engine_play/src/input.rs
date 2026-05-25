//! `ControlsMapper` — the pure keys + `ControlsDescriptor` → input-write
//! resolver. No window/GPU dependency, so it is fully unit-testable.
//!
//! Each [`KeyBinding`](engine_play_api::KeyBinding) in `Hold` mode contributes
//! its `value` to its target field for every held key. Values for the same
//! field sum. The paired movement axes (`*.move_x` / `*.move_y`) are normalized
//! to a unit vector when both are nonzero, so a diagonal (`w`+`d`) yields
//! `±1/√2` on each axis rather than `±1`. `Press`-mode bindings are *edge*
//! events the loop tracks separately (see [`resolve_pressed`]); the held-set
//! `resolve` ignores them.

use std::collections::{HashMap, HashSet};

use engine_play_api::{BindMode, ControlsDescriptor};

/// Stateless resolver from held/pressed keys + a `ControlsDescriptor` to the
/// `(field, value)` input writes the player feeds into `set_input`.
pub struct ControlsMapper;

/// The suffix of a field that participates in 2-axis movement normalization.
fn axis_suffix(field: &str) -> Option<&'static str> {
    if field.ends_with(".move_x") || field == "move_x" {
        Some("x")
    } else if field.ends_with(".move_y") || field == "move_y" {
        Some("y")
    } else {
        None
    }
}

/// The block prefix of a `"<block>.<field>"` name (`""` if unprefixed). Used to
/// pair `move_x`/`move_y` only within the same block.
fn block_of(field: &str) -> &str {
    field.rsplit_once('.').map(|(b, _)| b).unwrap_or("")
}

impl ControlsMapper {
    /// Resolve the `Hold`-mode bindings against the `held` key set into the
    /// `(field, value)` writes to apply this frame.
    ///
    /// Algorithm: for each `Hold` binding whose key is held, sum its `value`
    /// into its field; then, per block, if both `move_x` and `move_y` are
    /// present and the vector is longer than ~0, scale both to unit length.
    /// Output is sorted by field name for deterministic assertions.
    pub fn resolve(desc: &ControlsDescriptor, held: &HashSet<String>) -> Vec<(String, f32)> {
        // Sum held Hold-mode bindings per field, preserving first-seen order.
        let mut sums: HashMap<String, f32> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for b in &desc.bindings {
            if b.mode != BindMode::Hold {
                continue;
            }
            // Every bound field is emitted (neutral 0.0 when its key isn't
            // held) so the runtime's input is cleared when keys release.
            if !sums.contains_key(&b.field) {
                sums.insert(b.field.clone(), 0.0);
                order.push(b.field.clone());
            }
            if held.contains(&b.key) {
                *sums.get_mut(&b.field).unwrap() += b.value;
            }
        }

        // Normalize each block's (move_x, move_y) pair to a unit vector.
        let blocks: HashSet<String> = order.iter().map(|f| block_of(f).to_string()).collect();
        for block in blocks {
            let fx = order
                .iter()
                .find(|f| block_of(f) == block && axis_suffix(f) == Some("x"))
                .cloned();
            let fy = order
                .iter()
                .find(|f| block_of(f) == block && axis_suffix(f) == Some("y"))
                .cloned();
            if let (Some(fx), Some(fy)) = (fx, fy) {
                let x = sums[&fx];
                let y = sums[&fy];
                let len = (x * x + y * y).sqrt();
                if len > 1e-6 {
                    sums.insert(fx, x / len);
                    sums.insert(fy, y / len);
                }
            }
        }

        let mut out: Vec<(String, f32)> = order.into_iter().map(|f| {
            let v = sums[&f];
            (f, v)
        }).collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Resolve the `Press`-mode bindings for keys that transitioned down this
    /// frame (`pressed`). One `(field, value)` write per fired binding; no
    /// summing or normalization (these are one-shot events).
    pub fn resolve_pressed(
        desc: &ControlsDescriptor,
        pressed: &HashSet<String>,
    ) -> Vec<(String, f32)> {
        desc.bindings
            .iter()
            .filter(|b| b.mode == BindMode::Press && pressed.contains(&b.key))
            .map(|b| (b.field.clone(), b.value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MOCK_CONTROLS;
    use engine_play_api::{BindMode, KeyBinding};

    fn held(keys: &[&str]) -> HashSet<String> {
        keys.iter().map(|s| s.to_string()).collect()
    }

    fn get(pairs: &[(String, f32)], field: &str) -> Option<f32> {
        pairs.iter().find(|(f, _)| f == field).map(|(_, v)| *v)
    }

    #[test]
    fn controls_map_to_inputs() {
        let desc = ControlsDescriptor::from_json(MOCK_CONTROLS).unwrap();

        // {w, d} held: pre-norm move_y=+1, move_x=+1 → post-norm both +1/√2.
        let out = ControlsMapper::resolve(&desc, &held(&["w", "d"]));
        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        assert!(
            (get(&out, "ctl.move_x").unwrap() - inv_sqrt2).abs() < 1e-5,
            "move_x = {:?}",
            get(&out, "ctl.move_x")
        );
        assert!(
            (get(&out, "ctl.move_y").unwrap() - inv_sqrt2).abs() < 1e-5,
            "move_y = {:?}",
            get(&out, "ctl.move_y")
        );

        // Single key: full magnitude on one axis, zero on the other.
        let out = ControlsMapper::resolve(&desc, &held(&["d"]));
        assert!((get(&out, "ctl.move_x").unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(get(&out, "ctl.move_y"), Some(0.0));

        // Opposing keys cancel: w+s → move_y 0.
        let out = ControlsMapper::resolve(&desc, &held(&["w", "s"]));
        assert_eq!(get(&out, "ctl.move_y"), Some(0.0));

        // Nothing held: every bound field resolves to its neutral 0.0.
        let out = ControlsMapper::resolve(&desc, &held(&[]));
        assert_eq!(get(&out, "ctl.move_x"), Some(0.0));
        assert_eq!(get(&out, "ctl.move_y"), Some(0.0));
    }

    #[test]
    fn press_mode_fires_on_edge_only() {
        let desc = ControlsDescriptor {
            bindings: vec![
                KeyBinding {
                    key: "space".into(),
                    field: "ctl.dash".into(),
                    value: 1.0,
                    mode: BindMode::Press,
                },
                KeyBinding {
                    key: "w".into(),
                    field: "ctl.move_y".into(),
                    value: 1.0,
                    mode: BindMode::Hold,
                },
            ],
        };
        // Press-mode binding ignored by `resolve` (held path).
        let out = ControlsMapper::resolve(&desc, &held(&["space"]));
        assert_eq!(get(&out, "ctl.dash"), None);
        // …and fires on the press edge.
        let pressed = ControlsMapper::resolve_pressed(&desc, &held(&["space"]));
        assert_eq!(pressed, vec![("ctl.dash".to_string(), 1.0)]);
        // No press edge this frame → nothing.
        assert!(ControlsMapper::resolve_pressed(&desc, &held(&[])).is_empty());
    }
}
