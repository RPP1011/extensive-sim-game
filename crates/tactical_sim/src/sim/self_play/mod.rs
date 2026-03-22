//! Self-play action space definitions (V5 pipeline).

pub mod actions;
pub mod actions_pointer;
pub mod actions_dual_head;

/// Max ability slots encoded.
const MAX_ABILITIES: usize = 8;

/// Number of discrete actions (legacy flat action space, used by combined recording).
/// 0: attack nearest, 1: attack weakest, 2: attack focus
/// 3-10: use ability 0-7
/// 11: move toward, 12: move away, 13: hold
const NUM_ACTIONS: usize = 14;

#[allow(unused_imports)]
pub use actions::{action_mask, action_to_intent, action_to_intent_with_focus, intent_to_action};
