mod abilities;
mod healer;
mod targeting;

pub(in crate::ai::squad) use abilities::choose_action;
pub(in crate::ai::squad) use healer::{healer_backline_position, healer_intent};
pub(in crate::ai::squad) use targeting::choose_target;

/// Sort two tag strings alphabetically for consistent reaction lookup.
fn sorted_tags<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a < b { (a, b) } else { (b, a) }
}
