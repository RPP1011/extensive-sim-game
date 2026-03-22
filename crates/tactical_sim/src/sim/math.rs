use super::types::{SimVec2, sim_vec2};

pub fn distance(a: SimVec2, b: SimVec2) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

pub fn move_towards(from: SimVec2, to: SimVec2, max_delta: f32) -> SimVec2 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON || max_delta <= f32::EPSILON {
        return from;
    }
    let step = len.min(max_delta);
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(from.x + nx * step, from.y + ny * step)
}

pub fn move_away(from: SimVec2, threat: SimVec2, max_delta: f32) -> SimVec2 {
    let dx = from.x - threat.x;
    let dy = from.y - threat.y;
    let len = (dx * dx + dy * dy).sqrt();
    if max_delta <= f32::EPSILON {
        return from;
    }
    if len <= f32::EPSILON {
        return sim_vec2(from.x + max_delta, from.y);
    }
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(from.x + nx * max_delta, from.y + ny * max_delta)
}

pub fn position_at_range(from: SimVec2, target: SimVec2, desired_range: f32) -> SimVec2 {
    let dx = from.x - target.x;
    let dy = from.y - target.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return sim_vec2(target.x + desired_range, target.y);
    }
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(target.x + nx * desired_range, target.y + ny * desired_range)
}
