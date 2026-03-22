//! Delivery method lowering (AST DeliveryNode → Delivery).

use super::ast::*;
use super::lower_effects::lower_effect;
use crate::effects::types::*;

pub(super) fn lower_delivery(node: &DeliveryNode) -> Result<Delivery, String> {
    match node.method.as_str() {
        "projectile" => {
            let mut speed = 8.0f32;
            let mut pierce = false;
            let mut width = 0.0f32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "speed" => speed = val.as_f64().unwrap_or(8.0) as f32,
                    "pierce" => pierce = true,
                    "width" => width = val.as_f64().unwrap_or(0.0) as f32,
                    _ => {}
                }
            }

            let on_hit = node.on_hit.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;
            let on_arrival = node.on_arrival.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Projectile { speed, pierce, width, on_hit, on_arrival })
        }
        "chain" => {
            let mut bounces = 3u32;
            let mut bounce_range = 3.0f32;
            let mut falloff = 0.0f32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "bounces" => bounces = val.as_f64().unwrap_or(3.0) as u32,
                    "range" => bounce_range = val.as_f64().unwrap_or(3.0) as f32,
                    "falloff" => falloff = val.as_f64().unwrap_or(0.0) as f32,
                    _ => {}
                }
            }

            let on_hit = node.on_hit.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Chain { bounces, bounce_range, falloff, on_hit })
        }
        "zone" => {
            let mut duration_ms = 0u32;
            let mut tick_interval_ms = 1000u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(1000),
                    _ => {}
                }
            }

            // Zone uses on_hit for tick effects
            Ok(Delivery::Zone { duration_ms, tick_interval_ms })
        }
        "channel" => {
            let mut duration_ms = 0u32;
            let mut tick_interval_ms = 500u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(500),
                    _ => {}
                }
            }

            Ok(Delivery::Channel { duration_ms, tick_interval_ms })
        }
        "tether" => {
            let mut max_range = 5.0f32;
            let mut tick_interval_ms = 500u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "max_range" => max_range = val.as_f64().unwrap_or(5.0) as f32,
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(500),
                    _ => {}
                }
            }

            let on_complete = node.on_complete.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Tether { max_range, tick_interval_ms, on_complete })
        }
        "trap" => {
            let mut duration_ms = 0u32;
            let mut trigger_radius = 1.5f32;
            let mut arm_time_ms = 0u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "trigger_radius" => trigger_radius = val.as_f64().unwrap_or(1.5) as f32,
                    "arm_time" => arm_time_ms = val.as_duration_ms().unwrap_or(0),
                    _ => {}
                }
            }

            Ok(Delivery::Trap { duration_ms, trigger_radius, arm_time_ms })
        }
        other => Err(format!("unknown delivery method: {other}")),
    }
}
