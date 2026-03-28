//! Ability Editor — design abilities by navigating a learned latent space.
//!
//! Uses a trained VAE to map 64 latent dimensions to 142-dim ability slot vectors,
//! which are then decoded to DSL text. Adjust sliders to explore the space of
//! possible abilities.
//!
//! Usage:
//!   cargo run --bin ability_editor --release
//!
//! Requires: generated/ability_vae_weights.json from training/train_ability_vae.py

use std::sync::Mutex;
use tack::{TackConfig, TackUi};

// ---------------------------------------------------------------------------
// VAE inference (pure Rust, no ML framework needed)
// ---------------------------------------------------------------------------

const SLOT_DIM: usize = 142;
const LATENT_DIM: usize = 64;
const HIDDEN_DIM: usize = 256;
const NUM_ARCHETYPES: usize = 19;

// Slot layout constants
const EFFECT_OFFSET: usize = 42;
const EFFECT_SLOT_DIM: usize = 25;
const NUM_EFFECT_SLOTS: usize = 4;
const NUM_EFFECT_TYPES: usize = 17;

const EFFECT_NAMES: &[&str] = &[
    "damage", "heal", "shield", "dot", "hot", "slow", "root",
    "stun", "silence", "knockback", "pull", "dash", "blink",
    "stealth", "buff", "debuff", "summon",
];

const TARGETING_NAMES: &[&str] = &[
    "enemy", "ally", "self", "self_aoe", "ground", "direction", "enemy_aoe", "ally_aoe",
];

const HINT_NAMES: &[&str] = &[
    "damage", "crowd_control", "defense", "utility", "heal",
];

const DELIVERY_NAMES: &[&str] = &[
    "instant", "projectile", "aoe", "cone", "line", "channel", "self_buff",
];

struct Vae {
    // Encoder: 3 hidden layers + mu head
    enc1_weight: Vec<Vec<f32>>,
    enc1_bias: Vec<f32>,
    enc2_weight: Vec<Vec<f32>>,
    enc2_bias: Vec<f32>,
    enc3_weight: Vec<Vec<f32>>,
    enc3_bias: Vec<f32>,
    enc_mu_weight: Vec<Vec<f32>>,
    enc_mu_bias: Vec<f32>,
    // Decoder: 3 hidden layers + output
    dec1_weight: Vec<Vec<f32>>,
    dec1_bias: Vec<f32>,
    dec2_weight: Vec<Vec<f32>>,
    dec2_bias: Vec<f32>,
    dec3_weight: Vec<Vec<f32>>,
    dec3_bias: Vec<f32>,
    dec_out_weight: Vec<Vec<f32>>,
    dec_out_bias: Vec<f32>,
    // Scaler params for de-standardization
    scaler_means: Vec<f32>,
    scaler_stds: Vec<f32>,
}

#[derive(Clone)]
struct Preset {
    name: String,
    z: Vec<f64>,
}

impl Vae {
    fn load(path: &str) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let raw: serde_json::Value = serde_json::from_str(&content).ok()?;
        let obj = raw.as_object()?;

        let get_matrix = |name: &str| -> Vec<Vec<f32>> {
            obj.get(name)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|row| {
                            row.as_array()
                                .map(|r| r.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        let get_bias = |name: &str| -> Vec<f32> {
            obj.get(name)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                .unwrap_or_default()
        };

        let get_vec = |name: &str| -> Vec<f32> {
            obj.get(name)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                .unwrap_or_default()
        };

        Some(Self {
            enc1_weight: get_matrix("enc1.weight"),
            enc1_bias: get_bias("enc1.bias"),
            enc2_weight: get_matrix("enc2.weight"),
            enc2_bias: get_bias("enc2.bias"),
            enc3_weight: get_matrix("enc3.weight"),
            enc3_bias: get_bias("enc3.bias"),
            enc_mu_weight: get_matrix("enc_mu.weight"),
            enc_mu_bias: get_bias("enc_mu.bias"),
            dec1_weight: get_matrix("dec1.weight"),
            dec1_bias: get_bias("dec1.bias"),
            dec2_weight: get_matrix("dec2.weight"),
            dec2_bias: get_bias("dec2.bias"),
            dec3_weight: get_matrix("dec3.weight"),
            dec3_bias: get_bias("dec3.bias"),
            dec_out_weight: get_matrix("dec_out.weight"),
            dec_out_bias: get_bias("dec_out.bias"),
            scaler_means: get_vec("_scaler_means"),
            scaler_stds: get_vec("_scaler_stds"),
        })
    }

    fn decode(&self, z: &[f64]) -> Vec<f32> {
        // Decoder input: z (64) + archetype (19 zeros — no archetype conditioning)
        let mut input = Vec::with_capacity(LATENT_DIM + NUM_ARCHETYPES);
        for &v in z {
            input.push(v as f32);
        }
        // Pass zero archetype vector
        for _ in 0..NUM_ARCHETYPES {
            input.push(0.0);
        }
        let h1 = linear_relu(&input, &self.dec1_weight, &self.dec1_bias);
        let h2 = linear_relu(&h1, &self.dec2_weight, &self.dec2_bias);
        let h3 = linear_relu(&h2, &self.dec3_weight, &self.dec3_bias);
        let standardized = linear(&h3, &self.dec_out_weight, &self.dec_out_bias);

        // De-standardize
        if self.scaler_means.len() == SLOT_DIM && self.scaler_stds.len() == SLOT_DIM {
            standardized.iter().enumerate().map(|(i, &v)| {
                v * self.scaler_stds[i] + self.scaler_means[i]
            }).collect()
        } else {
            standardized
        }
    }

    fn encode(&self, slots: &[f32]) -> Vec<f64> {
        // Standardize first
        let standardized: Vec<f32> = if self.scaler_means.len() == SLOT_DIM {
            slots.iter().enumerate().map(|(i, &v)| {
                if self.scaler_stds[i].abs() > 1e-6 {
                    (v - self.scaler_means[i]) / self.scaler_stds[i]
                } else {
                    0.0
                }
            }).collect()
        } else {
            slots.to_vec()
        };

        let mut input = Vec::with_capacity(SLOT_DIM + NUM_ARCHETYPES);
        input.extend_from_slice(&standardized);
        for _ in 0..NUM_ARCHETYPES {
            input.push(0.0);
        }
        let h1 = linear_relu(&input, &self.enc1_weight, &self.enc1_bias);
        let h2 = linear_relu(&h1, &self.enc2_weight, &self.enc2_bias);
        let h3 = linear_relu(&h2, &self.enc3_weight, &self.enc3_bias);
        let mu = linear(&h3, &self.enc_mu_weight, &self.enc_mu_bias);
        mu.into_iter().map(|v| v as f64).collect()
    }
}

fn curated_presets(vae: &Vae) -> Vec<Preset> {
    // Build slot vectors using the CORRECT layout:
    // [0:3] output_type, [3:11] targeting, [11:15] range/cd/cast/cost
    // [15:20] hint, [20:27] delivery, [27:33] delivery params
    // [33:37] charges/toggle/recast/unstoppable, [37:42] charge params
    // [42:67] effect 0 (25d), [67:92] effect 1, [92:117] effect 2, [117:142] effect 3
    // Each effect: [0:17] type one-hot, [17] param/155, [18] duration/10000, [19:24] area, [24] condition

    let mut presets = Vec::new();

    // Damage nuke: enemy, ranged, damage, single-target damage effect
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;  // active
        s[3] = 1.0;  // target: enemy
        s[11] = 0.5; // range 5.0
        s[12] = 0.4; // cooldown 12s
        s[13] = 0.1; // cast 300ms
        s[15] = 1.0; // hint: damage
        s[20] = 1.0; // delivery: instant
        // Effect 0: damage (type index 0)
        s[EFFECT_OFFSET + 0] = 1.0;  // damage type
        s[EFFECT_OFFSET + 17] = 0.5; // param: 77 damage
        presets.push(Preset {
            name: "Damage Nuke".into(),
            z: vae.encode(&s),
        });
    }

    // Burst Heal: ally target, heal effect
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[4] = 1.0;  // target: ally
        s[11] = 0.4; // range 4.0
        s[12] = 0.2; // cooldown 6s
        s[15 + 4] = 1.0; // hint: heal
        s[20] = 1.0; // instant
        s[EFFECT_OFFSET + 1] = 1.0;  // heal type
        s[EFFECT_OFFSET + 17] = 0.4; // 62 heal
        presets.push(Preset {
            name: "Burst Heal".into(),
            z: vae.encode(&s),
        });
    }

    // Tank Shield: self target, shield
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[5] = 1.0;  // target: self
        s[12] = 0.3; // cooldown 9s
        s[15 + 2] = 1.0; // hint: defense
        s[20] = 1.0;
        s[EFFECT_OFFSET + 2] = 1.0;  // shield type
        s[EFFECT_OFFSET + 17] = 0.6; // 93 shield
        s[EFFECT_OFFSET + 18] = 0.4; // 4000ms duration
        presets.push(Preset {
            name: "Tank Shield".into(),
            z: vae.encode(&s),
        });
    }

    // Stun: enemy, melee, crowd control
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[3] = 1.0;
        s[11] = 0.15; // range 1.5
        s[12] = 0.3;
        s[15 + 1] = 1.0; // hint: cc
        s[20] = 1.0;
        s[EFFECT_OFFSET + 7] = 1.0;  // stun type
        s[EFFECT_OFFSET + 18] = 0.15; // 1500ms
        presets.push(Preset {
            name: "Concussion".into(),
            z: vae.encode(&s),
        });
    }

    // AoE Fire: ground target, area damage
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[7] = 1.0;  // target: ground
        s[11] = 0.6; // range 6
        s[12] = 0.4;
        s[13] = 0.13; // cast 400ms
        s[15] = 1.0; // hint: damage
        s[20] = 1.0;
        s[EFFECT_OFFSET + 0] = 1.0;  // damage
        s[EFFECT_OFFSET + 17] = 0.3; // moderate damage
        s[EFFECT_OFFSET + 19] = 1.0; // area: circle
        s[EFFECT_OFFSET + 20] = 0.3; // radius 3.0
        presets.push(Preset {
            name: "Fire Storm".into(),
            z: vae.encode(&s),
        });
    }

    // Dash + damage: rogue mobility
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[3] = 1.0;  // enemy
        s[11] = 0.5;
        s[12] = 0.2;
        s[15 + 3] = 1.0; // hint: utility
        s[20] = 1.0;
        // Effect 0: dash
        s[EFFECT_OFFSET + 11] = 1.0; // dash type
        // Effect 1: damage
        s[EFFECT_OFFSET + EFFECT_SLOT_DIM + 0] = 1.0; // damage
        s[EFFECT_OFFSET + EFFECT_SLOT_DIM + 17] = 0.3;
        presets.push(Preset {
            name: "Shadow Strike".into(),
            z: vae.encode(&s),
        });
    }

    // DoT: poison
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[3] = 1.0;
        s[11] = 0.4;
        s[12] = 0.25;
        s[15] = 1.0; // damage
        s[20] = 1.0;
        s[EFFECT_OFFSET + 3] = 1.0;  // dot type
        s[EFFECT_OFFSET + 17] = 0.2; // damage per tick
        s[EFFECT_OFFSET + 18] = 0.3; // 3000ms duration
        presets.push(Preset {
            name: "Poison".into(),
            z: vae.encode(&s),
        });
    }

    // Buff: self speed boost
    {
        let mut s = [0.0f32; SLOT_DIM];
        s[0] = 1.0;
        s[5] = 1.0;  // self
        s[12] = 0.3;
        s[15 + 3] = 1.0; // utility
        s[20] = 1.0;
        s[EFFECT_OFFSET + 14] = 1.0; // buff type
        s[EFFECT_OFFSET + 17] = 0.3;
        s[EFFECT_OFFSET + 18] = 0.4; // 4s
        presets.push(Preset {
            name: "Speed Boost".into(),
            z: vae.encode(&s),
        });
    }

    presets
}

fn linear(input: &[f32], weight: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let out_dim = weight.len();
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = bias.get(i).copied().unwrap_or(0.0);
        if let Some(row) = weight.get(i) {
            for (j, &x) in input.iter().enumerate() {
                if let Some(&w) = row.get(j) {
                    sum += x * w;
                }
            }
        }
        output[i] = sum;
    }
    output
}

fn linear_relu(input: &[f32], weight: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut out = linear(input, weight, bias);
    for v in &mut out {
        *v = v.max(0.0);
    }
    out
}

// ---------------------------------------------------------------------------
// Slot vector → DSL text (using correct slot layout)
// ---------------------------------------------------------------------------

fn slots_to_dsl(slots: &[f32]) -> String {
    if slots.len() < SLOT_DIM { return "// invalid slot vector".to_string(); }

    // Targeting [3:11]
    let targeting = argmax(&slots[3..11])
        .and_then(|i| TARGETING_NAMES.get(i).copied())
        .unwrap_or("enemy");

    // Range, cooldown, cast [11:15]
    let range = (slots[11] * 10.0).max(0.5);
    let cooldown_ms = (slots[12] * 30000.0).max(1000.0);
    let cast_ms = slots[13] * 3000.0;

    // Hint [15:20]
    let hint = argmax(&slots[15..20])
        .and_then(|i| HINT_NAMES.get(i).copied())
        .unwrap_or("damage");

    // Delivery [20:27]
    let delivery_idx = argmax(&slots[20..27]);

    // Effects
    let mut effects = Vec::new();
    for slot_i in 0..NUM_EFFECT_SLOTS {
        let off = EFFECT_OFFSET + slot_i * EFFECT_SLOT_DIM;
        let type_vec = &slots[off..off + NUM_EFFECT_TYPES];
        let max_type = type_vec.iter().cloned().fold(0.0f32, f32::max);
        if max_type < 0.3 { continue; }

        let type_idx = type_vec.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let effect_name = EFFECT_NAMES[type_idx];
        let param = slots[off + 17] * 155.0;
        let duration_ms = slots[off + 18] * 10000.0;

        // Area
        let has_area = slots[off + 19..off + 24].iter().any(|&v| v > 0.3);
        let area_str = if has_area {
            let area_val = slots[off + 20]; // circle radius
            if area_val > 0.1 {
                format!(" in circle({:.1})", area_val * 10.0)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let effect_str = match effect_name {
            "damage" => format!("    damage {:.0}{}", param.max(5.0), area_str),
            "heal" => format!("    heal {:.0}{}", param.max(5.0), area_str),
            "shield" => format!("    shield {:.0} for {:.0}ms{}", param.max(5.0), duration_ms.max(1000.0), area_str),
            "dot" => format!("    damage {:.0}/1s for {:.0}ms{}", (param / 3.0).max(3.0), duration_ms.max(1000.0), area_str),
            "hot" => format!("    heal {:.0}/1s for {:.0}ms{}", (param / 3.0).max(3.0), duration_ms.max(1000.0), area_str),
            "slow" => format!("    slow {:.2} for {:.0}ms{}", (param / 155.0).clamp(0.1, 0.8), duration_ms.max(500.0), area_str),
            "root" => format!("    root {:.0}ms{}", duration_ms.max(500.0), area_str),
            "stun" => format!("    stun {:.0}ms{}", duration_ms.max(500.0), area_str),
            "silence" => format!("    silence {:.0}ms{}", duration_ms.max(500.0), area_str),
            "knockback" => format!("    knockback {:.1}{}", (param / 30.0).max(1.0), area_str),
            "pull" => format!("    pull {:.1}{}", (param / 30.0).max(1.0), area_str),
            "dash" => {
                if param > 80.0 { "    dash to_target".into() }
                else { format!("    dash {:.1}", (param / 20.0).max(1.0)) }
            }
            "blink" => format!("    blink {:.1}", (param / 20.0).max(2.0)),
            "stealth" => format!("    stealth {:.0}ms", duration_ms.max(1000.0)),
            "buff" => format!("    buff damage_output {:.2} for {:.0}ms", (param / 155.0).clamp(0.05, 0.5), duration_ms.max(1000.0)),
            "debuff" => format!("    debuff move_speed {:.2} for {:.0}ms", (param / 155.0).clamp(0.05, 0.5), duration_ms.max(1000.0)),
            "summon" => "    summon \"minion\"".into(),
            _ => format!("    // unknown effect: {}", effect_name),
        };
        effects.push(effect_str);
    }

    if effects.is_empty() {
        effects.push("    damage 10".into());
    }

    // Format cooldown
    let cd_str = if cooldown_ms >= 1000.0 && cooldown_ms % 1000.0 < 100.0 {
        format!("{:.0}s", cooldown_ms / 1000.0)
    } else {
        format!("{:.0}ms", cooldown_ms)
    };

    // Format cast
    let cast_str = if cast_ms < 50.0 {
        "0ms".to_string()
    } else if cast_ms >= 1000.0 && cast_ms % 1000.0 < 100.0 {
        format!("{:.0}s", cast_ms / 1000.0)
    } else {
        format!("{:.0}ms", cast_ms)
    };

    // Build header
    let mut header = format!("    target: {}", targeting);
    if targeting != "self" && targeting != "self_aoe" {
        header.push_str(&format!(", range: {:.1}", range));
    }
    header.push_str(&format!("\n    cooldown: {}, cast: {}", cd_str, cast_str));
    header.push_str(&format!("\n    hint: {}", hint));

    // Delivery block
    let delivery = delivery_idx.and_then(|i| DELIVERY_NAMES.get(i).copied());
    let use_delivery_block = matches!(delivery, Some("projectile" | "channel"));

    if use_delivery_block {
        let deliver_keyword = delivery.unwrap();
        let speed = if deliver_keyword == "projectile" { " speed: 12.0 " } else { "" };
        format!(
            "ability Generated {{\n{}\n\n    deliver {} {{{} }} {{\n        on_hit {{\n{}\n        }}\n    }}\n}}",
            header, deliver_keyword,
            if speed.is_empty() { "" } else { speed },
            effects.iter().map(|e| format!("    {}", e)).collect::<Vec<_>>().join("\n")
        )
    } else {
        format!(
            "ability Generated {{\n{}\n\n{}\n}}",
            header,
            effects.join("\n")
        )
    }
}

fn argmax(slice: &[f32]) -> Option<usize> {
    if slice.is_empty() { return None; }
    let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val < 0.2 { return None; }
    slice.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// Cluster map: encode dataset abilities, project to 2D, color by effect type
// ---------------------------------------------------------------------------

struct ClusterPoint {
    x: f64,
    y: f64,
    label: String,
    effect_type: usize, // primary effect type index
}

/// Load .ability files, parse slot vectors, encode to latent, project to 2D.
fn build_cluster_map(vae: &Vae) -> (Vec<ClusterPoint>, [f64; 4]) {
    let dataset_path = std::path::Path::new("generated/ability_dataset.npz");

    // Try loading from npz via a pre-computed JSON cache, or just use presets
    // For now, generate a grid of latent points to show the space structure
    let mut points = Vec::new();

    // Sample the latent space on a grid using first 2 dims, keeping others at 0
    // This shows what the decoder produces across the space
    let steps = 20;
    for xi in 0..steps {
        for yi in 0..steps {
            let x = (xi as f64 / steps as f64) * 6.0 - 3.0;
            let y = (yi as f64 / steps as f64) * 6.0 - 3.0;
            let mut z = vec![0.0f64; LATENT_DIM];
            z[0] = x;
            z[1] = y;
            let slots = vae.decode(&z);

            // Find primary effect type
            let mut best_type = 0;
            let mut best_conf = 0.0f32;
            for slot_i in 0..1 { // just first effect slot
                let off = EFFECT_OFFSET + slot_i * EFFECT_SLOT_DIM;
                for t in 0..NUM_EFFECT_TYPES {
                    if slots[off + t] > best_conf {
                        best_conf = slots[off + t];
                        best_type = t;
                    }
                }
            }

            let label = EFFECT_NAMES.get(best_type).copied().unwrap_or("unknown").to_string();
            points.push(ClusterPoint { x, y, label, effect_type: best_type });
        }
    }

    let bounds = [-3.5, 3.5, -3.5, 3.5]; // x_min, x_max, y_min, y_max
    (points, bounds)
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

fn main() {
    let vae: Option<Vae> = Vae::load("generated/ability_vae_weights.json");
    let has_vae = vae.is_some();

    let presets: Vec<Preset> = if let Some(ref v) = vae {
        curated_presets(v)
    } else {
        Vec::new()
    };

    // Build cluster map at startup
    let cluster_data: Option<(Vec<ClusterPoint>, [f64; 4])> = if let Some(ref v) = vae {
        Some(build_cluster_map(v))
    } else {
        None
    };

    let preset_names: Vec<String> = presets.iter().map(|p| p.name.clone()).collect();
    let preset_name_refs: Vec<&'static str> = preset_names.iter()
        .map(|s| &*Box::leak(s.clone().into_boxed_str()))
        .collect();
    let vae = Mutex::new(vae);

    TackConfig::new("Ability Editor", move |ui: &mut TackUi| {
        ui.title("Ability Editor");

        if !has_vae {
            ui.warning("VAE weights not found at generated/ability_vae_weights.json");
            ui.text("Run: uv run --with torch --with numpy python training/train_ability_vae.py");
            return;
        }

        // Buttons: randomize, reset
        ui.horizontal(|hui| {
            if hui.button("Randomize") {
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let mut rng = seed;
                for i in 0..LATENT_DIM {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let val = ((rng >> 33) as f64 / (1u64 << 31) as f64) * 4.0 - 2.0;
                    hui.set_f64(&format!("z__dim_{}", i), val);
                }
            }
            if hui.button("Reset") {
                for i in 0..LATENT_DIM {
                    hui.set_f64(&format!("z__dim_{}", i), 0.0);
                }
            }
        });

        // Preset selector + interpolation
        if !presets.is_empty() {
            ui.divider();
            ui.subheader("Presets");
            ui.horizontal(|hui| {
                let preset_a = hui.selectbox("preset_a", &preset_name_refs);
                if hui.button("Load A") {
                    if let Some(p) = presets.get(preset_a) {
                        for (i, &v) in p.z.iter().enumerate() {
                            hui.set_f64(&format!("z__dim_{}", i), v);
                        }
                    }
                }
            });

            ui.subheader("Interpolate A <-> B");
            ui.horizontal(|hui| {
                let preset_b = hui.selectbox("preset_b", &preset_name_refs);
                if hui.button("Load B") {
                    if let Some(p) = presets.get(preset_b) {
                        for (i, &v) in p.z.iter().enumerate() {
                            hui.set_f64(&format!("z__dim_{}", i), v);
                        }
                    }
                }
            });
            let t = ui.slider_f64("Interpolation", 0.0, 1.0, 0.0);

            let a_sel = presets.get(
                ui.selectbox_with_default("preset_a", &preset_name_refs, 0)
            );
            let b_sel = presets.get(
                ui.selectbox_with_default("preset_b", &preset_name_refs, 0)
            );
            if let (Some(a), Some(b)) = (a_sel, b_sel) {
                if t > 0.01 && t < 0.99 {
                    for i in 0..LATENT_DIM {
                        let interp = a.z[i] * (1.0 - t) + b.z[i] * t;
                        ui.set_f64(&format!("z__dim_{}", i), interp);
                    }
                }
            }
        }

        ui.divider();

        // Two columns: left = sliders, right = DSL output
        ui.two_columns(|left, right| {
            left.subheader("Latent Dimensions");
            let z = left.latent_sliders("z", LATENT_DIM, (-3.0, 3.0), None);

            let v = vae.lock().unwrap();
            let slots = if let Some(ref d) = *v {
                d.decode(&z)
            } else {
                vec![0.0f32; SLOT_DIM]
            };
            drop(v);

            right.header("Generated Ability");
            let dsl = slots_to_dsl(&slots);
            right.code(&dsl);

            right.divider();

            // Show decoded effects summary
            right.subheader("Decoded Effects");
            for slot_i in 0..NUM_EFFECT_SLOTS {
                let off = EFFECT_OFFSET + slot_i * EFFECT_SLOT_DIM;
                let type_vec = &slots[off..off + NUM_EFFECT_TYPES];
                let max_val = type_vec.iter().cloned().fold(0.0f32, f32::max);
                if max_val < 0.3 { continue; }
                let type_idx = type_vec.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                let param = slots[off + 17] * 155.0;
                let dur = slots[off + 18] * 10000.0;
                right.caption(&format!(
                    "Effect {}: {} (param={:.0}, dur={:.0}ms, conf={:.0}%)",
                    slot_i, EFFECT_NAMES[type_idx], param, dur, max_val * 100.0
                ));
            }

            // Cluster map: show latent space colored by effect type
            if let Some((ref cluster, _bounds)) = cluster_data {
                right.divider();
                right.subheader("Latent Space Map (dims 0-1)");

                // Group points by effect type for scatter_colored
                use std::collections::HashMap;
                let mut groups_map: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
                for pt in cluster.iter() {
                    groups_map.entry(pt.effect_type).or_default().push((pt.x, pt.y));
                }

                let colors = [
                    tack::Color32::from_rgb(239, 68, 68),   // damage - red
                    tack::Color32::from_rgb(34, 197, 94),    // heal - green
                    tack::Color32::from_rgb(59, 130, 246),   // shield - blue
                    tack::Color32::from_rgb(249, 115, 22),   // dot - orange
                    tack::Color32::from_rgb(16, 185, 129),   // hot - teal
                    tack::Color32::from_rgb(168, 85, 247),   // slow - purple
                    tack::Color32::from_rgb(236, 72, 153),   // root - pink
                    tack::Color32::from_rgb(245, 158, 11),   // stun - amber
                    tack::Color32::from_rgb(107, 114, 128),  // silence - gray
                    tack::Color32::from_rgb(139, 92, 246),   // knockback - violet
                    tack::Color32::from_rgb(6, 182, 212),    // pull - cyan
                    tack::Color32::from_rgb(132, 204, 22),   // dash - lime
                    tack::Color32::from_rgb(14, 165, 233),   // blink - sky
                    tack::Color32::from_rgb(100, 116, 139),  // stealth - slate
                    tack::Color32::from_rgb(251, 191, 36),   // buff - yellow
                    tack::Color32::from_rgb(244, 63, 94),    // debuff - rose
                    tack::Color32::from_rgb(162, 28, 175),   // summon - fuchsia
                ];

                let groups_owned: Vec<(String, Vec<(f64, f64)>, tack::Color32)> = groups_map.into_iter()
                    .map(|(type_idx, pts)| {
                        let label = EFFECT_NAMES.get(type_idx).copied().unwrap_or("?").to_string();
                        let color = colors.get(type_idx).copied().unwrap_or(tack::Color32::GRAY);
                        (label, pts, color)
                    })
                    .collect();

                let groups: Vec<tack::ScatterGroup> = groups_owned.iter()
                    .map(|(label, pts, color)| {
                        tack::ScatterGroup {
                            label: label.as_str(),
                            points: pts.as_slice(),
                            color: *color,
                        }
                    })
                    .collect();

                right.scatter_colored("cluster_map", &groups, 300.0);

                // Show current position as crosshair
                right.caption(&format!(
                    "Current position: ({:.2}, {:.2})",
                    z[0], z[1]
                ));
            }
        });
    })
    .size(1400.0, 900.0)
    .run()
    .unwrap();
}
