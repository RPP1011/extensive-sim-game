//! Ability Editor — design combat abilities by navigating a learned latent space.
//!
//! Uses a trained VAE to map 32 latent dimensions to 142-dim ability slot vectors,
//! which are then decoded to DSL text. Adjust sliders to explore the space of
//! possible abilities.
//!
//! Usage:
//!   cargo run --example ability_editor --release
//!
//! Requires: generated/ability_vae_weights.json from training/train_ability_vae.py

use std::sync::Mutex;
use tack::{TackConfig, TackUi};

// ---------------------------------------------------------------------------
// VAE inference (pure Rust, no ML framework needed)
// ---------------------------------------------------------------------------

const SLOT_DIM: usize = 142;
const LATENT_DIM: usize = 32;
const NUM_ARCHETYPES: usize = 19;

const ARCHETYPE_NAMES: &[&str] = &[
    "artificer", "assassin", "bard", "berserker", "cleric",
    "druid", "fighter", "guardian", "knight", "mage",
    "monk", "necromancer", "paladin", "ranger", "rogue",
    "shaman", "tank", "warlock", "warrior",
];

struct Vae {
    // Encoder
    enc1_weight: Vec<Vec<f32>>,
    enc1_bias: Vec<f32>,
    enc2_weight: Vec<Vec<f32>>,
    enc2_bias: Vec<f32>,
    enc_mu_weight: Vec<Vec<f32>>,
    enc_mu_bias: Vec<f32>,
    // Decoder
    dec1_weight: Vec<Vec<f32>>,
    dec1_bias: Vec<f32>,
    dec2_weight: Vec<Vec<f32>>,
    dec2_bias: Vec<f32>,
    dec_out_weight: Vec<Vec<f32>>,
    dec_out_bias: Vec<f32>,
}

/// A named preset ability with its latent vector.
#[derive(Clone)]
struct Preset {
    name: String,
    archetype_idx: usize,
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

        Some(Self {
            enc1_weight: get_matrix("enc1.weight"),
            enc1_bias: get_bias("enc1.bias"),
            enc2_weight: get_matrix("enc2.weight"),
            enc2_bias: get_bias("enc2.bias"),
            enc_mu_weight: get_matrix("enc_mu.weight"),
            enc_mu_bias: get_bias("enc_mu.bias"),
            dec1_weight: get_matrix("dec1.weight"),
            dec1_bias: get_bias("dec1.bias"),
            dec2_weight: get_matrix("dec2.weight"),
            dec2_bias: get_bias("dec2.bias"),
            dec_out_weight: get_matrix("dec_out.weight"),
            dec_out_bias: get_bias("dec_out.bias"),
        })
    }

    /// Encode a slot vector to its latent mean (deterministic, no sampling).
    fn encode(&self, slots: &[f32], archetype_idx: usize) -> Vec<f64> {
        let mut input = Vec::with_capacity(SLOT_DIM + NUM_ARCHETYPES);
        input.extend_from_slice(slots);
        for i in 0..NUM_ARCHETYPES {
            input.push(if i == archetype_idx { 1.0 } else { 0.0 });
        }
        let h1 = linear_relu(&input, &self.enc1_weight, &self.enc1_bias);
        let h2 = linear_relu(&h1, &self.enc2_weight, &self.enc2_bias);
        let mu = linear(&h2, &self.enc_mu_weight, &self.enc_mu_bias);
        mu.into_iter().map(|v| v as f64).collect()
    }

    fn decode(&self, z: &[f64], archetype_idx: usize) -> Vec<f32> {
        let mut input = Vec::with_capacity(LATENT_DIM + NUM_ARCHETYPES);
        for &v in z {
            input.push(v as f32);
        }
        for i in 0..NUM_ARCHETYPES {
            input.push(if i == archetype_idx { 1.0 } else { 0.0 });
        }
        let h1 = linear_relu(&input, &self.dec1_weight, &self.dec1_bias);
        let h2 = linear_relu(&h1, &self.dec2_weight, &self.dec2_bias);
        linear(&h2, &self.dec_out_weight, &self.dec_out_bias)
    }
}

/// Build presets from the dataset JSONL (grammar-walked abilities).
fn load_dataset_presets(vae: &Vae, path: &str, max: usize) -> Vec<Preset> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut presets = Vec::new();
    for line in content.lines().take(max * 10) {
        let parsed: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let archetype = parsed["archetype"].as_str().unwrap_or("");
        let arch_idx = ARCHETYPE_NAMES.iter().position(|&a| a == archetype).unwrap_or(0);
        let slots: Vec<f32> = parsed["slots"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .unwrap_or_default();
        if slots.len() < SLOT_DIM { continue; }

        let level = parsed["level"].as_u64().unwrap_or(1);
        let is_passive = parsed["is_passive"].as_bool().unwrap_or(false);
        if is_passive { continue; } // skip passives for presets

        let z = vae.encode(&slots, arch_idx);
        presets.push(Preset {
            name: format!("{} L{}", archetype, level),
            archetype_idx: arch_idx,
            z,
        });
        if presets.len() >= max { break; }
    }
    presets
}

/// Curated presets from known interesting abilities.
fn curated_presets(vae: &Vae) -> Vec<Preset> {
    // Manually defined slot vectors for iconic ability archetypes.
    // These are approximate — the VAE will smooth them into valid abilities.
    let examples: &[(&str, usize, &[f32])] = &[
        // High damage nuke: enemy target, long range, high damage
        ("Damage Nuke", 3, &{  // berserker
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;  // active
            s[1] = 1.0;  // target enemy
            s[11] = 0.5; // range 3.0
            s[12] = 0.4; // cooldown 12s
            s[13] = 0.2; // cast 300ms
            s[15] = 1.0; // hint: damage
            s[29] = 0.8; // high damage
            s
        }),
        // Shield ability: self target, big shield
        ("Tank Shield", 7, &{  // guardian
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;
            s[3] = 1.0;  // target self
            s[12] = 0.3;
            s[17] = 1.0; // hint: defense
            s[60] = 0.9; // big shield
            s
        }),
        // Heal: ally target, heal effect
        ("Burst Heal", 4, &{  // cleric
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;
            s[2] = 1.0;  // target ally
            s[11] = 0.6; // range 3.5
            s[12] = 0.25;
            s[19] = 1.0; // hint: heal
            s[59] = 0.8; // big heal
            s
        }),
        // Stealth + dash: rogue mobility
        ("Shadow Step", 14, &{  // rogue
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;
            s[6] = 1.0;  // direction
            s[12] = 0.2;
            s[18] = 1.0; // hint: utility
            s[54] = 0.6; // stealth
            s[84] = 0.7; // dash
            s
        }),
        // AoE damage: ground target, circle area
        ("Fire Storm", 9, &{  // mage
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;
            s[5] = 1.0;  // ground target
            s[11] = 0.8; // range 5.0
            s[12] = 0.5; // cooldown 15s
            s[15] = 1.0; // hint: damage
            s[29] = 0.5; // moderate damage
            s
        }),
        // Stun: crowd control
        ("Concussion", 8, &{  // knight
            let mut s = [0.0f32; SLOT_DIM];
            s[0] = 1.0;
            s[1] = 1.0;  // enemy
            s[11] = 0.3;
            s[12] = 0.3;
            s[16] = 1.0; // hint: cc
            s[63] = 0.7; // stun
            s
        }),
    ];

    examples.iter().map(|(name, arch_idx, slots)| {
        let z = vae.encode(slots.as_ref(), *arch_idx);
        Preset {
            name: name.to_string(),
            archetype_idx: *arch_idx,
            z,
        }
    }).collect()
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
// Slot vector → DSL text
// ---------------------------------------------------------------------------

fn slots_to_dsl(slots: &[f32], archetype: &str) -> String {
    if slots.len() < SLOT_DIM { return "// invalid".to_string(); }

    let targeting = if slots[1] > 0.3 { "enemy" }
        else if slots[2] > 0.3 { "ally" }
        else if slots[3] > 0.3 { "self" }
        else if slots[4] > 0.3 { "self_aoe" }
        else if slots[5] > 0.3 { "ground" }
        else if slots[6] > 0.3 { "direction" }
        else { "enemy" };

    let range = (slots[11] * 6.0).max(0.5);
    let cooldown = (slots[12] * 30.0).max(1.0);
    let cast_time = slots[13] * 1500.0;

    let hint = if slots[15] > 0.3 { "damage" }
        else if slots[16] > 0.3 { "crowd_control" }
        else if slots[17] > 0.3 { "defense" }
        else if slots[18] > 0.3 { "utility" }
        else if slots[19] > 0.3 { "heal" }
        else { "damage" };

    let mut effects = Vec::new();
    let dmg = slots.get(29).copied().unwrap_or(0.0) * 200.0;
    if dmg > 5.0 { effects.push(format!("    damage {:.0}", dmg)); }
    let heal = slots.get(59).copied().unwrap_or(0.0) * 200.0;
    if heal > 5.0 { effects.push(format!("    heal {:.0}", heal)); }
    let shield = slots.get(60).copied().unwrap_or(0.0) * 200.0;
    if shield > 5.0 { effects.push(format!("    shield {:.0} for 3s", shield)); }
    let stun = slots.get(63).copied().unwrap_or(0.0);
    if stun > 0.2 { effects.push(format!("    stun {:.0}ms", stun * 3000.0)); }
    let slow = slots.get(61).copied().unwrap_or(0.0);
    if slow > 0.1 { effects.push(format!("    slow {:.0}% for 2s", slow * 100.0)); }
    let dash = slots.get(84).copied().unwrap_or(0.0);
    if dash > 0.1 { effects.push(format!("    dash {:.1}", dash * 8.0)); }
    let stealth = slots.get(54).copied().unwrap_or(0.0);
    if stealth > 0.2 { effects.push(format!("    stealth for {:.0}ms", stealth * 5000.0)); }
    if effects.is_empty() { effects.push(format!("    damage {:.0}", dmg.max(10.0))); }

    format!(
        "ability generated_{} {{\n    target: {}, range: {:.1}\n    cooldown: {:.0}s, cast: {:.0}ms\n    hint: {}\n\n{}\n}}",
        archetype, targeting, range, cooldown, cast_time, hint, effects.join("\n")
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

fn main() {
    let paths = [
        "generated/ability_vae_weights.json",
        "../game/generated/ability_vae_weights.json",
    ];
    let vae: Option<Vae> = paths.iter().find_map(|p| Vae::load(p));
    let has_vae = vae.is_some();

    // Build presets
    let presets: Vec<Preset> = if let Some(ref v) = vae {
        let mut p = curated_presets(v);
        let dataset_paths = [
            "generated/ability_dataset_slots.jsonl",
            "../game/generated/ability_dataset_slots.jsonl",
        ];
        for dp in &dataset_paths {
            let dataset_presets = load_dataset_presets(v, dp, 20);
            if !dataset_presets.is_empty() {
                p.extend(dataset_presets);
                break;
            }
        }
        p
    } else {
        Vec::new()
    };

    let preset_names: Vec<String> = presets.iter().map(|p| p.name.clone()).collect();
    // Leak the names so the &str refs live for 'static (they're app-lifetime anyway)
    let preset_name_refs: Vec<&'static str> = preset_names.iter()
        .map(|s| &*Box::leak(s.clone().into_boxed_str()))
        .collect();
    let vae = Mutex::new(vae);

    TackConfig::new("Ability Editor", move |ui: &mut TackUi| {
        // Archetype selector
        ui.subheader("Archetype");
        let arch_idx = ui.selectbox("archetype", ARCHETYPE_NAMES);
        let archetype = ARCHETYPE_NAMES[arch_idx];

        if !has_vae {
            ui.warning("VAE weights not found. Run train_ability_vae.py first.");
        }

        ui.divider();

        // Buttons: randomize, reset, presets
        ui.horizontal(|hui| {
            if hui.button("🎲 Randomize") {
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
            if hui.button("🔄 Reset") {
                for i in 0..LATENT_DIM {
                    hui.set_f64(&format!("z__dim_{}", i), 0.0);
                }
            }
        });

        // Preset selector
        if !presets.is_empty() {
            ui.divider();
            ui.subheader("Presets");
            ui.horizontal(|hui| {
                let preset_a = hui.selectbox("preset_a", &preset_name_refs);
                if hui.button("⬅ Load A") {
                    if let Some(p) = presets.get(preset_a) {
                        for (i, &v) in p.z.iter().enumerate() {
                            hui.set_f64(&format!("z__dim_{}", i), v);
                        }
                    }
                }
            });

            // Interpolation
            ui.subheader("Interpolate A ↔ B");
            ui.horizontal(|hui| {
                let preset_b = hui.selectbox("preset_b", &preset_name_refs);
                if hui.button("➡ Load B") {
                    if let Some(p) = presets.get(preset_b) {
                        for (i, &v) in p.z.iter().enumerate() {
                            hui.set_f64(&format!("z__dim_{}", i), v);
                        }
                    }
                }
            });
            let t = ui.slider("interp_t", 0.0, 1.0);
            let preset_a_idx = ui.selectbox_with_default("preset_a_hidden", &[""], 0);
            let _ = preset_a_idx; // just to keep state alive

            // Apply interpolation when t changes
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

        // Two columns: left = sliders, right = DSL
        ui.two_columns(|left, right| {
            left.subheader("Latent Dimensions");
            let z = left.latent_sliders("z", LATENT_DIM, (-3.0, 3.0), None);

            let v = vae.lock().unwrap();
            let slots = if let Some(ref d) = *v {
                d.decode(&z, arch_idx)
            } else {
                vec![0.0f32; SLOT_DIM]
            };
            drop(v);

            right.header("Generated Ability DSL");
            let dsl = slots_to_dsl(&slots, archetype);
            right.code(&dsl);

            right.divider();

            right.subheader("Slot Vector");
            let nonzero = slots.iter().filter(|&&v| v.abs() > 0.05).count();
            right.caption(&format!("{} active dims out of {}", nonzero, SLOT_DIM));

            let mut top: Vec<(usize, f32)> = slots.iter().enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .filter(|(_, v)| *v > 0.05)
                .collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top.truncate(15);

            for (idx, val) in &top {
                let bar = "█".repeat((val * 15.0).min(30.0) as usize);
                right.caption(&format!("[{:>3}] {:>5.2} {}", idx, val, bar));
            }
        });
    })
    .size(1400.0, 900.0)
    .run()
    .unwrap();
}
