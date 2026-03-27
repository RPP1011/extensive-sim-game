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

struct VaeDecoder {
    dec1_weight: Vec<Vec<f32>>,
    dec1_bias: Vec<f32>,
    dec2_weight: Vec<Vec<f32>>,
    dec2_bias: Vec<f32>,
    dec_out_weight: Vec<Vec<f32>>,
    dec_out_bias: Vec<f32>,
}

impl VaeDecoder {
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
            dec1_weight: get_matrix("dec1.weight"),
            dec1_bias: get_bias("dec1.bias"),
            dec2_weight: get_matrix("dec2.weight"),
            dec2_bias: get_bias("dec2.bias"),
            dec_out_weight: get_matrix("dec_out.weight"),
            dec_out_bias: get_bias("dec_out.bias"),
        })
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
    let decoder: Option<VaeDecoder> = paths.iter()
        .find_map(|p| VaeDecoder::load(p));

    let has_vae = decoder.is_some();
    let decoder = Mutex::new(decoder);

    TackConfig::new("Ability Editor", move |ui: &mut TackUi| {
        // Archetype selector at top
        ui.subheader("Archetype");
        let arch_idx = ui.selectbox("archetype", ARCHETYPE_NAMES);
        let archetype = ARCHETYPE_NAMES[arch_idx];

        if !has_vae {
            ui.warning("VAE weights not found. Run train_ability_vae.py first.");
        }

        ui.divider();

        // Randomize and reset buttons
        ui.horizontal(|hui| {
            if hui.button("🎲 Randomize") {
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let mut rng = seed;
                for i in 0..LATENT_DIM {
                    // Simple LCG for random values in [-2, 2]
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

        ui.divider();

        // Two columns side by side: left = latent sliders, right = DSL output
        ui.two_columns(|left, right| {
            // Left column: latent sliders
            left.subheader("Latent Dimensions");
            let z = left.latent_sliders("z", LATENT_DIM, (-3.0, 3.0), None);

            // Decode
            let dec = decoder.lock().unwrap();
            let slots = if let Some(ref d) = *dec {
                d.decode(&z, arch_idx)
            } else {
                vec![0.0f32; SLOT_DIM]
            };
            drop(dec);

            // Right column: DSL output + slot stats
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
