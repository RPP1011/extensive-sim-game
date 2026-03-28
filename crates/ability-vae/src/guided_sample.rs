//! Constraint-based ability generation.
//!
//! Parse natural-language-ish constraints into partial grammar space vectors,
//! then use the flow model to fill in unconstrained dimensions.
//!
//! Usage:
//!   let constraints = parse_constraints("fire damage AoE with stun, 8s cooldown");
//!   let abilities = sample_with_constraints(&flow_model, &constraints, 10, device);

use burn::prelude::*;
use super::grammar_space::*;

/// A partial specification of an ability in grammar space.
/// `None` = unconstrained (let the model decide).
/// `Some(v)` = fixed to this value.
pub struct Constraints {
    pub dims: [Option<f32>; GRAMMAR_DIM],
}

impl Constraints {
    pub fn empty() -> Self {
        Self { dims: [None; GRAMMAR_DIM] }
    }

    pub fn set(&mut self, dim: usize, val: f32) {
        if dim < GRAMMAR_DIM {
            self.dims[dim] = Some(val.clamp(0.0, 1.0));
        }
    }

    /// How many dimensions are constrained.
    pub fn n_fixed(&self) -> usize {
        self.dims.iter().filter(|d| d.is_some()).count()
    }
}

// Dimension indices (re-export from grammar_space for external use)
const D_TYPE: usize = 0;
const D_DOMAIN: usize = 1;
const D_TARGET: usize = 2;
const D_RANGE: usize = 3;
const D_COOLDOWN: usize = 4;
const D_CAST: usize = 5;
const D_HINT: usize = 6;
const D_COST: usize = 7;
const D_DELIVERY: usize = 8;
const D_N_EFFECTS: usize = 11;
const D_E0_TYPE: usize = 12;
const D_E0_AREA: usize = 15;
const D_E0_TAG: usize = 17;

// ---------------------------------------------------------------------------
// Keyword → constraint parser
// ---------------------------------------------------------------------------

/// Parse a natural language description into grammar space constraints.
///
/// Handles synonyms, phrases, intensity modifiers, and compositional descriptions.
///
/// Examples:
///   "a powerful fire spell that hits all nearby enemies"
///   "quick melee strike that stuns on hit"
///   "healing aura that protects allies from fear"
///   "trade embargo against hostile faction"
///   "passive: whenever you kill an enemy, gain a shield"
///   "ultimate ability that rains meteors, massive AoE damage with stun"
///   "army-wide buff that makes soldiers fight harder"
pub fn parse_constraints(text: &str) -> Constraints {
    let mut c = Constraints::empty();
    let t = text.to_lowercase();

    // --- Ability type ---
    if has_any(&t, &["passive", "aura", "whenever", "always active", "on hit", "on kill",
                      "each time", "every time", "triggers when", "procs when"]) {
        c.set(D_TYPE, 0.75);
    } else {
        c.set(D_TYPE, 0.25);
    }

    // --- Domain ---
    let is_campaign = has_any(&t, &[
        "campaign", "economy", "trade", "diplomacy", "faction", "region", "guild",
        "market", "territory", "army", "kingdom", "political", "espionage",
        "fortif", "embargo", "alliance", "treaty", "merchant", "commerce",
        "scout", "forage", "supply", "morale", "leadership", "command",
    ]);
    let is_combat = has_any(&t, &[
        "damage", "heal", "stun", "slow", "root", "silence", "shield", "projectile",
        "melee", "ranged", "attack", "strike", "slash", "spell", "cast", "nuke",
        "burst", "dot", "hot", "tank", "dps", "crowd control", "cc",
        "knockback", "pull", "dash", "blink", "stealth", "fear",
    ]);
    if is_campaign && !is_combat {
        c.set(D_DOMAIN, 0.75);
    } else if is_combat {
        c.set(D_DOMAIN, 0.25);
    }
    let campaign = c.dims[D_DOMAIN].map_or(false, |v| v > 0.5);

    // --- Targeting (NL phrases → target mode) ---
    if campaign {
        if has_any(&t, &["all factions", "every faction", "faction"]) { c.set(D_TARGET, 0.07); }
        else if has_any(&t, &["region", "territory", "land"]) { c.set(D_TARGET, 0.19); }
        else if has_any(&t, &["market", "trade", "commerce"]) { c.set(D_TARGET, 0.31); }
        else if has_any(&t, &["party", "group", "squad", "team"]) { c.set(D_TARGET, 0.44); }
        else if has_any(&t, &["guild", "organization"]) { c.set(D_TARGET, 0.56); }
        else if has_any(&t, &["self", "yourself", "personal"]) { c.set(D_TARGET, 0.69); }
        else if has_any(&t, &["global", "world", "everywhere", "all", "army", "kingdom"]) { c.set(D_TARGET, 0.81); }
    } else {
        if has_any(&t, &["enemies", "all enemies", "every enemy", "foes"]) {
            c.set(D_TARGET, 0.07); // enemy
            if has_any(&t, &["all enemies", "every enemy", "all foes", "nearby enemies"]) {
                c.set(D_E0_AREA, 0.75); // imply AoE
            }
        } else if has_any(&t, &["enemy", "target", "foe", "opponent", "hostile"]) { c.set(D_TARGET, 0.07); }
        else if has_any(&t, &["allies", "all allies", "friendly", "teammates", "party members"]) {
            c.set(D_TARGET, 0.21);
            if has_any(&t, &["all allies", "every ally", "nearby allies"]) {
                c.set(D_E0_AREA, 0.75);
            }
        }
        else if has_any(&t, &["ally", "friend", "teammate"]) { c.set(D_TARGET, 0.21); }
        else if has_any(&t, &["self", "yourself", "personal", "own"]) { c.set(D_TARGET, 0.36); }
        else if has_any(&t, &["around you", "around the caster", "nearby", "surrounding"]) { c.set(D_TARGET, 0.50); }
        else if has_any(&t, &["ground", "location", "area", "zone", "place"]) { c.set(D_TARGET, 0.64); }
        else if has_any(&t, &["direction", "skillshot", "aimed"]) { c.set(D_TARGET, 0.78); }
        else if has_any(&t, &["global", "everyone", "map-wide", "everywhere"]) { c.set(D_TARGET, 0.93); }
    }

    // --- Range ---
    if has_any(&t, &["melee", "close range", "point blank", "adjacent", "touch"]) {
        c.set(D_RANGE, 0.1);
    } else if has_any(&t, &["mid range", "medium range", "moderate range"]) {
        c.set(D_RANGE, 0.45);
    } else if has_any(&t, &["long range", "ranged", "far", "distant", "sniper", "across the field"]) {
        c.set(D_RANGE, 0.85);
    }

    // --- Cooldown (intensity-based) ---
    if has_any(&t, &["spam", "spammable", "no cooldown", "rapid", "quick cooldown"]) {
        c.set(D_COOLDOWN, 0.05); // ~1-2s
    } else if has_any(&t, &["short cooldown", "low cooldown", "fast", "quick"]) {
        c.set(D_COOLDOWN, 0.2); // ~3-5s
    } else if has_any(&t, &["moderate cooldown", "medium cooldown"]) {
        c.set(D_COOLDOWN, 0.4); // ~8-12s
    } else if has_any(&t, &["long cooldown", "high cooldown", "powerful", "ultimate"]) {
        c.set(D_COOLDOWN, 0.75); // ~25-40s
    } else if has_any(&t, &["once per battle", "once per fight"]) {
        c.set(D_COOLDOWN, 0.95); // ~55s+
    }
    // Parse explicit durations: "8s cooldown", "cooldown of 15 seconds"
    parse_explicit_cooldown(&t, &mut c);

    // --- Hint (from NL synonyms) ---
    let hint_matches: &[(&[&str], usize)] = if campaign {
        &[
            (&["economy", "trade", "gold", "money", "merchant", "market", "commerce"], 0),
            (&["diplomacy", "political", "alliance", "treaty", "faction", "negotiate"], 1),
            (&["stealth", "sneak", "hidden", "covert", "spy", "espionage", "infiltrate"], 2),
            (&["leadership", "command", "rally", "inspire", "morale", "army", "troops"], 3),
            (&["utility", "support", "general", "misc"], 4),
            (&["defense", "protect", "fortif", "guard", "ward", "sanctuary"], 5),
            (&["heal", "mend", "restore", "cure", "purif"], 6),
        ]
    } else {
        &[
            (&["damage", "attack", "strike", "nuke", "blast", "destroy", "kill", "dps",
               "hurt", "slash", "smash", "burn", "shock", "meteor", "rain"], 0),
            (&["heal", "restore", "mend", "cure", "regenerat", "recover", "patch up",
               "bandage", "rejuvenate", "revive"], 1),
            (&["crowd control", " cc ", "stun", "root", "silence", "fear", "disable",
               "incapacitat", "lock down", "immobilize", "freeze", "petrif"], 2),
            (&["defense", "protect", "shield", "block", "absorb", "tank", "armor",
               "barrier", "ward", "fortif", "resist", "mitigat"], 3),
            (&["utility", "support", "buff", "debuff", "speed", "mobility", "teleport",
               "stealth", "invis", "dispel", "cleans"], 4),
        ]
    };
    for &(keywords, idx) in hint_matches {
        if keywords.iter().any(|kw| t.contains(kw)) {
            let n = if campaign { 7 } else { 5 };
            c.set(D_HINT, (idx as f32 + 0.5) / n as f32);
            break;
        }
    }

    // --- Delivery (NL phrases) ---
    if has_any(&t, &["projectile", "missile", "bolt", "arrow", "shoot", "fire at",
                      "launch", "throw", "hurl"]) {
        c.set(D_DELIVERY, 0.5); // projectile
    } else if has_any(&t, &["chain", "bounce", "jump between", "arc", "lightning"]) {
        c.set(D_DELIVERY, 0.64); // chain
    } else if has_any(&t, &["zone", "ground effect", "persistent area", "field",
                             "rain", "meteor", "blizzard", "storm"]) {
        c.set(D_DELIVERY, 0.78); // zone
    } else if has_any(&t, &["trap", "mine", "placed", "triggered"]) {
        c.set(D_DELIVERY, 0.92); // trap
    } else if has_any(&t, &["instant", "immediate", "direct"]) && !campaign {
        c.set(D_DELIVERY, 0.14); // none (direct)
    }

    // --- Area (NL phrases) ---
    if has_any(&t, &["aoe", "area of effect", "splash", "all nearby", "everyone around",
                      "circle", "radius", "explosion", "eruption", "nova", "surrounding"]) {
        c.set(D_E0_AREA, 0.75);
    } else if has_any(&t, &["cone", "breath", "fan", "spray", "wave"]) {
        c.set(D_E0_AREA, 0.85);
    } else if has_any(&t, &["line", "beam", "piercing", "through enemies", "straight line"]) {
        c.set(D_E0_AREA, 0.95);
    } else if has_any(&t, &["single target", "single-target", "focused", "one enemy"]) {
        c.set(D_E0_AREA, 0.1);
    }

    // --- Element/tag ---
    let elements: &[(&[&str], usize)] = &[
        (&["physical", "martial", "weapon", "blade", "sword", "punch", "brute"], 3),
        (&["magic", "arcane", "mystic", "mana", "spell"], 4),
        (&["fire", "flame", "burn", "inferno", "blaze", "meteor", "scorch", "ember"], 5),
        (&["ice", "frost", "frozen", "cold", "blizzard", "chill", "glacial"], 6),
        (&["dark", "shadow", "void", "necrotic", "death", "curse", "wither"], 7),
        (&["holy", "light", "divine", "sacred", "radiant", "sanctif", "celestial", "blessed"], 8),
        (&["poison", "toxic", "venom", "plague", "disease", "corrosi", "acid"], 9),
    ];
    for &(keywords, idx) in elements {
        if keywords.iter().any(|kw| t.contains(kw)) {
            c.set(D_E0_TAG, (idx as f32 + 0.5) / 10.0);
            break;
        }
    }

    // --- Tag power (intensity modifiers) ---
    if has_any(&t, &["weak", "minor", "slight", "small", "little", "basic"]) {
        c.set(D_E0_TAG + 1, 0.2); // low power tag
    } else if has_any(&t, &["powerful", "strong", "massive", "devastating", "overwhelming",
                             "ultimate", "legendary", "mythic", "supreme"]) {
        c.set(D_E0_TAG + 1, 0.9); // high power tag
    }

    // --- Effect type (NL → grammar space index) ---
    let combat_effects: &[(&[&str], f32)] = &[
        // Instant (no params)
        (&["dispel", "cleans", "purge", "remove buff"], 0.01),
        (&["swap", "switch position", "trade places"], 0.02),
        (&["reset cooldown", "refresh"], 0.03),
        // Amount-based
        (&["damage", "attack", "strike", "hit", "nuke", "blast", "smash"], 0.08),
        (&["heal", "restore", "mend", "cure", "patch"], 0.10),
        (&["shield", "barrier", "absorb", "block"], 0.12),
        (&["knockback", "push", "repel", "slam back"], 0.14),
        (&["pull", "drag", "yank", "hook", "grapple"], 0.16),
        (&["dash", "charge", "rush", "lunge", "leap"], 0.18),
        (&["blink", "teleport", "flash", "warp", "phase"], 0.20),
        // Duration-based CC
        (&["stun", "daze", "knock out", "incapacitat"], 0.30),
        (&["root", "immobilize", "snare", "entangle", "pin"], 0.32),
        (&["silence", "mute", "prevent casting", "spell lock"], 0.34),
        (&["fear", "terrif", "frighten", "scare", "flee"], 0.36),
        (&["taunt", "provoke", "force attack", "aggro"], 0.38),
        (&["stealth", "invisible", "vanish", "hide", "cloak"], 0.40),
        (&["blind", "obscure", "miss chance"], 0.42),
        (&["charm", "mesmerize", "entrance", "captivate"], 0.44),
        (&["suppress", "lockdown", "disable completely"], 0.46),
        (&["confuse", "disorient", "befuddle"], 0.48),
        (&["polymorph", "transform enemy", "turn into", "sheep"], 0.50),
        (&["banish", "remove from combat", "exile"], 0.52),
        (&["cooldown lock", "ability lock", "prevent abilities"], 0.54),
        // Amount + duration
        (&["slow", "sluggish", "impede", "hinder movement"], 0.58),
        (&["damage amp", "damage boost", "vulnerability"], 0.60),
        (&["lifesteal", "life drain", "vampiric", "siphon life"], 0.62),
        (&["mana burn", "resource drain", "exhaust mana"], 0.64),
        // Meta-effects
        (&["amplif", "empower", "strengthen abilities", "boost power"], 0.68),
        (&["echo", "double cast", "twin cast", "cast twice"], 0.69),
        (&["instant cast", "no cast time", "quicken"], 0.70),
        (&["free cast", "no cost", "zero mana"], 0.71),
        (&["spell shield", "ability block", "negate spell", "anti-magic"], 0.72),
        // Buff/debuff
        (&["buff", "enhance", "augment", "bolster", "empower ally"], 0.78),
        (&["debuff", "weaken", "reduce stats", "diminish", "cripple"], 0.82),
        // Over time
        (&["dot", "damage over time", "bleed", "burn damage", "poison damage", "tick damage"], 0.88),
        (&["hot", "heal over time", "regenerat", "rejuven", "renew"], 0.92),
    ];
    for &(keywords, val) in combat_effects {
        if keywords.iter().any(|kw| t.contains(kw)) {
            if !campaign {
                c.set(D_E0_TYPE, val);
            }
            break;
        }
    }

    // Campaign effects
    let campaign_effects: &[(&[&str], f32)] = &[
        (&["appraise", "evaluate", "assess value"], 0.02),
        (&["beast lore", "monster knowledge", "identify weakness"], 0.04),
        (&["rally", "boost morale", "inspire troops", "war cry"], 0.50),
        (&["fortif", "defend region", "strengthen walls"], 0.55),
        (&["sanctuary", "safe zone", "no combat zone", "peaceful"], 0.60),
        (&["ghost walk", "invisible travel", "move undetected"], 0.65),
        (&["ceasefire", "stop fighting", "halt hostilities", "peace"], 0.68),
        (&["trade embargo", "block trade", "economic sanctions"], 0.70),
        (&["corner market", "monopol", "control trade"], 0.73),
        (&["broker alliance", "form alliance", "create pact"], 0.76),
    ];
    if campaign {
        for &(keywords, val) in campaign_effects {
            if keywords.iter().any(|kw| t.contains(kw)) {
                c.set(D_E0_TYPE, val);
                break;
            }
        }
    }

    // --- Param intensity (damage/heal amount) ---
    let e0_param_dim = D_E0_TYPE + 1; // D_E0_PARAM = 13
    if has_any(&t, &["weak", "minor", "small", "little", "light", "basic", "low"]) {
        c.set(e0_param_dim, 0.15);
    } else if has_any(&t, &["moderate", "medium", "decent", "solid", "good"]) {
        c.set(e0_param_dim, 0.45);
    } else if has_any(&t, &["strong", "powerful", "heavy", "big", "large", "high"]) {
        c.set(e0_param_dim, 0.7);
    } else if has_any(&t, &["massive", "devastating", "overwhelming", "enormous", "ultimate",
                             "legendary", "mythic", "supreme", "catastroph"]) {
        c.set(e0_param_dim, 0.95);
    }

    // --- Number of effects ---
    if has_any(&t, &["simple", "basic", "single effect", "just", "only"]) {
        c.set(D_N_EFFECTS, 0.1);
    } else if has_any(&t, &["combo", "combination", "multi", "complex", "and then",
                             "followed by", "with", "that also", "plus"]) {
        c.set(D_N_EFFECTS, 0.75);
    } else if has_any(&t, &["ultimate", "legendary", "mythic", "supreme"]) {
        c.set(D_N_EFFECTS, 0.9);
    }

    // --- Passive triggers (NL) ---
    if c.dims[D_TYPE].map_or(false, |v| v > 0.5) {
        let trigger_matches: &[(&[&str], f32)] = &[
            (&["when you deal damage", "on hit", "when hitting", "on damage dealt"], 0.05),
            (&["when hit", "when damaged", "when taking damage", "on damage taken"], 0.15),
            (&["on kill", "when you kill", "after killing", "upon slaying"], 0.25),
            (&["when healed", "on heal", "after healing", "when receiving heal"], 0.35),
            (&["when casting", "on ability use", "after using", "when you cast"], 0.45),
            (&["on trade", "when trading", "after a trade", "on sale"], 0.55),
            (&["on quest", "after quest", "quest complete", "mission done"], 0.65),
            (&["on level up", "when leveling", "after leveling"], 0.75),
            (&["on crisis", "when crisis", "during crisis", "emergency"], 0.85),
        ];
        for &(keywords, val) in trigger_matches {
            if keywords.iter().any(|kw| t.contains(kw)) {
                c.set(44, val); // D_TRIGGER
                break;
            }
        }
    }

    c
}

fn parse_explicit_cooldown(t: &str, c: &mut Constraints) {
    // Match patterns like "8s cooldown", "cooldown 15s", "15 second cooldown"
    let re_patterns: &[(&str, f32)] = &[
        ("1s", 1.0), ("2s", 2.0), ("3s", 3.0), ("4s", 4.0), ("5s", 5.0),
        ("6s", 6.0), ("8s", 8.0), ("10s", 10.0), ("12s", 12.0), ("15s", 15.0),
        ("20s", 20.0), ("25s", 25.0), ("30s", 30.0), ("45s", 45.0), ("60s", 60.0),
    ];
    for &(pat, secs) in re_patterns {
        if t.contains(pat) && (t.contains("cooldown") || t.contains("cd")) {
            let ms = secs * 1000.0;
            let val = ((ms.ln() - 1000.0f32.ln()) / (60000.0f32.ln() - 1000.0f32.ln())).clamp(0.0, 1.0);
            c.set(D_COOLDOWN, val);
            return;
        }
    }
    // "N second cooldown" patterns
    for word in t.split_whitespace() {
        if let Ok(n) = word.parse::<f32>() {
            if n >= 1.0 && n <= 120.0 && t.contains("second") && t.contains("cooldown") {
                let ms = n * 1000.0;
                let val = ((ms.ln() - 1000.0f32.ln()) / (60000.0f32.ln() - 1000.0f32.ln())).clamp(0.0, 1.0);
                c.set(D_COOLDOWN, val);
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constrained sampling
// ---------------------------------------------------------------------------

/// Sample abilities with partial constraints using the flow model.
///
/// Fixed dimensions are clamped at every Euler step.
/// Free dimensions follow the flow.
pub fn sample_with_constraints<B: burn::prelude::Backend>(
    model: &super::diffusion::FlowModel<B>,
    constraints: &Constraints,
    n_samples: usize,
    n_steps: usize,
    device: &B::Device,
) -> Vec<[f32; GRAMMAR_DIM]> {
    // Start from noise
    let mut x = Tensor::<B, 2>::random(
        [n_samples, GRAMMAR_DIM],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    // Apply constraints to initial noise (replace constrained dims)
    x = apply_constraints_tensor(x, constraints, n_samples, device);

    let uncond_labels = Tensor::<B, 2>::zeros([n_samples, 10], device);
    let dt = 1.0 / n_steps as f32;

    for step in 0..n_steps {
        let t_val = 1.0 - step as f32 / n_steps as f32;
        let t = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![t_val; n_samples], [n_samples]),
            device,
        );

        let v = model.forward(x.clone(), t, uncond_labels.clone());
        x = x - v * dt;

        // Re-apply constraints after each step
        x = apply_constraints_tensor(x, constraints, n_samples, device);
    }

    x = x.clamp(0.0, 1.0);

    let data: Vec<f32> = x.to_data().to_vec().unwrap();
    let mut results = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in 0..GRAMMAR_DIM {
            v[d] = data[i * GRAMMAR_DIM + d];
        }
        results.push(v);
    }
    results
}

/// Sample variations around an existing ability.
pub fn sample_variations(
    base: &[f32; GRAMMAR_DIM],
    n_samples: usize,
    noise_scale: f32,
) -> Vec<[f32; GRAMMAR_DIM]> {
    let mut rng: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let mut results = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut v = *base;
        for d in &mut v {
            // Box-Muller for Gaussian noise
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (rng >> 33) as f32 / (1u64 << 31) as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng >> 33) as f32 / (1u64 << 31) as f32;
            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            *d = (*d + z * noise_scale).clamp(0.0, 1.0);
        }
        results.push(v);
    }
    results
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_any(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| text.contains(kw))
}

fn parse_seconds(s: &str) -> Option<f32> {
    if s.ends_with('s') && !s.ends_with("ms") {
        s[..s.len()-1].parse().ok()
    } else if s.ends_with("ms") {
        s[..s.len()-2].parse::<f32>().ok().map(|v| v / 1000.0)
    } else {
        None
    }
}

fn apply_constraints_tensor<B: burn::prelude::Backend>(
    x: Tensor<B, 2>,
    constraints: &Constraints,
    n_samples: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data: Vec<f32> = x.to_data().to_vec().unwrap();
    for i in 0..n_samples {
        for (d, constraint) in constraints.dims.iter().enumerate() {
            if let Some(val) = constraint {
                data[i * GRAMMAR_DIM + d] = *val;
            }
        }
    }
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [n_samples, GRAMMAR_DIM]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constraints() {
        let c = parse_constraints("fire damage AoE with stun, 8s cooldown");
        assert!(c.n_fixed() > 3, "should fix several dims, got {}", c.n_fixed());
        assert!(c.dims[D_DOMAIN].unwrap() < 0.5, "should be combat");
        assert!(c.dims[D_E0_TAG].is_some(), "should have fire tag");
    }

    #[test]
    fn test_parse_campaign() {
        let c = parse_constraints("campaign economy trade_embargo");
        assert!(c.dims[D_DOMAIN].unwrap() > 0.5, "should be campaign");
    }

    #[test]
    fn test_nl_descriptions() {
        // Natural language → constraints
        let c = parse_constraints("a powerful ice spell that freezes all nearby enemies");
        assert!(c.dims[D_DOMAIN].unwrap() < 0.5, "should be combat");
        assert!(c.dims[D_E0_TAG].is_some(), "should detect ice");
        assert!(c.dims[D_E0_AREA].is_some(), "should detect AoE from 'all nearby'");

        let c = parse_constraints("quick melee strike that bleeds the target");
        assert!(c.dims[D_RANGE].is_some(), "should detect melee range");
        assert!(c.dims[D_COOLDOWN].is_some(), "should detect quick cooldown");

        let c = parse_constraints("passive: whenever you kill an enemy, gain a shield");
        assert!(c.dims[D_TYPE].unwrap() > 0.5, "should be passive");

        let c = parse_constraints("ultimate devastating holy nova that stuns everyone");
        assert!(c.dims[D_COOLDOWN].unwrap() > 0.5, "ultimate = long cooldown");
        assert!(c.dims[D_N_EFFECTS].unwrap() > 0.5, "ultimate = complex");

        let c = parse_constraints("army-wide buff that makes troops fight harder");
        assert!(c.dims[D_DOMAIN].unwrap() > 0.5, "should be campaign");
    }

    #[test]
    fn test_variations() {
        let base = [0.5f32; GRAMMAR_DIM];
        let vars = sample_variations(&base, 10, 0.1);
        assert_eq!(vars.len(), 10);
        for v in &vars {
            let dsl = decode(v);
            assert!(parse_abilities(&dsl).is_ok(), "variation should parse: {}", dsl);
        }
    }

    #[test]
    fn test_variations_parse_rate() {
        // Encode a real ability, generate variations, check parse rate
        let base = [0.3f32; GRAMMAR_DIM]; // arbitrary point
        let vars = sample_variations(&base, 100, 0.2);
        let ok = vars.iter().filter(|v| {
            let dsl = decode(v);
            parse_abilities(&dsl).is_ok()
        }).count();
        assert_eq!(ok, 100, "all variations should parse");
    }

    use tactical_sim::effects::dsl::parse_abilities;
}
