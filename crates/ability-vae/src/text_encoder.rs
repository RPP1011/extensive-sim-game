//! Static text embeddings with Matryoshka Representation Learning.
//!
//! A bag-of-words embedding model: each word maps to a learned vector,
//! sentence embedding = mean of word vectors. Trained with contrastive loss
//! at multiple truncation points (MRL) so any prefix is a useful embedding.
//!
//! Architecture:
//!   text → tokenize (whitespace + lowercase) → embedding lookup → mean pool → [D]
//!
//! Training:
//!   Pairs of (description, grammar_space_vector).
//!   Loss: cosine similarity + MSE at matryoshka dims [32, 64, 128].

use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::prelude::*;

use std::collections::HashMap;

/// Matryoshka truncation points.
const MRL_DIMS: &[usize] = &[64, 128, 256];

/// Full embedding dimension.
const EMBED_DIM: usize = 256;

const N_HEADS: usize = 8;
const N_LAYERS: usize = 6;
const D_FF: usize = 512;
const MAX_SEQ: usize = 64;

// ---------------------------------------------------------------------------
// Tokenizer (simple whitespace + cleanup)
// ---------------------------------------------------------------------------

/// Simple word-level tokenizer for ability descriptions.
pub struct WordTokenizer {
    word2id: HashMap<String, u32>,
    vocab_size: usize,
}

impl WordTokenizer {
    /// Build vocabulary from a list of texts.
    pub fn fit(texts: &[String], min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in Self::tokenize_text(text) {
                *freq.entry(word).or_default() += 1;
            }
        }

        let mut word2id = HashMap::new();
        word2id.insert("<pad>".to_string(), 0);
        word2id.insert("<unk>".to_string(), 1);

        let mut id = 2u32;
        // Sort by frequency descending for determinism
        let mut words: Vec<_> = freq.into_iter().filter(|(_, c)| *c >= min_freq).collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (word, _) in words {
            word2id.insert(word, id);
            id += 1;
        }

        let vocab_size = word2id.len();
        Self { word2id, vocab_size }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        Self::tokenize_text(text)
            .into_iter()
            .map(|w| *self.word2id.get(&w).unwrap_or(&1))
            .collect()
    }

    /// Tokenize: lowercase, split into words, then extract character n-grams.
    /// Each word produces its full form + all 3-grams and 4-grams.
    /// "flame" → ["flame", "<fl", "fla", "lam", "ame", "me>", "<fla", "flam", "lame", "ame>"]
    fn tokenize_text(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| w.len() >= 2)
        {
            // Add the full word
            tokens.push(word.to_string());

            // Add character n-grams (3 and 4) with boundary markers
            let padded = format!("<{}>", word);
            let chars: Vec<char> = padded.chars().collect();
            for n in 3..=4 {
                for i in 0..chars.len().saturating_sub(n - 1) {
                    let ngram: String = chars[i..i + n].iter().collect();
                    tokens.push(ngram);
                }
            }
        }
        tokens
    }

    /// Export vocab for Rust inference.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.word2id
    }
}

// ---------------------------------------------------------------------------
// Static Embedding Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct StaticEmbedder<B: Backend> {
    token_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    encoder: TransformerEncoder<B>,
    proj: Linear<B>,
    #[module(skip)]
    embed_dim: usize,
}

impl<B: Backend> StaticEmbedder<B> {
    pub fn new(vocab_size: usize, embed_dim: usize, device: &B::Device) -> Self {
        let encoder = TransformerEncoderConfig::new(embed_dim, D_FF, N_HEADS, N_LAYERS)
            .with_dropout(0.1)
            .init(device);

        Self {
            token_emb: EmbeddingConfig::new(vocab_size, embed_dim).init(device),
            pos_emb: EmbeddingConfig::new(MAX_SEQ, embed_dim).init(device),
            encoder,
            proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            embed_dim,
        }
    }

    /// Encode a batch of token ID sequences → [B, D] via transformer + [CLS] pooling.
    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, Int>,  // [B, max_len] padded with 0
        lengths: Tensor<B, 1, Int>,     // [B] actual lengths
    ) -> Tensor<B, 2> {
        let [batch, max_len] = token_ids.dims();
        let device = token_ids.device();

        // Token + positional embeddings
        let tok_emb = self.token_emb.forward(token_ids);
        let seq_len = max_len.min(MAX_SEQ);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch as i64, seq_len as i64]);
        let pos_emb = self.pos_emb.forward(positions);
        let x = tok_emb.slice([0..batch, 0..seq_len]) + pos_emb;

        // Padding mask
        let range = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch as i64, seq_len as i64]);
        let lengths_expanded = lengths.unsqueeze_dim::<2>(1)
            .expand([batch as i64, seq_len as i64]);
        let pad_mask = range.lower(lengths_expanded); // [B, seq_len] true where valid

        // Transformer encoder
        let enc_input = TransformerEncoderInput::new(x).mask_pad(pad_mask);
        let encoded = self.encoder.forward(enc_input); // [B, seq_len, D]

        // [CLS] pooling: take first token
        let cls = encoded.slice([0..batch, 0..1]).reshape([batch, self.embed_dim]);
        self.proj.forward(cls)
    }

    /// Embed a single text using the tokenizer.
    pub fn embed_text(
        &self,
        text: &str,
        tokenizer: &WordTokenizer,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let ids = tokenizer.encode(text);
        let len = ids.len().min(MAX_SEQ);
        let ids_tensor = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                ids.into_iter().take(len).map(|x| x as i64).collect::<Vec<_>>(),
                [len],
            ),
            device,
        ).unsqueeze::<2>();
        let lengths = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![len as i64], [1]),
            device,
        );
        self.forward(ids_tensor, lengths).reshape([self.embed_dim])
    }
}

// ---------------------------------------------------------------------------
// Matryoshka contrastive training loss
// ---------------------------------------------------------------------------

/// Compute MRL loss with InfoNCE in-batch negatives.
///
/// For each (text_emb[i], target[i]) positive pair, all other targets in the
/// batch serve as negatives. This pushes apart embeddings that should be different
/// (e.g., fire vs ice abilities) while pulling together matching pairs.
pub fn mrl_loss<B: Backend>(
    text_emb: Tensor<B, 2>,    // [B, EMBED_DIM] from StaticEmbedder
    target: Tensor<B, 2>,      // [B, target_dim] grammar space vectors (padded to EMBED_DIM)
) -> (Tensor<B, 1>, f64) {
    let [batch, _] = text_emb.dims();
    let device = text_emb.device();

    if batch < 2 {
        return (Tensor::zeros([1], &device), 0.0);
    }

    let mut total_loss = Tensor::<B, 1>::zeros([1], &device);
    let mut loss_val = 0.0f64;
    let temperature = 0.05f32;

    for &dim in MRL_DIMS {
        let emb_trunc = text_emb.clone().slice([0..batch, 0..dim]);
        let tgt_trunc = target.clone().slice([0..batch, 0..dim]);

        let emb_norm = l2_normalize(emb_trunc);
        let tgt_norm = l2_normalize(tgt_trunc.clone());

        // InfoNCE: similarity matrix [B, B] between all text embs and all targets
        // sim[i][j] = cosine(text_emb[i], target[j]) / temperature
        let sim_matrix = emb_norm.clone().matmul(tgt_norm.transpose());  // [B, B]
        let sim_scaled = sim_matrix / temperature;

        // Labels: diagonal (each text should match its own target)
        let labels = Tensor::<B, 1, Int>::arange(0..batch as i64, &device); // [B]

        // Cross-entropy loss: -log(exp(sim[i][i]) / sum_j(exp(sim[i][j])))
        let log_softmax = burn::tensor::activation::log_softmax(sim_scaled, 1); // [B, B]
        let correct = log_softmax.gather(1, labels.unsqueeze_dim::<2>(1)); // [B, 1]
        let nce_loss = correct.neg().mean();

        // Also keep MSE for direct regression (helps with continuous dims)
        let diff = emb_norm - l2_normalize(tgt_trunc);
        let mse = (diff.clone() * diff).mean() * 0.1; // lower weight than NCE

        let dim_loss = nce_loss + mse;
        let v: f64 = dim_loss.clone().into_scalar().elem();
        loss_val += v;

        total_loss = total_loss + dim_loss.unsqueeze();
    }

    (total_loss, loss_val / MRL_DIMS.len() as f64)
}

fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = (x.clone() * x.clone()).sum_dim(1).sqrt().clamp_min(1e-8); // [B, 1]
    x / norm
}

// ---------------------------------------------------------------------------
// Description generator: AbilityDef → natural language
// ---------------------------------------------------------------------------

/// Generate multiple diverse natural language descriptions from a grammar space vector.
/// Produces 3-4 descriptions with varied vocabulary and phrasing.
pub fn describe_ability(v: &[f32; super::grammar_space::GRAMMAR_DIM]) -> Vec<String> {
    let is_passive = v[0] > 0.5;
    let is_campaign = v[1] > 0.5;
    let seed = (v[0] * 1000.0 + v[3] * 777.0 + v[6] * 333.0 + v[12] * 111.0) as usize;

    let mut descriptions = Vec::new();

    // Element
    let elem_groups: &[&[&str]] = &[
        &[], &[], &[],
        &["physical", "martial", "brute", "weapon"],
        &["magic", "arcane", "mystic", "spell"],
        &["fire", "flame", "blazing", "burning", "inferno"],
        &["ice", "frost", "frozen", "glacial", "cold"],
        &["dark", "shadow", "void", "necrotic", "cursed"],
        &["holy", "divine", "sacred", "radiant", "light"],
        &["poison", "toxic", "venomous", "plague", "acid"],
    ];
    let ei = ((v[17] * elem_groups.len() as f32) as usize).min(elem_groups.len() - 1);
    let ew = if !elem_groups[ei].is_empty() { elem_groups[ei][seed % elem_groups[ei].len()] } else { "" };

    // Hint
    let hint_groups: &[&[&str]] = if is_campaign {
        &[&["economic", "trade", "commerce"], &["diplomatic", "political", "negotiation"],
          &["stealth", "covert", "espionage"], &["leadership", "commanding", "rallying"],
          &["utility", "support"], &["defensive", "fortification"], &["healing", "medical"]]
    } else {
        &[&["damage", "attack", "offensive", "destructive", "dps", "nuke"],
          &["healing", "restorative", "curative", "mending", "regeneration"],
          &["crowd control", "disabling", "cc", "lockdown", "stun"],
          &["defensive", "protective", "shielding", "tanking", "armor"],
          &["utility", "support", "tactical", "buff", "mobility"]]
    };
    let hi = ((v[6] * hint_groups.len() as f32) as usize).min(hint_groups.len() - 1);
    let hw = hint_groups[hi][(seed + 1) % hint_groups[hi].len()];

    // Target
    let tgt_groups: &[&[&str]] = if is_campaign {
        &[&["faction", "rival faction"], &["region", "territory"],
          &["market", "trade route"], &["party", "squad", "group"],
          &["guild", "organization"], &["self", "personal"],
          &["global", "army-wide", "kingdom-wide"], &["adventurer", "hero"]]
    } else {
        &[&["enemy", "foe", "hostile", "opponent", "target"],
          &["ally", "friendly", "teammate", "companion"],
          &["self", "yourself", "caster"], &["surrounding area", "nearby"],
          &["ground", "location", "zone"], &["directional", "aimed"],
          &["everyone", "all units", "map-wide"], &["global"]]
    };
    let ti = ((v[2] * tgt_groups.len() as f32) as usize).min(tgt_groups.len() - 1);
    let tw = tgt_groups[ti][(seed + 2) % tgt_groups[ti].len()];

    // Intensity
    let iw = if v[13] > 0.8 { "devastating" }
        else if v[13] > 0.6 { "powerful" }
        else if v[13] > 0.4 { "solid" }
        else if v[13] < 0.2 { "minor" }
        else { "moderate" };

    let cd = if v[4] < 0.15 { "spammable" }
        else if v[4] < 0.3 { "quick" }
        else if v[4] > 0.8 { "ultimate" }
        else if v[4] > 0.6 { "powerful" }
        else { "" };

    let area = if v[15] > 0.7 { "AoE" } else if v[15] > 0.5 { "area" } else { "single target" };

    let range = if v[3] < 0.15 { "melee" } else if v[3] > 0.75 { "long range" } else { "" };

    // D1: keyword-rich
    let mut d1 = Vec::new();
    if !cd.is_empty() { d1.push(cd); }
    if !ew.is_empty() { d1.push(ew); }
    d1.push(hw);
    d1.push(if is_passive { "passive" } else { "ability" });
    d1.push(area);
    if !range.is_empty() { d1.push(range); }
    descriptions.push(d1.join(" "));

    // D2: natural sentence
    let verb = if is_campaign {
        match hi { 0 => "manipulates trade with", 1 => "negotiates with", 2 => "infiltrates",
                   3 => "commands", _ => "affects" }
    } else {
        match hi { 0 => "deals damage to", 1 => "heals", 2 => "disables",
                   3 => "protects", _ => "supports" }
    };
    let mut d2 = format!("{} {} {} that {} {}", iw, ew, if is_passive { "passive" } else { "ability" }, verb, tw);
    if v[15] > 0.5 { d2.push_str(" in an area"); }
    descriptions.push(d2.trim().to_string());

    // D3: concise
    let mut d3 = Vec::new();
    if !ew.is_empty() { d3.push(ew.to_string()); }
    d3.push(hw.to_string());
    d3.push(area.to_string());
    if is_passive { d3.push("passive".to_string()); }
    descriptions.push(d3.join(" "));

    // D4: RPG flavor
    if !ew.is_empty() && !is_campaign {
        let flavor = match ei {
            5 => format!("unleash {} flames upon your {}", iw, tw),
            6 => format!("freeze {} with {} ice", tw, iw),
            7 => format!("{} dark energy consumes {}", iw, tw),
            8 => format!("call {} holy light against {}", iw, tw),
            9 => format!("coat {} in {} venom", tw, iw),
            3 => format!("deliver a {} strike to {}", iw, tw),
            4 => format!("channel {} arcane power at {}", iw, tw),
            _ => format!("{} {} blast targeting {}", iw, ew, tw),
        };
        descriptions.push(flavor);
    }

    descriptions
}

// ---------------------------------------------------------------------------
// Batch preparation
// ---------------------------------------------------------------------------

/// Prepare a batch of (text_ids, lengths, target_vectors) for training.
pub fn prepare_batch<B: Backend>(
    descriptions: &[Vec<u32>],
    targets: &[[f32; super::grammar_space::GRAMMAR_DIM]],
    indices: &[usize],
    max_len: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>, Tensor<B, 2>) {
    let batch = indices.len();
    let dim = super::grammar_space::GRAMMAR_DIM;

    let mut ids_data = vec![0i64; batch * max_len];
    let mut len_data = vec![0i64; batch];
    let mut target_data = vec![0.0f32; batch * EMBED_DIM];

    for (bi, &idx) in indices.iter().enumerate() {
        let desc = &descriptions[idx];
        let len = desc.len().min(max_len);
        len_data[bi] = len as i64;
        for ti in 0..len {
            ids_data[bi * max_len + ti] = desc[ti] as i64;
        }
        // Pad target to EMBED_DIM (grammar dim may be smaller)
        for d in 0..dim.min(EMBED_DIM) {
            target_data[bi * EMBED_DIM + d] = targets[idx][d];
        }
    }

    let ids = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(ids_data, [batch * max_len]),
        device,
    ).reshape([batch, max_len]);

    let lengths = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(len_data, [batch]),
        device,
    );

    let targets = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(target_data, [batch, EMBED_DIM]),
        device,
    );

    (ids, lengths, targets)
}

pub const STATIC_EMBED_DIM: usize = EMBED_DIM;
