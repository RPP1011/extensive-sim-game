//! Token-sequence VAE model using burn's transformer modules.
//!
//! Encoder: tokens → Embedding + PosEnc → TransformerEncoder → [CLS] pool → MLP → (μ, σ)
//! Decoder: z → memory → TransformerDecoder (cross-attn, causal) → vocab logits

use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig,
    transformer::{
        TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
        TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
        TransformerDecoderAutoregressiveCache,
    },
};
use burn::prelude::*;
use burn::tensor::activation;

use tactical_sim::sim::ability_transformer::tokenizer_vocab::VOCAB;

pub struct VaeOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub recon_loss_val: f64,
    pub kl_loss_val: f64,
}

#[derive(Module, Debug)]
pub struct AbilityVAE<B: Backend> {
    // Shared
    token_emb: Embedding<B>,
    pos_emb: Embedding<B>,

    // Encoder
    encoder: TransformerEncoder<B>,
    enc_pool: Linear<B>,
    enc_mu: Linear<B>,
    enc_logvar: Linear<B>,

    // Decoder (burn's built-in with cross-attention + autoregressive cache)
    decoder: TransformerDecoder<B>,
    dec_norm: LayerNorm<B>,
    z_to_memory: Linear<B>,
    output_proj: Linear<B>,

    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    d_model: usize,
    #[module(skip)]
    latent_dim: usize,
    #[module(skip)]
    max_seq_len: usize,
}

impl<B: Backend> AbilityVAE<B> {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        latent_dim: usize,
        max_seq_len: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
            .with_dropout(0.1)
            .init(device);

        let decoder = TransformerDecoderConfig::new(d_model, d_ff, n_heads, n_layers)
            .with_dropout(0.1)
            .init(device);

        Self {
            token_emb: EmbeddingConfig::new(vocab_size, d_model).init(device),
            pos_emb: EmbeddingConfig::new(max_seq_len, d_model).init(device),
            encoder,
            enc_pool: LinearConfig::new(d_model, d_model).init(device),
            enc_mu: LinearConfig::new(d_model, latent_dim).init(device),
            enc_logvar: LinearConfig::new(d_model, latent_dim).init(device),
            decoder,
            dec_norm: LayerNormConfig::new(d_model).init(device),
            z_to_memory: LinearConfig::new(latent_dim, d_model).init(device),
            output_proj: LinearConfig::new(d_model, vocab_size).init(device),
            vocab_size,
            d_model,
            latent_dim,
            max_seq_len,
        }
    }

    pub fn num_params(&self) -> usize {
        self.vocab_size * self.d_model * 2
            + self.d_model * self.latent_dim * 2
            + self.d_model * self.vocab_size
    }

    fn embed_tokens(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = token_ids.dims();
        let device = token_ids.device();
        let tok_emb = self.token_emb.forward(token_ids);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch as i64, seq_len as i64]);
        let pos_emb = self.pos_emb.forward(positions);
        tok_emb + pos_emb
    }

    fn encode(
        &self,
        token_ids: Tensor<B, 2, Int>,
        mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _seq_len] = token_ids.dims();
        let x = self.embed_tokens(token_ids);
        let enc_input = TransformerEncoderInput::new(x).mask_pad(mask);
        let encoded = self.encoder.forward(enc_input);

        // Use [CLS] token (position 0) as sequence representation
        let cls = encoded.slice([0..batch, 0..1]).reshape([batch, self.d_model]);
        let pooled = activation::relu(self.enc_pool.forward(cls));
        let mu = self.enc_mu.forward(pooled.clone());
        let logvar = self.enc_logvar.forward(pooled);
        (mu, logvar)
    }

    fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = (logvar * 0.5).exp();
        let eps = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0, 1.0));
        mu + eps * std
    }

    /// Decode with teacher forcing (for training).
    fn decode_teacher(
        &self,
        z: Tensor<B, 2>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = target_ids.dims();
        let device = z.device();

        // Memory: project z to [B, 1, D] for cross-attention
        let memory = self.z_to_memory.forward(z).unsqueeze_dim::<3>(1);

        // Target embeddings
        let target_emb = self.embed_tokens(target_ids);

        // Causal mask for self-attention
        let mask = burn::nn::attention::generate_autoregressive_mask::<B>(
            batch, seq_len, &device,
        );

        let dec_input = TransformerDecoderInput::new(target_emb, memory)
            .target_mask_attn(mask);
        let decoded = self.decoder.forward(dec_input);
        let decoded = self.dec_norm.forward(decoded);
        self.output_proj.forward(decoded)
    }

    pub fn forward_train(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
        mask: Tensor<B, 2, Bool>,
        kl_weight: f32,
        word_dropout: f32,
    ) -> VaeOutput<B> {
        let (mu, logvar) = self.encode(input_ids, mask.clone());
        let z = self.reparameterize(mu.clone(), logvar.clone());

        // Word dropout: replace random tokens with [MASK]=2
        let decoder_input = if word_dropout > 0.0 {
            let [batch, seq_len] = target_ids.dims();
            let device = target_ids.device();
            let drop_mask = Tensor::<B, 2>::random(
                [batch, seq_len],
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &device,
            ).lower_elem(word_dropout);
            // Keep position 0 ([CLS])
            let keep_first = Tensor::<B, 2, Bool>::ones([batch, 1], &device).bool_not();
            let drop_rest = drop_mask.slice([0..batch, 1..seq_len]);
            let drop_mask = Tensor::cat(vec![keep_first, drop_rest], 1);
            target_ids.clone().mask_fill(drop_mask, 2)
        } else {
            target_ids.clone()
        };

        let logits = self.decode_teacher(z, decoder_input);

        // Cross-entropy loss
        let [batch, seq_len, _] = logits.dims();
        let logits_flat = logits.reshape([batch * seq_len, self.vocab_size]);
        let targets_flat = target_ids.reshape([batch * seq_len]);

        let log_probs = activation::log_softmax(logits_flat, 1);
        let target_lp = log_probs.gather(1, targets_flat.unsqueeze_dim::<2>(1));
        let target_lp = target_lp.reshape([batch * seq_len]);

        let mask_flat = mask.reshape([batch * seq_len]).float();
        let masked_nll = target_lp.neg() * mask_flat.clone();
        let recon_loss = masked_nll.sum() / mask_flat.sum().clamp_min(1.0);

        let kl = (mu.powf_scalar(2.0) + logvar.clone().exp() - logvar - 1.0) * 0.5;
        let kl_loss = kl.mean();

        let recon_val: f64 = recon_loss.clone().into_scalar().elem();
        let kl_val: f64 = kl_loss.clone().into_scalar().elem();
        let loss = recon_loss + kl_loss * kl_weight;

        VaeOutput {
            loss: loss.unsqueeze(),
            recon_loss_val: recon_val,
            kl_loss_val: kl_val,
        }
    }

    /// Autoregressive generation (no KV cache — rebuilds sequence each step).
    /// Batch size should be small (1-4) for reasonable speed.
    pub fn generate(&self, z: Tensor<B, 2>, max_len: usize) -> Vec<Vec<u32>> {
        let [batch, _] = z.dims();
        let device = z.device();
        let cls_id = 1u32;
        let pad_id = 0u32;

        let mut generated: Vec<Vec<u32>> = (0..batch).map(|_| vec![cls_id]).collect();
        let mut done = vec![false; batch];

        for step in 0..max_len {
            let current_len = step + 1;
            let mut seq_data = vec![pad_id as i64; batch * current_len];
            for (bi, seq) in generated.iter().enumerate() {
                for (ti, &tok) in seq.iter().enumerate().take(current_len) {
                    seq_data[bi * current_len + ti] = tok as i64;
                }
            }

            let seq_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(seq_data, [batch * current_len]),
                &device,
            ).reshape([batch, current_len]);

            // Full decoder pass
            let logits = self.decode_teacher(z.clone(), seq_tensor);

            // Take last position
            let last_logits = logits.slice([0..batch, step..step + 1])
                .reshape([batch, self.vocab_size]);
            let next_tokens = last_logits.argmax(1);
            let next_data: Vec<i64> = next_tokens.reshape([batch]).to_data().to_vec().unwrap();

            for (bi, &tok) in next_data.iter().enumerate() {
                if !done[bi] {
                    let tok_u32 = tok as u32;
                    generated[bi].push(tok_u32);
                    if tok_u32 == pad_id { done[bi] = true; }
                }
            }

            if done.iter().all(|&d| d) { break; }
        }

        generated
    }

    /// Convert token IDs back to DSL text.
    pub fn tokens_to_dsl(token_ids: &[u32]) -> String {
        let mut parts = Vec::new();

        for &id in token_ids {
            let id = id as usize;
            if id >= VOCAB.len() { continue; }
            let tok = VOCAB[id];

            match tok {
                "[PAD]" | "[CLS]" | "[MASK]" | "[SEP]" => continue,
                "[UNK]" => { parts.push("???".to_string()); continue; }
                "[NAME]" => { parts.push("Generated".to_string()); continue; }
                "[STR]" => { parts.push("\"minion\"".to_string()); continue; }
                "[TAG]" => { parts.push("MAGIC".to_string()); continue; }
                _ => {}
            }

            let text = match tok {
                "NUM_0" => "0", "NUM_1" => "1", "NUM_2" => "2", "NUM_3" => "3",
                "NUM_4" => "4", "NUM_5" => "5", "NUM_6" => "6", "NUM_7" => "7",
                "NUM_8" => "8", "NUM_9" => "9", "NUM_10" => "10",
                "NUM_SMALL" => "30", "NUM_MED" => "100", "NUM_LARGE" => "500",
                "NUM_HUGE" => "2000",
                "FRAC_TINY" => "0.15", "FRAC_LOW" => "0.3", "FRAC_MID" => "0.5",
                "FRAC_HIGH" => "0.7", "FRAC_MAX" => "0.9",
                "DUR_INSTANT" => "100ms", "DUR_SHORT" => "1s", "DUR_MED" => "5s",
                "DUR_LONG" => "15s", "DUR_VLONG" => "45s",
                "TICK_SHORT" => "30t", "TICK_MED" => "100t", "TICK_LONG" => "300t",
                "TICK_VLONG" => "750t", "TICK_EPIC" => "2000t",
                _ => tok,
            };

            parts.push(text.to_string());
        }

        // Reconstruct with formatting
        let mut output = String::new();
        let mut indent = 0i32;
        for (i, part) in parts.iter().enumerate() {
            if part == "}" {
                indent = (indent - 1).max(0);
                output.push('\n');
                for _ in 0..indent { output.push_str("    "); }
                output.push('}');
                continue;
            }
            if part == "{" {
                output.push_str(" {");
                indent += 1;
                continue;
            }

            let needs_newline = matches!(
                part.as_str(),
                "ability" | "passive" | "target" | "cooldown" | "cast" | "hint" | "cost"
                | "deliver" | "on_hit" | "on_arrival" | "damage" | "heal" | "shield"
                | "stun" | "slow" | "root" | "silence" | "knockback" | "pull" | "dash"
                | "blink" | "stealth" | "buff" | "debuff" | "summon" | "fear" | "taunt"
                | "blind" | "charm" | "suppress" | "execute" | "lifesteal" | "reflect"
                | "corner_market" | "forge_trade_route" | "destabilize" | "rally"
                | "ghost_walk" | "shadow_step" | "blood_oath" | "fortify" | "sanctuary"
                | "appraise" | "reveal" | "prophecy" | "inspire" | "ceasefire"
                | "demand_audience" | "broker_alliance" | "claim_territory" | "purify"
                | "field_triage" | "forage" | "beast_lore" | "war_cry" | "hold_the_line"
                | "domain" | "trigger"
            ) && i > 0 && parts.get(i.wrapping_sub(1)).map_or(false, |p| p != ":");

            if needs_newline {
                output.push('\n');
                for _ in 0..indent { output.push_str("    "); }
            } else if i > 0 && !output.ends_with(' ') && !output.ends_with('\n')
                && part != ":" && part != "," && part != "%" && part != "+"
                && !parts.get(i.wrapping_sub(1)).map_or(false, |p| p == ":" || p == "," || p == "(")
                && part != ")" && part != "]"
            {
                output.push(' ');
            }

            output.push_str(part);
        }

        output.trim().to_string()
    }

    /// Encode → decode roundtrip (deterministic, uses mu not sampled z).
    pub fn roundtrip(
        &self,
        token_ids: Tensor<B, 2, Int>,
        mask: Tensor<B, 2, Bool>,
    ) -> Vec<Vec<u32>> {
        let (mu, _logvar) = self.encode(token_ids, mask);
        self.generate(mu, self.max_seq_len)
    }
}
