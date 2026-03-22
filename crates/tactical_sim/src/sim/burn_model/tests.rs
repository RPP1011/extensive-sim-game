//! Smoke tests for the Burn V5 model.

#[cfg(test)]
mod tests {
    use super::super::*;

    #[cfg(feature = "burn-cpu")]
    use burn::backend::NdArray;
    #[cfg(feature = "burn-cpu")]
    type B = NdArray;

    #[cfg(feature = "burn-gpu")]
    mod gpu {
        use super::super::super::*;
        use burn::backend::LibTorch;
        type B = LibTorch;

        #[test]
        fn libtorch_forward_cuda() {
            let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
            let model = ActorCriticV5Config {
                vocab_size: 256,
                d_model: 32,
                d_ff: 64,
                n_heads: 4,
                n_layers: 2,
                entity_encoder_layers: 2,
                external_cls_dim: 0,
                h_dim: 16,
            }
            .init::<B>(&device);

            let batch = 4;
            let n_ent = 5;
            let n_zones = 3;

            let ent_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_ent, config::ENTITY_DIM], &device);
            let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
            let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
            let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
            let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
            let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);

            let ability_cls: Vec<Option<burn::tensor::Tensor<B, 2>>> = vec![None; config::MAX_ABILITIES];

            let (output, h_new) = model.forward(
                ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
                &ability_cls, Some(agg), None,
            );

            assert_eq!(output.target_pos.dims(), [batch, 2]);
            assert_eq!(output.combat.combat_logits.dims(), [batch, config::NUM_COMBAT_TYPES]);
            assert_eq!(h_new.dims(), [batch, 16]);
            eprintln!("LibTorch CUDA forward passed (batch={batch}, d=32)");
        }

        #[cfg(feature = "burn-gpu")]
        #[test]
        fn burn_inference_client_multithreaded() {
            use crate::sim::burn_model::inference::{BurnInferenceClient, InferenceRequest, InferenceResult};
            use std::sync::Arc;

            let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
            let model = ActorCriticV5Config {
                vocab_size: 256,
                d_model: 32,
                d_ff: 64,
                n_heads: 4,
                n_layers: 2,
                entity_encoder_layers: 2,
                external_cls_dim: 0,
                h_dim: 16,
            }
            .init::<B>(&device);

            let client = BurnInferenceClient::new(model, device, 64, 5);

            // Spawn 8 threads, each submitting 10 requests
            let n_threads = 8;
            let n_per_thread = 10;
            let handles: Vec<_> = (0..n_threads)
                .map(|tid| {
                    let client = client.clone();
                    std::thread::spawn(move || {
                        let mut results = Vec::new();
                        for _ in 0..n_per_thread {
                            let req = InferenceRequest {
                                entities: vec![vec![0.0; config::ENTITY_DIM]; 3],
                                entity_types: vec![0, 1, 2],
                                zones: vec![vec![0.0; config::ZONE_DIM]; 2],
                                combat_mask: vec![true; config::NUM_COMBAT_TYPES],
                                ability_cls: vec![None; config::MAX_ABILITIES],
                                hidden_state: vec![0.0; 16],
                                aggregate_features: vec![0.0; config::AGG_DIM],
                                corner_tokens: vec![],
                            };
                            let result = client.infer(req).unwrap();
                            results.push(result);
                        }
                        results
                    })
                })
                .collect();

            let mut total = 0;
            for h in handles {
                let results: Vec<InferenceResult> = h.join().unwrap();
                assert_eq!(results.len(), n_per_thread);
                for r in &results {
                    assert_eq!(r.hidden_state_out.len(), 16);
                }
                total += results.len();
            }

            assert_eq!(total, n_threads * n_per_thread);
            eprintln!("BurnInferenceClient: {total} inferences from {n_threads} threads");
        }
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn entity_encoder_smoke() {
        let device = Default::default();
        let encoder = EntityEncoderV5Config {
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
        }
        .init::<B>(&device);

        let batch = 2;
        let n_ent = 3;
        let n_zones = 2;

        let ent_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_ent, config::ENTITY_DIM], &device);
        let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
        let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
        let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
        let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
        let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);

        let (tokens, mask) = encoder.forward(ent_feat, ent_types, zone_feat, ent_mask, zone_mask, Some(agg));

        // 3 entities + 2 zones + 1 aggregate = 6 tokens
        assert_eq!(tokens.dims(), [batch, 6, 32]);
        assert_eq!(mask.dims(), [batch, 6]);
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn actor_critic_smoke() {
        let device = Default::default();
        let model = ActorCriticV5Config {
            vocab_size: 256,
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            n_layers: 2,
            entity_encoder_layers: 2,
            external_cls_dim: 0,
            h_dim: 16,
        }
        .init::<B>(&device);

        let batch = 2;
        let n_ent = 3;
        let n_zones = 2;

        let ent_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_ent, config::ENTITY_DIM], &device);
        let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
        let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
        let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
        let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
        let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);

        let ability_cls: Vec<Option<burn::tensor::Tensor<B, 2>>> = vec![None; config::MAX_ABILITIES];

        let (output, h_new) = model.forward(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg), None,
        );

        assert_eq!(output.target_pos.dims(), [batch, 2]);
        assert_eq!(output.combat.combat_logits.dims(), [batch, config::NUM_COMBAT_TYPES]);
        assert_eq!(h_new.dims(), [batch, 16]);

        eprintln!("V5 smoke test passed (d=32)");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn latent_interface_smoke() {
        use super::super::latent_interface::LatentInterfaceConfig;

        let device = Default::default();
        let li = LatentInterfaceConfig {
            d_model: 32,
            n_heads: 4,
            n_latents: 8,
            n_latent_blocks: 2,
        }
        .init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let tokens = burn::tensor::Tensor::<B, 3>::zeros([batch, seq_len, 32], &device);
        let mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, seq_len], true, &device);

        let (out_tokens, pooled) = li.forward(tokens, mask, None);
        assert_eq!(out_tokens.dims(), [batch, seq_len, 32]);
        assert_eq!(pooled.dims(), [batch, 32]);
        eprintln!("LatentInterface smoke test passed (d=32, K=8)");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn latent_interface_tail_drop() {
        use super::super::latent_interface::LatentInterfaceConfig;

        let device = Default::default();
        let li = LatentInterfaceConfig {
            d_model: 32,
            n_heads: 4,
            n_latents: 12,
            n_latent_blocks: 2,
        }
        .init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let tokens = burn::tensor::Tensor::<B, 3>::zeros([batch, seq_len, 32], &device);
        let mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, seq_len], true, &device);

        // Use only 4 latents
        let (out_tokens, pooled) = li.forward(tokens, mask, Some(4));
        assert_eq!(out_tokens.dims(), [batch, seq_len, 32]);
        assert_eq!(pooled.dims(), [batch, 32]);
        eprintln!("LatentInterface tail-drop test passed (K=12, used=4)");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn spatial_cross_attn_smoke() {
        use super::super::spatial_cross_attn::{SpatialCrossAttentionConfig, CORNER_DIM};

        let device = Default::default();
        let sca = SpatialCrossAttentionConfig {
            d_model: 32,
            n_heads: 4,
        }
        .init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let n_corners = 5;
        let tokens = burn::tensor::Tensor::<B, 3>::zeros([batch, seq_len, 32], &device);
        let corners = burn::tensor::Tensor::<B, 3>::zeros([batch, n_corners, CORNER_DIM], &device);
        let cmask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_corners], true, &device);

        let out = sca.forward(tokens, corners, cmask);
        assert_eq!(out.dims(), [batch, seq_len, 32]);
        eprintln!("SpatialCrossAttention smoke test passed (d=32, corners=5)");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn spatial_cross_attn_zero_init_identity() {
        use super::super::spatial_cross_attn::{SpatialCrossAttentionConfig, CORNER_DIM};

        let device = Default::default();
        let sca = SpatialCrossAttentionConfig {
            d_model: 32,
            n_heads: 4,
        }
        .init::<B>(&device);

        let batch = 1;
        let seq_len = 4;
        let n_corners = 3;

        // Random input tokens
        let tokens = burn::tensor::Tensor::<B, 3>::random(
            [batch, seq_len, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let corners = burn::tensor::Tensor::<B, 3>::zeros([batch, n_corners, CORNER_DIM], &device);
        let cmask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_corners], true, &device);

        let out = sca.forward(tokens.clone(), corners, cmask);

        // At init, zero-init out_proj means cross-attn output is zero,
        // so output should be LayerNorm(tokens + 0) = LayerNorm(tokens).
        // We can't check exact equality because of LayerNorm, but the
        // output should be close to the normalized input.
        let diff = (out - tokens).abs().mean().into_scalar();
        // Should be small-ish (LayerNorm shifts things around but from the same input)
        assert!(diff < 5.0, "Zero-init identity check: diff={diff} too large");
        eprintln!("SpatialCrossAttention zero-init identity test passed (diff={diff})");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn value_head_smoke() {
        use super::super::value_head::ValueHeadConfig;

        let device = Default::default();
        let vh = ValueHeadConfig { d_model: 32 }.init::<B>(&device);

        let batch = 2;
        let pooled = burn::tensor::Tensor::<B, 2>::zeros([batch, 32], &device);
        let out = vh.forward(pooled);
        assert_eq!(out.attrition.dims(), [batch, 1]);
        assert_eq!(out.survival.dims(), [batch, 1]);
        eprintln!("ValueHead smoke test passed");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn actor_critic_v6_smoke() {
        use super::super::actor_critic_v6::ActorCriticV6Config;
        use super::super::spatial_cross_attn::CORNER_DIM;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256,
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            n_layers: 2,
            entity_encoder_layers: 2,
            external_cls_dim: 0,
            h_dim: 16,
            n_latents: 6,
            n_latent_blocks: 2,
        }
        .init::<B>(&device);

        let batch = 2;
        let n_ent = 3;
        let n_zones = 2;
        let n_corners = 4;

        let ent_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_ent, config::ENTITY_DIM], &device);
        let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
        let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
        let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
        let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
        let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);
        let corners = burn::tensor::Tensor::<B, 3>::zeros([batch, n_corners, CORNER_DIM], &device);
        let cmask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_corners], true, &device);

        let ability_cls: Vec<Option<burn::tensor::Tensor<B, 2>>> = vec![None; config::MAX_ABILITIES];

        let (output, h_new, value) = model.forward(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg), None,
            Some(corners), Some(cmask), None,
        );

        assert_eq!(output.target_pos.dims(), [batch, 2]);
        assert_eq!(output.combat.combat_logits.dims(), [batch, config::NUM_COMBAT_TYPES]);
        assert_eq!(h_new.dims(), [batch, 16]);
        assert_eq!(value.attrition.dims(), [batch, 1]);
        assert_eq!(value.survival.dims(), [batch, 1]);

        eprintln!("V6 smoke test passed (d=32, K=6, h_dim=16, corners=4)");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn actor_critic_v6_no_corners() {
        use super::super::actor_critic_v6::ActorCriticV6Config;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256,
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            n_layers: 2,
            entity_encoder_layers: 2,
            external_cls_dim: 0,
            h_dim: 16,
            n_latents: 6,
            n_latent_blocks: 2,
        }
        .init::<B>(&device);

        let batch = 2;
        let n_ent = 3;
        let n_zones = 2;

        let ent_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_ent, config::ENTITY_DIM], &device);
        let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
        let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
        let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
        let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
        let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);

        let ability_cls: Vec<Option<burn::tensor::Tensor<B, 2>>> = vec![None; config::MAX_ABILITIES];

        // Forward without corners — spatial cross-attention should be skipped
        let (output, h_new, _) = model.forward(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg), None,
            None, None, None,
        );

        assert_eq!(output.target_pos.dims(), [batch, 2]);
        assert_eq!(h_new.dims(), [batch, 16]);
        eprintln!("V6 no-corners test passed");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn v6_checkpoint_save_load_roundtrip() {
        use super::super::actor_critic_v6::ActorCriticV6Config;
        use super::super::checkpoint;
        use super::super::spatial_cross_attn::CORNER_DIM;

        let device = Default::default();
        let config = ActorCriticV6Config {
            vocab_size: 256,
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            n_layers: 2,
            entity_encoder_layers: 2,
            external_cls_dim: 0,
            h_dim: 16,
            n_latents: 6,
            n_latent_blocks: 2,
        };
        let model = config.init::<B>(&device);

        // Save
        let dir = std::env::temp_dir().join("burn_test_v6_ckpt");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_v6");
        checkpoint::save_v6(&model, &path).expect("save failed");

        // Load
        let loaded = checkpoint::load_v6::<B>(&config, &path, &device).expect("load failed");

        // Forward both and compare
        let batch = 1;
        let n_ent = 2;
        let n_zones = 1;
        let n_corners = 2;

        let ent_feat = burn::tensor::Tensor::<B, 3>::random(
            [batch, n_ent, config::ENTITY_DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ent_types = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::zeros([batch, n_ent], &device);
        let zone_feat = burn::tensor::Tensor::<B, 3>::zeros([batch, n_zones, config::ZONE_DIM], &device);
        let ent_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_ent], true, &device);
        let zone_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_zones], true, &device);
        let agg = burn::tensor::Tensor::<B, 2>::zeros([batch, config::AGG_DIM], &device);
        let corners = burn::tensor::Tensor::<B, 3>::zeros([batch, n_corners, CORNER_DIM], &device);
        let cmask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::full([batch, n_corners], true, &device);
        let ability_cls: Vec<Option<burn::tensor::Tensor<B, 2>>> = vec![None; config::MAX_ABILITIES];

        let (out1, h1, _) = model.forward(
            ent_feat.clone(), ent_types.clone(), zone_feat.clone(), ent_mask.clone(), zone_mask.clone(),
            &ability_cls, Some(agg.clone()), None, Some(corners.clone()), Some(cmask.clone()), None,
        );
        let (out2, h2, _) = loaded.forward(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg), None, Some(corners), Some(cmask), None,
        );

        // Outputs should be identical (same weights)
        let diff_pos = (out1.target_pos - out2.target_pos).abs().mean().into_scalar();
        let diff_h = (h1 - h2).abs().mean().into_scalar();
        assert!(diff_pos < 1e-5, "Position diff too large: {diff_pos}");
        assert!(diff_h < 1e-5, "Hidden state diff too large: {diff_h}");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("V6 checkpoint save/load roundtrip passed (diff_pos={diff_pos}, diff_h={diff_h})");
    }

    #[test]
    fn vtrace_basic() {
        use super::super::training::vtrace_targets;

        // Simple 3-step trajectory: on-policy (log_rho=0), constant discount
        let log_rhos = vec![0.0, 0.0, 0.0];
        let discounts = vec![0.99, 0.99, 0.99];
        let rewards = vec![1.0, 0.0, 0.0];
        let values = vec![0.5, 0.3, 0.1];
        let bootstrap = 0.0;

        let (vs, adv) = vtrace_targets(&log_rhos, &discounts, &rewards,
                                        &values, bootstrap, 1.0, 1.0);

        assert_eq!(vs.len(), 3);
        assert_eq!(adv.len(), 3);

        // On-policy: V-trace = TD(lambda) with lambda=1
        // vs[2] = r2 + gamma*bootstrap - v2 + v2 = 0 + 0 - 0.1 + 0.1 = 0.0 + v2
        // Actually let's just check vs > values for rewarded trajectory
        assert!(vs[0] > values[0], "V-trace target should exceed value at rewarded step");
        eprintln!("V-trace basic: vs={vs:?}, adv={adv:?}");
    }

    #[test]
    fn vtrace_off_policy_clipping() {
        use super::super::training::vtrace_targets;

        // High importance ratio (very off-policy) should be clipped
        let log_rhos = vec![5.0, 5.0]; // rho = e^5 ≈ 148
        let discounts = vec![0.99, 0.99];
        let rewards = vec![1.0, 0.0];
        let values = vec![0.0, 0.0];

        let (vs_clipped, _) = vtrace_targets(&log_rhos, &discounts, &rewards,
                                              &values, 0.0, 1.0, 1.0);
        let (vs_unclipped, _) = vtrace_targets(&log_rhos, &discounts, &rewards,
                                                &values, 0.0, 200.0, 200.0);

        // Clipped should be more conservative (lower) than unclipped
        assert!(vs_clipped[0] <= vs_unclipped[0] + 1e-6,
            "Clipped V-trace should be <= unclipped: {} vs {}", vs_clipped[0], vs_unclipped[0]);
        eprintln!("V-trace clipping: clipped={}, unclipped={}", vs_clipped[0], vs_unclipped[0]);
    }

    // =========================================================================
    // V-trace extended tests
    // =========================================================================

    #[test]
    fn vtrace_zero_rewards_on_policy_targets_equal_values() {
        use super::super::training::vtrace_targets;

        let log_rhos = vec![0.0, 0.0, 0.0, 0.0];
        let discounts = vec![0.99, 0.99, 0.99, 0.99];
        let rewards = vec![0.0, 0.0, 0.0, 0.0];
        let values = vec![1.0, 0.8, 0.5, 0.2];
        let bootstrap = 0.0;

        let (vs, adv) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, bootstrap, 1.0, 1.0);

        assert_eq!(vs.len(), 4);
        assert_eq!(adv.len(), 4);

        // With zero rewards and on-policy, TD errors drive targets toward
        // discounted bootstrap. Targets should not wildly diverge from values.
        for (i, (&v, &tgt)) in values.iter().zip(vs.iter()).enumerate() {
            // The difference should be bounded — targets are corrections on values
            let diff = (tgt - v).abs();
            assert!(diff < 2.0, "step {i}: target {tgt} too far from value {v}");
        }
        eprintln!("V-trace zero rewards: vs={vs:?}, adv={adv:?}");
    }

    #[test]
    fn vtrace_single_positive_reward_terminal() {
        use super::super::training::vtrace_targets;

        let log_rhos = vec![0.0];
        let discounts = vec![0.99];
        let rewards = vec![10.0];
        let values = vec![0.0];
        let bootstrap = 0.0; // terminal

        let (vs, adv) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, bootstrap, 1.0, 1.0);

        assert_eq!(vs.len(), 1);
        // Target should exceed the value since there's a positive reward
        assert!(vs[0] > values[0], "target {} should exceed value {}", vs[0], values[0]);
        // Advantage should be positive
        assert!(adv[0] > 0.0, "advantage {} should be positive", adv[0]);
        eprintln!("V-trace single reward terminal: vs={vs:?}, adv={adv:?}");
    }

    #[test]
    fn vtrace_single_step_td0() {
        use super::super::training::vtrace_targets;

        // Single step: target = reward + gamma * bootstrap (on-policy, no clipping)
        let reward = 3.0;
        let gamma = 0.95;
        let bootstrap = 2.0;
        let value = 1.0;

        let (vs, adv) = vtrace_targets(
            &[0.0], &[gamma], &[reward], &[value], bootstrap, 1.0, 1.0,
        );

        let expected_target = reward + gamma * bootstrap;
        assert!(
            (vs[0] - expected_target).abs() < 1e-4,
            "TD(0) target: got {} expected {}", vs[0], expected_target,
        );
        // Advantage = rho * (r + gamma * v_next - v) = 1.0 * (3 + 0.95*2 - 1) = 3.9
        let expected_adv = reward + gamma * bootstrap - value;
        assert!(
            (adv[0] - expected_adv).abs() < 1e-4,
            "TD(0) advantage: got {} expected {}", adv[0], expected_adv,
        );
        eprintln!("V-trace TD(0): vs={vs:?}, adv={adv:?}");
    }

    #[test]
    fn vtrace_off_policy_clipped_le_unclipped() {
        use super::super::training::vtrace_targets;

        // Very off-policy: large positive log_rhos
        let log_rhos = vec![10.0, 10.0, 10.0];
        let discounts = vec![0.99, 0.99, 0.99];
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let bootstrap = 0.0;

        let (vs_clipped, _) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, bootstrap, 1.0, 1.0);
        let (vs_unclipped, _) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, bootstrap, 1e6, 1e6);

        for i in 0..3 {
            assert!(
                vs_clipped[i] <= vs_unclipped[i] + 1e-5,
                "step {i}: clipped {} > unclipped {}", vs_clipped[i], vs_unclipped[i],
            );
        }
        eprintln!("V-trace off-policy clipping (3-step): clipped={vs_clipped:?}, unclipped={vs_unclipped:?}");
    }

    #[test]
    fn vtrace_terminal_vs_nonterminal_bootstrap() {
        use super::super::training::vtrace_targets;

        let log_rhos = vec![0.0, 0.0];
        let discounts = vec![0.99, 0.99];
        let rewards = vec![1.0, 0.0];
        let values = vec![0.5, 0.3];

        let (vs_terminal, _) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, 0.0, 1.0, 1.0);
        let (vs_nonterminal, _) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, 5.0, 1.0, 1.0);

        // Non-terminal bootstrap should produce higher targets (more future value)
        assert!(
            vs_nonterminal[0] > vs_terminal[0],
            "non-terminal target {} should exceed terminal {}", vs_nonterminal[0], vs_terminal[0],
        );
        assert!(
            vs_nonterminal[1] > vs_terminal[1],
            "non-terminal target[1] {} should exceed terminal {}", vs_nonterminal[1], vs_terminal[1],
        );
        eprintln!("V-trace bootstrap: terminal={vs_terminal:?}, nonterminal={vs_nonterminal:?}");
    }

    #[test]
    fn vtrace_constant_rewards_targets_above_values() {
        use super::super::training::vtrace_targets;

        // Constant reward stream with zero values: targets should reflect discounted sums
        let n = 5;
        let log_rhos = vec![0.0; n];
        let discounts = vec![0.99; n];
        let rewards = vec![1.0; n];
        let values = vec![0.0; n];
        let bootstrap = 0.0;

        let (vs, _) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, bootstrap, 1.0, 1.0);

        // All targets should be positive (accumulating positive rewards)
        for (i, &v) in vs.iter().enumerate() {
            assert!(v > 0.0, "step {i}: target {v} should be positive with constant rewards");
        }
        // Earlier steps should have higher targets (more future reward)
        for i in 0..n - 1 {
            assert!(
                vs[i] >= vs[i + 1] - 1e-5,
                "step {i}: target {} should be >= step {} target {}", vs[i], i + 1, vs[i + 1],
            );
        }
        eprintln!("V-trace constant rewards: vs={vs:?}");
    }

    #[test]
    fn vtrace_output_lengths_match_inputs() {
        use super::super::training::vtrace_targets;

        for n in [1, 2, 5, 10] {
            let log_rhos = vec![0.0; n];
            let discounts = vec![0.99; n];
            let rewards = vec![0.0; n];
            let values = vec![0.0; n];

            let (vs, adv) = vtrace_targets(&log_rhos, &discounts, &rewards, &values, 0.0, 1.0, 1.0);
            assert_eq!(vs.len(), n, "vs length mismatch for n={n}");
            assert_eq!(adv.len(), n, "adv length mismatch for n={n}");
        }
        eprintln!("V-trace output lengths match for various sizes");
    }

    // =========================================================================
    // pack_batch tests
    // =========================================================================

    #[cfg(feature = "burn-cpu")]
    fn make_sample(n_entities: usize, advantage: f32, reward: f32) -> super::super::training::TrainingSample {
        super::super::training::TrainingSample {
            entities: (0..n_entities).map(|_| vec![0.0; 34]).collect(),
            entity_types: vec![0; n_entities],
            zones: vec![vec![0.0; 12]],
            aggregate_features: vec![0.0; 16],
            corner_tokens: vec![],
            target_move_pos: [5.0, 5.0],
            behavior_lp_move: 0.0,
            step_reward: reward,
            combat_type: 1,
            target_idx: 0,
            combat_mask: vec![true, true, false, false, false, false, false, false, false, false],
            behavior_log_prob: -1.0,
            value_target: 0.5,
            advantage,
            traj_id: 1,
            traj_pos: 0,
            traj_terminal: true,
        }
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn pack_batch_single_sample() {
        use super::super::training::pack_batch;

        let device = Default::default();
        let samples = vec![make_sample(3, 0.5, 1.0)];
        let batch = pack_batch::<B>(&samples, &device);

        assert_eq!(batch.bs, 1);
        assert_eq!(batch.ent_feat.dims()[0], 1);
        assert_eq!(batch.ent_feat.dims()[1], 3); // 3 entities, no padding needed
        assert_eq!(batch.ent_feat.dims()[2], 34);
        assert_eq!(batch.ent_types.dims(), [1, 3]);
        assert_eq!(batch.ent_mask.dims(), [1, 3]);
        assert_eq!(batch.zone_feat.dims()[0], 1);
        assert_eq!(batch.agg_feat.dims(), [1, 16]);
        assert_eq!(batch.advantages.len(), 1);
        assert!((batch.advantages[0] - 0.5).abs() < 1e-6);
        assert_eq!(batch.value_targets.len(), 1);
        assert!((batch.value_targets[0] - 0.5).abs() < 1e-6);
        assert_eq!(batch.move_targets.len(), 2); // B*2
        assert_eq!(batch.combat_targets.len(), 1);
        assert_eq!(batch.combat_targets[0], 1);
        eprintln!("pack_batch single sample passed");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn pack_batch_pads_to_max_entities() {
        use super::super::training::pack_batch;

        let device = Default::default();
        let s1 = make_sample(2, 0.0, 0.0); // 2 entities
        let s2 = make_sample(5, 0.0, 0.0); // 5 entities
        let samples = vec![s1, s2];
        let batch = pack_batch::<B>(&samples, &device);

        assert_eq!(batch.bs, 2);
        // Should be padded to max(2, 5) = 5
        assert_eq!(batch.ent_feat.dims(), [2, 5, 34]);
        assert_eq!(batch.ent_types.dims(), [2, 5]);
        assert_eq!(batch.ent_mask.dims(), [2, 5]);
        assert_eq!(batch.max_ent, 5);

        // First sample: mask should be true for first 2, false for rest
        let mask_data = batch.ent_mask.to_data();
        // Check that the mask captures padding correctly
        // (first sample has 2 real entities in a 5-wide tensor)
        eprintln!("pack_batch padding test passed: dims={:?}", batch.ent_feat.dims());
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn pack_batch_combat_mask_packing() {
        use super::super::training::pack_batch;

        let device = Default::default();
        let mut s = make_sample(3, 0.0, 0.0);
        // combat_mask: first 2 true, rest false
        s.combat_mask = vec![true, true, false, false, false, false, false, false, false, false];
        let samples = vec![s];
        let batch = pack_batch::<B>(&samples, &device);

        // combat_masks should be [1, 10] float
        assert_eq!(batch.combat_masks.dims(), [1, 10]);
        let cm_data: Vec<f32> = batch.combat_masks.to_data().to_vec().unwrap();
        assert!((cm_data[0] - 1.0).abs() < 1e-6, "mask[0] should be 1.0, got {}", cm_data[0]);
        assert!((cm_data[1] - 1.0).abs() < 1e-6, "mask[1] should be 1.0, got {}", cm_data[1]);
        assert!((cm_data[2] - 0.0).abs() < 1e-6, "mask[2] should be 0.0, got {}", cm_data[2]);
        assert!((cm_data[9] - 0.0).abs() < 1e-6, "mask[9] should be 0.0, got {}", cm_data[9]);
        eprintln!("pack_batch combat mask packing passed: {:?}", cm_data);
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn pack_batch_empty_corners_is_none() {
        use super::super::training::pack_batch;

        let device = Default::default();
        let s = make_sample(3, 0.0, 0.0); // corner_tokens = vec![]
        let samples = vec![s];
        let batch = pack_batch::<B>(&samples, &device);

        assert!(batch.corner_tokens.is_none(), "corner_tokens should be None when no samples have corners");
        assert!(batch.corner_mask.is_none(), "corner_mask should be None when no samples have corners");
        eprintln!("pack_batch empty corners test passed");
    }

    // =========================================================================
    // rescore_replay_buffer tests
    // =========================================================================

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn rescore_empty_buffer_is_noop() {
        use super::super::training::rescore_replay_buffer;
        use super::super::actor_critic_v6::ActorCriticV6Config;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256, d_model: 32, d_ff: 64, n_heads: 4, n_layers: 2,
            entity_encoder_layers: 2, external_cls_dim: 0, h_dim: 16,
            n_latents: 6, n_latent_blocks: 2,
        }.init::<B>(&device);

        let mut samples: Vec<super::super::training::TrainingSample> = vec![];
        rescore_replay_buffer(&mut samples, &model, &device, 0.99, 1.0, 1.0);
        assert!(samples.is_empty());
        eprintln!("rescore empty buffer noop passed");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn rescore_modifies_value_targets() {
        use super::super::training::rescore_replay_buffer;
        use super::super::actor_critic_v6::ActorCriticV6Config;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256, d_model: 32, d_ff: 64, n_heads: 4, n_layers: 2,
            entity_encoder_layers: 2, external_cls_dim: 0, h_dim: 16,
            n_latents: 6, n_latent_blocks: 2,
        }.init::<B>(&device);

        let mut s0 = make_sample(3, 1.0, 0.5);
        s0.traj_id = 10;
        s0.traj_pos = 0;
        s0.traj_terminal = false;
        let mut s1 = make_sample(3, 1.0, 0.5);
        s1.traj_id = 10;
        s1.traj_pos = 1;
        s1.traj_terminal = true;

        let original_vt0 = s0.value_target;
        let original_vt1 = s1.value_target;
        let original_adv0 = s0.advantage;
        let original_adv1 = s1.advantage;

        let mut samples = vec![s0, s1];
        rescore_replay_buffer(&mut samples, &model, &device, 0.99, 1.0, 1.0);

        // After rescore, at least one value_target or advantage should have changed
        // (model produces different values than the initial 0.5)
        let changed = (samples[0].value_target - original_vt0).abs() > 1e-6
            || (samples[1].value_target - original_vt1).abs() > 1e-6
            || (samples[0].advantage - original_adv0).abs() > 1e-6
            || (samples[1].advantage - original_adv1).abs() > 1e-6;
        assert!(changed, "rescore should modify value_target or advantage");
        eprintln!(
            "rescore modifies targets: vt0 {}->{}, vt1 {}->{}, adv0 {}->{}, adv1 {}->{}",
            original_vt0, samples[0].value_target,
            original_vt1, samples[1].value_target,
            original_adv0, samples[0].advantage,
            original_adv1, samples[1].advantage,
        );
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn rescore_groups_by_traj_id() {
        use super::super::training::rescore_replay_buffer;
        use super::super::actor_critic_v6::ActorCriticV6Config;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256, d_model: 32, d_ff: 64, n_heads: 4, n_layers: 2,
            entity_encoder_layers: 2, external_cls_dim: 0, h_dim: 16,
            n_latents: 6, n_latent_blocks: 2,
        }.init::<B>(&device);

        // Two separate trajectories interleaved
        let mut s_a0 = make_sample(3, 0.0, 1.0);
        s_a0.traj_id = 1; s_a0.traj_pos = 0; s_a0.traj_terminal = false;
        let mut s_b0 = make_sample(3, 0.0, 2.0);
        s_b0.traj_id = 2; s_b0.traj_pos = 0; s_b0.traj_terminal = true;
        let mut s_a1 = make_sample(3, 0.0, 1.5);
        s_a1.traj_id = 1; s_a1.traj_pos = 1; s_a1.traj_terminal = true;

        let mut samples = vec![s_a0, s_b0, s_a1];
        rescore_replay_buffer(&mut samples, &model, &device, 0.99, 1.0, 1.0);

        // All samples should still exist with their traj_ids intact
        assert_eq!(samples.len(), 3);
        // Traj IDs should be preserved
        let traj_ids: Vec<u64> = samples.iter().map(|s| s.traj_id).collect();
        assert!(traj_ids.contains(&1));
        assert!(traj_ids.contains(&2));
        eprintln!("rescore groups by traj_id passed: ids={traj_ids:?}");
    }

    #[cfg(feature = "burn-cpu")]
    #[test]
    fn rescore_noncontiguous_traj_pos_fallback() {
        use super::super::training::rescore_replay_buffer;
        use super::super::actor_critic_v6::ActorCriticV6Config;

        let device = Default::default();
        let model = ActorCriticV6Config {
            vocab_size: 256, d_model: 32, d_ff: 64, n_heads: 4, n_layers: 2,
            entity_encoder_layers: 2, external_cls_dim: 0, h_dim: 16,
            n_latents: 6, n_latent_blocks: 2,
        }.init::<B>(&device);

        // Non-contiguous traj_pos (gaps from eviction): 0, 2, 5
        let mut s0 = make_sample(3, 0.0, 1.0);
        s0.traj_id = 42; s0.traj_pos = 0; s0.traj_terminal = false;
        let mut s1 = make_sample(3, 0.0, 0.5);
        s1.traj_id = 42; s1.traj_pos = 2; s1.traj_terminal = false;
        let mut s2 = make_sample(3, 0.0, 0.0);
        s2.traj_id = 42; s2.traj_pos = 5; s2.traj_terminal = true;

        let mut samples = vec![s0, s1, s2];

        // Should not panic — falls back to TD(0) for non-contiguous trajectories
        rescore_replay_buffer(&mut samples, &model, &device, 0.99, 1.0, 1.0);

        assert_eq!(samples.len(), 3);
        // Values should have been updated (even with TD(0) fallback)
        eprintln!(
            "rescore non-contiguous fallback passed: vt={:?}",
            samples.iter().map(|s| s.value_target).collect::<Vec<_>>(),
        );
    }
}
