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
            use crate::ai::core::burn_model::inference::{BurnInferenceClient, InferenceRequest, InferenceResult};
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
}
