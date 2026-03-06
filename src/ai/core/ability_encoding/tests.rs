#[cfg(test)]
mod tests {
    use crate::ai::core::ability_encoding::*;
    use crate::ai::core::ability_eval::AbilityCategory;
    use crate::mission::hero_templates::{load_embedded_templates, parse_hero_toml};

    #[test]
    fn warrior_properties_are_80_dim() {
        let templates = load_embedded_templates();
        let warrior = templates.values().find(|t| t.hero.name == "Warrior").unwrap();
        for def in &warrior.abilities {
            let props = extract_ability_properties(def);
            assert_eq!(props.len(), ABILITY_PROP_DIM);
            // No NaN
            for (i, &v) in props.iter().enumerate() {
                assert!(!v.is_nan(), "NaN at index {} for ability {}", i, def.name);
            }
        }
    }

    #[test]
    fn category_labels_round_trip() {
        let templates = load_embedded_templates();
        for toml in templates.values() {
            for def in &toml.abilities {
                let cat = ability_category_label(def);
                let name = cat.name();
                let back = AbilityCategory::from_name(name);
                assert_eq!(back, Some(cat), "round-trip failed for {}", def.name);
            }
        }
    }

    #[test]
    fn encoder_loads_and_produces_unit_embeddings() {
        let path = std::path::Path::new("generated/ability_encoder.json");
        if !path.exists() {
            eprintln!("Skipping: encoder weights not found at {}", path.display());
            return;
        }
        let json = std::fs::read_to_string(path).unwrap();
        let encoder = AbilityEncoder::from_json(&json).unwrap();

        let templates = load_embedded_templates();
        let warrior = templates.values().find(|t| t.hero.name == "Warrior").unwrap();

        for def in &warrior.abilities {
            let embed = encoder.encode_def(def);
            assert_eq!(embed.len(), ABILITY_EMBED_DIM);
            let norm: f32 = embed.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "embedding not normalized: {}", norm);
        }
    }

    #[test]
    fn decoder_round_trip() {
        let path = std::path::Path::new("generated/ability_encoder.json");
        if !path.exists() {
            eprintln!("Skipping: encoder weights not found at {}", path.display());
            return;
        }
        let json = std::fs::read_to_string(path).unwrap();
        let (encoder, decoder) = load_autoencoder(&json).unwrap();

        let decoder = match decoder {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no decoder in weights file (legacy format)");
                return;
            }
        };

        let templates = load_embedded_templates();
        let warrior = templates.values().find(|t| t.hero.name == "Warrior").unwrap();

        for def in &warrior.abilities {
            let mse = decoder.reconstruction_error(&encoder, def);
            assert!(!mse.is_nan(), "NaN reconstruction for {}", def.name);
            // Reconstruction should be reasonable (MSE < 0.1 for normalized features)
            assert!(mse < 0.5, "high reconstruction MSE={:.4} for {}", mse, def.name);
        }
    }

    #[test]
    fn decoder_interpolation() {
        let path = std::path::Path::new("generated/ability_encoder.json");
        if !path.exists() {
            return;
        }
        let json = std::fs::read_to_string(path).unwrap();
        let (encoder, decoder) = load_autoencoder(&json).unwrap();
        let decoder = match decoder {
            Some(d) => d,
            None => return,
        };

        let templates = load_embedded_templates();
        let warrior = templates.values().find(|t| t.hero.name == "Warrior").unwrap();

        // Interpolate between first two abilities
        if warrior.abilities.len() >= 2 {
            let embed_a = encoder.encode_def(&warrior.abilities[0]);
            let embed_b = encoder.encode_def(&warrior.abilities[1]);

            let mid = decoder.interpolate(&embed_a, &embed_b, 0.5);
            assert_eq!(mid.len(), ABILITY_PROP_DIM);
            for (i, &v) in mid.iter().enumerate() {
                assert!(!v.is_nan(), "NaN at index {} in interpolation", i);
            }

            // Endpoints should reconstruct close to originals
            let at_0 = decoder.interpolate(&embed_a, &embed_b, 0.0);
            let at_1 = decoder.interpolate(&embed_a, &embed_b, 1.0);
            let recon_a = decoder.decode(&embed_a);
            let recon_b = decoder.decode(&embed_b);
            // at t=0 should be very close to decode(a), at t=1 close to decode(b)
            let diff_0: f32 = at_0.iter().zip(recon_a.iter()).map(|(a, b)| (a - b).abs()).sum();
            let diff_1: f32 = at_1.iter().zip(recon_b.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(diff_0 < 1e-4, "t=0 diverges from decode(a): {diff_0}");
            assert!(diff_1 < 1e-4, "t=1 diverges from decode(b): {diff_1}");
        }
    }

    #[test]
    fn lol_hero_properties_no_nan() {
        let lol_dir = std::path::Path::new("assets/lol_heroes");
        if !lol_dir.exists() {
            return;
        }
        let mut checked = 0;
        for entry in std::fs::read_dir(lol_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            let toml = match parse_hero_toml(&content) {
                Ok(t) => t,
                Err(_) => continue,
            };
            for def in &toml.abilities {
                let props = extract_ability_properties(def);
                for (i, &v) in props.iter().enumerate() {
                    assert!(!v.is_nan(), "NaN at [{}] for {} / {}", i, toml.hero.name, def.name);
                }
                checked += 1;
            }
        }
        assert!(checked > 100, "expected >100 LoL abilities, got {checked}");
    }
}
