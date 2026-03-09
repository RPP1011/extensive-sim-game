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
