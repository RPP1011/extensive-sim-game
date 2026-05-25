use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct UiData {
    vals: HashMap<String, f32>,
}

impl UiData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, v: f32) -> &mut Self {
        self.vals.insert(key.to_string(), v);
        self
    }

    pub fn get(&self, key: &str) -> f32 {
        self.vals.get(key).copied().unwrap_or(0.0)
    }

    /// Substitute `{key}` placeholders (rendered as rounded ints) in a template.
    pub fn fill(&self, template: &str) -> String {
        let mut out = String::new();
        let mut rest = template;
        while let Some(open) = rest.find('{') {
            out.push_str(&rest[..open]);
            if let Some(close) = rest[open..].find('}') {
                let key = &rest[open + 1..open + close];
                out.push_str(&format!("{}", self.get(key).round() as i64));
                rest = &rest[open + close + 1..];
            } else {
                out.push_str(&rest[open..]);
                rest = "";
            }
        }
        out.push_str(rest);
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fill_substitutes_int_keys() {
        let mut d = super::UiData::new();
        d.set("level", 3.0).set("kills", 42.7);
        assert_eq!(d.fill("Lv {level}  Kills {kills}"), "Lv 3  Kills 43");
    }

    #[test]
    fn fill_missing_key_is_zero() {
        assert_eq!(super::UiData::new().fill("hp {hp}"), "hp 0");
    }
}
