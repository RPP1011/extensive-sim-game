use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct UiData {
    vals: HashMap<String, f32>,
    /// S13: the TEXT channel. `UiData` was f32-only, which is why S12's
    /// campaign HUD could print "day 7" but never the day's WORD, the open
    /// petition's ask, or the selected colonist's NAME — the numbers a
    /// simulation carries are not the whole of what a player must read.
    /// A key present here wins over the numeric one in [`Self::fill`].
    texts: HashMap<String, String>,
}

impl UiData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, v: f32) -> &mut Self {
        self.vals.insert(key.to_string(), v);
        self
    }

    /// Publish a STRING under `key`. `{key}` in a template then substitutes
    /// the string verbatim instead of a rounded number.
    pub fn set_text(&mut self, key: &str, v: impl Into<String>) -> &mut Self {
        self.texts.insert(key.to_string(), v.into());
        self
    }

    pub fn get(&self, key: &str) -> f32 {
        self.vals.get(key).copied().unwrap_or(0.0)
    }

    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.texts.get(key).map(String::as_str)
    }

    /// Substitute `{key}` placeholders in a template: a TEXT value verbatim
    /// if one was published under that key, else the numeric value as a
    /// rounded int (the pre-S13 behaviour, unchanged for numeric keys).
    pub fn fill(&self, template: &str) -> String {
        let mut out = String::new();
        let mut rest = template;
        while let Some(open) = rest.find('{') {
            out.push_str(&rest[..open]);
            if let Some(close) = rest[open..].find('}') {
                let key = &rest[open + 1..open + close];
                match self.texts.get(key) {
                    Some(s) => out.push_str(s),
                    None => out.push_str(&format!("{}", self.get(key).round() as i64)),
                }
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

    #[test]
    fn text_keys_substitute_verbatim_and_win_over_numbers() {
        let mut d = super::UiData::new();
        d.set("who", 3.0).set_text("who", "Alard the Quiet");
        d.set_text("ask", "the abbey asks for 5 hands");
        assert_eq!(
            d.fill("{who}: {ask} ({days})"),
            "Alard the Quiet: the abbey asks for 5 hands (0)"
        );
        assert_eq!(d.get_text("ask"), Some("the abbey asks for 5 hands"));
        assert_eq!(d.get_text("nope"), None);
    }
}
