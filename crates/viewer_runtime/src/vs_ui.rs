//! Host-side game-UI state for the playable `vs_viewer`.
//!
//! Owns the player's upgrade levels ([`PlayerProgress`]), the HUD/menu/death
//! [`UiModel`], and the per-frame [`UiData`] built from sim readback. The
//! viewer pushes WASD into `config.ctl.move_*` each tick and mirrors the
//! progress levels into the weapon `config.ctl.*_level` fields; `engine_ui`
//! renders the model + data and returns the [`UiAction`]s the viewer applies.

use engine_ui::{Card, NamedScreen, Screen, UiAction, UiData, UiModel, Widget};

/// XP needed per level. Mirrors `config.vs.xp_per_level` (5.0) in
/// `assets/sim/vampire_survivors.sim` — keep in sync.
pub const XP_PER_LEVEL: f32 = 5.0;

/// Player starting/full HP. Mirrors `config.vs.player_hp` (100.0) — used as
/// the HP-bar denominator. Keep in sync.
pub const PLAYER_HP_MAX: f32 = 100.0;

/// The full upgrade pool: (display label, `config.ctl` level key).
pub const UPGRADE_POOL: &[(&str, &str)] = &[
    ("Bolt Damage +", "bolt_level"),
    ("Bolt Rate +", "bolt_rate_level"),
    ("Nova +", "nova_level"),
    ("Move Speed +", "move_level"),
    ("Garlic +", "garlic_level"),
    ("Whip +", "whip_level"),
];

/// Player upgrade levels + run tallies. Mirrors the `config.ctl.*_level`
/// fields (Plan 1 field-name contract). `bolt_rate_level` is `u32` (an integer
/// modulo divisor); the rest are `f32` (feed float arithmetic / `> 0.0` gates).
#[derive(Default, Clone)]
pub struct PlayerProgress {
    pub bolt_level: f32,
    pub bolt_rate_level: u32,
    pub nova_level: f32,
    pub move_level: f32,
    pub garlic_level: f32,
    pub whip_level: f32,
    /// Last level the player was observed at (for level-up edge detection).
    pub last_level: u32,
    pub kills: u32,
}

impl PlayerProgress {
    /// Discrete level from cumulative XP.
    pub fn level(xp: f32) -> u32 {
        (xp / XP_PER_LEVEL).floor() as u32
    }

    /// Returns true if a new level was reached since the last check (the edge
    /// that opens the level-up menu). Always advances `last_level` to the
    /// current level so repeated calls within a level don't re-fire.
    pub fn check_level_up(&mut self, xp: f32) -> bool {
        let lv = Self::level(xp);
        let up = lv > self.last_level;
        self.last_level = lv;
        up
    }

    /// Apply a menu pick: raise the matching upgrade level.
    pub fn apply(&mut self, action: &UiAction) {
        if let UiAction::Increment(k) = action {
            match k.as_str() {
                "bolt_level" => self.bolt_level += 1.0,
                "bolt_rate_level" => self.bolt_rate_level += 1,
                "nova_level" => self.nova_level += 1.0,
                "move_level" => self.move_level += 1.0,
                "garlic_level" => self.garlic_level += 1.0,
                "whip_level" => self.whip_level += 1.0,
                _ => {}
            }
        }
    }
}

/// Seeded 3-card draw for a level-up. Picks 3 distinct upgrades using a
/// per-level seeded PCG step (deterministic for a given seed + level).
pub fn menu_screen(seed: u64, level: u32) -> NamedScreen {
    let mut idxs: Vec<usize> = Vec::with_capacity(3);
    let mut s = seed ^ (level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let pool_len = UPGRADE_POOL.len();
    // Distinct draw; the pool is >= 3 so this always terminates quickly.
    while idxs.len() < 3 && idxs.len() < pool_len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = (s >> 33) as usize % pool_len;
        if !idxs.contains(&i) {
            idxs.push(i);
        }
    }
    let cards = idxs
        .into_iter()
        .map(|i| {
            let (label, key) = UPGRADE_POOL[i];
            Card {
                label: label.into(),
                action: UiAction::Increment(key.into()),
            }
        })
        .collect();
    NamedScreen {
        name: "level_up".into(),
        screen: Screen::Menu {
            title: "Level Up!".into(),
            cards,
        },
    }
}

/// The death-summary screen (shown when the player dies; offers restart).
pub fn death_screen() -> NamedScreen {
    NamedScreen {
        name: "dead".into(),
        screen: Screen::End {
            title: "You Died".into(),
            summary: vec![
                ("Time".into(), "time".into()),
                ("Level".into(), "level".into()),
                ("Kills".into(), "kills".into()),
            ],
            restart_label: "Restart (R)".into(),
        },
    }
}

/// The static HUD model (no modal screens; the viewer injects the level-up /
/// death screens into `screens` as needed).
pub fn hud_model() -> UiModel {
    UiModel {
        hud: vec![
            Widget::Bar {
                label: "HP".into(),
                value: "hp".into(),
                max: "hp_max".into(),
                color: [220, 40, 40],
            },
            Widget::Bar {
                label: "XP".into(),
                value: "xp_into".into(),
                max: "xp_per_level".into(),
                color: [40, 160, 240],
            },
            Widget::Text {
                template: "Lv {level}   Kills {kills}   {time}s   Enemies {enemies}".into(),
            },
        ],
        screens: vec![],
    }
}

/// Populate `d` with this frame's HUD values from sim readback.
pub fn build_data(
    d: &mut UiData,
    hp: f32,
    hp_max: f32,
    xp: f32,
    kills: u32,
    tick: u64,
    enemies: usize,
) {
    d.set("hp", hp)
        .set("hp_max", hp_max)
        .set("level", PlayerProgress::level(xp) as f32)
        .set("xp_into", xp % XP_PER_LEVEL)
        .set("xp_per_level", XP_PER_LEVEL)
        .set("kills", kills as f32)
        .set("time", (tick as f32) * 0.1)
        .set("enemies", enemies as f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_up_menu_logic() {
        let mut p = PlayerProgress::default();
        // Below the first threshold: no level-up.
        assert!(!p.check_level_up(0.0));
        assert!(!p.check_level_up(XP_PER_LEVEL - 1.0));
        assert_eq!(p.last_level, 0);

        // Crossing the first threshold opens the menu (returns true once).
        assert!(p.check_level_up(XP_PER_LEVEL));
        assert_eq!(p.last_level, 1);
        // Same level again: no re-fire.
        assert!(!p.check_level_up(XP_PER_LEVEL + 1.0));

        // Applying an Increment raises the matching field.
        assert_eq!(p.bolt_level, 0.0);
        p.apply(&UiAction::Increment("bolt_level".into()));
        assert_eq!(p.bolt_level, 1.0);
        p.apply(&UiAction::Increment("bolt_rate_level".into()));
        assert_eq!(p.bolt_rate_level, 1);
        // Restart is not an Increment — leaves levels untouched.
        p.apply(&UiAction::Restart);
        assert_eq!(p.bolt_level, 1.0);
    }

    #[test]
    fn menu_draws_three_distinct_cards() {
        let s = menu_screen(0x5_F00D_CAFE_0001, 1);
        if let Screen::Menu { cards, .. } = s.screen {
            assert_eq!(cards.len(), 3, "level-up menu draws 3 cards");
            // Distinct actions.
            let mut keys: Vec<String> = cards
                .iter()
                .map(|c| match &c.action {
                    UiAction::Increment(k) => k.clone(),
                    UiAction::Restart => "restart".into(),
                })
                .collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), 3, "3 distinct upgrade picks");
        } else {
            panic!("expected a Menu screen");
        }
    }

    #[test]
    fn build_data_fills_hud_keys() {
        let mut d = UiData::new();
        // 12 XP at 5/level -> level 2, 2.0 into the current level.
        build_data(&mut d, 30.0, 100.0, 12.0, 7, 100, 42);
        assert_eq!(d.get("hp"), 30.0);
        assert_eq!(d.get("hp_max"), 100.0);
        assert_eq!(d.get("level"), 2.0);
        assert_eq!(d.get("xp_into"), 2.0);
        assert_eq!(d.get("xp_per_level"), XP_PER_LEVEL);
        assert_eq!(d.get("kills"), 7.0);
        assert!((d.get("time") - 10.0).abs() < 1e-4);
        assert_eq!(d.get("enemies"), 42.0);
    }
}
