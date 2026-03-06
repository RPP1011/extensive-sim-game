//! CoverageTracker with stats tracking.

use std::collections::HashMap;

use super::metadata::{ALL_HEROES, ROOM_TYPES};
use super::super::types::ScenarioCfg;

// ---------------------------------------------------------------------------
// Coverage tracker
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct CoverageTracker {
    hero_count: HashMap<String, u32>,
    pair_count: HashMap<(String, String), u32>,
    room_type_count: HashMap<String, u32>,
    team_size_count: HashMap<(usize, usize), u32>,
    total: u32,
}

impl CoverageTracker {
    pub fn record(&mut self, cfg: &ScenarioCfg) {
        self.total += 1;
        for h in &cfg.hero_templates {
            *self.hero_count.entry(h.clone()).or_default() += 1;
        }
        let heroes = &cfg.hero_templates;
        for i in 0..heroes.len() {
            for j in (i + 1)..heroes.len() {
                let (a, b) = if heroes[i] <= heroes[j] {
                    (heroes[i].clone(), heroes[j].clone())
                } else {
                    (heroes[j].clone(), heroes[i].clone())
                };
                *self.pair_count.entry((a, b)).or_default() += 1;
            }
        }
        *self.room_type_count.entry(cfg.room_type.clone()).or_default() += 1;
        *self.team_size_count.entry((cfg.hero_count, cfg.enemy_count)).or_default() += 1;
    }

    pub fn hero_appearances(&self, name: &str) -> u32 {
        self.hero_count.get(name).copied().unwrap_or(0)
    }

    pub fn least_seen_hero(&self) -> &'static str {
        ALL_HEROES.iter()
            .min_by_key(|h| self.hero_count.get(h.name).copied().unwrap_or(0))
            .map(|h| h.name)
            .unwrap_or("warrior")
    }

    pub fn least_seen_room(&self) -> &'static str {
        ROOM_TYPES.iter()
            .min_by_key(|r| self.room_type_count.get(**r).copied().unwrap_or(0))
            .copied()
            .unwrap_or("Entry")
    }

    pub fn pair_coverage(&self) -> (usize, usize) {
        let total_pairs = ALL_HEROES.len() * (ALL_HEROES.len() - 1) / 2;
        (self.pair_count.len(), total_pairs)
    }

    pub fn print_summary(&self) {
        println!("\n--- Coverage Summary ({} scenarios) ---", self.total);

        let mut hero_list: Vec<_> = ALL_HEROES.iter()
            .map(|h| (h.name, self.hero_count.get(h.name).copied().unwrap_or(0)))
            .collect();
        hero_list.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\nHero appearances:");
        for (name, count) in &hero_list {
            let bar: String = std::iter::repeat('#').take((*count as usize) / 4).collect();
            println!("  {:<16} {:>4}  {}", name, count, bar);
        }

        let (covered, total) = self.pair_coverage();
        println!("\nPair coverage: {covered}/{total} ({:.1}%)", covered as f64 / total as f64 * 100.0);

        println!("\nRoom types:");
        for room in ROOM_TYPES {
            let count = self.room_type_count.get(*room).copied().unwrap_or(0);
            let bar: String = std::iter::repeat('#').take(count as usize / 4).collect();
            println!("  {:<12} {:>4}  {}", room, count, bar);
        }

        println!("\nTeam sizes:");
        let mut sizes: Vec<_> = self.team_size_count.iter().collect();
        sizes.sort_by_key(|((h, e), _)| (*h, *e));
        for ((h, e), count) in &sizes {
            println!("  {}v{:<4} {:>4}", h, e, count);
        }

        let min_hero = hero_list.last().map(|(_, c)| *c).unwrap_or(0);
        let max_hero = hero_list.first().map(|(_, c)| *c).unwrap_or(0);
        let spread = if max_hero > 0 { min_hero as f64 / max_hero as f64 } else { 0.0 };
        println!("\nHero balance: min={min_hero} max={max_hero} ratio={spread:.2} (1.0 = perfect)");
    }
}
