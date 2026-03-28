use bevy_game::headless_campaign::state::{CampaignState, AdventurerStatus};
use bevy_game::headless_campaign::{step_world, seed_world_population};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_ticks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let npc_count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let mut state = CampaignState::default_test_campaign(42);
    seed_world_population(&mut state, npc_count);

    println!("=== NPC Economy World Pre-Generation ===");
    println!("Adventurers: {}", state.adventurers.len());
    println!("Locations: {}", state.overworld.locations.len());
    println!("Guild gold: {:.1}", state.guild.gold);
    println!();

    println!("Running {} ticks with {} NPCs...", max_ticks, npc_count);
    for i in 0..max_ticks {
        let events = step_world(&mut state);
        if i % 500 == 0 {
            let alive = state.adventurers.iter()
                .filter(|a| a.status != AdventurerStatus::Dead).count();
            let total_gold: f32 = state.adventurers.iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .map(|a| a.gold).sum();
            let auto_parties = state.parties.iter().filter(|p| p.autonomous).count();
            let settlements = state.overworld.locations.iter()
                .filter(|l| l.location_type == bevy_game::headless_campaign::state::LocationType::Settlement)
                .count();
            let seeking = state.adventurers.iter()
                .filter(|a| matches!(a.economic_intent, bevy_game::headless_campaign::state::EconomicIntent::SeekingParty { .. }))
                .count();
            println!("Tick {:>4} — alive:{}, npc_gold:{:.0}, guild:{:.0}, events:{}, parties:{}, settlements:{}, seeking:{}",
                state.tick, alive, total_gold, state.guild.gold, events.len(), auto_parties, settlements, seeking);
        }
    }

    println!();

    // Final state
    let alive: Vec<_> = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .collect();

    println!("=== After {} ticks ===", state.tick);
    println!("Alive: {}", alive.len());

    let with_home = alive.iter().filter(|a| a.home_location_id.is_some()).count();
    println!("With home: {with_home}");

    let total_npc_gold: f32 = alive.iter().map(|a| a.gold).sum();
    let avg_gold = if alive.is_empty() { 0.0 } else { total_npc_gold / alive.len() as f32 };
    let max_gold = alive.iter().map(|a| a.gold).fold(f32::NEG_INFINITY, f32::max);
    let min_gold = alive.iter().map(|a| a.gold).fold(f32::INFINITY, f32::min);
    println!("NPC gold — total:{total_npc_gold:.1}, avg:{avg_gold:.1}, min:{min_gold:.1}, max:{max_gold:.1}");
    let total_treasury: f32 = state.overworld.locations.iter().map(|l| l.treasury).sum();
    let total_gold = total_npc_gold + state.guild.gold + total_treasury;
    println!("Guild gold: {:.1}, settlement treasuries: {:.1}, TOTAL SYSTEM GOLD: {:.1}",
        state.guild.gold, total_treasury, total_gold);

    // Intent breakdown
    let mut intent_counts = std::collections::HashMap::new();
    for a in &alive {
        let intent = format!("{:?}", a.economic_intent);
        let key = intent.split('{').next().unwrap_or(&intent).trim().to_string();
        *intent_counts.entry(key).or_insert(0u32) += 1;
    }
    println!("Intents: {:?}", intent_counts);

    // Per-NPC details
    println!();
    for a in alive.iter().take(10) {
        let classes: Vec<String> = a.classes.iter().map(|c| format!("{}(lv{})", c.class_name, c.level)).collect();
        println!("  {} lv{} — gold:{:.1}, home:{:?}, classes:[{}], stress:{:.0}, injury:{:.0}",
            a.name, a.level, a.gold,
            a.home_location_id,
            classes.join(", "),
            a.stress, a.injury);
    }

    // Location stats
    println!();
    for loc in &state.overworld.locations {
        let demand_sum: f32 = loc.service_demand.iter().sum();
        let s = &loc.stockpile;
        println!("  {} ({:?}) — pop:{}, safety:{:.0}, threat:{:.0}, treasury:{:.0}",
            loc.name, loc.location_type, loc.resident_ids.len(),
            loc.safety_level, loc.threat_level, loc.treasury);
        println!("    stockpile: food:{:.0} iron:{:.0} wood:{:.0} herbs:{:.0} hide:{:.0} crystal:{:.0} equip:{:.1} med:{:.1}",
            s.food, s.iron, s.wood, s.herbs, s.hide, s.crystal, s.equipment, s.medicine);
        println!("    prices: food:{:.2} iron:{:.2} wood:{:.2} herbs:{:.2} equip:{:.2} med:{:.2}",
            loc.local_prices[0], loc.local_prices[1], loc.local_prices[2],
            loc.local_prices[3], loc.local_prices[6], loc.local_prices[7]);
    }

    // Autonomous parties
    let auto_parties: Vec<_> = state.parties.iter().filter(|p| p.autonomous).collect();
    println!();
    println!("Autonomous Parties: {}", auto_parties.len());
    for p in auto_parties.iter().take(10) {
        let dist = p.destination.map(|d| {
            let dx = d.0 - p.position.0;
            let dy = d.1 - p.position.1;
            (dx*dx + dy*dy).sqrt()
        });
        println!("  Party {} ({:?}) — members:{}, status:{:?}, dist_to_dest:{:.1?}, pos:({:.1},{:.1})",
            p.id, p.party_type, p.member_ids.len(), p.status,
            dist, p.position.0, p.position.1);
    }

    println!("Patronage Contracts: {}", state.patronage_contracts.len());
}
