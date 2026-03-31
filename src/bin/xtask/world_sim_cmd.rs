use std::process::ExitCode;

use bevy_game::world_sim::*;
use bevy_game::world_sim::state::{DungeonSite, SubBiome, MemEventType, TradeRoute};
use bevy_game::world_sim::runtime::WorldSim;
// Trace recording is done inline (not via WorldSimTraceRecorder) to avoid borrow conflicts.

use super::cli::WorldSimArgs;

/// 4 seasons × 1200 ticks per season.
const TICKS_PER_YEAR: u64 = 4800;

pub fn run_world_sim(mut args: WorldSimArgs) -> ExitCode {
    // --warm implies rich mode
    if args.warm {
        args.rich = true;
    }

    let state = if let Some(ref path) = args.load {
        // Resume from saved world state.
        match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str::<WorldState>(&json) {
                Ok(s) => {
                    println!("Loaded world state from {} (tick {}, {} entities)",
                        path, s.tick, s.entities.len());
                    s
                }
                Err(e) => {
                    eprintln!("Failed to deserialize {}: {}", path, e);
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                eprintln!("Failed to read {}: {}", path, e);
                return ExitCode::FAILURE;
            }
        }
    } else if args.peaceful {
        build_peaceful_world(&args)
    } else {
        build_world(&args)
    };
    let npc_count = state.entities.iter().filter(|e| e.kind == EntityKind::Npc).count();
    let monster_count = state.entities.iter().filter(|e| e.kind == EntityKind::Monster).count();
    println!("=== World Sim ===");
    println!("Entities: {} ({} NPCs, {} monsters)", state.entities.len(), npc_count, monster_count);
    println!("Settlements: {}", state.settlements.len());
    println!("Regions: {}", state.regions.len());
    println!("Factions: {}", state.factions.len());
    println!("Trade routes: {}", state.trade_routes.len());
    println!("Grids: {}", state.grids.len());
    if let Some(dur) = args.duration_secs {
        println!("Duration: {}s", dur);
    } else {
        println!("Ticks: {}", args.ticks);
    }
    if args.rich { println!("Mode: resource-rich"); }
    println!();

    // Creation myth — unique origin story from seed.
    print_creation_myth(&state, args.seed);

    // World generation summary
    print_world_summary(&state);

    let mut sim = WorldSim::new(state);

    // WebSocket mode: stream TraceFrame JSON to browser visualizer
    if let Some(port) = args.ws {
        return run_ws_server(&mut sim, port, &args);
    }

    let mut last_profile = TickProfile::default();
    let mut total_ticks = 0u64;

    let wall_start = std::time::Instant::now();
    let deadline = args.duration_secs.map(|s| wall_start + std::time::Duration::from_secs(s));
    let max_ticks = if deadline.is_some() { u64::MAX } else { args.ticks };

    let mut next_report = std::time::Instant::now() + std::time::Duration::from_secs(10);

    // Trace recording: capture snapshots every 100 ticks.
    let recording_trace = args.trace.is_some();
    let snapshot_interval: u64 = 100;
    let mut trace_snapshots: Vec<(u64, WorldState)> = Vec::new();
    let mut trace_chronicle: Vec<state::ChronicleEntry> = Vec::new();
    let mut last_chronicle_len = sim.state().chronicle.len();
    if recording_trace {
        trace_snapshots.push((sim.state().tick, sim.state().clone()));
        trace_chronicle.extend(sim.state().chronicle.iter().cloned());
    }

    loop {
        if total_ticks >= max_ticks { break; }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl { break; }
        }

        let profile = sim.tick();
        total_ticks += 1;

        // Peaceful mode NPC debug: print goal stack + inventory every 50 ticks
        if args.peaceful && total_ticks % 50 == 0 {
            let st = sim.state();
            for e in &st.entities {
                if e.kind != EntityKind::Npc || !e.alive { continue; }
                let npc = match &e.npc { Some(n) => n, None => continue };
                let goals: Vec<String> = npc.goal_stack.goals.iter().map(|g| {
                    let plan_info = if g.plan.is_empty() {
                        String::new()
                    } else {
                        format!(" plan[{}/{}]", g.plan_index, g.plan.len())
                    };
                    format!("{:?}(p{}){}", g.kind, g.priority, plan_info)
                }).collect();
                let inv = e.inventory.as_ref().map(|i| {
                    let commodities: Vec<String> = ["food","iron","wood","herbs","hide","crystal","equip","med"]
                        .iter().enumerate()
                        .filter(|(ci, _)| i.commodities[*ci] > 0.01)
                        .map(|(ci, name)| format!("{}:{:.1}", name, i.commodities[ci]))
                        .collect();
                    if commodities.is_empty() { "empty".to_string() } else { commodities.join(", ") }
                }).unwrap_or_else(|| "no inv".to_string());
                eprintln!("[t{}] {} pos=({:.0},{:.0}) goals=[{}] inv=[{}] gold={:.1} hunger={:.0} shelter={:.0}",
                    total_ticks, npc.name, e.pos.0, e.pos.1,
                    goals.join(" | "), inv, npc.gold, npc.needs.hunger, npc.needs.shelter);
            }
        }

        // Divine intervention — trigger at tick 1000 (early enough to see effects).
        if total_ticks == 1000 {
            if let Some(ref cmd) = args.intervene {
                apply_intervention(sim.state_mut(), cmd, total_ticks);
            }
        }

        // Capture trace data if recording.
        if recording_trace {
            let current_tick = sim.state().tick;
            if current_tick % snapshot_interval == 0 {
                trace_snapshots.push((current_tick, sim.state().clone()));
            }
            // Capture new chronicle entries.
            let chronicle = &sim.state().chronicle;
            if chronicle.len() > last_chronicle_len {
                trace_chronicle.extend(chronicle[last_chronicle_len..].iter().cloned());
            } else if chronicle.len() < last_chronicle_len {
                // Ring buffer drained — capture entries with current tick.
                for entry in chronicle.iter() {
                    if entry.tick == current_tick {
                        trace_chronicle.push(entry.clone());
                    }
                }
            }
            last_chronicle_len = chronicle.len();
        }

        // Report every 500 ticks (or 10s wall time, whichever comes first).
        let now = std::time::Instant::now();
        if total_ticks % 500 == 0 || now >= next_report {
            let elapsed = wall_start.elapsed().as_secs_f64();
            let st = sim.state();
            let alive_npcs = st.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
            let alive_monsters = st.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();
            let tps = total_ticks as f64 / elapsed;

            // Compute global avg morale, fear, pride across all alive NPCs.
            let npc_data: Vec<&bevy_game::world_sim::state::NpcData> = st.entities.iter()
                .filter(|e| e.alive && e.kind == EntityKind::Npc)
                .filter_map(|e| e.npc.as_ref())
                .collect();
            let nc = npc_data.len().max(1) as f32;
            let avg_morale = npc_data.iter().map(|n| n.morale).sum::<f32>() / nc;
            let avg_fear = npc_data.iter().map(|n| n.emotions.fear).sum::<f32>() / nc;
            let avg_pride = npc_data.iter().map(|n| n.emotions.pride).sum::<f32>() / nc;
            let avg_anger = npc_data.iter().map(|n| n.emotions.anger).sum::<f32>() / nc;
            let avg_safety = npc_data.iter().map(|n| n.needs.safety).sum::<f32>() / nc;

            println!("[{:.0}s] tick {} | npcs:{} monsters:{} | morale:{:.0} fear:{:.2} pride:{:.2} anger:{:.2} safety:{:.0} | {:.0} t/s",
                elapsed, total_ticks, alive_npcs, alive_monsters,
                avg_morale, avg_fear, avg_pride, avg_anger, avg_safety, tps);
            next_report = now + std::time::Duration::from_secs(5);
        }

        last_profile = profile;
    }
    let wall_elapsed = wall_start.elapsed();

    println!("\n=== Results ===");
    println!("{}", sim.profile_acc);
    let tps = total_ticks as f64 / wall_elapsed.as_secs_f64();
    let game_hours = (total_ticks as f64 * 0.1) / 3600.0;
    println!("Wall time: {:.2}s | {} ticks ({:.1} game-hours) | {:.0} ticks/sec",
        wall_elapsed.as_secs_f64(), total_ticks, game_hours, tps);

    println!("\n--- Last tick breakdown ---");
    println!("  Compute: {}µs", last_profile.compute_us);
    println!("  Merge: {}µs ({} deltas)", last_profile.merge_us, last_profile.delta_count);
    println!("  Apply: {}µs (hp:{}µs mv:{}µs econ:{}µs)",
        last_profile.apply_us, last_profile.apply_hp_us, last_profile.apply_movement_us,
        last_profile.apply_economy_us);
    println!("  Post-apply: {}µs (30 systems)", last_profile.postapply_us);

    // World summary
    let s = sim.state();
    let alive_npcs = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
    let alive_monsters = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();
    let item_entities = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Item).count();
    let equipped_items = s.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Item)
        .filter(|e| e.item.as_ref().map(|i| i.owner_id.is_some()).unwrap_or(false))
        .count();
    let unowned_items = item_entities - equipped_items;
    println!("\n--- World state ---");
    println!("Alive: {} NPCs, {} monsters", alive_npcs, alive_monsters);
    println!("Items: {} total ({} equipped, {} unowned)", item_entities, equipped_items, unowned_items);
    // Item rarity breakdown
    {
        let mut rarity_counts = [0u32; 5];
        for e in s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Item) {
            if let Some(item) = &e.item {
                rarity_counts[item.rarity as usize] += 1;
            }
        }
        if item_entities > 0 {
            println!("  Rarity: {} common, {} uncommon, {} rare, {} epic, {} legendary",
                rarity_counts[0], rarity_counts[1], rarity_counts[2], rarity_counts[3], rarity_counts[4]);
        }
    }
    // Forge worker stats
    {
        use bevy_game::world_sim::state::BuildingType;
        let forge_buildings = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.building_type == BuildingType::Forge).unwrap_or(false))
            .count();
        let complete_forges = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.building_type == BuildingType::Forge && b.construction_progress >= 1.0).unwrap_or(false))
            .count();
        let forge_workers = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc)
            .filter(|e| {
                e.npc.as_ref().and_then(|n| n.work_building_id).and_then(|wid| {
                    s.entity(wid).and_then(|we| we.building.as_ref())
                        .map(|b| b.building_type == BuildingType::Forge)
                }).unwrap_or(false)
            })
            .count();
        let work_states: Vec<&str> = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc)
            .filter_map(|e| e.npc.as_ref())
            .map(|n| match &n.work_state {
                bevy_game::world_sim::state::WorkState::Idle => "idle",
                bevy_game::world_sim::state::WorkState::TravelingToWork { .. } => "traveling",
                bevy_game::world_sim::state::WorkState::Working { .. } => "working",
                bevy_game::world_sim::state::WorkState::CarryingToStorage { .. } => "carrying",
            })
            .collect();
        let idle = work_states.iter().filter(|&&s| s == "idle").count();
        let traveling = work_states.iter().filter(|&&s| s == "traveling").count();
        let working = work_states.iter().filter(|&&s| s == "working").count();
        let carrying = work_states.iter().filter(|&&s| s == "carrying").count();
        println!("Forges: {} ({} complete) | Forge workers: {}", forge_buildings, complete_forges, forge_workers);
        println!("Work states: {} idle, {} traveling, {} working, {} carrying", idle, traveling, working, carrying);
    }
    // Observable actions breakdown
    {
        use bevy_game::world_sim::state::NpcAction;
        let mut action_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for e in s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc) {
            if let Some(npc) = &e.npc {
                let label = match &npc.action {
                    NpcAction::Idle => "idle",
                    NpcAction::Walking { .. } => "walking",
                    NpcAction::Eating { .. } => "eating",
                    NpcAction::Working { .. } => "working",
                    NpcAction::Hauling { .. } => "hauling",
                    NpcAction::Fighting { .. } => "fighting",
                    NpcAction::Socializing { .. } => "socializing",
                    NpcAction::Resting { .. } => "resting",
                    NpcAction::Building { .. } => "building",
                    NpcAction::Fleeing => "fleeing",
                    NpcAction::Trading { .. } => "trading",
                };
                *action_counts.entry(label).or_default() += 1;
            }
        }
        let mut sorted: Vec<_> = action_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let parts: Vec<String> = sorted.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
        println!("Actions: {}", parts.join(", "));
    }
    // Building storage stats
    {
        use bevy_game::world_sim::state::BuildingType;
        let storage_buildings = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.storage_capacity > 0.0 && b.construction_progress >= 1.0).unwrap_or(false))
            .count();
        let total_stored: f32 = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter_map(|e| e.building.as_ref())
            .map(|b| b.storage_used())
            .sum();
        let total_capacity: f32 = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter_map(|e| e.building.as_ref())
            .filter(|b| b.construction_progress >= 1.0)
            .map(|b| b.storage_capacity)
            .sum();
        println!("Storage buildings: {} | Stored: {:.0}/{:.0} ({:.0}%)",
            storage_buildings, total_stored, total_capacity,
            if total_capacity > 0.0 { total_stored / total_capacity * 100.0 } else { 0.0 });
        // Treasury buildings
        let treasury_count = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.building_type == BuildingType::Treasury).unwrap_or(false))
            .count();
        let treasury_gold: f32 = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.building_type == BuildingType::Treasury).unwrap_or(false))
            .filter_map(|e| e.inventory.as_ref())
            .map(|inv| inv.gold)
            .sum();
        let treasury_commodities: f32 = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter(|e| e.building.as_ref().map(|b| b.building_type == BuildingType::Treasury).unwrap_or(false))
            .filter_map(|e| e.inventory.as_ref())
            .map(|inv| inv.total_stored())
            .sum();
        println!("Treasuries: {} | Gold: {:.0} | Commodities: {:.0}",
            treasury_count, treasury_gold, treasury_commodities);
    }

    let total_food: f32 = s.settlements.iter().map(|s| s.stockpile[commodity::FOOD]).sum();
    let total_treasury: f32 = s.settlements.iter().map(|s| s.treasury).sum();
    let total_gold: f32 = s.entities.iter()
        .filter_map(|e| e.npc.as_ref())
        .map(|n| n.gold)
        .sum();
    println!("Economy: {:.0} food, {:.0} treasury, {:.0} NPC gold", total_food, total_treasury, total_gold);

    // Top/bottom settlements
    let mut settlements: Vec<&SettlementState> = s.settlements.iter().collect();
    settlements.sort_by(|a, b| b.treasury.partial_cmp(&a.treasury).unwrap());
    println!("\nRichest settlements:");
    for s in settlements.iter().take(5) {
        println!("  {} ({}) — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
            s.name, s.specialty, s.stockpile[commodity::FOOD], s.stockpile[commodity::IRON], s.stockpile[commodity::WOOD], s.treasury);
    }
    if settlements.len() > 5 {
        println!("Poorest settlements:");
        for s in settlements.iter().rev().take(3) {
            println!("  {} ({}) — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
                s.name, s.specialty, s.stockpile[commodity::FOOD], s.stockpile[commodity::IRON], s.stockpile[commodity::WOOD], s.treasury);
        }
    }

    // --- Per-settlement morale/emotion breakdown ---
    println!("\n--- Settlement morale & emotions ---");
    for settlement in &s.settlements {
        let range = s.group_index.settlement_entities(settlement.id);
        let npcs: Vec<&bevy_game::world_sim::state::Entity> = s.entities[range.clone()].iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
            .collect();
        let n = npcs.len();
        if n == 0 {
            println!("  {} — 0 NPCs (depopulated)", settlement.name);
            continue;
        }
        let avg = |f: fn(&bevy_game::world_sim::state::NpcData) -> f32| -> f32 {
            npcs.iter()
                .filter_map(|e| e.npc.as_ref())
                .map(f)
                .sum::<f32>() / n as f32
        };
        let morale = avg(|n| n.morale);
        let stress = avg(|n| n.stress);
        let hunger = avg(|n| n.needs.hunger);
        let safety = avg(|n| n.needs.safety);
        let purpose = avg(|n| n.needs.purpose);
        let fear = avg(|n| n.emotions.fear);
        let anger = avg(|n| n.emotions.anger);
        let pride = avg(|n| n.emotions.pride);
        let joy = avg(|n| n.emotions.joy);
        let anxiety = avg(|n| n.emotions.anxiety);
        let grief = avg(|n| n.emotions.grief);
        let risk = avg(|n| n.personality.risk_tolerance);

        println!("  {} ({} NPCs) threat:{:.2}", settlement.name, n, settlement.threat_level);
        println!("    morale:{:.0} stress:{:.0} | hunger:{:.0} safety:{:.0} purpose:{:.0}",
            morale, stress, hunger, safety, purpose);
        println!("    emotions — fear:{:.2} anger:{:.2} pride:{:.2} joy:{:.2} anxiety:{:.2} grief:{:.2}",
            fear, anger, pride, joy, anxiety, grief);
        println!("    personality — risk_tolerance:{:.2}", risk);
    }

    // --- Survivor profiles: top 5 NPCs by level ---
    {
        let mut survivors: Vec<&bevy_game::world_sim::state::Entity> = s.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
            .collect();
        survivors.sort_by(|a, b| b.level.cmp(&a.level));
        if !survivors.is_empty() {
            println!("\n--- Top survivors ---");
            for e in survivors.iter().take(5) {
                let npc = e.npc.as_ref().unwrap();
                let display = bevy_game::world_sim::naming::entity_display_name(e);
                let home = npc.home_settlement_id
                    .and_then(|sid| s.settlement(sid))
                    .map(|s| s.name.as_str())
                    .unwrap_or("homeless");
                let class_str = if npc.classes.is_empty() {
                    "no class".to_string()
                } else {
                    npc.classes.iter()
                        .map(|c| format!("{} L{}", c.display_name, c.level))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                println!("  {} ({}) lv{} hp:{:.0}/{:.0} atk:{:.1} armor:{:.1} spd:{:.1}",
                    display, npc.archetype, e.level, e.hp, e.max_hp,
                    e.attack_damage, e.armor, e.move_speed);
                println!("    home: {} | gold: {:.0} | classes: [{}]", home, npc.gold, class_str);
                println!("    morale:{:.0} stress:{:.0} | hunger:{:.0} safety:{:.0} shelter:{:.0} social:{:.0} purpose:{:.0} esteem:{:.0}",
                    npc.morale, npc.stress, npc.needs.hunger, npc.needs.safety,
                    npc.needs.shelter, npc.needs.social, npc.needs.purpose, npc.needs.esteem);
                println!("    emotions — fear:{:.2} anger:{:.2} pride:{:.2} joy:{:.2} anxiety:{:.2} grief:{:.2}",
                    npc.emotions.fear, npc.emotions.anger, npc.emotions.pride,
                    npc.emotions.joy, npc.emotions.anxiety, npc.emotions.grief);
                println!("    personality — risk:{:.2} social:{:.2} ambition:{:.2} compassion:{:.2} curiosity:{:.2}",
                    npc.personality.risk_tolerance, npc.personality.social_drive,
                    npc.personality.ambition, npc.personality.compassion, npc.personality.curiosity);
                // Memory: count events by type
                let mem = &npc.memory;
                let friend_deaths = mem.events.iter().filter(|e| matches!(e.event_type, bevy_game::world_sim::state::MemEventType::FriendDied(_))).count();
                let attacks = mem.events.iter().filter(|e| matches!(e.event_type, bevy_game::world_sim::state::MemEventType::WasAttacked)).count();
                let wins = mem.events.iter().filter(|e| matches!(e.event_type, bevy_game::world_sim::state::MemEventType::WonFight)).count();
                let starved = mem.events.iter().filter(|e| matches!(e.event_type, bevy_game::world_sim::state::MemEventType::Starved)).count();
                let beliefs = mem.beliefs.len();
                // Equipped items
                let weapon_name = npc.equipped_items.weapon_id
                    .and_then(|iid| s.entities.iter().find(|e| e.id == iid))
                    .and_then(|e| e.item.as_ref())
                    .map(|i| format!("{} (q{:.1})", i.name, i.effective_quality()))
                    .unwrap_or_else(|| "none".to_string());
                let armor_name = npc.equipped_items.armor_id
                    .and_then(|iid| s.entities.iter().find(|e| e.id == iid))
                    .and_then(|e| e.item.as_ref())
                    .map(|i| format!("{} (q{:.1})", i.name, i.effective_quality()))
                    .unwrap_or_else(|| "none".to_string());
                println!("    memory: {} friend deaths, {} attacks, {} wins, {} starved | {} beliefs",
                    friend_deaths, attacks, wins, starved, beliefs);
                println!("    weapon: {} | armor: {}", weapon_name, armor_name);
            }
        }
    }

    // Behavior profile samples — show top tags for a few alive NPCs
    // Sample NPCs evenly across the entity array (different settlements).
    let alive_all: Vec<&bevy_game::world_sim::state::Entity> = s.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
        .collect();
    let stride = (alive_all.len() / 10).max(1);
    let alive_npcs: Vec<&bevy_game::world_sim::state::Entity> = alive_all.iter()
        .step_by(stride)
        .take(10)
        .copied()
        .collect();

    if !alive_npcs.is_empty() {
        println!("\n--- NPC behavior profiles (sample) ---");
        // Build reverse tag lookup
        use bevy_game::world_sim::state::tags;
        let tag_names: &[(u32, &str)] = &[
            (tags::MELEE, "melee"), (tags::RANGED, "ranged"), (tags::COMBAT, "combat"),
            (tags::DEFENSE, "defense"), (tags::TACTICS, "tactics"), (tags::MINING, "mining"),
            (tags::SMITHING, "smithing"), (tags::CRAFTING, "crafting"), (tags::ENCHANTMENT, "enchantment"),
            (tags::ALCHEMY, "alchemy"), (tags::TRADE, "trade"), (tags::DIPLOMACY, "diplomacy"),
            (tags::LEADERSHIP, "leadership"), (tags::NEGOTIATION, "negotiation"),
            (tags::RESEARCH, "research"), (tags::LORE, "lore"), (tags::MEDICINE, "medicine"),
            (tags::HERBALISM, "herbalism"), (tags::NAVIGATION, "navigation"),
            (tags::ENDURANCE, "endurance"), (tags::RESILIENCE, "resilience"),
            (tags::STEALTH, "stealth"), (tags::SURVIVAL, "survival"), (tags::AWARENESS, "awareness"),
            (tags::FAITH, "faith"), (tags::RITUAL, "ritual"), (tags::LABOR, "labor"),
            (tags::TEACHING, "teaching"), (tags::DISCIPLINE, "discipline"),
            (tags::FARMING, "farming"), (tags::WOODWORK, "woodwork"), (tags::EXPLORATION, "exploration"),
            (tags::DECEPTION, "deception"),
        ];
        let resolve_tag = |hash: u32| -> &str {
            tag_names.iter().find(|(h, _)| *h == hash).map(|(_, n)| *n).unwrap_or("unknown")
        };

        for npc_entity in &alive_npcs {
            let npc = npc_entity.npc.as_ref().unwrap();
            let mut tag_pairs: Vec<(&str, f32)> = npc.behavior_profile.iter()
                .map(|&(h, v)| (resolve_tag(h), v))
                .filter(|(_, v)| *v > 0.0)
                .collect();
            tag_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_tags: Vec<String> = tag_pairs.iter().take(5)
                .map(|(name, val)| format!("{}:{:.0}", name, val))
                .collect();

            // Resolve class name hashes
            let class_name_table: &[(u32, &str)] = &[
                (bevy_game::world_sim::state::tag(b"Warrior"), "Warrior"),
                (bevy_game::world_sim::state::tag(b"Ranger"), "Ranger"),
                (bevy_game::world_sim::state::tag(b"Guardian"), "Guardian"),
                (bevy_game::world_sim::state::tag(b"Healer"), "Healer"),
                (bevy_game::world_sim::state::tag(b"Merchant"), "Merchant"),
                (bevy_game::world_sim::state::tag(b"Scholar"), "Scholar"),
                (bevy_game::world_sim::state::tag(b"Rogue"), "Rogue"),
                (bevy_game::world_sim::state::tag(b"Artisan"), "Artisan"),
                (bevy_game::world_sim::state::tag(b"Diplomat"), "Diplomat"),
                (bevy_game::world_sim::state::tag(b"Commander"), "Commander"),
                (bevy_game::world_sim::state::tag(b"Farmer"), "Farmer"),
                (bevy_game::world_sim::state::tag(b"Miner"), "Miner"),
                (bevy_game::world_sim::state::tag(b"Woodsman"), "Woodsman"),
                (bevy_game::world_sim::state::tag(b"Alchemist"), "Alchemist"),
                (bevy_game::world_sim::state::tag(b"Herbalist"), "Herbalist"),
                (bevy_game::world_sim::state::tag(b"Explorer"), "Explorer"),
                (bevy_game::world_sim::state::tag(b"Mentor"), "Mentor"),
            ];
            let resolve_class = |hash: u32| -> &str {
                class_name_table.iter().find(|(h, _)| *h == hash).map(|(_, n)| *n).unwrap_or("?")
            };
            let classes: Vec<String> = npc.classes.iter()
                .map(|c| {
                    if c.display_name.is_empty() {
                        format!("{} L{}", resolve_class(c.class_name_hash), c.level)
                    } else {
                        format!("{} L{}", c.display_name, c.level)
                    }
                })
                .collect();

            let display_name = bevy_game::world_sim::naming::entity_display_name(npc_entity);
            println!("  #{} {} ({}) lv{} | tags: {} | classes: [{}]",
                npc_entity.id,
                display_name,
                if npc.archetype.is_empty() { "?" } else { &npc.archetype },
                npc_entity.level,
                if top_tags.is_empty() { "none".to_string() } else { top_tags.join(", ") },
                classes.join(", "),
            );
        }
    }

    // Chronicle output
    if args.chronicle || args.warm {
        print_chronicle(s);
    }

    // Narrative history summary
    if args.history || args.warm {
        print_history(s);
    }

    // Export history as markdown
    if let Some(ref path) = args.export_history {
        export_history_markdown(s, path);
    }

    // Warm-up summary
    if args.warm {
        print_warm_summary(s);
    }

    // Serialize world state to disk if --output specified.
    // Save trace file if --trace specified.
    if let Some(ref trace_path) = args.trace {
        let trace = bevy_game::world_sim::trace::WorldSimTrace {
            seed: args.seed,
            snapshots: trace_snapshots,
            chronicle_log: trace_chronicle,
            total_ticks: s.tick,
        };
        match trace.save_to_file(trace_path) {
            Ok(_) => println!("\nTrace written to {} ({} snapshots, {} chronicle entries)",
                trace_path, trace.snapshots.len(), trace.chronicle_log.len()),
            Err(e) => eprintln!("Failed to write trace: {}", e),
        }
    }

    if let Some(ref path) = args.output {
        match serde_json::to_string(s) {
            Ok(json) => {
                match std::fs::write(path, &json) {
                    Ok(_) => {
                        let size_kb = json.len() / 1024;
                        println!("\nWorld state written to {} ({} KB)", path, size_kb);
                    }
                    Err(e) => eprintln!("Failed to write {}: {}", path, e),
                }
            }
            Err(e) => eprintln!("Failed to serialize world state: {}", e),
        }
    }

    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Chronicle output (Phase 3)
// ---------------------------------------------------------------------------

fn chronicle_category_name(cat: &bevy_game::world_sim::state::ChronicleCategory) -> &'static str {
    use bevy_game::world_sim::state::ChronicleCategory::*;
    match cat {
        Battle => "Battle",
        Quest => "Quest",
        Diplomacy => "Diplomacy",
        Economy => "Economy",
        Death => "Death",
        Discovery => "Discovery",
        Crisis => "Crisis",
        Achievement => "Achievement",
        Narrative => "Narrative",
    }
}

fn print_chronicle(state: &WorldState) {
    println!("\n=== Chronicle ===");

    if state.chronicle.is_empty() {
        println!("  (no chronicle entries)");
        return;
    }

    // Print last 20 entries
    let start = state.chronicle.len().saturating_sub(20);
    println!("Last {} entries (of {} total):", state.chronicle.len() - start, state.chronicle.len());
    for entry in &state.chronicle[start..] {
        println!("  [tick {}] [{}] {}",
            entry.tick,
            chronicle_category_name(&entry.category),
            entry.text,
        );
    }

    // Summary counts by category
    use bevy_game::world_sim::state::ChronicleCategory;
    let categories = [
        ChronicleCategory::Battle,
        ChronicleCategory::Quest,
        ChronicleCategory::Diplomacy,
        ChronicleCategory::Economy,
        ChronicleCategory::Death,
        ChronicleCategory::Discovery,
        ChronicleCategory::Crisis,
        ChronicleCategory::Achievement,
        ChronicleCategory::Narrative,
    ];
    println!("\nChronicle summary ({} total entries):", state.chronicle.len());
    for cat in &categories {
        let count = state.chronicle.iter().filter(|e| e.category == *cat).count();
        if count > 0 {
            println!("  {}: {}", chronicle_category_name(cat), count);
        }
    }
}

// ---------------------------------------------------------------------------
// Warm-up summary (Phase 5)
// ---------------------------------------------------------------------------

fn npc_display_name(npc: &bevy_game::world_sim::state::NpcData, entity_id: u32) -> String {
    if !npc.name.is_empty() {
        npc.name.clone()
    } else {
        format!("Entity #{}", entity_id)
    }
}

fn name_matches(name: &str, query: &str) -> bool {
    name.to_lowercase().contains(&query.to_lowercase())
}

fn apply_intervention(state: &mut WorldState, cmd: &str, tick: u64) {
    let parts: Vec<&str> = cmd.splitn(2, ':').collect();
    let action = parts[0];
    let target = parts.get(1).copied().unwrap_or("");

    match action {
        "bless" => {
            // Bless a settlement: +50 food, +30 treasury, +20 morale to all NPCs.
            let found = state.settlements.iter_mut().find(|s|
                name_matches(&s.name, &target));
            if let Some(settlement) = found {
                let sid = settlement.id;
                settlement.stockpile[bevy_game::world_sim::commodity::FOOD] += 50.0;
                settlement.treasury += 30.0;
                let name = settlement.name.clone();
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("DIVINE INTERVENTION: The gods have blessed {}! Food and gold rain from the heavens.", name),
                    entity_ids: vec![],
                });
                for entity in &mut state.entities {
                    if let Some(npc) = &mut entity.npc {
                        if npc.home_settlement_id == Some(sid) {
                            npc.morale = (npc.morale + 20.0).min(100.0);
                            npc.emotions.joy = 1.0;
                        }
                    }
                }
                eprintln!("[divine] Blessed {}", name);
            }
        }
        "curse" => {
            // Curse a region: +50 monster density, +0.5 threat.
            let found = state.regions.iter_mut().find(|r|
                name_matches(&r.name, &target));
            if let Some(region) = found {
                region.monster_density += 50.0;
                region.threat_level += 0.5;
                let name = region.name.clone();
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("DIVINE INTERVENTION: A curse falls upon {}! Monsters surge from the darkness.", name),
                    entity_ids: vec![],
                });
                eprintln!("[divine] Cursed {}", name);
            }
        }
        "champion" => {
            // Spawn a powerful champion NPC at a settlement.
            let sid = state.settlements.iter()
                .find(|s| name_matches(&s.name, &target))
                .map(|s| (s.id, s.pos));
            if let Some((sid, pos)) = sid {
                let id = state.next_entity_id();
                let mut champion = Entity::new_npc(id, pos);
                champion.hp = 500.0;
                champion.max_hp = 500.0;
                champion.attack_damage = 50.0;
                champion.armor = 20.0;
                champion.level = 30;
                if let Some(npc) = &mut champion.npc {
                    npc.name = format!("Divine Champion");
                    npc.home_settlement_id = Some(sid);
                    npc.morale = 100.0;
                    npc.emotions.pride = 1.0;
                    npc.accumulate_tags(&{
                        let mut a = ActionTags::empty();
                        a.add(tags::COMBAT, 50.0);
                        a.add(tags::LEADERSHIP, 30.0);
                        a.add(tags::FAITH, 30.0);
                        a.add(tags::DEFENSE, 20.0);
                        a
                    });
                }
                state.entities.push(champion);
                let sname = state.settlements.iter().find(|s| s.id == sid)
                    .map(|s| s.name.clone()).unwrap_or_default();
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("DIVINE INTERVENTION: A Champion descends upon {}! The gods answer the people's prayers.", sname),
                    entity_ids: vec![id],
                });
                eprintln!("[divine] Spawned champion at {}", sname);
            }
        }
        "prophecy" => {
            // Force-fulfill the first unfulfilled prophecy.
            for p in &mut state.prophecies {
                if !p.fulfilled {
                    p.fulfilled = true;
                    p.fulfilled_tick = Some(tick);
                    let text = p.text.clone();
                    state.chronicle.push(ChronicleEntry {
                        tick,
                        category: ChronicleCategory::Narrative,
                        text: format!("DIVINE INTERVENTION: A PROPHECY FULFILLED BY DIVINE WILL: \"{}\"", text),
                        entity_ids: vec![],
                    });
                    eprintln!("[divine] Fulfilled prophecy: {}", text);
                    break;
                }
            }
        }
        "plague" => {
            // Plague a settlement: damage all NPCs, drain food.
            let found = state.settlements.iter().find(|s|
                name_matches(&s.name, &target))
                .map(|s| (s.id, s.name.clone()));
            if let Some((sid, sname)) = found {
                for entity in &mut state.entities {
                    if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                    if let Some(npc) = &mut entity.npc {
                        if npc.home_settlement_id == Some(sid) {
                            entity.hp -= entity.max_hp * 0.3;
                            npc.emotions.fear = 1.0;
                            npc.emotions.anxiety = 1.0;
                        }
                    }
                }
                if let Some(s) = state.settlements.iter_mut().find(|s| s.id == sid) {
                    s.stockpile[bevy_game::world_sim::commodity::FOOD] *= 0.3;
                }
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("DIVINE INTERVENTION: A terrible plague descends upon {}! The people suffer greatly.", sname),
                    entity_ids: vec![],
                });
                eprintln!("[divine] Plagued {}", sname);
            }
        }
        _ => {
            eprintln!("[divine] Unknown intervention: {}", cmd);
        }
    }
}

fn print_history(state: &WorldState) {
    println!("\n{}", "=".repeat(60));
    println!("=== WORLD HISTORY ===");
    println!("{}\n", "=".repeat(60));

    let total_ticks = state.tick;
    let years = total_ticks / TICKS_PER_YEAR; // 4 seasons × 1200 ticks
    let pop = state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
    let monsters = state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();
    let settlements = state.settlements.len();

    println!("After {} years ({} ticks), the world has {} souls across {} settlements ({} monsters roam).\n",
        years, total_ticks, pop, settlements, monsters);

    // --- Legends ---
    let legends: Vec<&Entity> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| n.name.contains("the Legendary")).unwrap_or(false))
        .collect();
    if !legends.is_empty() {
        println!("LIVING LEGENDS:");
        for e in &legends {
            let npc = e.npc.as_ref().unwrap();
            let classes = npc.classes.iter()
                .map(|c| format!("{} L{}", c.display_name, c.level))
                .collect::<Vec<_>>().join(", ");
            let oaths_fulfilled = npc.oaths.iter().filter(|o| o.fulfilled).count();
            println!("  {} — [{}] | {} oaths fulfilled", npc.name, classes, oaths_fulfilled);
        }
        println!();
    }

    // --- Marriages & Families ---
    let married = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| n.spouse_id.is_some()).unwrap_or(false))
        .count();
    let children_born = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| !n.parents.is_empty()).unwrap_or(false))
        .count();
    if married > 0 || children_born > 0 {
        println!("FAMILIES: {} married NPCs, {} children born.", married, children_born);
    }

    // --- Wars & Betrayals ---
    let wars: Vec<String> = state.chronicle.iter()
        .filter(|e| e.text.contains("WAR!"))
        .map(|e| format!("  [tick {}] {}", e.tick, e.text))
        .collect();
    let peace: Vec<String> = state.chronicle.iter()
        .filter(|e| e.text.contains("PEACE!"))
        .map(|e| format!("  [tick {}] {}", e.tick, e.text))
        .collect();
    let betrayals: Vec<String> = state.chronicle.iter()
        .filter(|e| e.text.contains("betrayed"))
        .map(|e| format!("  [tick {}] {}", e.tick, e.text))
        .collect();
    if !wars.is_empty() || !betrayals.is_empty() {
        println!("\nCONFLICTS:");
        for w in &wars { println!("{}", w); }
        for p in &peace { println!("{}", p); }
        for b in &betrayals { println!("{}", b); }
    }

    // --- Prophecies ---
    if !state.prophecies.is_empty() {
        println!("\nPROPHECIES:");
        for p in &state.prophecies {
            let status = if p.fulfilled {
                format!("FULFILLED (tick {})", p.fulfilled_tick.unwrap_or(0))
            } else {
                "unfulfilled".into()
            };
            println!("  \"{}\" — {}", p.text, status);
        }
    }

    // --- Dungeons ---
    let explored: Vec<String> = state.regions.iter()
        .flat_map(|r| r.dungeon_sites.iter().map(move |d| (r, d)))
        .filter(|(_, d)| d.explored_depth > 0)
        .map(|(r, d)| format!("  {} ({}) — depth {}/{} {}",
            d.name, r.name, d.explored_depth, d.max_depth,
            if d.is_cleared { "CLEARED" } else { "" }))
        .collect();
    if !explored.is_empty() {
        println!("\nDUNGEONS EXPLORED:");
        for d in &explored { println!("{}", d); }
    }

    // --- Haunted sites ---
    let hauntings: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| e.text.contains("haunted"))
        .collect();
    if !hauntings.is_empty() {
        println!("\nHAUNTED SITES:");
        for h in &hauntings { println!("  [tick {}] {}", h.tick, h.text); }
    }

    // --- Most dramatic chronicle entries ---
    println!("\nMOST NOTABLE EVENTS:");
    // Prioritize: prophecies, legends, wars, betrayals, weddings, births, named monsters.
    let notable_keywords = ["PROPHECY", "LEGEND", "WAR!", "PEACE!", "betrayed", "wed", "born",
        "earned the name", "haunted", "founded", "Oathkeeper", "succession"];
    let mut notable: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| notable_keywords.iter().any(|k| e.text.contains(k)))
        .collect();
    notable.sort_by_key(|e| e.tick);
    notable.dedup_by_key(|e| e.tick); // one per tick
    for entry in notable.iter().take(20) {
        println!("  [tick {}] {}", entry.tick, entry.text);
    }

    println!();
}

fn print_creation_myth(state: &WorldState, seed: u64) {
    let h = |salt: u64| -> usize {
        ((seed.wrapping_mul(6364136223846793005).wrapping_add(salt)) >> 33) as usize
    };

    // Pick narrative elements from the world's initial conditions.
    let dominant_terrain = {
        let mut counts = std::collections::HashMap::new();
        for r in &state.regions { *counts.entry(r.terrain.name()).or_insert(0) += 1; }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(t, _)| t).unwrap_or("unknown")
    };

    let num_factions = state.factions.len();
    let has_ocean = state.regions.iter().any(|r| r.terrain == Terrain::DeepOcean);
    let has_volcano = state.regions.iter().any(|r| r.terrain == Terrain::Volcano);
    let has_ruins = state.regions.iter().any(|r| r.terrain == Terrain::AncientRuins);
    let has_death_zone = state.regions.iter().any(|r| r.terrain == Terrain::DeathZone);
    let num_rivers = state.regions.iter().filter(|r| r.has_river).count();

    let origin = match h(1) % 5 {
        0 => "In the beginning, the world was forged from silence and stone.",
        1 => "The gods shaped the land from their dreams, and life followed.",
        2 => "From the void came fire, and from fire came earth, and from earth came everything.",
        3 => "The world was born from the collision of two great spirits — one of order, one of chaos.",
        _ => "Before memory, before names, the land simply was — waiting for those who would give it meaning.",
    };

    let landscape = match dominant_terrain {
        "Plains" => "Vast grasslands stretch to every horizon, dotted with settlements of ambitious souls.",
        "Forest" => "Ancient forests blanket the world, their canopies hiding both beauty and danger.",
        "Mountains" => "Great mountain ranges carve the sky, their peaks hiding treasures and terrors.",
        "Jungle" => "Thick jungle covers the land, where temples lie hidden and the air hums with life.",
        "Desert" => "Sun-scorched deserts dominate, where only the resilient survive.",
        "Coast" => "The sea defines this world — coastal settlements cling to the shore, watching the deep.",
        _ => "The land is varied and wild, each region a world unto itself.",
    };

    let conflict = match h(3) % 4 {
        0 => format!("{} factions rose to claim dominion, and the struggle began.", num_factions),
        1 => format!("From {} settlements, {} powers emerged — each with ambitions that would reshape the world.", state.settlements.len(), num_factions),
        2 => format!("{} banners were raised. Peace was never more than a season away from breaking.", num_factions),
        _ => format!("The people divided into {} factions, each certain that their way was the only way.", num_factions),
    };

    let special = if has_death_zone {
        " In the cursed lands, something ancient and terrible stirs."
    } else if has_volcano {
        " The volcanos rumble with a fury older than memory."
    } else if has_ruins {
        " Ruins of a forgotten civilization dot the landscape, holding secrets of the past."
    } else if has_ocean {
        " Beyond the shore, the deep ocean conceals monsters that sailors dare not name."
    } else if num_rivers > 0 {
        " Rivers carve through the heartland, giving life and dividing nations."
    } else {
        ""
    };

    let prophecy_hint = if !state.prophecies.is_empty() {
        format!(" The seers speak of {} prophecies yet to be fulfilled.", state.prophecies.iter().filter(|p| !p.fulfilled).count())
    } else {
        String::new()
    };

    println!("--- Creation Myth ---");
    println!("{} {} {} {}{}\n", origin, landscape, conflict, special, prophecy_hint);
}

fn export_history_markdown(state: &WorldState, path: &str) {
    let mut out = String::new();

    let years = state.tick / TICKS_PER_YEAR;
    let pop = state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
    let monsters = state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();

    out.push_str(&format!("# World History\n\n"));
    out.push_str(&format!("*{} years have passed. {} souls endure across {} settlements. {} monsters roam the wilds.*\n\n",
        years, pop, state.settlements.len(), monsters));

    // Terrain
    out.push_str("## Geography\n\n");
    for terrain in Terrain::ALL {
        let regions: Vec<&str> = state.regions.iter()
            .filter(|r| r.terrain == terrain)
            .map(|r| r.name.as_str())
            .collect();
        if !regions.is_empty() {
            out.push_str(&format!("- **{}**: {}\n", terrain, regions.join(", ")));
        }
    }

    // Chokepoints
    let chokepoints: Vec<&str> = state.regions.iter()
        .filter(|r| r.is_chokepoint)
        .map(|r| r.name.as_str())
        .collect();
    if !chokepoints.is_empty() {
        out.push_str(&format!("\n**Strategic chokepoints:** {}\n", chokepoints.join(", ")));
    }

    // Legends
    let legends: Vec<String> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| n.name.contains("the Legendary")).unwrap_or(false))
        .map(|e| {
            let npc = e.npc.as_ref().unwrap();
            let classes = npc.classes.iter()
                .map(|c| format!("{} L{}", c.display_name, c.level))
                .collect::<Vec<_>>().join(", ");
            format!("- **{}** — [{}]", npc.name, classes)
        })
        .collect();
    if !legends.is_empty() {
        out.push_str("\n## Living Legends\n\n");
        for l in &legends { out.push_str(&format!("{}\n", l)); }
    }

    // Families
    let married = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| n.spouse_id.is_some()).unwrap_or(false))
        .count();
    let children = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| !n.parents.is_empty()).unwrap_or(false))
        .count();
    out.push_str(&format!("\n## Society\n\n"));
    out.push_str(&format!("- **{} marriages**, **{} children** born\n", married / 2, children));

    // Settlements
    out.push_str("\n## Settlements\n\n");
    for s in &state.settlements {
        let pop = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc
                && e.npc.as_ref().map(|n| n.home_settlement_id == Some(s.id)).unwrap_or(false))
            .count();
        let status = if pop == 0 { " *(depopulated)*" } else { "" };
        out.push_str(&format!("- **{}** ({}) — {} NPCs, {:.0} treasury{}\n",
            s.name, s.specialty, pop, s.treasury, status));
    }

    // Prophecies
    out.push_str("\n## Prophecies\n\n");
    for p in &state.prophecies {
        let status = if p.fulfilled { "**FULFILLED**" } else { "*unfulfilled*" };
        out.push_str(&format!("- *\"{}\"* — {}\n", p.text, status));
    }

    // Notable chronicle
    let notable_keywords = ["PROPHECY", "LEGEND", "WAR!", "PEACE!", "betrayed", "wed", "born",
        "earned the name", "haunted", "founded", "Oathkeeper", "succession", "DIVINE"];
    let notable: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| notable_keywords.iter().any(|k| e.text.contains(k)))
        .collect();
    if !notable.is_empty() {
        out.push_str("\n## Chronicle\n\n");
        for entry in notable.iter().take(30) {
            let year = entry.tick / TICKS_PER_YEAR;
            out.push_str(&format!("- **Year {}:** {}\n", year, entry.text));
        }
    }

    // Top survivors
    out.push_str("\n## Notable Figures\n\n");
    let mut survivors: Vec<&Entity> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
        .collect();
    survivors.sort_by(|a, b| b.level.cmp(&a.level));
    for e in survivors.iter().take(5) {
        let npc = e.npc.as_ref().unwrap();
        let classes = npc.classes.iter()
            .map(|c| format!("{} L{}", c.display_name, c.level))
            .collect::<Vec<_>>().join(", ");
        let friend_deaths = npc.memory.events.iter()
            .filter(|ev| matches!(ev.event_type, MemEventType::FriendDied(_)))
            .count();
        out.push_str(&format!("### {} (Level {})\n\n", npc.name, e.level));
        out.push_str(&format!("*Classes:* {}\n\n", classes));
        out.push_str(&format!("*{} friend deaths witnessed. Compassion: {:.2}. Risk tolerance: {:.2}.*\n\n",
            friend_deaths, npc.personality.compassion, npc.personality.risk_tolerance));
        if let Some(sid) = npc.spouse_id {
            let spouse_name = state.entities.iter()
                .find(|e2| e2.id == sid)
                .and_then(|e2| e2.npc.as_ref())
                .map(|n| n.name.as_str())
                .unwrap_or("(deceased)");
            out.push_str(&format!("*Married to {}.*\n\n", spouse_name));
        }
    }

    // Write file
    match std::fs::write(path, &out) {
        Ok(_) => println!("\nHistory exported to {}", path),
        Err(e) => eprintln!("Failed to export history: {}", e),
    }
}

fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else {
        let end = s.char_indices().take(max).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(max);
        &s[..end]
    }
}

fn print_warm_summary(state: &WorldState) {
    println!("\n=== Warm-Up Summary ===");

    // --- Settlements ranked by prosperity ---
    println!("\n--- Settlements by prosperity ---");
    let mut settlements: Vec<&SettlementState> = state.settlements.iter().collect();
    // Prosperity = treasury + population * 10 (rough heuristic)
    settlements.sort_by(|a, b| {
        let pa = a.treasury + a.population as f32 * 10.0;
        let pb = b.treasury + b.population as f32 * 10.0;
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, s) in settlements.iter().enumerate() {
        let prosperity = s.treasury + s.population as f32 * 10.0;
        let bld_info = s.city_grid_idx
            .and_then(|idx| state.city_grids.get(idx))
            .map(|g| {
                let counts = g.building_counts();
                let total: u32 = counts.iter().sum();
                format!(" bld:{}", total)
            })
            .unwrap_or_default();
        println!("  {}. {} ({}) -- pop:{} ${:.0}{} prosperity:{:.0}",
            rank + 1, s.name, s.specialty, s.population, s.treasury, bld_info, prosperity);
    }

    // --- Top 5 notable NPCs ---
    println!("\n--- Top 5 notable NPCs ---");
    let mut npc_entries: Vec<(u32, &bevy_game::world_sim::state::Entity)> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
        .map(|e| (e.id, e))
        .collect();
    // Sort by (level DESC, num_classes DESC)
    npc_entries.sort_by(|a, b| {
        let a_npc = a.1.npc.as_ref().unwrap();
        let b_npc = b.1.npc.as_ref().unwrap();
        let level_cmp = b.1.level.cmp(&a.1.level);
        if level_cmp != std::cmp::Ordering::Equal {
            return level_cmp;
        }
        b_npc.classes.len().cmp(&a_npc.classes.len())
    });

    let class_name_table: &[(u32, &str)] = &[
        (bevy_game::world_sim::state::tag(b"Warrior"), "Warrior"),
        (bevy_game::world_sim::state::tag(b"Ranger"), "Ranger"),
        (bevy_game::world_sim::state::tag(b"Guardian"), "Guardian"),
        (bevy_game::world_sim::state::tag(b"Healer"), "Healer"),
        (bevy_game::world_sim::state::tag(b"Merchant"), "Merchant"),
        (bevy_game::world_sim::state::tag(b"Scholar"), "Scholar"),
        (bevy_game::world_sim::state::tag(b"Rogue"), "Rogue"),
        (bevy_game::world_sim::state::tag(b"Artisan"), "Artisan"),
        (bevy_game::world_sim::state::tag(b"Diplomat"), "Diplomat"),
        (bevy_game::world_sim::state::tag(b"Commander"), "Commander"),
        (bevy_game::world_sim::state::tag(b"Farmer"), "Farmer"),
        (bevy_game::world_sim::state::tag(b"Miner"), "Miner"),
        (bevy_game::world_sim::state::tag(b"Woodsman"), "Woodsman"),
        (bevy_game::world_sim::state::tag(b"Alchemist"), "Alchemist"),
        (bevy_game::world_sim::state::tag(b"Herbalist"), "Herbalist"),
        (bevy_game::world_sim::state::tag(b"Explorer"), "Explorer"),
        (bevy_game::world_sim::state::tag(b"Mentor"), "Mentor"),
    ];
    let resolve_class = |hash: u32| -> &str {
        class_name_table.iter().find(|(h, _)| *h == hash).map(|(_, n)| *n).unwrap_or("?")
    };

    for (_, entity) in npc_entries.iter().take(5) {
        let npc = entity.npc.as_ref().unwrap();
        let name = npc_display_name(npc, entity.id);
        let classes: Vec<String> = npc.classes.iter()
            .map(|c| {
                if c.display_name.is_empty() {
                    format!("{} L{}", resolve_class(c.class_name_hash), c.level)
                } else {
                    format!("{} L{}", c.display_name, c.level)
                }
            })
            .collect();
        let home = npc.home_settlement_id
            .and_then(|sid| state.settlements.iter().find(|s| s.id == sid))
            .map(|s| s.name.as_str())
            .unwrap_or("unknown");
        println!("  {} -- lv{} ({}) [{}] of {}",
            name, entity.level,
            if npc.archetype.is_empty() { "?" } else { &npc.archetype },
            if classes.is_empty() { "no classes".to_string() } else { classes.join(", ") },
            home,
        );
    }

    // --- Legends: NPCs most mentioned in the chronicle ---
    println!("\n--- Legends (most chronicled) ---");
    {
        let mut mention_counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for entry in &state.chronicle {
            for &eid in &entry.entity_ids {
                *mention_counts.entry(eid).or_default() += 1;
            }
        }
        let mut legends: Vec<(u32, u32)> = mention_counts.into_iter().collect();
        legends.sort_by(|a, b| b.1.cmp(&a.1));
        let mut printed = 0;
        for (eid, count) in &legends {
            if printed >= 5 { break; }
            if let Some(entity) = state.entity(*eid) {
                // Skip low-level monsters (not legendary).
                if entity.kind == EntityKind::Monster && entity.level < 10 { continue; }
                let name = bevy_game::world_sim::naming::entity_display_name(entity);
                let detail = if entity.alive {
                    let npc_info = entity.npc.as_ref().map(|n| {
                        format!("lv{} {} at {}", entity.level, n.archetype,
                            n.home_settlement_id.and_then(|sid| state.settlement(sid))
                                .map(|s| s.name.as_str()).unwrap_or("?"))
                    }).unwrap_or_else(|| format!("lv{} alive", entity.level));
                    npc_info
                } else {
                    "deceased".to_string()
                };
                // Build a short biography from their chronicle entries.
                let bio_entries: Vec<&str> = state.chronicle.iter()
                    .filter(|e| e.entity_ids.contains(eid) && e.category != ChronicleCategory::Death)
                    .map(|e| e.text.as_str())
                    .collect();
                let bio = if bio_entries.len() > 2 {
                    // Pick first and last non-death entries as biography bookends.
                    format!("First known: \"{}\". Last known: \"{}\"",
                        truncate_str(bio_entries[0], 60),
                        truncate_str(bio_entries[bio_entries.len() - 1], 60))
                } else if !bio_entries.is_empty() {
                    format!("Known for: \"{}\"", truncate_str(bio_entries[0], 80))
                } else {
                    String::new()
                };

                println!("  {} — {} mentions ({})", name, count, detail);
                if !bio.is_empty() {
                    println!("    {}", bio);
                }
                printed += 1;
            }
        }
    }

    // --- World Ages (named based on current state) ---
    println!("\n--- World Ages ---");
    {
        // Name the current era based on world state.
        let total_pop: u32 = state.settlements.iter().map(|s| s.population).sum();
        let max_faction_territory = state.factions.iter()
            .map(|f| state.settlements.iter().filter(|s| s.faction_id == Some(f.id)).count())
            .max().unwrap_or(0);
        let dominant_faction = state.factions.iter()
            .max_by_key(|f| state.settlements.iter().filter(|s| s.faction_id == Some(f.id)).count())
            .map(|f| f.name.as_str()).unwrap_or("none");
        let total_treasury: f32 = state.settlements.iter().map(|s| s.treasury).sum();
        let avg_threat: f32 = state.regions.iter().map(|r| r.threat_level).sum::<f32>()
            / state.regions.len().max(1) as f32;

        // Chronicle-based category counts for naming.
        let crisis_count = state.chronicle.iter().filter(|e| e.category == ChronicleCategory::Crisis).count();
        let battle_count = state.chronicle.iter().filter(|e| e.category == ChronicleCategory::Battle).count();
        let quest_count = state.chronicle.iter().filter(|e| e.category == ChronicleCategory::Quest).count();

        let era_name = if max_faction_territory >= 8 {
            format!("The Empire of {}", dominant_faction)
        } else if crisis_count > 10 {
            "The Age of Strife".to_string()
        } else if avg_threat > 30.0 {
            "The Dark Times".to_string()
        } else if battle_count > 300 {
            "The Age of War".to_string()
        } else if total_treasury > 50000.0 {
            "The Age of Prosperity".to_string()
        } else if quest_count > 10 {
            "The Age of Adventure".to_string()
        } else if total_pop > 2500 {
            "The Age of Expansion".to_string()
        } else {
            "The Age of Foundations".to_string()
        };

        let year = state.tick / (bevy_game::world_sim::systems::seasons::TICKS_PER_SEASON * 4) + 1;
        let season = bevy_game::world_sim::systems::seasons::current_season(state.tick);
        let season_name = match season {
            bevy_game::world_sim::systems::seasons::Season::Spring => "Spring",
            bevy_game::world_sim::systems::seasons::Season::Summer => "Summer",
            bevy_game::world_sim::systems::seasons::Season::Autumn => "Autumn",
            bevy_game::world_sim::systems::seasons::Season::Winter => "Winter",
        };
        println!("  Current era: {} (Year {}, {}, pop {}, avg threat {:.0})",
            era_name, year, season_name, total_pop, avg_threat);
    }

    // --- Faction power standings ---
    println!("\n--- Faction power standings ---");
    let mut factions: Vec<&bevy_game::world_sim::state::FactionState> = state.factions.iter().collect();
    factions.sort_by(|a, b| b.military_strength.partial_cmp(&a.military_strength).unwrap_or(std::cmp::Ordering::Equal));
    for f in &factions {
        let settlement_count = state.settlements.iter().filter(|s| s.faction_id == Some(f.id)).count();
        let stance = match f.diplomatic_stance {
            DiplomaticStance::Friendly => "friendly",
            DiplomaticStance::Neutral => "neutral",
            DiplomaticStance::Hostile => "hostile",
            DiplomaticStance::AtWar => "at war",
            DiplomaticStance::Coalition => "coalition",
        };
        println!("  {} -- mil:{:.0} treasury:{:.0} territory:{} stance:{} tech:{}",
            f.name, f.military_strength, f.treasury, settlement_count, stance, f.tech_level);
    }

    // --- Chronicle highlights (diverse, not just deaths) ---
    println!("\n--- Chronicle highlights ---");
    if state.chronicle.is_empty() {
        println!("  (no entries)");
    } else {
        // Pick the most recent entry from each non-Death category, then fill remaining
        // slots with the most recent entries overall. This ensures variety.
        use bevy_game::world_sim::state::ChronicleCategory;
        let categories = [
            ChronicleCategory::Crisis, ChronicleCategory::Quest,
            ChronicleCategory::Diplomacy, ChronicleCategory::Achievement,
            ChronicleCategory::Battle, ChronicleCategory::Discovery,
            ChronicleCategory::Economy, ChronicleCategory::Narrative,
        ];
        let mut highlights: Vec<&ChronicleEntry> = Vec::new();
        let mut used_indices = std::collections::HashSet::new();

        // One entry per non-Death category (most recent)
        for cat in &categories {
            if let Some((idx, entry)) = state.chronicle.iter().enumerate().rev()
                .find(|(_, e)| std::mem::discriminant(&e.category) == std::mem::discriminant(cat))
            {
                if used_indices.insert(idx) {
                    highlights.push(entry);
                }
            }
        }

        // Fill remaining slots with most recent entries (any category)
        let _remaining = 12usize.saturating_sub(highlights.len());
        for (idx, entry) in state.chronicle.iter().enumerate().rev() {
            if highlights.len() >= 12 { break; }
            if used_indices.insert(idx) {
                highlights.push(entry);
            }
        }

        // Sort by tick for display
        highlights.sort_by_key(|e| e.tick);
        for entry in &highlights {
            println!("  [tick {}] [{}] {}",
                entry.tick,
                chronicle_category_name(&entry.category),
                entry.text,
            );
        }
    }

    // --- Narrative history ---
    print_world_history(state);
}

/// Print a compact narrative history derived from the chronicle and world state.
fn print_world_history(state: &WorldState) {
    println!("\n--- World History ---");

    let year = state.tick / (bevy_game::world_sim::systems::seasons::TICKS_PER_SEASON * 4) + 1;

    // Count key events.
    let total_deaths = state.world_events.iter()
        .filter(|e| matches!(e, WorldEvent::EntityDied { .. }))
        .count();
    let conquests: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| e.category == ChronicleCategory::Crisis && e.text.contains("conquered"))
        .collect();
    let artifacts: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| e.category == ChronicleCategory::Discovery && e.text.contains("forged"))
        .collect();
    let coups = state.chronicle.iter()
        .filter(|e| e.category == ChronicleCategory::Crisis && e.text.contains("Coup"))
        .count();

    // Dominant faction.
    let dominant = state.factions.iter()
        .max_by_key(|f| state.settlements.iter().filter(|s| s.faction_id == Some(f.id)).count())
        .map(|f| (f.name.as_str(), state.settlements.iter().filter(|s| s.faction_id == Some(f.id)).count()));

    // Most prosperous settlement.
    let richest = state.settlements.iter()
        .max_by(|a, b| a.treasury.partial_cmp(&b.treasury).unwrap_or(std::cmp::Ordering::Equal));

    // Most populous settlement.
    let largest = state.settlements.iter()
        .max_by_key(|s| s.population);

    // Build narrative.
    print!("  In {} years of history, ", year);

    if let Some((name, count)) = dominant {
        if count >= 5 {
            print!("{} rose to dominate {} of {} settlements. ", name, count, state.settlements.len());
        } else {
            print!("no single power dominated the {} settlements. ", state.settlements.len());
        }
    }

    println!("{} souls perished in the struggles that shaped this world.", total_deaths);

    if !conquests.is_empty() {
        println!("  {} settlements changed hands through conquest.", conquests.len());
    }
    if !artifacts.is_empty() {
        let artifact_names: Vec<&str> = artifacts.iter().take(3)
            .filter_map(|e| e.text.split("forged ").nth(1))
            .map(|s| s.split(" at ").next().unwrap_or(s))
            .collect();
        if !artifact_names.is_empty() {
            println!("  {} artifacts were forged, including {}.", artifacts.len(),
                artifact_names.join(", "));
        }
    }
    if coups > 0 {
        println!("  {} coups overthrew sitting rulers.", coups);
    }
    if let Some(s) = richest {
        if s.treasury > 1000.0 {
            println!("  {} grew to be the wealthiest settlement ({:.0} gold).", s.name, s.treasury);
        }
    }
    if let Some(s) = largest {
        if s.population > 200 {
            println!("  {} became the most populous settlement ({} citizens).", s.name, s.population);
        }
    }

    // Nemesis threat.
    let nemeses: Vec<_> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Monster && e.level >= 20)
        .collect();
    if !nemeses.is_empty() {
        let top = nemeses.iter().max_by_key(|e| e.level).unwrap();
        let name = bevy_game::world_sim::naming::entity_display_name(top);
        println!("  {} (level {}) terrorizes the land as the greatest monster threat.",
            name, top.level);
    }
}

/// Print a summary of the generated world before simulation starts.
fn print_world_summary(state: &WorldState) {
    println!("--- Terrain distribution ---");
    for terrain in Terrain::ALL {
        let count = state.regions.iter().filter(|r| r.terrain == terrain).count();
        if count > 0 {
            let elev_info: Vec<String> = state.regions.iter()
                .filter(|r| r.terrain == terrain)
                .map(|r| {
                    let elev_name = match r.elevation {
                        0 => "sea",
                        1 => "low",
                        2 => "mid",
                        3 => "high",
                        4 => if r.is_floating { "sky" } else { "summit" },
                        _ => "?",
                    };
                    format!("{}({}{})", r.name, elev_name, r.sub_biome.suffix())
                })
                .collect();
            println!("  {}: {} region(s) — {}", terrain, count, elev_info.join(", "));
        }
    }

    // Water features.
    let river_regions = state.regions.iter().filter(|r| r.has_river).count();
    let lake_regions = state.regions.iter().filter(|r| r.has_lake).count();
    let coastal_regions = state.regions.iter().filter(|r| r.is_coastal).count();
    if river_regions > 0 || lake_regions > 0 || coastal_regions > 0 {
        println!("\n--- Water features ---");
        if river_regions > 0 {
            let names: Vec<&str> = state.regions.iter()
                .filter(|r| r.has_river)
                .map(|r| r.name.as_str())
                .collect();
            println!("  Rivers flow through: {}", names.join(" → "));
        }
        if lake_regions > 0 {
            let names: Vec<&str> = state.regions.iter()
                .filter(|r| r.has_lake)
                .map(|r| r.name.as_str())
                .collect();
            println!("  Lakes in: {}", names.join(", "));
        }
        println!("  Coastal regions: {}", coastal_regions);
    }

    // Region connections + chokepoints.
    let chokepoints: Vec<&str> = state.regions.iter()
        .filter(|r| r.is_chokepoint)
        .map(|r| r.name.as_str())
        .collect();
    if !chokepoints.is_empty() {
        println!("\n--- Strategic chokepoints ---");
        println!("  {}", chokepoints.join(", "));
    }

    // Dungeon sites.
    let total_dungeons: usize = state.regions.iter().map(|r| r.dungeon_sites.len()).sum();
    let cleared_dungeons: usize = state.regions.iter()
        .flat_map(|r| r.dungeon_sites.iter())
        .filter(|d| d.is_cleared).count();
    if total_dungeons > 0 {
        println!("\n--- Dungeons ---");
        println!("  {} total ({} cleared)", total_dungeons, cleared_dungeons);
        for region in &state.regions {
            for d in &region.dungeon_sites {
                let status = if d.is_cleared { "CLEARED" }
                    else if d.explored_depth > 0 { "explored" }
                    else { "unexplored" };
                println!("  {} ({}) — depth {}/{}, {}, threat {:.1}×",
                    d.name, region.name, d.explored_depth, d.max_depth, status, d.threat_mult);
            }
        }
    }

    println!("\n--- Faction territories ---");
    for faction in &state.factions {
        let settlements: Vec<&str> = state.settlements.iter()
            .filter(|s| s.faction_id == Some(faction.id))
            .map(|s| s.name.as_str())
            .collect();
        let stance = match faction.diplomatic_stance {
            DiplomaticStance::Friendly => "friendly",
            DiplomaticStance::Neutral => "neutral",
            DiplomaticStance::Hostile => "hostile",
            DiplomaticStance::AtWar => "at war",
            DiplomaticStance::Coalition => "coalition",
        };
        println!("  {} (id:{}, {}) — {} settlement(s): {}",
            faction.name, faction.id, stance,
            settlements.len(), settlements.join(", "));
    }

    println!("\n--- Settlement specialties ---");
    for spec in SettlementSpecialty::ALL {
        let count = state.settlements.iter().filter(|s| s.specialty == spec).count();
        if count > 0 {
            let names: Vec<&str> = state.settlements.iter()
                .filter(|s| s.specialty == spec)
                .map(|s| s.name.as_str())
                .collect();
            println!("  {}: {} — {}", spec, count, names.join(", "));
        }
    }

    if !state.trade_routes.is_empty() {
        println!("\n--- Trade routes ---");
        for route in &state.trade_routes { let (a, b) = (route.settlement_a, route.settlement_b);
            let name_a = state.settlements.iter().find(|s| s.id == a).map(|s| s.name.as_str()).unwrap_or("?");
            let name_b = state.settlements.iter().find(|s| s.id == b).map(|s| s.name.as_str()).unwrap_or("?");
            println!("  {} <-> {}", name_a, name_b);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Simple LCG helpers for deterministic world generation
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn lcg_u32(state: &mut u64) -> u32 {
    lcg_next(state);
    (*state >> 32) as u32
}

fn lcg_f32(state: &mut u64) -> f32 {
    lcg_u32(state) as f32 / u32::MAX as f32
}

fn lcg_range(state: &mut u64, lo: usize, hi: usize) -> usize {
    if hi <= lo { return lo; }
    lo + (lcg_u32(state) as usize % (hi - lo))
}

// ---------------------------------------------------------------------------
// Terrain assignment: simple 2D hash noise
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Perlin-style noise for terrain generation
// ---------------------------------------------------------------------------

/// Integer hash → float in [0, 1).
fn noise_hash(x: i32, y: i32, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(374761393)
        .wrapping_add((y as u64).wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(1013904223));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h = h ^ (h >> 16);
    (h & 0x7fffffff) as f32 / 0x7fffffff as f32
}

/// Smoothstep-interpolated 2D value noise.
fn value_noise(x: f32, y: f32, seed: u64) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = noise_hash(ix, iy, seed);
    let b = noise_hash(ix + 1, iy, seed);
    let c = noise_hash(ix, iy + 1, seed);
    let d = noise_hash(ix + 1, iy + 1, seed);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

/// Fractal Brownian Motion — multi-octave noise in [0, 1).
fn fbm(x: f32, y: f32, seed: u64, octaves: u32) -> f32 {
    let mut v = 0.0f32;
    let mut amp = 0.5;
    let mut freq = 1.0f32;
    for i in 0..octaves {
        v += amp * value_noise(x * freq, y * freq, seed.wrapping_add(i as u64 * 7919));
        amp *= 0.5;
        freq *= 2.03;
    }
    v
}

/// Assign terrain from continuous noise fields (temperature, moisture, elevation).
/// Spatially coherent — nearby regions get similar terrain.
fn assign_terrain(x: f32, y: f32, seed: u64) -> Terrain {
    // Three orthogonal noise fields at different scales
    let scale = 0.008; // world-scale frequency
    let temperature = fbm(x * scale, y * scale, seed, 5);                    // 0=cold, 1=hot
    let moisture    = fbm(x * scale, y * scale, seed.wrapping_add(99991), 5); // 0=dry, 1=wet
    let elevation   = fbm(x * scale, y * scale, seed.wrapping_add(77773), 4); // 0=low, 1=high

    // Small-scale detail noise for rare biome placement
    let detail = fbm(x * scale * 3.0, y * scale * 3.0, seed.wrapping_add(55537), 3);

    // Map (temperature, moisture, elevation) → terrain type
    if elevation > 0.75 {
        // High altitude
        if temperature < 0.3 { return Terrain::Glacier; }
        if detail > 0.85 { return Terrain::FlyingIslands; }
        return Terrain::Mountains;
    }
    if elevation < 0.15 {
        // Low altitude — water
        if moisture > 0.6 { return Terrain::DeepOcean; }
        if detail > 0.7 { return Terrain::CoralReef; }
        return Terrain::Coast;
    }

    // Mid elevations — climate-driven
    match (temperature > 0.5, moisture > 0.5) {
        (true, true) => {
            // Hot + wet
            if moisture > 0.7 { Terrain::Swamp }
            else { Terrain::Jungle }
        }
        (true, false) => {
            // Hot + dry
            if detail > 0.8 { Terrain::Volcano }
            else if moisture < 0.25 { Terrain::Desert }
            else { Terrain::Badlands }
        }
        (false, true) => {
            // Cold + wet
            if temperature < 0.25 { Terrain::Tundra }
            else { Terrain::Forest }
        }
        (false, false) => {
            // Cold + dry
            if detail > 0.85 { Terrain::AncientRuins }
            else if detail > 0.8 { Terrain::DeathZone }
            else if temperature < 0.3 { Terrain::Caverns }
            else { Terrain::Plains }
        }
    }
}

// ---------------------------------------------------------------------------
// Settlement specialty selection based on terrain
// ---------------------------------------------------------------------------

fn choose_specialty(terrain: Terrain, rng: &mut u64) -> SettlementSpecialty {
    // Terrain-weighted specialty selection.
    let candidates: &[SettlementSpecialty] = match terrain {
        Terrain::Plains => &[
            SettlementSpecialty::FarmingVillage,
            SettlementSpecialty::FarmingVillage,
            SettlementSpecialty::General,
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::Forest => &[
            SettlementSpecialty::General,
            SettlementSpecialty::CraftingGuild,
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::ScholarCity,
        ],
        Terrain::Mountains => &[
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::CraftingGuild,
        ],
        Terrain::Coast => &[
            SettlementSpecialty::PortTown,
            SettlementSpecialty::PortTown,
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::FarmingVillage,
        ],
        Terrain::Swamp => &[
            SettlementSpecialty::CraftingGuild,
            SettlementSpecialty::General,
            SettlementSpecialty::ScholarCity,
        ],
        Terrain::Desert => &[
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::ScholarCity,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::Tundra => &[
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::General,
            SettlementSpecialty::MiningTown,
        ],
        Terrain::Jungle => &[
            SettlementSpecialty::FarmingVillage,
            SettlementSpecialty::CraftingGuild,
            SettlementSpecialty::ScholarCity,
        ],
        Terrain::Caverns => &[
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::Badlands => &[
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::General,
        ],
        Terrain::Glacier => &[
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::AncientRuins => &[
            SettlementSpecialty::ScholarCity,
            SettlementSpecialty::MilitaryOutpost,
        ],
        // Non-settleable terrains — shouldn't be called, but provide fallback.
        Terrain::Volcano | Terrain::DeepOcean | Terrain::FlyingIslands
        | Terrain::DeathZone | Terrain::CoralReef => &[
            SettlementSpecialty::MilitaryOutpost,
        ],
    };
    let idx = lcg_range(rng, 0, candidates.len());
    candidates[idx]
}

// ---------------------------------------------------------------------------
// Faction name generation
// ---------------------------------------------------------------------------

const FACTION_PREFIXES: &[&str] = &[
    "Iron", "Silver", "Golden", "Shadow", "Storm", "Crimson", "Emerald", "Azure",
    "Obsidian", "Ivory", "Scarlet", "Jade", "Onyx", "Amber", "Cobalt", "Marble",
];

const FACTION_SUFFIXES: &[&str] = &[
    "Dominion", "Alliance", "Compact", "League", "Order", "Pact", "Realm", "Confederacy",
    "Republic", "Empire", "Syndicate", "Covenant", "Tribunal", "Council", "Principality",
];

fn generate_faction_name(rng: &mut u64) -> String {
    let prefix = FACTION_PREFIXES[lcg_range(rng, 0, FACTION_PREFIXES.len())];
    let suffix = FACTION_SUFFIXES[lcg_range(rng, 0, FACTION_SUFFIXES.len())];
    format!("{} {}", prefix, suffix)
}

// ---------------------------------------------------------------------------
// Settlement name generation
// ---------------------------------------------------------------------------

const SETTLEMENT_PREFIXES: &[&str] = &[
    "Oak", "Iron", "River", "Stone", "Green", "Red", "North", "South", "East", "West",
    "High", "Low", "Dark", "Bright", "Old", "New", "White", "Black", "Grey", "Blue",
    "Pine", "Elm", "Ash", "Birch", "Cedar", "Maple", "Willow", "Holly",
];

const SETTLEMENT_SUFFIXES: &[&str] = &[
    "haven", "ford", "bridge", "gate", "watch", "hold", "stead", "dale", "vale",
    "port", "keep", "wick", "bury", "moor", "fell", "marsh", "town", "field",
    "grove", "wood", "crest", "peak", "hollow", "bend", "crossing",
];

fn generate_settlement_name(rng: &mut u64) -> String {
    let prefix = SETTLEMENT_PREFIXES[lcg_range(rng, 0, SETTLEMENT_PREFIXES.len())];
    let suffix = SETTLEMENT_SUFFIXES[lcg_range(rng, 0, SETTLEMENT_SUFFIXES.len())];
    format!("{}{}", prefix, suffix)
}

// ---------------------------------------------------------------------------
// Region name generation
// ---------------------------------------------------------------------------

const REGION_NAMES: &[&str] = &[
    "Ashenvale", "Thornmarch", "Windrift", "Sunreach", "Mistwood", "Frostpeak",
    "Dusthollow", "Irondeep", "Greenmantle", "Stormveil", "Shadowfen", "Goldfields",
    "Silverbark", "Blackmoor", "Crystalshore", "Dragonspine", "Serpentine Reach",
    "Ember Wastes", "Coral Basin", "Howling Steppe", "Silent Vale", "Thundercleft",
    "Pilgrim's Rest", "Wraith Hollow", "Dawn Plateau", "Twilight Fen",
];

fn generate_region_name(idx: usize, rng: &mut u64) -> String {
    if idx < REGION_NAMES.len() {
        REGION_NAMES[idx].to_string()
    } else {
        // Fallback for large worlds.
        format!("Region_{}", lcg_u32(rng) % 10000)
    }
}

// ---------------------------------------------------------------------------
// Archetype selection helpers
// ---------------------------------------------------------------------------

const MIXED_ARCHETYPES: &[&str] = &[
    "warrior", "ranger", "mage", "cleric", "rogue", "merchant", "farmer", "smith",
];

/// Pick an archetype from the weighted distribution for a settlement specialty.
///
/// Uses the `weighted_archetypes()` cumulative table. Rolls 0..99;
/// if the roll exceeds all entries, falls through to the mixed pool.
fn pick_weighted_archetype(specialty: SettlementSpecialty, rng: &mut u64) -> &'static str {
    let roll = lcg_u32(rng) % 100;
    let table = specialty.weighted_archetypes();
    for &(archetype, cumulative) in table {
        if roll < cumulative {
            return archetype;
        }
    }
    // Mixed pool fallback
    MIXED_ARCHETYPES[lcg_range(rng, 0, MIXED_ARCHETYPES.len())]
}

// ---------------------------------------------------------------------------
// build_world — terrain-aware, faction-controlled world generation
// ---------------------------------------------------------------------------

/// Grid layout dimensions shared across build_world helpers.
struct GridLayout {
    cols: usize,
    rows: usize,
    spacing: f32,
}

/// Peaceful mode: single forest settlement, no monsters, no gold.
/// NPCs start with nothing and must gather + build from scratch.
fn build_peaceful_world(args: &WorldSimArgs) -> WorldState {
    let mut state = WorldState::new(args.seed);
    let mut rng = args.seed;
    let npcs = args.entities.min(50); // small village

    // Single forest region
    state.regions.push(RegionState {
        id: 0,
        name: "Greenwood".into(),
        terrain: Terrain::Forest,
        pos: (100.0, 100.0),
        monster_density: 0.0,
        faction_id: Some(0),
        threat_level: 0.0,
        has_river: true,
        has_lake: false,
        is_coastal: false,
        river_connections: vec![],
        dungeon_sites: vec![],
        sub_biome: SubBiome::Standard,
        neighbors: vec![],
        is_chokepoint: false,
        elevation: 1,
        is_floating: false,
        unrest: 0.0,
        control: 1.0,
    });

    // Single faction
    state.factions.push(FactionState {
        id: 0,
        name: "Forest Folk".into(),
        diplomatic_stance: DiplomaticStance::Friendly,
        military_strength: 0.0,
        max_military_strength: 10.0,
        treasury: 0.0, // no gold!
        territory_size: 1,
        relationship_to_guild: 0.0,
        at_war_with: vec![],
        coup_risk: 0.0,
        escalation_level: 0,
        tech_level: 0,
        recent_actions: vec![],
    });

    // Single settlement — zero treasury, zero stockpile
    let mut settlement = SettlementState::new(0, "Willowgrove".into(), (100.0, 100.0));
    settlement.faction_id = Some(0);
    settlement.treasury = 0.0; // no gold
    // Zero stockpile — NPCs must gather everything
    for c in settlement.stockpile.iter_mut() { *c = 0.0; }

    let grid = LocalGrid {
        id: 0,
        fidelity: bevy_game::world_sim::fidelity::Fidelity::Medium,
        center: settlement.pos,
        radius: 30.0,
        entity_ids: Vec::new(),
    };
    state.grids.push(grid);
    settlement.grid_id = Some(0);
    state.settlements.push(settlement);

    // NPCs — zero gold, no equipment, just people
    let mut id = 0u32;
    let archetypes = ["farmer", "woodcutter", "miner", "herbalist", "builder",
                      "smith", "hunter", "healer", "merchant", "scholar"];
    for i in 0..npcs {
        let angle = (i as f32 / npcs as f32) * std::f32::consts::TAU;
        let dist = 5.0 + lcg_f32(&mut rng) * 15.0;
        let px = 100.0 + angle.cos() * dist;
        let py = 100.0 + angle.sin() * dist;

        let mut npc = Entity::new_npc(id, (px, py));
        npc.grid_id = Some(0);
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(0);
        npc_data.faction_id = Some(0);
        npc_data.gold = 0.0; // no gold!
        npc_data.morale = 60.0;
        npc_data.archetype = archetypes[i % archetypes.len()].to_string();
        npc_data.name = bevy_game::world_sim::naming::generate_personal_name(id, args.seed);
        npc.inventory = Some(bevy_game::world_sim::state::Inventory::default());

        state.entities.push(npc);
        id += 1;
    }

    state.next_id = id;
    state
}

fn build_world(args: &WorldSimArgs) -> WorldState {
    let mut state = WorldState::new(args.seed);
    let npcs = args.entities.saturating_sub(args.monsters);
    let num_factions = args.factions.max(1);

    let terrain_seed = args.terrain_seed.unwrap_or(args.seed);
    let mut rng = terrain_seed;

    let cols = (args.settlements as f32).sqrt().ceil() as usize;
    let rows = (args.settlements + cols - 1) / cols;
    let spacing = if args.rich { 200.0 } else { 100.0 };
    let layout = GridLayout { cols, rows, spacing };

    create_regions(&mut state, &layout, &mut rng);
    create_water_features(&mut state, &layout);
    assign_elevations(&mut state);
    assign_sub_biomes(&mut state);
    build_adjacency_graph(&mut state, &layout);
    create_dungeon_sites(&mut state, &layout);
    create_factions(&mut state, num_factions, &mut rng);
    create_settlements(&mut state, args, &layout, num_factions, npcs, &mut rng);
    // Update faction territory sizes now that settlements are assigned.
    for faction in &mut state.factions {
        faction.territory_size = state.settlements.iter()
            .filter(|s| s.faction_id == Some(faction.id))
            .count() as u32;
    }
    create_trade_routes(&mut state, &layout, args.trade_routes);
    let mut next_id = populate_npcs(&mut state, args, npcs, &mut rng);
    next_id = spawn_monsters(&mut state, &layout, args, next_id, &mut rng);
    spawn_sea_monsters(&mut state, &layout, next_id, &mut rng);

    state
}

// ---------------------------------------------------------------------------
// build_world helpers — one per generation phase
// ---------------------------------------------------------------------------

/// Phase 1: Create regions with terrain types.
fn create_regions(state: &mut WorldState, layout: &GridLayout, rng: &mut u64) {
    let num_regions = layout.cols * layout.rows;
    let terrain_seed = *rng;
    for i in 0..num_regions {
        let row = i / layout.cols;
        let col = i % layout.cols;
        // Jittered position — break the grid for organic Voronoi boundaries
        let base_x = col as f32 * layout.spacing + layout.spacing * 0.5;
        let base_y = row as f32 * layout.spacing + layout.spacing * 0.5;
        let jitter = layout.spacing * 0.35;
        let jx = (lcg_f32(rng) - 0.5) * 2.0 * jitter;
        let jy = (lcg_f32(rng) - 0.5) * 2.0 * jitter;
        let pos = (base_x + jx, base_y + jy);
        // Terrain from continuous noise fields — nearby regions get similar biomes
        let terrain = assign_terrain(pos.0, pos.1, terrain_seed);
        state.regions.push(RegionState {
            id: i as u32,
            name: generate_region_name(i, rng),
            terrain,
            pos,
            monster_density: terrain.threat_multiplier()
                * Terrain::elevation_threat_mult(terrain.base_elevation())
                * (0.1 + lcg_f32(rng) * 0.3),
            faction_id: None,
            threat_level: (terrain.threat_multiplier() - 1.0).max(0.0) * 0.2
                + (terrain.base_elevation() as f32 - 1.0).max(0.0) * 0.1,
            has_river: false,
            has_lake: false,
            is_coastal: false,
            river_connections: Vec::new(),
            dungeon_sites: Vec::new(),
            sub_biome: SubBiome::Standard,
            neighbors: Vec::new(),
            is_chokepoint: false,
            elevation: terrain.base_elevation(),
            is_floating: terrain == Terrain::FlyingIslands,
            unrest: lcg_f32(rng) * 0.2,
            control: 0.5 + lcg_f32(rng) * 0.5,
        });
    }
}

/// Phase 1b: Generate water features — rivers, lakes, ocean borders.
fn create_water_features(state: &mut WorldState, layout: &GridLayout) {
    let GridLayout { cols, rows, .. } = *layout;
    let num_regions = state.regions.len();

    // Ocean borders: edge regions and DeepOcean regions mark neighbors as coastal.
    for i in 0..num_regions {
        let row = i / cols;
        let col = i % cols;
        let is_edge = row == 0 || col == 0 || row >= rows - 1 || col >= cols - 1;
        let is_ocean = state.regions[i].terrain == Terrain::DeepOcean
            || state.regions[i].terrain == Terrain::CoralReef;

        if is_edge || is_ocean {
            // Mark self and neighbors as coastal.
            if state.regions[i].terrain == Terrain::Coast || is_edge {
                state.regions[i].is_coastal = true;
            }
            for &(dr, dc) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = row as i32 + dr;
                let nc = col as i32 + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    let ni = nr as usize * cols + nc as usize;
                    if ni < num_regions {
                        state.regions[ni].is_coastal = true;
                    }
                }
            }
        }
    }

    // Rivers: start from Mountains/Glacier, flow downhill toward Coast/Plains.
    // Each river is a chain of connected regions.
    let mut river_count = 0;
    for i in 0..num_regions {
        if river_count >= 3 { break; } // max 3 rivers
        let terrain = state.regions[i].terrain;
        if !matches!(terrain, Terrain::Mountains | Terrain::Glacier | Terrain::Volcano) { continue; }
        // Deterministic: only some mountain regions spawn rivers.
        let roll = (i as u32).wrapping_mul(2654435761).wrapping_add(0x81B3);
        if roll % 3 != 0 { continue; }

        // Flow downhill: prefer Plains, Coast, Forest, Swamp.
        let mut current = i;
        let mut path = vec![i];
        for _ in 0..6 { // max river length
            let crow = current / cols;
            let ccol = current % cols;
            let mut best_next = None;
            let mut best_score = 0.0f32;
            for &(dr, dc) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = crow as i32 + dr;
                let nc = ccol as i32 + dc;
                if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols { continue; }
                let ni = nr as usize * cols + nc as usize;
                if ni >= num_regions || path.contains(&ni) { continue; }
                // Score: prefer lower-threat (lower elevation proxy).
                let score = 1.0 / state.regions[ni].terrain.threat_multiplier()
                    + if state.regions[ni].terrain == Terrain::Coast { 2.0 } else { 0.0 }
                    + if state.regions[ni].terrain == Terrain::Plains { 1.0 } else { 0.0 };
                if score > best_score {
                    best_score = score;
                    best_next = Some(ni);
                }
            }
            match best_next {
                Some(ni) => {
                    path.push(ni);
                    current = ni;
                    if state.regions[ni].is_coastal { break; } // reached the sea
                }
                None => break,
            }
        }

        if path.len() >= 2 {
            // Mark river regions and connections.
            for wi in 0..path.len() {
                let ri = path[wi];
                state.regions[ri].has_river = true;
                if wi + 1 < path.len() {
                    let next_id = state.regions[path[wi + 1]].id;
                    state.regions[ri].river_connections.push(next_id);
                }
                if wi > 0 {
                    let prev_id = state.regions[path[wi - 1]].id;
                    state.regions[ri].river_connections.push(prev_id);
                }
            }
            river_count += 1;
        }
    }

    // Lakes: valleys between mountains (non-mountain regions adjacent to 2+ mountain regions).
    for i in 0..num_regions {
        let row = i / cols;
        let col = i % cols;
        if matches!(state.regions[i].terrain, Terrain::Mountains | Terrain::DeepOcean | Terrain::Volcano) {
            continue;
        }
        let mut mountain_neighbors = 0;
        for &(dr, dc) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                let ni = nr as usize * cols + nc as usize;
                if ni < num_regions && matches!(state.regions[ni].terrain,
                    Terrain::Mountains | Terrain::Glacier) {
                    mountain_neighbors += 1;
                }
            }
        }
        if mountain_neighbors >= 2 {
            state.regions[i].has_lake = true;
            // Lakes boost food production.
        }
    }
}

/// Phase 1c: Elevation variation — Mountains/Glacier get random height tiers.
fn assign_elevations(state: &mut WorldState) {
    for i in 0..state.regions.len() {
        let terrain = state.regions[i].terrain;
        let base = terrain.base_elevation();
        let variation: u8 = match terrain {
            Terrain::Mountains => {
                // Mountains vary: foothills(2), highlands(2), peaks(3), summit(3-4).
                let h = (i as u32).wrapping_mul(2654435761);
                match h % 4 {
                    0 => 2, // foothills
                    1 => 2, // highlands
                    2 => 3, // peaks
                    _ => { if h % 7 == 0 { 4 } else { 3 } } // rare summit
                }
            }
            Terrain::Glacier => {
                let h = (i as u32).wrapping_mul(1103515245);
                if h % 3 == 0 { 4 } else { 3 } // some glacier summits
            }
            Terrain::Volcano => 3, // always peaks
            _ => base,
        };
        state.regions[i].elevation = variation;
    }
}

/// Phase 1d: Sub-biome assignment — terrain variants for richer geography.
fn assign_sub_biomes(state: &mut WorldState) {
    for i in 0..state.regions.len() {
        let terrain = state.regions[i].terrain;
        let h = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
        state.regions[i].sub_biome = match terrain {
            Terrain::Forest => match h % 10 {
                0..=3 => SubBiome::LightForest,    // 40%
                4..=6 => SubBiome::DenseForest,    // 30%
                _ => SubBiome::AncientForest,      // 30%
            },
            Terrain::Desert => match h % 3 {
                0 => SubBiome::SandDunes,
                _ => SubBiome::RockyDesert,
            },
            Terrain::Mountains if state.regions[i].elevation >= 3 => {
                if h % 5 == 0 { SubBiome::HotSprings } else { SubBiome::Standard }
            },
            Terrain::Swamp => {
                if h % 4 == 0 { SubBiome::GlowingMarsh } else { SubBiome::Standard }
            },
            Terrain::Jungle => {
                if h % 3 == 0 { SubBiome::TempleJungle } else { SubBiome::Standard }
            },
            _ => SubBiome::Standard,
        };
    }
}

/// Phase 1e: Build region adjacency graph + detect chokepoints.
fn build_adjacency_graph(state: &mut WorldState, layout: &GridLayout) {
    let GridLayout { cols, rows, .. } = *layout;
    let num_regions = state.regions.len();
    for i in 0..num_regions {
        let row = i / cols;
        let col = i % cols;
        let mut neighbors = Vec::new();

        // 4-connected neighbors (cardinal directions).
        for &(dr, dc) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                let ni = nr as usize * cols + nc as usize;
                if ni < num_regions {
                    neighbors.push(state.regions[ni].id);
                }
            }
        }

        state.regions[i].neighbors = neighbors;
    }

    // Detect chokepoints: regions with <=2 passable neighbors.
    for i in 0..num_regions {
        let passable_neighbors = state.regions[i].neighbors.iter()
            .filter(|&&nid| {
                state.regions.iter()
                    .find(|r| r.id == nid)
                    .map(|r| r.terrain.travel_speed() > 0.0)
                    .unwrap_or(false)
            })
            .count();
        // A chokepoint is a passable region with only 1-2 passable neighbors.
        let is_passable = state.regions[i].terrain.travel_speed() > 0.0;
        state.regions[i].is_chokepoint = is_passable && passable_neighbors <= 2
            && passable_neighbors >= 1;
    }
}

/// Phase 1f: Generate dungeon sites in appropriate regions.
fn create_dungeon_sites(state: &mut WorldState, layout: &GridLayout) {
    let GridLayout { cols, spacing, .. } = *layout;
    let dungeon_prefixes = ["Sunken", "Lost", "Forgotten", "Ancient", "Cursed", "Hidden", "Ruined", "Shadowed"];
    let dungeon_suffixes = ["Halls", "Depths", "Catacombs", "Labyrinth", "Sanctum", "Vault", "Crypts", "Cavern"];

    for i in 0..state.regions.len() {
        let terrain = state.regions[i].terrain;
        if !terrain.has_dungeons() { continue; }

        // Number of dungeons based on terrain.
        let num_dungeons = match terrain {
            Terrain::AncientRuins => 2,
            Terrain::Caverns => 2,
            Terrain::Volcano => 1,
            Terrain::DeathZone => 1,
            _ => 0,
        };

        let row = i / cols;
        let col = i % cols;
        let region_center = (col as f32 * spacing, row as f32 * spacing);

        for d in 0..num_dungeons {
            let h = (i as u32).wrapping_mul(2654435761).wrapping_add(d as u32);
            let jx = (h % 100) as f32 * 0.4 - 20.0;
            let jy = ((h / 100) % 100) as f32 * 0.4 - 20.0;
            let pos = (region_center.0 + jx, region_center.1 + jy);

            let max_depth = match terrain {
                Terrain::AncientRuins => 3 + (h % 3) as u8,  // 3-5 floors
                Terrain::Caverns => 4 + (h % 4) as u8,       // 4-7 floors
                Terrain::Volcano => 2 + (h % 2) as u8,       // 2-3 floors
                Terrain::DeathZone => 5 + (h % 3) as u8,     // 5-7 floors
                _ => 3,
            };

            let prefix = dungeon_prefixes[(h as usize / 10) % dungeon_prefixes.len()];
            let suffix = dungeon_suffixes[(h as usize / 100) % dungeon_suffixes.len()];
            let name = format!("The {} {}", prefix, suffix);

            state.regions[i].dungeon_sites.push(DungeonSite {
                pos,
                name,
                explored_depth: 0,
                max_depth,
                is_cleared: false,
                last_explored_tick: 0,
                threat_mult: terrain.threat_multiplier(),
            });
        }
    }
}

/// Phase 2: Create factions with diplomatic stances and initial wars.
fn create_factions(state: &mut WorldState, num_factions: usize, rng: &mut u64) {
    let stances = [
        DiplomaticStance::Friendly,
        DiplomaticStance::Neutral,
        DiplomaticStance::Neutral,
        DiplomaticStance::Hostile,
        DiplomaticStance::Neutral,
        DiplomaticStance::Coalition,
    ];

    for f in 0..num_factions {
        let stance = stances[f % stances.len()];
        let relationship = match stance {
            DiplomaticStance::Friendly => 30.0 + lcg_f32(rng) * 40.0,
            DiplomaticStance::Hostile => -60.0 + lcg_f32(rng) * 30.0,
            DiplomaticStance::AtWar => -80.0 + lcg_f32(rng) * 20.0,
            DiplomaticStance::Coalition => 50.0 + lcg_f32(rng) * 30.0,
            DiplomaticStance::Neutral => -10.0 + lcg_f32(rng) * 20.0,
        };
        state.factions.push(FactionState {
            id: f as u32,
            name: generate_faction_name(rng),
            relationship_to_guild: relationship,
            military_strength: 20.0 + lcg_f32(rng) * 80.0,
            max_military_strength: 100.0,
            territory_size: 0, // computed after assignment
            diplomatic_stance: stance,
            treasury: 500.0 + lcg_f32(rng) * 2000.0,
            at_war_with: Vec::new(),
            coup_risk: lcg_f32(rng) * 0.3,
            escalation_level: 0,
            tech_level: 1 + (lcg_u32(rng) % 3) as u32,
            recent_actions: Vec::new(),
        });
    }

    // Set some faction wars: hostile factions fight each other.
    let hostile_ids: Vec<u32> = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::Hostile)
        .map(|f| f.id)
        .collect();
    if hostile_ids.len() >= 2 {
        // First hostile faction is at war with the second.
        let a = hostile_ids[0];
        let b = hostile_ids[1 % hostile_ids.len()];
        if a != b {
            if let Some(fa) = state.faction_mut(a) { fa.at_war_with.push(b); }
            if let Some(fb) = state.faction_mut(b) { fb.at_war_with.push(a); }
        }
    }
}

/// Phase 3: Create settlements with terrain-aware specialties, city grids, and local grids.
fn create_settlements(
    state: &mut WorldState,
    args: &WorldSimArgs,
    layout: &GridLayout,
    num_factions: usize,
    npcs: usize,
    rng: &mut u64,
) {
    let GridLayout { cols, spacing, .. } = *layout;
    let rich = args.rich;

    let mut used_names = std::collections::HashSet::new();
    for i in 0..args.settlements {
        let row = i / cols;
        let col = i % cols;
        // Jitter settlement positions so the world isn't a perfect grid.
        let jx = (lcg_f32(rng) - 0.5) * spacing * 0.6;
        let jy = (lcg_f32(rng) - 0.5) * spacing * 0.6;
        let pos = (col as f32 * spacing + jx, row as f32 * spacing + jy);
        let region_idx = row * cols + col;
        let region_idx = region_idx.min(state.regions.len().saturating_sub(1));
        let terrain = state.regions[region_idx].terrain;

        // Skip non-settleable terrain — find nearest settleable region.
        let region_idx = if !terrain.is_settleable() {
            state.regions.iter().enumerate()
                .filter(|(_, r)| r.terrain.is_settleable())
                .min_by_key(|(ri, _)| ((*ri as i32 - region_idx as i32).abs()) as u32)
                .map(|(ri, _)| ri)
                .unwrap_or(region_idx)
        } else { region_idx };
        let terrain = state.regions[region_idx].terrain;

        // Generate a unique settlement name (retry on collision).
        let mut name = generate_settlement_name(rng);
        let mut attempts = 0;
        while used_names.contains(&name) && attempts < 50 {
            name = generate_settlement_name(rng);
            attempts += 1;
        }
        if used_names.contains(&name) {
            // Fallback: append index to guarantee uniqueness.
            name = format!("{} {}", name, i);
        }
        used_names.insert(name.clone());
        let mut settlement = SettlementState::new(i as u32, name, pos);

        // Assign specialty based on terrain.
        settlement.specialty = choose_specialty(terrain, rng);

        // Set context tags based on specialty (tag-based action system).
        settlement.context_tags = match settlement.specialty {
            SettlementSpecialty::MiningTown => vec![(tags::MINING, 0.3), (tags::ENDURANCE, 0.1)],
            SettlementSpecialty::TradeHub => vec![(tags::TRADE, 0.3), (tags::NEGOTIATION, 0.2)],
            SettlementSpecialty::MilitaryOutpost => vec![(tags::TACTICS, 0.2), (tags::DISCIPLINE, 0.1)],
            SettlementSpecialty::FarmingVillage => vec![(tags::FARMING, 0.3), (tags::LABOR, 0.2)],
            SettlementSpecialty::ScholarCity => vec![(tags::RESEARCH, 0.3), (tags::LORE, 0.2)],
            SettlementSpecialty::PortTown => vec![(tags::TRADE, 0.2), (tags::NAVIGATION, 0.2)],
            SettlementSpecialty::CraftingGuild => vec![(tags::CRAFTING, 0.3), (tags::SMITHING, 0.2)],
            SettlementSpecialty::General => vec![(tags::LABOR, 0.1)],
        };

        // Assign faction ownership: distribute settlements round-robin among factions.
        let faction_id = (i % num_factions) as u32;
        settlement.faction_id = Some(faction_id);

        // Set production rates based on terrain + specialty.
        // Base stockpile from terrain.
        let base_amount = if rich { 500.0 } else { 100.0 };
        for &(commodity, mult) in terrain.primary_commodities() {
            settlement.stockpile[commodity] = base_amount * mult;
        }
        // Specialty bonuses on top.
        for &(commodity, mult) in settlement.specialty.production_bonuses() {
            settlement.stockpile[commodity] += base_amount * (mult - 1.0);
        }

        // Set prices: cheap what you produce, expensive what you don't.
        for c in 0..NUM_COMMODITIES {
            let terrain_produces = terrain.primary_commodities().iter().any(|&(ci, _)| ci == c);
            let specialty_produces = settlement.specialty.production_bonuses().iter().any(|&(ci, _)| ci == c);
            settlement.prices[c] = if terrain_produces && specialty_produces {
                0.4 + lcg_f32(rng) * 0.2
            } else if terrain_produces || specialty_produces {
                0.7 + lcg_f32(rng) * 0.3
            } else {
                1.5 + lcg_f32(rng) * 1.0
            };
        }

        // Treasury based on specialty and terrain.
        settlement.treasury = match settlement.specialty {
            SettlementSpecialty::TradeHub | SettlementSpecialty::PortTown => {
                200.0 + lcg_f32(rng) * 500.0
            }
            SettlementSpecialty::MiningTown | SettlementSpecialty::CraftingGuild => {
                150.0 + lcg_f32(rng) * 300.0
            }
            _ => 50.0 + lcg_f32(rng) * 200.0,
        };
        if rich { settlement.treasury *= 3.0; }

        // Vary initial population by specialty — Trade Hubs and Farming Villages
        // are larger, Military Outposts are smaller, etc.
        let base_pop = (npcs / args.settlements) as f32;
        let pop_mult = match settlement.specialty {
            SettlementSpecialty::TradeHub => 1.4,
            SettlementSpecialty::FarmingVillage => 1.3,
            SettlementSpecialty::PortTown => 1.2,
            SettlementSpecialty::ScholarCity => 1.1,
            SettlementSpecialty::General => 1.0,
            SettlementSpecialty::CraftingGuild => 0.9,
            SettlementSpecialty::MiningTown => 0.8,
            SettlementSpecialty::MilitaryOutpost => 0.7,
        };
        // Add +-10% jitter so no two settlements are exactly the same.
        let jitter = 0.9 + lcg_f32(rng) * 0.2;
        settlement.population = (base_pop * pop_mult * jitter) as u32;

        // Also tag the region with the faction.
        if region_idx < state.regions.len() && state.regions[region_idx].faction_id.is_none() {
            state.regions[region_idx].faction_id = Some(faction_id);
        }

        state.settlements.push(settlement);

        // Create city grid for this settlement.
        let terrain_name = state.regions[region_idx].terrain.name();
        let grid = bevy_game::world_sim::city_grid::CityGrid::new(128, 128, i as u32, terrain_name, args.seed + i as u64);
        let influence = bevy_game::world_sim::city_grid::InfluenceMap::new(128, 128);
        let grid_idx = state.city_grids.len();
        state.city_grids.push(grid);
        state.influence_maps.push(influence);
        // Update the settlement with grid index.
        state.settlements.last_mut().unwrap().city_grid_idx = Some(grid_idx);

        state.grids.push(LocalGrid {
            id: i as u32,
            fidelity: Fidelity::Medium,
            center: pos,
            radius: if rich { 50.0 } else { 30.0 },
            entity_ids: Vec::new(),
        });
    }

}

/// Phase 4: Trade routes between nearby settlements.
fn create_trade_routes(state: &mut WorldState, layout: &GridLayout, enabled: bool) {
    if !enabled { return; }
    let max_dist = layout.spacing * 1.8; // connect settlements within ~2 grid cells
    for i in 0..state.settlements.len() {
        for j in (i + 1)..state.settlements.len() {
            let a = &state.settlements[i];
            let b = &state.settlements[j];
            let dx = a.pos.0 - b.pos.0;
            let dy = a.pos.1 - b.pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= max_dist {
                state.trade_routes.push(TradeRoute { settlement_a: a.id, settlement_b: b.id, established_tick: 0, total_profit: 0.0, trade_count: 0, strength: 0.5 });
            }
        }
    }
}

/// Phase 5: Distribute NPCs with specialty-aware archetypes. Returns next entity id.
fn populate_npcs(
    state: &mut WorldState,
    args: &WorldSimArgs,
    npcs: usize,
    rng: &mut u64,
) -> u32 {
    let rich = args.rich;
    let mut id = 0u32;

    for i in 0..npcs {
        let settlement_idx = i % args.settlements;
        let s = &state.settlements[settlement_idx];
        let specialty = s.specialty;
        let settlement_faction = s.faction_id;
        let settlement_treasury = s.treasury;

        let jitter_x = ((id as f32 * 7.13).sin()) * 5.0;
        let jitter_y = ((id as f32 * 3.71).cos()) * 5.0;

        let mut npc = Entity::new_npc(id, (s.pos.0 + jitter_x, s.pos.1 + jitter_y));
        npc.grid_id = Some(settlement_idx as u32);
        npc.level = 1 + (id % 10) as u32;

        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.name = bevy_game::world_sim::naming::generate_personal_name(id, args.seed);
        npc_data.home_settlement_id = Some(settlement_idx as u32);
        npc_data.faction_id = settlement_faction;

        // Choose archetype from weighted distribution matching settlement specialty.
        npc_data.archetype = pick_weighted_archetype(specialty, rng).to_string();

        // Set production based on terrain region.
        let region_idx = settlement_idx.min(state.regions.len().saturating_sub(1));
        let terrain = if region_idx < state.regions.len() {
            state.regions[region_idx].terrain
        } else {
            Terrain::Plains
        };

        if rich {
            // Varied production: terrain + specialty influenced.
            let terrain_commodities = terrain.primary_commodities();
            if !terrain_commodities.is_empty() {
                let primary = terrain_commodities[id as usize % terrain_commodities.len()];
                npc_data.behavior_production = vec![(primary.0, 0.15)];
                // Add a secondary from specialty if available.
                let spec_bonuses = specialty.production_bonuses();
                if !spec_bonuses.is_empty() {
                    let secondary = spec_bonuses[id as usize % spec_bonuses.len()];
                    if secondary.0 != primary.0 {
                        npc_data.behavior_production.push((secondary.0, 0.05));
                    }
                }
            } else {
                // Sparse terrain: low production.
                npc_data.behavior_production = vec![(0, 0.05)];
            }

            // Varied starting gold based on settlement wealth.
            let wealth_factor = settlement_treasury / 200.0;
            npc_data.gold = 5.0 + wealth_factor * lcg_f32(rng) * 20.0;
            npc_data.morale = 30.0 + lcg_f32(rng) * 50.0;
            npc_data.loyalty = 20.0 + lcg_f32(rng) * 60.0;
        } else {
            let terrain_commodities = terrain.primary_commodities();
            if !terrain_commodities.is_empty() {
                let primary = terrain_commodities[id as usize % terrain_commodities.len()];
                npc_data.behavior_production = vec![(primary.0, 0.1)];
            } else {
                npc_data.behavior_production = vec![(0, 0.05)];
            }
            npc_data.gold = 2.0 + lcg_f32(rng) * 8.0;
            npc_data.morale = 40.0 + lcg_f32(rng) * 30.0;
            npc_data.loyalty = 30.0 + lcg_f32(rng) * 40.0;
        }

        // ~80% of NPCs are producers (flee from combat), ~20% are idle defenders.
        // Military outposts have more defenders.
        let defender_chance = match specialty {
            SettlementSpecialty::MilitaryOutpost => 0.4,
            _ => 0.15,
        };
        let role_roll = (id as f32 * 2.71828).fract();
        if role_roll < defender_chance {
            npc_data.economic_intent = EconomicIntent::Idle; // will fight
        } else {
            npc_data.economic_intent = EconomicIntent::Produce; // will flee
        }

        state.grids[settlement_idx].entity_ids.push(id);
        state.entities.push(npc);
        id += 1;
    }

    id
}

/// Phase 6: Spawn monsters in the wilderness (density weighted by region). Returns next entity id.
fn spawn_monsters(
    state: &mut WorldState,
    layout: &GridLayout,
    args: &WorldSimArgs,
    mut id: u32,
    rng: &mut u64,
) -> u32 {
    let max_extent = (layout.cols as f32) * layout.spacing;
    let rich = args.rich;
    for i in 0..args.monsters {
        let angle = (i as f32 / args.monsters.max(1) as f32) * std::f32::consts::TAU;
        let base_radius = max_extent * 0.6 + lcg_f32(rng) * max_extent * 0.4;
        let cx = max_extent / 2.0;
        let cy = max_extent / 2.0;
        let pos = (cx + base_radius * angle.cos(), cy + base_radius * angle.sin());
        let level = 1 + (i % 10) as u32;
        let mut monster = Entity::new_monster(id, pos, level);
        if rich {
            monster.hp *= 2.0;
            monster.max_hp *= 2.0;
            monster.attack_damage *= 1.5;
        }
        state.entities.push(monster);
        id += 1;
    }
    id
}

/// Phase 7: Spawn sea monsters in DeepOcean/CoralReef regions and along coastal edges.
fn spawn_sea_monsters(
    state: &mut WorldState,
    layout: &GridLayout,
    mut id: u32,
    rng: &mut u64,
) {
    let GridLayout { cols, spacing, .. } = *layout;
    let max_extent = (cols as f32) * spacing;

    let ocean_regions: Vec<(f32, f32)> = state.regions.iter()
        .filter(|r| matches!(r.terrain, Terrain::DeepOcean | Terrain::CoralReef))
        .map(|r| {
            // Approximate region position from ID.
            let row = r.id as usize / cols;
            let col = r.id as usize % cols;
            (col as f32 * spacing, row as f32 * spacing)
        })
        .collect();

    // Also spawn sea monsters near coastal edges for any world.
    let edge_positions: Vec<(f32, f32)> = (0..4)
        .map(|i| {
            let t = i as f32 / 4.0;
            match i {
                0 => (t * max_extent, 0.0),        // top edge
                1 => (max_extent, t * max_extent),  // right edge
                2 => (t * max_extent, max_extent),  // bottom edge
                _ => (0.0, t * max_extent),         // left edge
            }
        })
        .collect();

    let sea_positions: Vec<(f32, f32)> = ocean_regions.iter()
        .chain(edge_positions.iter())
        .copied()
        .collect();

    let num_sea_monsters = (sea_positions.len() * 2).min(10).max(2);
    for i in 0..num_sea_monsters {
        let base_pos = sea_positions[i % sea_positions.len()];
        let jx = lcg_f32(rng) * 40.0 - 20.0;
        let jy = lcg_f32(rng) * 40.0 - 20.0;
        let pos = (base_pos.0 + jx, base_pos.1 + jy);
        let level = 20 + (i as u32 * 5); // high level: 20-65
        let mut monster = Entity::new_monster(id, pos, level);
        // Sea monsters are much tougher than regular monsters.
        monster.hp *= 5.0;
        monster.max_hp *= 5.0;
        monster.attack_damage *= 3.0;
        monster.attack_range = 5.0; // long reach
        monster.move_speed = 4.0;   // fast in water
        state.entities.push(monster);
        id += 1;
    }
}

// ---------------------------------------------------------------------------
// WebSocket server — streams TraceFrame JSON to the web visualizer
// ---------------------------------------------------------------------------

fn run_ws_server(sim: &mut WorldSim, port: u16, args: &WorldSimArgs) -> ExitCode {
    use std::net::TcpListener;
    use tungstenite::{accept, Message};

    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind WebSocket server on {}: {}", addr, e);
            return ExitCode::FAILURE;
        }
    };
    println!("WebSocket server listening on ws://localhost:{}", port);
    println!("Open web/index.html in a browser to visualize.");
    println!("Press Ctrl+C to stop.\n");

    // Accept connections in a loop (one at a time for simplicity)
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => { eprintln!("Accept error: {}", e); continue; }
        };
        println!("Client connected from {}", stream.peer_addr().unwrap_or_else(|_| "unknown".parse().unwrap()));

        let mut ws = match accept(stream) {
            Ok(ws) => ws,
            Err(e) => { eprintln!("WebSocket handshake failed: {}", e); continue; }
        };

        let max_ticks = args.ticks;
        let tick_interval = std::time::Duration::from_millis(50); // 20 fps

        let mut ticks_this_session = 0u64;
        let mut chronicle_snapshot: Vec<bevy_game::world_sim::state::ChronicleEntry> = Vec::new();
        let mut ticks_per_frame: u64 = 5;
        let mut paused = false;
        let mut selected_entity_id: Option<u32> = None;

        loop {
            let start = std::time::Instant::now();

            // Advance simulation (skip if paused)
            if !paused {
            for _ in 0..ticks_per_frame {
                if ticks_this_session >= max_ticks { break; }
                sim.tick();
                ticks_this_session += 1;
            }
            } // end if !paused

            // Build frame
            let state = sim.state();
            let chronicle = &state.chronicle;
            // Collect new chronicle entries
            let new_entries: Vec<_> = chronicle.iter()
                .filter(|e| {
                    let dominated = chronicle_snapshot.iter().any(|prev| prev.tick == e.tick && prev.text == e.text);
                    !dominated
                })
                .cloned()
                .collect();
            chronicle_snapshot.extend(new_entries.iter().cloned());
            // Keep chronicle bounded
            if chronicle_snapshot.len() > 500 {
                chronicle_snapshot.drain(..chronicle_snapshot.len() - 500);
            }

            let frame = bevy_game::world_sim::visualizer::generate_frame_with_selection(
                state,
                &new_entries,
                100, // event window
                max_ticks,
                selected_entity_id,
            );

            // Serialize and send
            let json = match serde_json::to_string(&frame) {
                Ok(j) => j,
                Err(e) => { eprintln!("Serialize error: {}", e); break; }
            };

            match ws.send(Message::Text(json.into())) {
                Ok(_) => {}
                Err(tungstenite::Error::ConnectionClosed) |
                Err(tungstenite::Error::AlreadyClosed) => {
                    println!("Client disconnected.");
                    break;
                }
                Err(e) => {
                    eprintln!("Send error: {}", e);
                    break;
                }
            }

            // Drain pending incoming messages non-blockingly.
            // Set socket to non-blocking for reads, then restore.
            if let Ok(raw) = ws.get_ref().try_clone() {
                let _ = raw.set_nonblocking(true);
            }
            let mut client_closed = false;
            loop {
                match ws.read() {
                    Ok(Message::Close(_)) => { println!("Client closed."); client_closed = true; break; }
                    Ok(Message::Text(t)) => {
                        // Parse JSON commands from client
                        if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&*t) {
                            match cmd.get("command").and_then(|c| c.as_str()) {
                                Some("speed") => {
                                    if let Some(v) = cmd.get("value").and_then(|v| v.as_u64()) {
                                        ticks_per_frame = v.max(1).min(50);
                                        println!("Speed: {} ticks/frame", ticks_per_frame);
                                    }
                                }
                                Some("pause") => { paused = true; println!("Paused"); }
                                Some("play") => { paused = false; println!("Resumed"); }
                                Some("select") => {
                                    selected_entity_id = cmd.get("entity_id")
                                        .and_then(|v| v.as_u64())
                                        .map(|v| v as u32);
                                    println!("Selected entity: {:?}", selected_entity_id);
                                }
                                Some("deselect") => {
                                    selected_entity_id = None;
                                    println!("Deselected");
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            if client_closed { break; }
            if let Ok(raw) = ws.get_ref().try_clone() {
                let _ = raw.set_nonblocking(false);
            }

            if ticks_this_session >= max_ticks {
                println!("Reached {} ticks, waiting for new client...", max_ticks);
                break;
            }

            // Rate limit
            let elapsed = start.elapsed();
            if elapsed < tick_interval {
                std::thread::sleep(tick_interval - elapsed);
            }
        }

        // Reset sim for next client
        let state = build_world(args);
        *sim = WorldSim::new(state);
    }

    ExitCode::SUCCESS
}
