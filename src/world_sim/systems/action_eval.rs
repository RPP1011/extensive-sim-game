//! Flat action-space utility evaluator.
//!
//! Replaces the GOAP goal stack with a single per-tick evaluation:
//! scan nearby entities, score each available action by utility,
//! pick the highest, execute immediately.
//!
//! Runs every ACTION_EVAL_INTERVAL ticks for alive NPCs and monsters.

use crate::world_sim::commodity;
use crate::world_sim::state::*;

/// How often to re-evaluate actions (ticks).
const ACTION_EVAL_INTERVAL: u64 = 5;

/// Distance threshold for harvesting a resource (squared).
const HARVEST_DIST_SQ: f32 = 25.0; // 5 units
/// Amount harvested per evaluation cycle.
const HARVEST_AMOUNT: f32 = 1.0;
/// Food consumed per meal (same as work.rs).
const FOOD_PER_MEAL: f32 = 0.1;
/// Hunger restored per meal.
const MEAL_HUNGER_RESTORE: f32 = 60.0;
/// Scan radius for nearby entities.
const SCAN_RADIUS: f32 = 50.0;
const SCAN_RADIUS_SQ: f32 = SCAN_RADIUS * SCAN_RADIUS;
/// Aggro range for attack actions.
const AGGRO_RANGE: f32 = 20.0;
const AGGRO_RANGE_SQ: f32 = AGGRO_RANGE * AGGRO_RANGE;
/// Construction progress per tick when building.
const BUILD_PROGRESS_PER_TICK: f32 = 0.05;

// ---------------------------------------------------------------------------
// Scored action candidates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum CandidateAction {
    Eat,
    Harvest { resource_idx: usize, resource_id: u32 },
    MoveToResource { resource_idx: usize, resource_id: u32, pos: (f32, f32) },
    MoveToPos { target: (f32, f32) },
    Work,
    MoveToWork { pos: (f32, f32) },
    BuildExisting { building_idx: usize, building_id: u32 },
    BuildNew,
    Attack { target_idx: usize, target_id: u32 },
    Flee { away_from: (f32, f32) },
    Idle,
}

// ---------------------------------------------------------------------------
// Lightweight entity snapshot for read-only spatial queries
// ---------------------------------------------------------------------------

struct EntitySnap {
    idx: usize,
    id: u32,
    kind: EntityKind,
    #[allow(dead_code)]
    team: WorldTeam,
    pos: (f32, f32),
    alive: bool,
    #[allow(dead_code)]
    hp: f32,
    #[allow(dead_code)]
    max_hp: f32,
    #[allow(dead_code)]
    attack_damage: f32,
    // Resource node info (if kind == Resource)
    resource_type: Option<ResourceType>,
    resource_remaining: f32,
    // Building info (if kind == Building)
    construction_progress: f32,
    #[allow(dead_code)]
    building_settlement_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Evaluate and act for every alive NPC and monster. Called post-apply from
/// runtime.rs, replacing evaluate_goals + evaluate_world_goap + advance_plans
/// + advance_eating.
pub fn evaluate_and_act(state: &mut WorldState) {
    if state.tick % ACTION_EVAL_INTERVAL != 0 { return; }

    let entity_count = state.entities.len();

    // --- Phase 1: collect read-only snapshots for spatial queries. ---
    let snaps: Vec<EntitySnap> = state.entities.iter().enumerate().map(|(idx, e)| {
        let (rt, rem) = e.resource.as_ref()
            .map(|r| (Some(r.resource_type), r.remaining))
            .unwrap_or((None, 0.0));
        let (cp, bsid) = e.building.as_ref()
            .map(|b| (b.construction_progress, b.settlement_id))
            .unwrap_or((1.0, None));
        EntitySnap {
            idx,
            id: e.id,
            kind: e.kind,
            team: e.team,
            pos: e.pos,
            alive: e.alive,
            hp: e.hp,
            max_hp: e.max_hp,
            attack_damage: e.attack_damage,
            resource_type: rt,
            resource_remaining: rem,
            construction_progress: cp,
            building_settlement_id: bsid,
        }
    }).collect();

    // --- Phase 2: for each NPC/monster, score and pick best action. ---
    struct DeferredAction {
        idx: usize,
        action: CandidateAction,
        npc_action: NpcAction,
        utility: f32,
        skip_execute: bool, // hysteresis: keep current action, only update intention ticks
    }

    let mut deferred: Vec<DeferredAction> = Vec::with_capacity(entity_count / 2);

    for i in 0..entity_count {
        let e = &state.entities[i];
        if !e.alive { continue; }

        match e.kind {
            EntityKind::Npc => {
                let npc = match &e.npc { Some(n) => n, None => continue };
                // Skip NPCs in adventuring parties.
                if npc.party_id.is_some() { continue; }
                // Skip NPCs on trade runs (handled by npc_decisions barter system).
                if matches!(npc.economic_intent, EconomicIntent::Trade { .. }) { continue; }
                // Skip NPCs currently adventuring.
                if matches!(npc.economic_intent, EconomicIntent::Adventuring { .. }) { continue; }
                // Skip NPCs in non-idle work states (let the work state machine finish).
                if !matches!(npc.work_state, WorkState::Idle) { continue; }

                let (action, npc_action, utility) = score_npc_actions(e, npc, &snaps);

                // --- Hysteresis: prefer continuing current intention ---
                let interrupt = is_interrupt(npc, e.hp, e.max_hp);
                if !interrupt {
                    if let Some((ref cur_action, cur_utility)) = npc.current_intention {
                        let same_kind = std::mem::discriminant(cur_action) == std::mem::discriminant(&npc_action);
                        if same_kind {
                            // Same action type — continue, just bump ticks.
                            deferred.push(DeferredAction { idx: i, action, npc_action, utility, skip_execute: false });
                            continue;
                        }
                        // Different action: apply continuation bonus + switching threshold.
                        // Bonus decays over 200 ticks to prevent permanent lock-in.
                        let bonus = 0.15 * (1.0 - (npc.intention_ticks as f32 / 200.0)).max(0.0);
                        let boosted_current = cur_utility + bonus;
                        if utility - boosted_current <= 0.2 {
                            // New action doesn't beat threshold — keep current, skip execution.
                            deferred.push(DeferredAction {
                                idx: i,
                                action: CandidateAction::Idle,
                                npc_action: cur_action.clone(),
                                utility: boosted_current,
                                skip_execute: true,
                            });
                            continue;
                        }
                    }
                }

                deferred.push(DeferredAction { idx: i, action, npc_action, utility, skip_execute: false });
            }
            EntityKind::Monster => {
                let (action, npc_action) = score_monster_actions(e, &snaps);
                deferred.push(DeferredAction { idx: i, action, npc_action, utility: 0.0, skip_execute: false });
            }
            _ => {}
        }
    }

    // --- Phase 3: execute deferred actions (mutable access to state). ---
    let tick = state.tick;
    for da in deferred {
        if !da.skip_execute {
            execute_action(state, da.idx, &da.action, tick);
        }
        // Update intention tracking on NPCs.
        if let Some(npc) = &mut state.entities[da.idx].npc {
            let switched = npc.current_intention.as_ref()
                .map(|(cur, _)| std::mem::discriminant(cur) != std::mem::discriminant(&da.npc_action))
                .unwrap_or(true);
            if switched {
                npc.intention_ticks = 0;
            } else {
                npc.intention_ticks += ACTION_EVAL_INTERVAL as u32;
            }
            npc.current_intention = Some((da.npc_action.clone(), da.utility));
            npc.action = da.npc_action;
        }
    }
}

// ---------------------------------------------------------------------------
// NPC scoring
// ---------------------------------------------------------------------------

/// Check whether the NPC is in an emergency that should bypass hysteresis.
fn is_interrupt(npc: &NpcData, hp: f32, max_hp: f32) -> bool {
    // Starving with food available
    if npc.needs.hunger < 15.0 { return true; }
    // Danger — safety critical
    if npc.needs.safety < 20.0 { return true; }
    // Low HP
    if max_hp > 0.0 && hp < max_hp * 0.3 { return true; }
    false
}

fn score_npc_actions(
    entity: &Entity,
    npc: &NpcData,
    snaps: &[EntitySnap],
) -> (CandidateAction, NpcAction, f32) {
    let pos = entity.pos;
    let needs = &npc.needs;
    let pers = &npc.personality;
    let emo = &npc.emotions;

    // Need urgencies: (100 - value) / 100.
    let hunger_urgency = (100.0 - needs.hunger) / 100.0;
    let safety_urgency = (100.0 - needs.safety) / 100.0;
    let shelter_urgency = (100.0 - needs.shelter) / 100.0;
    let purpose_urgency = (100.0 - needs.purpose) / 100.0;

    // Emotion modifiers.
    let grief_dampen = 1.0 - emo.grief * 0.5; // grief reduces all utilities
    let anger_boost = emo.anger;
    let fear_boost = emo.fear;

    // Personality modifiers per action type.
    let ambition_mod = 1.0 + (pers.ambition - 0.5) * 0.4;       // 0.8-1.2
    let _risk_mod = 1.0 + (pers.risk_tolerance - 0.5) * 0.4;    // 0.8-1.2
    let _compassion_mod = 1.0 + (pers.compassion - 0.5) * 0.2;  // 0.9-1.1
    let curiosity_mod = 1.0 + (pers.curiosity - 0.5) * 0.2;     // 0.9-1.1

    let mut best_utility = 0.01_f32; // Idle baseline
    let mut best_action = CandidateAction::Idle;
    let mut best_npc_action = NpcAction::Idle;

    // --- Eat (requires food in inventory) ---
    let has_food = entity.inventory.as_ref()
        .map(|inv| inv.commodities[commodity::FOOD] >= FOOD_PER_MEAL)
        .unwrap_or(false);
    if has_food {
        let utility = hunger_urgency * 1.5 * grief_dampen;
        if utility > best_utility {
            best_utility = utility;
            best_action = CandidateAction::Eat;
            best_npc_action = NpcAction::Eating { ticks_remaining: 1, building_id: 0 };
        }
    }

    // --- Scan nearby entities ---
    for snap in snaps {
        if !snap.alive { continue; }
        let dx = snap.pos.0 - pos.0;
        let dy = snap.pos.1 - pos.1;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq > SCAN_RADIUS_SQ { continue; }

        let dist = dist_sq.sqrt();

        match snap.kind {
            EntityKind::Resource => {
                if snap.resource_remaining <= 0.0 { continue; }
                let rt: ResourceType = match snap.resource_type {
                    Some(t) => t,
                    None => continue,
                };
                let commodity_idx = rt.commodity();

                // Compute commodity need based on what this resource produces.
                let commodity_need = match commodity_idx {
                    c if c == commodity::FOOD => hunger_urgency,
                    c if c == commodity::WOOD => shelter_urgency * 0.5,
                    c if c == commodity::IRON => purpose_urgency * 0.4,
                    c if c == commodity::HERBS => safety_urgency * 0.3,
                    _ => 0.2, // generic want
                };

                let distance_factor = 1.0 / (1.0 + dist / 10.0);

                if dist_sq <= HARVEST_DIST_SQ {
                    // Harvest (in range)
                    let utility = commodity_need * distance_factor * curiosity_mod * grief_dampen;
                    if utility > best_utility {
                        best_utility = utility;
                        best_action = CandidateAction::Harvest {
                            resource_idx: snap.idx,
                            resource_id: snap.id,
                        };
                        best_npc_action = NpcAction::Harvesting { resource_id: snap.id };
                    }
                } else {
                    // MoveTo resource (out of range)
                    let utility = commodity_need * 0.8 * distance_factor * curiosity_mod * grief_dampen;
                    if utility > best_utility {
                        best_utility = utility;
                        best_action = CandidateAction::MoveToResource {
                            resource_idx: snap.idx,
                            resource_id: snap.id,
                            pos: snap.pos,
                        };
                        best_npc_action = NpcAction::Walking { destination: snap.pos };
                    }
                }
            }

            EntityKind::Monster => {
                // Attack hostile monster if within aggro range.
                if dist_sq <= AGGRO_RANGE_SQ {
                    let utility = (safety_urgency * 0.7 + anger_boost * 0.3)
                        * _risk_mod * grief_dampen;
                    if utility > best_utility {
                        best_utility = utility;
                        best_action = CandidateAction::Attack {
                            target_idx: snap.idx,
                            target_id: snap.id,
                        };
                        best_npc_action = NpcAction::Fighting { target_id: snap.id };
                    }
                }
            }

            EntityKind::Building => {
                // Incomplete building nearby: advance construction (blueprint system).
                // ANY NPC with wood can contribute — multiple workers stack.
                if snap.construction_progress < 1.0 {
                    let has_wood = entity.inventory.as_ref()
                        .map(|inv| inv.commodities[commodity::WOOD] >= 0.1)
                        .unwrap_or(false);
                    if has_wood {
                        // Utility: homeless NPCs want shelter, housed NPCs help for purpose/social.
                        let base = if npc.home_building_id.is_none() {
                            shelter_urgency * 0.9
                        } else {
                            purpose_urgency * 0.3 + 0.1 // community contribution
                        };
                        let distance_factor = 1.0 / (1.0 + dist / 10.0);

                        if dist_sq <= HARVEST_DIST_SQ {
                            // Close enough to build
                            let utility = base * distance_factor * ambition_mod * grief_dampen;
                            if utility > best_utility {
                                best_utility = utility;
                                best_action = CandidateAction::BuildExisting {
                                    building_idx: snap.idx,
                                    building_id: snap.id,
                                };
                                best_npc_action = NpcAction::Building {
                                    building_id: snap.id,
                                    ticks_remaining: 1,
                                };
                            }
                        } else {
                            // Walk toward the building to help
                            let utility = base * 0.8 * distance_factor * ambition_mod * grief_dampen;
                            if utility > best_utility {
                                best_utility = utility;
                                best_action = CandidateAction::MoveToPos { target: snap.pos };
                                best_npc_action = NpcAction::Walking { destination: snap.pos };
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }

    // --- Work (requires work_building_id set) ---
    if let Some(work_bid) = npc.work_building_id {
        if let Some(work_snap) = snaps.iter().find(|s| s.id == work_bid && s.alive) {
            let dx = work_snap.pos.0 - pos.0;
            let dy = work_snap.pos.1 - pos.1;
            let dist_sq = dx * dx + dy * dy;
            let dist = dist_sq.sqrt();
            let distance_factor = 1.0 / (1.0 + dist / 10.0);

            if dist < 5.0 {
                // At work building: work
                let utility = (purpose_urgency * 0.6 + 0.2) * ambition_mod * grief_dampen;
                if utility > best_utility {
                    best_utility = utility;
                    best_action = CandidateAction::Work;
                    best_npc_action = NpcAction::Working {
                        ticks_remaining: 10,
                        building_id: work_bid,
                        activity: WorkActivity::Crafting,
                    };
                }
            } else {
                // Move to work building
                let utility = (purpose_urgency * 0.6 + 0.2) * distance_factor * ambition_mod * grief_dampen;
                if utility > best_utility {
                    best_utility = utility;
                    best_action = CandidateAction::MoveToWork { pos: work_snap.pos };
                    best_npc_action = NpcAction::Walking { destination: work_snap.pos };
                }
            }
        }
    }

    // --- Place blueprint (requires ANY wood to start, no home) ---
    // NPC places a building shell (0% progress). Materials deposited over time
    // via BuildExisting action. Any NPC can contribute.
    if npc.home_building_id.is_none() {
        let has_any_wood = entity.inventory.as_ref()
            .map(|inv| inv.commodities[commodity::WOOD] >= 1.0)
            .unwrap_or(false);
        if has_any_wood {
            let utility = shelter_urgency * 0.8 * ambition_mod * grief_dampen;
            if utility > best_utility {
                best_utility = utility;
                best_action = CandidateAction::BuildNew;
                best_npc_action = NpcAction::Building {
                    building_id: 0,
                    ticks_remaining: 1,
                };
            }
        }
    }

    // --- Flee (fear override) ---
    if fear_boost > 0.5 && entity.hp < entity.max_hp * 0.3 {
        let mut nearest_hostile: Option<(f32, f32)> = None;
        let mut nearest_dist_sq = f32::MAX;
        for snap in snaps {
            if !snap.alive { continue; }
            if snap.kind != EntityKind::Monster { continue; }
            let dx = snap.pos.0 - pos.0;
            let dy = snap.pos.1 - pos.1;
            let d2 = dx * dx + dy * dy;
            if d2 < nearest_dist_sq {
                nearest_dist_sq = d2;
                nearest_hostile = Some(snap.pos);
            }
        }
        if let Some(hostile_pos) = nearest_hostile {
            let utility = 0.9 * grief_dampen;
            if utility > best_utility {
                best_utility = utility;
                best_action = CandidateAction::Flee { away_from: hostile_pos };
                best_npc_action = NpcAction::Fleeing;
            }
        }
    }

    (best_action, best_npc_action, best_utility)
}

// ---------------------------------------------------------------------------
// Monster scoring
// ---------------------------------------------------------------------------

fn score_monster_actions(
    entity: &Entity,
    snaps: &[EntitySnap],
) -> (CandidateAction, NpcAction) {
    let pos = entity.pos;
    let hp_ratio = if entity.max_hp > 0.0 { entity.hp / entity.max_hp } else { 1.0 };

    // Flee: hp < 20%
    if hp_ratio < 0.2 {
        let mut nearest_hostile: Option<(f32, f32)> = None;
        let mut nearest_dist_sq = f32::MAX;
        for snap in snaps {
            if !snap.alive || snap.kind != EntityKind::Npc { continue; }
            let dx = snap.pos.0 - pos.0;
            let dy = snap.pos.1 - pos.1;
            let d2 = dx * dx + dy * dy;
            if d2 < nearest_dist_sq {
                nearest_dist_sq = d2;
                nearest_hostile = Some(snap.pos);
            }
        }
        if let Some(hostile_pos) = nearest_hostile {
            return (
                CandidateAction::Flee { away_from: hostile_pos },
                NpcAction::Fleeing,
            );
        }
    }

    // Attack: find nearest NPC within aggro range.
    let mut best_attack: Option<(f32, usize, u32)> = None;
    for snap in snaps {
        if !snap.alive || snap.kind != EntityKind::Npc { continue; }
        let dx = snap.pos.0 - pos.0;
        let dy = snap.pos.1 - pos.1;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq <= AGGRO_RANGE_SQ {
            let utility = 0.8;
            match &best_attack {
                Some((u, _, _)) if utility <= *u => {}
                _ => { best_attack = Some((utility, snap.idx, snap.id)); }
            }
        }
    }

    if let Some((_, target_idx, target_id)) = best_attack {
        return (
            CandidateAction::Attack { target_idx, target_id },
            NpcAction::Fighting { target_id },
        );
    }

    // Idle: monsters like to stand around.
    (CandidateAction::Idle, NpcAction::Idle)
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

fn execute_action(state: &mut WorldState, entity_idx: usize, action: &CandidateAction, tick: u64) {
    match action {
        CandidateAction::Eat => {
            let entity = &mut state.entities[entity_idx];
            if let Some(inv) = &mut entity.inventory {
                inv.commodities[commodity::FOOD] -= FOOD_PER_MEAL;
            }
            if let Some(npc) = &mut entity.npc {
                npc.needs.hunger = (npc.needs.hunger + MEAL_HUNGER_RESTORE).min(100.0);
            }
        }

        CandidateAction::Harvest { resource_idx, .. } => {
            let res_idx = *resource_idx;
            let resource_remaining = state.entities.get(res_idx)
                .and_then(|e| e.resource.as_ref())
                .map(|r| r.remaining)
                .unwrap_or(0.0);
            if resource_remaining <= 0.0 { return; }

            let commodity_idx = state.entities[res_idx].resource.as_ref()
                .map(|r| r.resource_type.commodity())
                .unwrap_or(0);

            let harvest = HARVEST_AMOUNT.min(resource_remaining);

            if let Some(res) = &mut state.entities[res_idx].resource {
                res.remaining -= harvest;
            }

            let entity = &mut state.entities[entity_idx];
            if let Some(inv) = &mut entity.inventory {
                inv.deposit(commodity_idx, harvest);
            } else {
                entity.inventory = Some(Inventory::default());
                entity.inventory.as_mut().unwrap().deposit(commodity_idx, harvest);
            }
        }

        CandidateAction::MoveToResource { pos, .. }
        | CandidateAction::MoveToWork { pos }
        | CandidateAction::MoveToPos { target: pos } => {
            state.entities[entity_idx].move_target = Some(*pos);
        }

        CandidateAction::Work => {
            let entity = &mut state.entities[entity_idx];
            if let Some(npc) = &mut entity.npc {
                npc.work_state = WorkState::Working {
                    building_id: npc.work_building_id.unwrap_or(0),
                    ticks_remaining: 10,
                };
            }
        }

        CandidateAction::BuildExisting { building_idx, building_id } => {
            let b_idx = *building_idx;
            let bid = *building_id;

            // Recipe-based construction: NPC deposits materials from inventory
            // into building storage. Progress = materials_deposited / materials_required.
            let building_type = state.entities[b_idx].building.as_ref()
                .map(|b| b.building_type).unwrap_or(BuildingType::House);
            let (wood_needed, iron_needed) = building_type.build_cost();

            // How much has been deposited so far? Tracked in building.storage.
            let (wood_deposited, iron_deposited) = state.entities[b_idx].building.as_ref()
                .map(|b| (b.storage[commodity::WOOD], b.storage[commodity::IRON]))
                .unwrap_or((0.0, 0.0));

            // Deposit what we can from NPC inventory.
            if wood_deposited < wood_needed {
                let can_give = state.entities[entity_idx].inventory.as_ref()
                    .map(|inv| inv.commodities[commodity::WOOD].min(1.0)) // up to 1.0 per tick
                    .unwrap_or(0.0);
                let need = (wood_needed - wood_deposited).min(can_give);
                if need > 0.0 {
                    if let Some(inv) = &mut state.entities[entity_idx].inventory {
                        inv.commodities[commodity::WOOD] -= need;
                    }
                    if let Some(bd) = &mut state.entities[b_idx].building {
                        bd.storage[commodity::WOOD] += need;
                    }
                    // material deposited
                }
            }
            if iron_deposited < iron_needed {
                let can_give = state.entities[entity_idx].inventory.as_ref()
                    .map(|inv| inv.commodities[commodity::IRON].min(1.0))
                    .unwrap_or(0.0);
                let need = (iron_needed - iron_deposited).min(can_give);
                if need > 0.0 {
                    if let Some(inv) = &mut state.entities[entity_idx].inventory {
                        inv.commodities[commodity::IRON] -= need;
                    }
                    if let Some(bd) = &mut state.entities[b_idx].building {
                        bd.storage[commodity::IRON] += need;
                    }
                    // material deposited
                }
            }

            // Recalculate progress from deposited materials.
            let total_needed = wood_needed + iron_needed;
            if total_needed > 0.0 {
                let total_deposited = state.entities[b_idx].building.as_ref()
                    .map(|b| b.storage[commodity::WOOD] + b.storage[commodity::IRON])
                    .unwrap_or(0.0);
                let progress = (total_deposited / total_needed).min(1.0);
                if let Some(bd) = &mut state.entities[b_idx].building {
                    bd.construction_progress = progress;
                }
            }

            // If all materials deposited, building is complete even without extra labor.
            // (Labor IS depositing materials — no separate construction phase.)
            let just_completed = state.entities[b_idx].building.as_ref()
                .map(|b| b.construction_progress >= 1.0).unwrap_or(false);

            if just_completed {
                let npc_id = state.entities[entity_idx].id;
                if let Some(bd) = &mut state.entities[b_idx].building {
                    bd.built_tick = tick;
                    if bd.owner_id.is_none() {
                        bd.owner_id = Some(npc_id);
                    }
                }
                if let Some(npc) = &mut state.entities[entity_idx].npc {
                    if npc.home_building_id.is_none() {
                        npc.home_building_id = Some(bid);
                    }
                }
                let builder_name = state.entities[entity_idx].npc.as_ref()
                    .map(|n| n.name.clone()).unwrap_or_default();
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Economy,
                    text: format!("{} completed construction of a {:?}", builder_name, building_type),
                    entity_ids: vec![npc_id, bid],
                });
            }
        }

        CandidateAction::BuildNew => {
            // Place a building blueprint at the NPC's position.
            // No materials consumed now — they're deposited via BuildExisting over time.
            let npc_pos = state.entities[entity_idx].pos;
            let npc_id = state.entities[entity_idx].id;

            // Spawn building entity.
            state.sync_next_id();
            let new_id = state.next_entity_id();
            let mut bld = Entity::new_building(new_id, npc_pos);
            bld.building = Some(BuildingData {
                building_type: BuildingType::House,
                settlement_id: state.entities[entity_idx].npc.as_ref()
                    .and_then(|n| n.home_settlement_id),
                grid_col: 0,
                grid_row: 0,
                footprint_w: 1,
                footprint_h: 1,
                tier: 0,
                room_seed: entity_hash(new_id, tick, 0x800E) as u64,
                rooms: BuildingType::House.default_rooms(),
                residential_capacity: BuildingType::House.residential_capacity(),
                work_capacity: BuildingType::House.work_capacity(),
                resident_ids: vec![npc_id],
                worker_ids: Vec::new(),
                construction_progress: 0.0,
                built_tick: tick,
                builder_id: Some(npc_id),
                temporary: false,
                ttl_ticks: None,
                name: format!("{}'s House", state.entities[entity_idx].npc.as_ref()
                    .map(|n| n.name.as_str()).unwrap_or("NPC")),
                storage: [0.0; crate::world_sim::NUM_COMMODITIES],
                storage_capacity: BuildingType::House.storage_capacity(),
                owner_id: Some(npc_id),
                builder_modifiers: Vec::new(),
                owner_modifiers: Vec::new(),
                worker_class_ticks: Vec::new(),
                specialization_tag: None,
                specialization_strength: 0.0,
                specialization_name: String::new(),
            });
            state.entities.push(bld);
            state.rebuild_entity_cache();

            // Assign the NPC to their new home.
            if let Some(npc) = &mut state.entities[entity_idx].npc {
                npc.home_building_id = Some(new_id);
            }

            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Economy,
                text: format!("{} built a house",
                    state.entities[entity_idx].npc.as_ref()
                        .map(|n| n.name.as_str()).unwrap_or("NPC")),
                entity_ids: vec![npc_id, new_id],
            });
        }

        CandidateAction::Attack { target_idx, .. } => {
            let target_pos = state.entities.get(*target_idx)
                .filter(|e| e.alive)
                .map(|e| e.pos);
            if let Some(tpos) = target_pos {
                state.entities[entity_idx].move_target = Some(tpos);
            }
        }

        CandidateAction::Flee { away_from } => {
            let entity = &mut state.entities[entity_idx];
            let dx = entity.pos.0 - away_from.0;
            let dy = entity.pos.1 - away_from.1;
            let mag = (dx * dx + dy * dy).sqrt().max(0.01);
            let flee_dist = 20.0;
            let flee_x = entity.pos.0 + dx / mag * flee_dist;
            let flee_y = entity.pos.1 + dy / mag * flee_dist;
            entity.move_target = Some((flee_x, flee_y));
        }

        CandidateAction::Idle => {}
    }
}
