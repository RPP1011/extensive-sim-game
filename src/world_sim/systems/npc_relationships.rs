#![allow(unused)]
//! NPC-to-NPC relationship evolution at settlements.
//!
//! For each settlement, NPC pairs develop friendships or rivalries based on
//! shared behavior tags. Strong overlap produces morale bonuses (friendship);
//! low overlap with high combat tags produces competitive rivalry bonuses.
//!
//! Cadence: every 50 ticks (relationship evaluation).
//!          every 500 ticks (chronicle recording for strongest friendship).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

/// Tick cadence for relationship evaluation.
const RELATIONSHIP_INTERVAL: u64 = 50;

/// Tick cadence for chronicle recording.
const CHRONICLE_INTERVAL: u64 = 500;

/// Max NPCs per settlement to evaluate (caps O(n^2) pair iteration).
const MAX_NPCS: usize = 32;

/// Number of top behavior tags to compare per NPC.
const TOP_TAGS: usize = 5;

/// Affinity above which a friendship bonus is applied.
const FRIENDSHIP_THRESHOLD: f32 = 0.6;

/// Affinity below which rivalry can trigger (also requires combat > 100).
const RIVALRY_THRESHOLD: f32 = 0.2;

/// Morale bonus for friends.
const FRIENDSHIP_MORALE: f32 = 0.5;

/// Combat behavior tag bonus for rivals (competitive motivation).
const RIVALRY_COMBAT_BONUS: f32 = 0.2;

/// Minimum combat behavior value for rivalry to trigger.
const RIVALRY_COMBAT_MIN: f32 = 100.0;

/// Compute NPC relationship deltas across all settlements.
pub fn compute_npc_relationships(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RELATIONSHIP_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_npc_relationships_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement relationship evaluation.
pub fn compute_npc_relationships_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % RELATIONSHIP_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Collect alive NPCs with behavior data, capped at MAX_NPCS.
    let mut npcs: Vec<&Entity> = Vec::with_capacity(MAX_NPCS);
    for entity in entities {
        if npcs.len() >= MAX_NPCS {
            break;
        }
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.npc.is_none() {
            continue;
        }
        npcs.push(entity);
    }

    if npcs.len() < 2 {
        return;
    }

    // Pre-compute top-5 behavior tags for each NPC (tag_hash, value), sorted by value descending.
    let top_tags: Vec<TopTags> = npcs.iter().map(|e| extract_top_tags(e)).collect();

    // Track strongest friendship for chronicle.
    let mut best_friendship: Option<(f32, u32, u32)> = None; // (affinity, id_a, id_b)

    // Iterate all pairs.
    for i in 0..npcs.len() {
        for j in (i + 1)..npcs.len() {
            let affinity = compute_affinity(&top_tags[i], &top_tags[j]);

            let id_a = npcs[i].id;
            let id_b = npcs[j].id;

            if affinity > FRIENDSHIP_THRESHOLD {
                // Friendship: morale bonus for both.
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: id_a,
                    field: EntityField::Morale,
                    value: FRIENDSHIP_MORALE,
                });
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: id_b,
                    field: EntityField::Morale,
                    value: FRIENDSHIP_MORALE,
                });

                // Track strongest for chronicle.
                let dominated = match best_friendship {
                    Some((best, _, _)) => affinity > best,
                    None => true,
                };
                if dominated {
                    best_friendship = Some((affinity, id_a, id_b));
                }
            } else if affinity < RIVALRY_THRESHOLD {
                // Rivalry: only if both have significant combat experience.
                let npc_a = npcs[i].npc.as_ref().unwrap();
                let npc_b = npcs[j].npc.as_ref().unwrap();
                let combat_a = npc_a.behavior_value(tags::COMBAT);
                let combat_b = npc_b.behavior_value(tags::COMBAT);

                if combat_a > RIVALRY_COMBAT_MIN && combat_b > RIVALRY_COMBAT_MIN {
                    // Competitive motivation: add combat behavior tags to both.
                    let mut tags_a = ActionTags::empty();
                    tags_a.add(tags::COMBAT, RIVALRY_COMBAT_BONUS);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: id_a,
                        tags: tags_a.tags,
                        count: tags_a.count,
                    });

                    let mut tags_b = ActionTags::empty();
                    tags_b.add(tags::COMBAT, RIVALRY_COMBAT_BONUS);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: id_b,
                        tags: tags_b.tags,
                        count: tags_b.count,
                    });
                }
            }
        }
    }

    // Chronicle: record strongest friendship every 500 ticks.
    if state.tick % CHRONICLE_INTERVAL == 0 {
        if let Some((affinity, id_a, id_b)) = best_friendship {
            let settlement = match state.settlement(settlement_id) {
                Some(s) => s,
                None => return,
            };
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Narrative,
                    text: format!(
                        "A strong friendship has formed in {} (affinity {:.0}%)",
                        settlement.name,
                        affinity * 100.0,
                    ),
                    entity_ids: vec![id_a, id_b],
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Pre-computed top behavior tags for one NPC.
struct TopTags {
    /// Up to TOP_TAGS entries: (tag_hash, value), sorted by value descending.
    entries: [(u32, f32); TOP_TAGS],
    /// How many are populated.
    len: usize,
    /// Sum of all behavior values (not just top 5) for normalization.
    total: f32,
}

/// Extract the top-N behavior tags from an NPC entity.
fn extract_top_tags(entity: &Entity) -> TopTags {
    let npc = entity.npc.as_ref().unwrap();
    let mut entries = [(0u32, 0.0f32); TOP_TAGS];
    let mut len = 0usize;
    let mut total = 0.0f32;

    for &(tag_hash, val) in &npc.behavior_profile {
        total += val;

        // Insert into sorted top-N (descending by value).
        if len < TOP_TAGS {
            entries[len] = (tag_hash, val);
            len += 1;
            // Bubble into sorted position.
            let mut k = len - 1;
            while k > 0 && entries[k].1 > entries[k - 1].1 {
                entries.swap(k, k - 1);
                k -= 1;
            }
        } else if val > entries[TOP_TAGS - 1].1 {
            // Replace the smallest in top-N.
            entries[TOP_TAGS - 1] = (tag_hash, val);
            // Bubble into sorted position.
            let mut k = TOP_TAGS - 1;
            while k > 0 && entries[k].1 > entries[k - 1].1 {
                entries.swap(k, k - 1);
                k -= 1;
            }
        }
    }

    TopTags { entries, len, total }
}

/// Compute affinity between two NPCs as dot product of their top-5 tags
/// normalized by the sum of their total behavior values.
///
/// Returns 0.0 if either NPC has no tags.
fn compute_affinity(a: &TopTags, b: &TopTags) -> f32 {
    if a.len == 0 || b.len == 0 {
        return 0.0;
    }

    let norm = (a.total + b.total).max(1.0);

    let mut dot = 0.0f32;
    for i in 0..a.len {
        let (tag_a, val_a) = a.entries[i];
        for j in 0..b.len {
            if b.entries[j].0 == tag_a {
                dot += val_a * b.entries[j].1;
                break;
            }
        }
    }

    dot / (norm * norm * 0.25)
}
