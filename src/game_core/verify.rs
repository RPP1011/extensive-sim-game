use super::verify_details;

// ---------------------------------------------------------------------------
// Campaign-layer violation types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum CampaignViolation {
    // --- Roster ---
    /// Two heroes share the same ID.
    DuplicateHeroId { hero_id: u32 },
    /// `next_id` is not greater than every existing hero ID.
    RosterNextIdTooLow { next_id: u32, max_existing: u32 },
    /// Hero stat is outside the valid [0, 100] range.
    HeroStatOutOfRange { hero_id: u32, stat: &'static str, value_x100: i32 },
    /// Recruit candidate ID collides with an active hero.
    RecruitIdCollidesWithHero { recruit_id: u32 },
    /// Player hero ID doesn't reference any hero in the roster.
    InvalidPlayerHeroId { id: u32 },
    /// Hero marked as both active and deserter.
    ActiveDeserter { hero_id: u32 },

    // --- Parties ---
    /// Party leader references a hero that doesn't exist or is inactive/deserted.
    InvalidPartyLeader { party_id: u32, leader_hero_id: u32 },
    /// Party `order_target_region_id` references a region that doesn't exist.
    PartyOrderTargetInvalid { party_id: u32, region_id: usize },
    /// Party's current region is out of bounds.
    PartyRegionOutOfBounds { party_id: u32, region_id: usize, num_regions: usize },
    /// Duplicate party IDs.
    DuplicatePartyId { party_id: u32 },
    /// No player-controlled party exists.
    NoPlayerParty,
    /// Multiple player-controlled parties.
    MultiplePlayerParties { count: usize },
    /// Party supply is negative.
    NegativePartySupply { party_id: u32, supply_x100: i32 },

    // --- Overworld ---
    /// Region neighbor references a region ID that doesn't exist.
    RegionNeighborOutOfBounds { region_id: usize, neighbor_id: usize, num_regions: usize },
    /// Region owner faction doesn't exist.
    RegionOwnerInvalid { region_id: usize, faction_id: usize, num_factions: usize },
    /// Region unrest is outside [0, 100].
    RegionUnrestOutOfRange { region_id: usize, value_x100: i32 },
    /// Region control is outside [0, 100].
    RegionControlOutOfRange { region_id: usize, value_x100: i32 },
    /// `current_region` index is out of bounds.
    CurrentRegionOutOfBounds { index: usize, num_regions: usize },

    // --- Diplomacy ---
    /// Diplomacy relation matrix is not square or doesn't match faction count.
    DiplomacyMatrixMismatch { rows: usize, expected: usize },
    /// Diplomacy diagonal entry is non-zero (faction relation to itself).
    DiplomacyDiagonalNonZero { faction_id: usize, value: i32 },
    /// Player faction ID is out of bounds.
    PlayerFactionOutOfBounds { player_faction_id: usize, num_factions: usize },

    // --- Missions ---
    /// Completed mission is still marked active.
    CompletedMissionStillActive { mission_name: String },
    /// Mission reactor integrity is negative.
    NegativeReactorIntegrity { mission_name: String, value_x100: i32 },
    /// Mission room index exceeds the room count.
    MissionRoomIndexOutOfBounds { mission_name: String, room_index: usize, num_rooms: usize },
    /// Mission bound to a region that doesn't exist.
    MissionBoundRegionInvalid { mission_name: String, region_id: usize },

    // --- Flashpoints ---
    /// Flashpoint stage is outside [1, 3].
    FlashpointStageInvalid { chain_id: u32, stage: u8 },
    /// Flashpoint references a region that doesn't exist.
    FlashpointRegionInvalid { chain_id: u32, region_id: usize },
    /// Flashpoint attacker or defender faction doesn't exist.
    FlashpointFactionInvalid { chain_id: u32, faction_id: usize },

    // --- Companion quests ---
    /// Quest references a hero that isn't in the roster.
    QuestHeroMissing { quest_id: u32, hero_id: u32 },
    /// Quest progress exceeds target.
    QuestProgressExceedsTarget { quest_id: u32, progress: u32, target: u32 },

    // --- Attention ---
    /// Attention energy exceeds max.
    AttentionEnergyExceedsMax { energy_x100: i32, max_x100: i32 },
    /// Attention energy is negative.
    NegativeAttentionEnergy { energy_x100: i32 },
}

#[derive(Debug, Clone)]
pub struct CampaignVerificationReport {
    pub violations: Vec<CampaignViolation>,
}

impl CampaignVerificationReport {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Run all campaign-layer verification checks against a save data snapshot.
pub fn verify_campaign(save: &super::save::CampaignSaveData) -> CampaignVerificationReport {
    let mut violations = Vec::new();

    violations.extend(verify_details::verify_roster(&save.campaign_roster));
    violations.extend(verify_details::verify_parties(
        &save.campaign_parties,
        &save.campaign_roster,
        &save.overworld_map,
    ));
    violations.extend(verify_details::verify_overworld(&save.overworld_map));
    violations.extend(verify_details::verify_diplomacy(
        &save.diplomacy_state,
        save.overworld_map.factions.len(),
    ));
    violations.extend(verify_details::verify_missions(
        &save.mission_snapshots,
        &save.overworld_map,
        &save.mission_map,
    ));
    violations.extend(verify_details::verify_flashpoints(
        &save.flashpoint_state,
        &save.overworld_map,
    ));
    violations.extend(verify_details::verify_companion_quests(
        &save.companion_story_state,
        &save.campaign_roster,
    ));
    violations.extend(verify_details::verify_attention(&save.attention_state));

    CampaignVerificationReport { violations }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::overworld_types::*;
    use super::super::roster_types::*;
    #[allow(unused_imports)]
    use super::super::types::*;

    fn minimal_roster() -> CampaignRoster {
        CampaignRoster {
            heroes: vec![
                HeroCompanion {
                    id: 1,
                    name: "Test Hero".into(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Vanguard,
                    loyalty: 50.0,
                    stress: 10.0,
                    fatigue: 5.0,
                    injury: 0.0,
                    resolve: 60.0,
                    active: true,
                    deserter: false,
                    xp: 0,
                    level: 1,
                    equipment: HeroEquipment::default(),
                    traits: Vec::new(),
                },
            ],
            recruit_pool: Vec::new(),
            player_hero_id: Some(1),
            next_id: 2,
            generation_counter: 0,
        }
    }

    #[test]
    fn clean_roster_passes() {
        let roster = minimal_roster();
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.is_empty(), "{:?}", violations);
    }

    #[test]
    fn detects_duplicate_hero_ids() {
        let mut roster = minimal_roster();
        let mut clone = roster.heroes[0].clone();
        clone.name = "Duplicate".into();
        roster.heroes.push(clone);
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::DuplicateHeroId { hero_id: 1 }
        )));
    }

    #[test]
    fn detects_next_id_too_low() {
        let mut roster = minimal_roster();
        roster.next_id = 1; // should be > 1
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::RosterNextIdTooLow { .. }
        )));
    }

    #[test]
    fn detects_hero_stat_out_of_range() {
        let mut roster = minimal_roster();
        roster.heroes[0].loyalty = 150.0;
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::HeroStatOutOfRange { hero_id: 1, stat: "loyalty", .. }
        )));
    }

    #[test]
    fn detects_active_deserter() {
        let mut roster = minimal_roster();
        roster.heroes[0].active = true;
        roster.heroes[0].deserter = true;
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::ActiveDeserter { hero_id: 1 }
        )));
    }

    #[test]
    fn detects_invalid_player_hero_id() {
        let mut roster = minimal_roster();
        roster.player_hero_id = Some(999);
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::InvalidPlayerHeroId { id: 999 }
        )));
    }

    #[test]
    fn detects_recruit_id_collision() {
        let mut roster = minimal_roster();
        roster.recruit_pool.push(RecruitCandidate {
            id: 1, // same as hero
            codename: "Spy".into(),
            origin_faction_id: 0,
            origin_region_id: 0,
            backstory: String::new(),
            archetype: PersonalityArchetype::Tactician,
            resolve: 50.0,
            loyalty_bias: 0.5,
            risk_tolerance: 0.5,
        });
        let violations = verify_details::verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::RecruitIdCollidesWithHero { recruit_id: 1 }
        )));
    }

    #[test]
    fn detects_diplomacy_diagonal_nonzero() {
        let diplomacy = DiplomacyState {
            player_faction_id: 0,
            relations: vec![vec![5, 10], vec![10, 0]],
        };
        let violations = verify_details::verify_diplomacy(&diplomacy, 2);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::DiplomacyDiagonalNonZero { faction_id: 0, value: 5 }
        )));
    }

    #[test]
    fn detects_attention_energy_exceeds_max() {
        let attention = AttentionState {
            global_energy: 150.0,
            max_energy: 100.0,
            ..AttentionState::default()
        };
        let violations = verify_details::verify_attention(&attention);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::AttentionEnergyExceedsMax { .. }
        )));
    }
}
