//! Loan and debt system — every 100 ticks.
//!
//! The guild can borrow gold from factions with interest.
//! Interest accrues over time; defaulting on loans damages faction
//! relations and credit rating.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{CampaignState, DiplomaticStance};

/// Cadence: every 100 ticks.
pub fn tick_loans(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 3 != 0 {
        return;
    }

    // dt in "loan periods" — each tick_loans call represents one period
    let dt = 1.0_f32;

    // Collect loan indices for processing (avoid borrow issues)
    let loan_count = state.loans.len();
    let mut defaulted_faction_ids: Vec<usize> = Vec::new();
    let mut paid_off_ids: Vec<u32> = Vec::new();

    for i in 0..loan_count {
        // --- Interest accrual ---
        let interest = state.loans[i].principal * state.loans[i].interest_rate * dt;
        state.loans[i].amount_owed += interest;

        // --- Auto-payment: if guild has gold > amount_owed * 2, pay 5% of owed ---
        let min_payment = state.loans[i].amount_owed * 0.05;
        if state.guild.gold > state.loans[i].amount_owed * 2.0 && min_payment > 0.0 {
            let payment = min_payment.min(state.loans[i].amount_owed);
            state.guild.gold -= payment;
            state.loans[i].amount_owed -= payment;
            state.loans[i].payments_made += 1;

            // Credit rating improves with on-time payments
            let old_rating = state.credit_rating;
            state.credit_rating = (state.credit_rating + 2.0).min(100.0);
            if (state.credit_rating - old_rating).abs() > 0.01 {
                events.push(WorldEvent::CreditRatingChanged {
                    old: old_rating,
                    new: state.credit_rating,
                });
            }

            events.push(WorldEvent::LoanRepaid {
                loan_id: state.loans[i].id,
                amount: payment,
                remaining: state.loans[i].amount_owed,
            });

            // Check if fully paid off
            if state.loans[i].amount_owed <= 0.01 {
                paid_off_ids.push(state.loans[i].id);
            }
        }

        // --- Due date check ---
        if state.tick > state.loans[i].due_tick && state.loans[i].amount_owed > 0.01 {
            let grace_end = state.loans[i].due_tick + 500;

            if state.tick <= grace_end {
                // Grace period: warning events (only every 100 ticks during grace)
                events.push(WorldEvent::CalamityWarning {
                    description: format!(
                        "Loan {} from faction {} is overdue! {:.0}g remaining. Grace period ends in {} ticks.",
                        state.loans[i].id,
                        state.loans[i].lender_faction_id,
                        state.loans[i].amount_owed,
                        grace_end.saturating_sub(state.tick),
                    ),
                });
            } else {
                // Default! Only fire once per loan per default cycle (mark via payments_made sentinel)
                // We use a simple approach: default fires once when grace expires
                if state.tick == grace_end + 100
                    || (state.tick > grace_end && (state.tick - grace_end) % 500 == 0)
                {
                    defaulted_faction_ids.push(state.loans[i].lender_faction_id);

                    events.push(WorldEvent::LoanDefaulted {
                        loan_id: state.loans[i].id,
                        lender_faction_id: state.loans[i].lender_faction_id,
                        amount_owed: state.loans[i].amount_owed,
                    });

                    // Credit rating drops
                    let old_rating = state.credit_rating;
                    state.credit_rating = (state.credit_rating - 20.0).max(0.0);
                    events.push(WorldEvent::CreditRatingChanged {
                        old: old_rating,
                        new: state.credit_rating,
                    });
                }
            }
        }
    }

    // Remove paid-off loans
    state.loans.retain(|l| !paid_off_ids.contains(&l.id));

    // Apply faction relation penalties for defaults
    for fid in &defaulted_faction_ids {
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *fid) {
            let old = faction.relationship_to_guild;
            faction.relationship_to_guild = (faction.relationship_to_guild - 30.0).max(-100.0);
            events.push(WorldEvent::FactionRelationChanged {
                faction_id: *fid,
                old,
                new: faction.relationship_to_guild,
            });

            // Faction may declare war if relation drops below -50
            if faction.relationship_to_guild < -50.0
                && faction.diplomatic_stance != DiplomaticStance::AtWar
            {
                faction.diplomatic_stance = DiplomaticStance::AtWar;
                let guild_fid = state.diplomacy.guild_faction_id;
                if !faction.at_war_with.contains(&guild_fid) {
                    faction.at_war_with.push(guild_fid);
                }
                events.push(WorldEvent::FactionActionTaken {
                    faction_id: *fid,
                    action: "Declared war over unpaid debts!".into(),
                });
            }
        }
    }
}

/// Determine the interest rate for a new loan based on credit rating.
pub fn interest_rate_for_credit(credit_rating: f32) -> f32 {
    if credit_rating > 70.0 {
        0.05 // 5%
    } else if credit_rating >= 30.0 {
        0.10 // 10%
    } else {
        0.20 // 20%
    }
}

/// Check if a faction is willing to lend to the guild.
pub fn faction_will_lend(faction_relation: f32, credit_rating: f32) -> bool {
    // Requires relation > 20 and credit rating > 10
    faction_relation > 20.0 && credit_rating > 10.0
}

/// Maximum loan amount a faction will offer based on credit rating and relation.
pub fn max_loan_amount(credit_rating: f32, faction_relation: f32) -> f32 {
    let base = 100.0;
    let credit_mult = if credit_rating > 70.0 {
        3.0
    } else if credit_rating >= 30.0 {
        2.0
    } else {
        1.0
    };
    let relation_mult = (faction_relation / 100.0).max(0.2);
    base * credit_mult * relation_mult
}
