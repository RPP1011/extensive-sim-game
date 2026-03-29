//! Bankruptcy cascade system — every 500 ticks.
//!
//! When a major guild or faction defaults on debts, creditors face liquidity
//! shortfalls and may themselves default, creating systemic economic crises.
//! Tracks inter-faction debt exposure, declares defaults when a faction's
//! treasury falls below obligations, and propagates losses through the
//! credit network.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{lcg_f32, CampaignState};

/// How often to evaluate debt exposure and cascade risk (in ticks).
const CASCADE_INTERVAL: u64 = 17;

/// Number of defaults in a single cycle that triggers a credit freeze.
const CREDIT_FREEZE_THRESHOLD: usize = 2;

/// Duration of a credit freeze in ticks.
const CREDIT_FREEZE_DURATION: u32 = 2000;

/// Natural decay of systemic risk per tick.
const SYSTEMIC_RISK_DECAY: f32 = 0.01;

/// Systemic risk increase per default event.
const SYSTEMIC_RISK_PER_DEFAULT: f32 = 0.05;

/// Fraction of military strength used as proxy for faction treasury.
const TREASURY_STRENGTH_FACTOR: f32 = 50.0;

/// Debt-to-treasury ratio threshold that triggers a default.
const DEFAULT_THRESHOLD: f32 = 0.8;

/// State for the bankruptcy cascade system.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct BankruptcyCascadeState {
    /// Default events that occurred in the current evaluation cycle.
    pub defaults_this_cycle: Vec<DefaultEvent>,
    /// Remaining ticks of credit freeze (0 = no freeze).
    pub credit_freeze_ticks: u32,
    /// Systemic risk score (0.0–1.0). Affects trade, caravans, hiring costs.
    pub systemic_risk: f32,
    /// Inter-faction debt obligations: (debtor_faction_id, creditor_faction_id, amount).
    pub faction_debts: Vec<FactionDebt>,
    /// Factions currently in default (cannot take new loans).
    pub defaulted_factions: Vec<u32>,
}

/// A single default event in the cascade.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DefaultEvent {
    pub debtor_faction: u32,
    pub creditor_faction: u32,
    pub amount: f32,
    pub tick: u32,
}

/// An inter-faction debt obligation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FactionDebt {
    pub debtor_faction_id: u32,
    pub creditor_faction_id: u32,
    pub amount: f32,
}

/// Evaluate debt exposure and trigger cascades every 500 ticks.
pub fn tick_bankruptcy_cascade(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // --- Systemic risk decays every tick ---
    if state.bankruptcy_cascade.systemic_risk > 0.0 {
        state.bankruptcy_cascade.systemic_risk =
            (state.bankruptcy_cascade.systemic_risk - SYSTEMIC_RISK_DECAY).max(0.0);
    }

    // --- Credit freeze countdown ---
    if state.bankruptcy_cascade.credit_freeze_ticks > 0 {
        state.bankruptcy_cascade.credit_freeze_ticks =
            state.bankruptcy_cascade.credit_freeze_ticks.saturating_sub(1);
    }

    // --- Main cascade evaluation every 500 ticks ---
    if state.tick % CASCADE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Seed faction debts from existing loan relationships if debts are empty
    // (bootstrap: derive inter-faction lending from the guild's loan book)
    seed_faction_debts(state);

    // Clear previous cycle's defaults
    state.bankruptcy_cascade.defaults_this_cycle.clear();

    // Evaluate each faction's solvency
    let mut new_defaults: Vec<(u32, Vec<(u32, f32)>)> = Vec::new(); // (debtor, vec<(creditor, amount)>)

    // Compute per-faction total debt obligations
    let faction_ids: Vec<usize> = state.factions.iter().map(|f| f.id).collect();
    for &fid in &faction_ids {
        let fid32 = fid as u32;

        // Skip already-defaulted factions
        if state.bankruptcy_cascade.defaulted_factions.contains(&fid32) {
            continue;
        }

        // Estimate faction treasury from military strength
        let treasury = state
            .factions
            .iter()
            .find(|f| f.id == fid)
            .map(|f| f.military_strength * TREASURY_STRENGTH_FACTOR)
            .unwrap_or(0.0);

        // Sum total debt obligations for this faction
        let total_debt: f32 = state
            .bankruptcy_cascade
            .faction_debts
            .iter()
            .filter(|d| d.debtor_faction_id == fid32)
            .map(|d| d.amount)
            .sum();

        // Default check: treasury < threshold * total_debt
        if total_debt > 0.0 && treasury < total_debt * DEFAULT_THRESHOLD {
            // Collect creditor losses
            let creditor_losses: Vec<(u32, f32)> = state
                .bankruptcy_cascade
                .faction_debts
                .iter()
                .filter(|d| d.debtor_faction_id == fid32)
                .map(|d| (d.creditor_faction_id, d.amount))
                .collect();

            new_defaults.push((fid32, creditor_losses));
        }
    }

    // Process defaults and cascade
    let mut cascade_chain = 0u32;
    let mut total_losses = 0.0f32;
    let mut iterations = 0;

    while !new_defaults.is_empty() && iterations < 10 {
        iterations += 1;
        let mut next_round_defaults: Vec<(u32, Vec<(u32, f32)>)> = Vec::new();

        for (debtor, creditor_losses) in &new_defaults {
            cascade_chain += 1;

            // Mark faction as defaulted
            if !state.bankruptcy_cascade.defaulted_factions.contains(debtor) {
                state.bankruptcy_cascade.defaulted_factions.push(*debtor);
            }

            // Calculate total debt for the event
            let faction_total_debt: f32 = creditor_losses.iter().map(|(_, amt)| *amt).sum();

            events.push(WorldEvent::FactionDefaulted {
                faction_id: *debtor,
                total_debt: faction_total_debt,
            });

            // Process each creditor's loss
            for &(creditor_id, amount) in creditor_losses {
                total_losses += amount;

                // Record the default event
                state.bankruptcy_cascade.defaults_this_cycle.push(DefaultEvent {
                    debtor_faction: *debtor,
                    creditor_faction: creditor_id,
                    amount,
                    tick: state.tick as u32,
                });

                // Increase systemic risk
                state.bankruptcy_cascade.systemic_risk =
                    (state.bankruptcy_cascade.systemic_risk + SYSTEMIC_RISK_PER_DEFAULT).min(1.0);

                // Apply loss to creditor faction's military strength (proxy for treasury drain)
                let strength_loss = amount / TREASURY_STRENGTH_FACTOR;
                if let Some(creditor) = state.factions.iter_mut().find(|f| f.id == creditor_id as usize) {
                    creditor.military_strength = (creditor.military_strength - strength_loss).max(0.0);
                }

                // Check if creditor now cascades into default
                let creditor_fid = creditor_id;
                if !state.bankruptcy_cascade.defaulted_factions.contains(&creditor_fid)
                    && !next_round_defaults.iter().any(|(d, _)| *d == creditor_fid)
                {
                    let creditor_treasury = state
                        .factions
                        .iter()
                        .find(|f| f.id == creditor_fid as usize)
                        .map(|f| f.military_strength * TREASURY_STRENGTH_FACTOR)
                        .unwrap_or(0.0);

                    let creditor_debt: f32 = state
                        .bankruptcy_cascade
                        .faction_debts
                        .iter()
                        .filter(|d| d.debtor_faction_id == creditor_fid)
                        .map(|d| d.amount)
                        .sum();

                    if creditor_debt > 0.0 && creditor_treasury < creditor_debt * DEFAULT_THRESHOLD {
                        let cascaded_losses: Vec<(u32, f32)> = state
                            .bankruptcy_cascade
                            .faction_debts
                            .iter()
                            .filter(|d| d.debtor_faction_id == creditor_fid)
                            .map(|d| (d.creditor_faction_id, d.amount))
                            .collect();

                        next_round_defaults.push((creditor_fid, cascaded_losses));
                    }
                }
            }

            // Remove debts owed by the defaulting faction
            state.bankruptcy_cascade.faction_debts.retain(|d| d.debtor_faction_id != *debtor);
        }

        new_defaults = next_round_defaults;
    }

    // Emit cascade event if chain > 1
    if cascade_chain > 1 {
        events.push(WorldEvent::CascadeTriggered {
            chain_length: cascade_chain,
            total_losses,
        });
    }

    // --- Credit freeze check ---
    let defaults_count = state.bankruptcy_cascade.defaults_this_cycle.len();
    if defaults_count >= CREDIT_FREEZE_THRESHOLD && state.bankruptcy_cascade.credit_freeze_ticks == 0 {
        state.bankruptcy_cascade.credit_freeze_ticks = CREDIT_FREEZE_DURATION;

        events.push(WorldEvent::CreditFreeze {
            duration_ticks: CREDIT_FREEZE_DURATION,
        });

        // Double loan interest rates during credit freeze
        for loan in &mut state.loans {
            loan.interest_rate *= 2.0;
        }
    }

    // --- Guild impact: check if guild has loans from defaulting factions ---
    let defaulted_set: Vec<u32> = state.bankruptcy_cascade.defaulted_factions.clone();
    for &dfid in &defaulted_set {
        // Guild loses access to credit from defaulted factions
        // (tracked via defaulted_factions list — loan system checks this)

        // If guild is owed money by defaulted faction (guild invested/lent to them),
        // reduce guild gold proportionally
        let guild_exposure: f32 = state
            .bankruptcy_cascade
            .defaults_this_cycle
            .iter()
            .filter(|d| d.debtor_faction == dfid)
            .map(|d| d.amount * 0.1) // Guild exposure is a fraction of inter-faction debt
            .sum();

        if guild_exposure > 0.0 {
            state.guild.gold = (state.guild.gold - guild_exposure).max(0.0);
            events.push(WorldEvent::GoldChanged {
                amount: -guild_exposure,
                reason: format!(
                    "Loss from faction {} bankruptcy",
                    dfid
                ),
            });
        }
    }

    // --- Systemic risk affects hiring costs and trade ---
    // (Other systems read state.bankruptcy_cascade.systemic_risk)
    // Reduce caravan progress based on systemic risk (slows trade)
    if state.bankruptcy_cascade.systemic_risk > 0.3 {
        let disruption = state.bankruptcy_cascade.systemic_risk;
        for caravan in &mut state.caravans {
            caravan.progress *= 1.0 - disruption * 0.1; // Slow caravans by up to 10%
        }
    }

    // Generate new inter-faction debts occasionally (economic activity)
    generate_new_debts(state);
}

/// Bootstrap inter-faction debt network from the existing loan book
/// and faction relationships.
fn seed_faction_debts(state: &mut CampaignState) {
    if !state.bankruptcy_cascade.faction_debts.is_empty() {
        return; // Already seeded
    }

    let faction_count = state.factions.len();
    if faction_count < 2 {
        return;
    }

    // Create some inter-faction debts based on faction relationships
    let faction_ids: Vec<usize> = state.factions.iter().map(|f| f.id).collect();
    for i in 0..faction_ids.len() {
        for j in (i + 1)..faction_ids.len() {
            let r = lcg_f32(&mut state.rng);
            // ~30% chance of a debt relationship between any two factions
            if r < 0.3 {
                let amount = lcg_f32(&mut state.rng) * 100.0 + 20.0;
                let (debtor, creditor) = if lcg_f32(&mut state.rng) < 0.5 {
                    (faction_ids[i] as u32, faction_ids[j] as u32)
                } else {
                    (faction_ids[j] as u32, faction_ids[i] as u32)
                };

                state.bankruptcy_cascade.faction_debts.push(FactionDebt {
                    debtor_faction_id: debtor,
                    creditor_faction_id: creditor,
                    amount,
                });
            }
        }
    }
}

/// Generate new inter-faction debts from ongoing economic activity.
fn generate_new_debts(state: &mut CampaignState) {
    let faction_count = state.factions.len();
    if faction_count < 2 {
        return;
    }

    let r = lcg_f32(&mut state.rng);
    // 20% chance of a new debt forming each cycle
    if r > 0.2 {
        return;
    }

    let faction_ids: Vec<usize> = state.factions.iter().map(|f| f.id).collect();
    let idx_a = (lcg_f32(&mut state.rng) * faction_ids.len() as f32) as usize % faction_ids.len();
    let mut idx_b = (lcg_f32(&mut state.rng) * faction_ids.len() as f32) as usize % faction_ids.len();
    if idx_b == idx_a {
        idx_b = (idx_a + 1) % faction_ids.len();
    }

    let amount = lcg_f32(&mut state.rng) * 60.0 + 10.0;
    state.bankruptcy_cascade.faction_debts.push(FactionDebt {
        debtor_faction_id: faction_ids[idx_a] as u32,
        creditor_faction_id: faction_ids[idx_b] as u32,
        amount,
    });
}
