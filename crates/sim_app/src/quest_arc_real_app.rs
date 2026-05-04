//! Quest Arc Real harness — drives `quest_arc_real_runtime` for 500
//! ticks with 30 Adventurers and reports per-N-tick stage-distribution
//! traces + final per-agent quests-completed counts.
//!
//! ## Predicted observable shape
//!
//! Each Adventurer cycles 0→1→2→3→4→0 driven by per-stage cooldown
//! gates (`world.tick % {3,4,3,5,1} == 0`). Worst-case latency for a
//! full quest: 3+4+3+5+1 = 16 ticks; phase-aligned best case is
//! shorter. Over 500 ticks, expect ~25-30 quests per agent ≈ 750-900
//! total.
//!
//! ## Stage register encoding
//!
//! mana = stage as f32:
//!   0.0 = Accept (waiting at town)
//!   1.0 = Hunt   (seeking monster)
//!   2.0 = Collect (picking up item)
//!   3.0 = Return  (walking home)
//!   4.0 = Complete (just before reset to 0)

use engine::CompiledSim;
use quest_arc_real_runtime::QuestArcRealState;

const SEED: u64 = 0xC0FFEE_F00D_BEEF;
const AGENT_COUNT: u32 = 30;
const MAX_TICKS: u64 = 500;
const TRACE_EVERY: u64 = 50;

const STAGE_NAMES: [&str; 5] = ["Accept", "Hunt", "Collect", "Return", "Complete"];

fn stage_distribution(mana: &[f32]) -> [u32; 5] {
    let mut counts = [0u32; 5];
    for &m in mana {
        // mana is exactly representable as 0.0..=4.0 in f32 (small
        // integers); round defensively to handle any +/-eps drift
        // from the CAS+add path even though the only writes are
        // either `mana = 0.0` or `mana = mana + 1.0` (both exact
        // in IEEE-754 for small integer values).
        let stage = m.round() as i32;
        if (0..=4).contains(&stage) {
            counts[stage as usize] += 1;
        }
    }
    counts
}

fn print_stage_line(tick: u64, mana: &[f32], total_quests: f32) {
    let counts = stage_distribution(mana);
    let total: u32 = counts.iter().sum();
    println!(
        "Tick {:>4}: stages [Accept={:>2} Hunt={:>2} Collect={:>2} Return={:>2} Complete={:>2}] (total agents={}) | quests_completed={:.0}",
        tick, counts[0], counts[1], counts[2], counts[3], counts[4], total, total_quests,
    );
}

fn main() {
    let mut sim = QuestArcRealState::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" QUEST ARC REAL — 30 Adventurers, 5-stage per-agent state machine");
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!("   stages: 0=Accept(cd3) → 1=Hunt(cd4) → 2=Collect(cd3) → 3=Return(cd5) → 4=Complete(cd1) → reset");
    println!("================================================================");

    // Initial state
    let mana0 = sim.read_mana();
    print_stage_line(0, &mana0, 0.0);

    // Track when we observe at least one full cycle (some agent
    // completing > 0 quests).
    let mut first_completion_tick: Option<u64> = None;

    for tick in 1..=MAX_TICKS {
        sim.step();

        if tick % TRACE_EVERY == 0 || tick == 1 || tick == 16 {
            let mana = sim.read_mana();
            let qc = sim.quests_completed().to_vec();
            let total: f32 = qc.iter().sum();
            print_stage_line(tick, &mana, total);
            if first_completion_tick.is_none() && total > 0.0 {
                first_completion_tick = Some(tick);
            }
        } else if first_completion_tick.is_none() {
            // Cheap check (no GPU sync of mana) — just peek at the
            // quests_completed view, which is mark_dirty'd each step
            // but only does a real readback on first access. To avoid
            // a per-tick sync we only check every 5 ticks until we
            // see the first completion.
            if tick % 5 == 0 {
                let qc_total: f32 = sim.quests_completed().iter().sum();
                if qc_total > 0.0 {
                    first_completion_tick = Some(tick);
                }
            }
        }
    }

    // Final readout.
    let final_mana = sim.read_mana();
    let final_qc = sim.quests_completed().to_vec();
    let final_sa = sim.stage_advances().to_vec();
    let total_quests: f32 = final_qc.iter().sum();
    let total_advances: f32 = final_sa.iter().sum();
    let final_stages = stage_distribution(&final_mana);

    println!();
    println!("================================================================");
    println!(" RESULTS — after {} ticks", MAX_TICKS);
    println!("================================================================");
    println!(" Final stage distribution:");
    for (i, name) in STAGE_NAMES.iter().enumerate() {
        let bar = "#".repeat(final_stages[i] as usize);
        println!("   {:>8} ({}): {:>2}  {}", name, i, final_stages[i], bar);
    }
    println!();
    println!(" Total quests completed : {:.0}", total_quests);
    println!(" Total stage advances   : {:.0}  (expected ≈ 4 × quests_completed = {:.0})",
        total_advances, total_quests * 4.0);
    if total_quests > 0.0 {
        println!(" Avg ticks per quest    : {:.1}", (MAX_TICKS as f32) / (total_quests / AGENT_COUNT as f32));
    }
    if let Some(t) = first_completion_tick {
        println!(" First quest completed  : tick {}", t);
    } else {
        println!(" First quest completed  : NEVER (no quests completed)");
    }

    println!();
    println!(" Per-agent quests_completed (30 slots):");
    for chunk in final_qc.chunks(10).enumerate() {
        let (i, c) = chunk;
        let s: String = c.iter().map(|v| format!("{:>4.0}", v)).collect::<Vec<_>>().join(" ");
        println!("   slots {:>2}..{:>2}: {}", i * 10, i * 10 + c.len(), s);
    }

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");

    // Demonstrates: state-machine progression actually happens (>0
    // stage advances) AND at least one full cycle (>0 quests
    // completed) AND multiple agents progressed (not just slot 0).
    let agents_with_completed_quest = final_qc.iter().filter(|&&v| v > 0.0).count();
    let agents_advanced = final_sa.iter().filter(|&&v| v > 0.0).count();

    if total_quests > 0.0 && agents_with_completed_quest >= (AGENT_COUNT as usize) / 2 {
        println!(
            "  (a) FULL FIRE — multi-stage state machine runs end-to-end. \
            {} of {} agents completed at least one full quest cycle ({:.0} \
            total quests, {:.0} total stage advances). Per-agent state \
            machine via mana-as-stage register works at scale.",
            agents_with_completed_quest, AGENT_COUNT,
            total_quests, total_advances,
        );
    } else if total_quests > 0.0 {
        println!(
            "  (a-partial) PARTIAL FIRE — {} agents completed a quest \
            but most did not. Likely the mask-kernel `target == self` \
            asymmetry (TODO task-5.7) bit harder than expected. \
            ({} agents advanced any stage; {} completed a full cycle.)",
            agents_with_completed_quest,
            agents_advanced, agents_with_completed_quest,
        );
    } else if total_advances > 0.0 {
        println!(
            "  (b) STAGE ADVANCES BUT NO COMPLETIONS — the cycle stalls \
            before reaching mana==4 OR the QuestCompleted chronicle \
            doesn't fold. ({:.0} stage advances total.)",
            total_advances,
        );
    } else {
        println!(
            "  (b) NO PROGRESSION — neither stage advances nor quest \
            completions accumulated. The verb cascade isn't reaching \
            either chronicle for any of the 5 verbs.",
        );
    }

    // Hard assert: state-machine progression must happen.
    assert!(
        total_advances > 0.0,
        "quest_arc_real_app: ASSERTION FAILED — no stage advances \
         recorded. The verb cascade did not reach the chronicle for \
         any of the 5 stage-gated verbs.",
    );
    assert!(
        total_quests > 0.0,
        "quest_arc_real_app: ASSERTION FAILED — no quest completed. \
         The state machine never reached mana==4 → QuestCompleted.",
    );
    // Demand at least 2 agents progressed — anything less suggests
    // we hit the slot-0-only mask asymmetry.
    assert!(
        agents_advanced >= 2,
        "quest_arc_real_app: ASSERTION FAILED — only {} agent(s) \
         advanced any stage. Expected most/all of {} adventurers to \
         progress. Likely the mask-kernel `target == self` asymmetry \
         (TODO task-5.7) bit — the verb predicates need to be \
         self-only (no `target == self` clause).",
        agents_advanced, AGENT_COUNT,
    );
}
