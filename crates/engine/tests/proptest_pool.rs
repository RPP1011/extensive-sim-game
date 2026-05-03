//! Property: `Pool<T>` maintains invariants across arbitrary sequences of
//! alloc/kill/repeat-kill ops. See Task 3 of Plan 2.75.
use engine::pool::{Pool, PoolId};
use proptest::prelude::*;

struct AgentTag;

#[derive(Copy, Clone, Debug)]
enum Op {
    Alloc,
    /// Kill a slot by *slot index* (0-based). If `slot` is out of range or
    /// already dead, it's a no-op — matches the pool's real contract.
    Kill(usize),
}

fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Alloc),
        (0usize..16).prop_map(Op::Kill),
    ]
}

/// Assert the three pool invariants. Returns on first violation via
/// `prop_assert!`.
fn assert_invariants(pool: &Pool<AgentTag>) -> Result<(), TestCaseError> {
    // (1) is_non_overlapping — no slot both alive and in freelist; no
    //     duplicates on the freelist.
    prop_assert!(
        pool.is_non_overlapping(),
        "invariant violated: alive ∩ freelist or freelist has duplicate"
    );
    // (2) freelist has no duplicates (explicit second check independent of
    //     is_non_overlapping's implementation, for redundancy).
    let mut seen = vec![false; pool.cap() as usize];
    for raw in pool.freelist_iter() {
        let slot = (raw - 1) as usize;
        prop_assert!(slot < pool.cap() as usize, "freelist contains out-of-range raw {}", raw);
        prop_assert!(!seen[slot], "freelist contains duplicate raw {}", raw);
        seen[slot] = true;
    }
    // (3) monotone counter identity: count_alive + freelist_len = next_raw - 1.
    let n_alive = pool.alive.iter().filter(|a| **a).count();
    let n_free  = pool.freelist_len();
    let n_ever  = pool.next_raw().saturating_sub(1) as usize;
    prop_assert_eq!(
        n_alive + n_free, n_ever,
        "monotone-counter identity broken: alive={} + free={} != next_raw-1={}",
        n_alive, n_free, n_ever,
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// Running an arbitrary length-1-200 sequence of alloc/kill ops against a
    /// fresh pool preserves: alive/freelist disjoint, no freelist dupes, and
    /// the monotone-counter identity `alive + free = next_raw - 1`.
    #[test]
    fn pool_invariants_hold_across_random_op_sequence(
        cap in 1u32..=16,
        ops in proptest::collection::vec(arb_op(), 1..200),
    ) {
        let mut pool: Pool<AgentTag> = Pool::new(cap);
        for op in ops {
            match op {
                Op::Alloc => { let _ = pool.alloc(); } // may fail at cap; OK.
                Op::Kill(slot) => {
                    if let Some(id) = PoolId::<AgentTag>::new((slot as u32) + 1) {
                        if (slot as u32) < cap {
                            pool.kill(id);
                        }
                    }
                }
            }
            assert_invariants(&pool)?;
        }
    }

    /// Allocating past `cap` returns `None` without corrupting invariants.
    /// Specifically exercises the overflow path that the unit tests touch
    /// once; proptest touches it many times with randomized priors.
    #[test]
    fn alloc_past_cap_returns_none_and_preserves_invariants(
        cap in 1u32..=8,
        extra_attempts in 1u32..=16,
    ) {
        let mut pool: Pool<AgentTag> = Pool::new(cap);
        for _ in 0..cap { prop_assert!(pool.alloc().is_some()); }
        for _ in 0..extra_attempts {
            prop_assert!(pool.alloc().is_none());
            assert_invariants(&pool)?;
        }
    }

}
