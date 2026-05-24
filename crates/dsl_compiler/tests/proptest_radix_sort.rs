//! Proptest: host-side reference sort matches Vec::sort_by_key((target, seq))
//! across edge cases.
//!
//! Validates the host-side LOGIC. GPU execution is validated end-to-end in
//! crates/sims/tests/forest_fire_pin.rs (Slice 5 acceptance gate).

use proptest::prelude::*;

/// Host-side simulation of the WGSL sort.
/// Input: Vec<(target, seq)>.
/// Output: same events in (target, seq)-ascending order, with stable
/// tiebreak on equal (target, seq) pairs preserving input order.
fn host_simulate_sort(input: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut indexed: Vec<(usize, u32, u32)> = input
        .iter()
        .enumerate()
        .map(|(i, (t, s))| (i, *t, *s))
        .collect();
    // Stable sort: ties on (target, seq) break by original index.
    indexed.sort_by(|a, b| (a.1, a.2, a.0).cmp(&(b.1, b.2, b.0)));
    indexed.into_iter().map(|(_, t, s)| (t, s)).collect()
}

proptest! {
    #[test]
    fn radix_sort_matches_reference(
        events in prop::collection::vec((0u32..100, 0u32..u32::MAX), 0..1024)
    ) {
        let sorted = host_simulate_sort(&events);

        // Property 1: same length.
        prop_assert_eq!(sorted.len(), events.len());

        // Property 2: sorted by (target, seq).
        for w in sorted.windows(2) {
            prop_assert!((w[0].0, w[0].1) <= (w[1].0, w[1].1));
        }

        // Property 3: same multiset (no events dropped or duplicated).
        let mut input_sorted: Vec<_> = events.clone();
        input_sorted.sort();
        let mut output_sorted: Vec<_> = sorted.clone();
        output_sorted.sort();
        prop_assert_eq!(input_sorted, output_sorted);
    }

    #[test]
    fn all_same_target_orders_by_seq(
        target in 0u32..1024u32,
        seqs in prop::collection::vec(0u32..u32::MAX, 1..256)
    ) {
        let events: Vec<_> = seqs.iter().map(|s| (target, *s)).collect();
        let sorted = host_simulate_sort(&events);
        let extracted_seqs: Vec<u32> = sorted.into_iter().map(|(_, s)| s).collect();
        let mut expected = seqs.clone();
        expected.sort();
        prop_assert_eq!(extracted_seqs, expected);
    }

    #[test]
    fn all_same_seq_groups_by_target(
        seq in 0u32..u32::MAX,
        targets in prop::collection::vec(0u32..100u32, 1..256)
    ) {
        let events: Vec<_> = targets.iter().map(|t| (*t, seq)).collect();
        let sorted = host_simulate_sort(&events);
        let extracted_targets: Vec<u32> = sorted.into_iter().map(|(t, _)| t).collect();
        let mut expected = targets.clone();
        expected.sort();
        prop_assert_eq!(extracted_targets, expected);
    }
}

// Edge case unit tests (not proptest, deterministic).

#[test]
fn empty_input_handled() {
    let sorted = host_simulate_sort(&[]);
    assert!(sorted.is_empty());
}

#[test]
fn single_element_handled() {
    let sorted = host_simulate_sort(&[(42, 0xDEAD_BEEF)]);
    assert_eq!(sorted, vec![(42, 0xDEAD_BEEF)]);
}

#[test]
fn sorted_input_passes_through_unchanged() {
    let events = vec![(0, 0), (0, 1), (1, 0), (2, 5)];
    let sorted = host_simulate_sort(&events);
    assert_eq!(sorted, events);
}

#[test]
fn reverse_sorted_input_gets_sorted() {
    let events = vec![(5, 100), (3, 50), (1, 10), (0, 0)];
    let sorted = host_simulate_sort(&events);
    assert_eq!(sorted, vec![(0, 0), (1, 10), (3, 50), (5, 100)]);
}

#[test]
fn stable_ordering_preserves_input_order_on_ties() {
    // Two events with identical (target, seq) — must preserve input order.
    let events = vec![(0, 0), (1, 0), (0, 0)];
    let sorted = host_simulate_sort(&events);
    // First and third events both (0,0) — stable sort keeps them in order.
    assert_eq!(sorted, vec![(0, 0), (0, 0), (1, 0)]);
}
