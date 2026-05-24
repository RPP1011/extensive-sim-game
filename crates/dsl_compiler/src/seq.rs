//! Shared deterministic seq computation for event payloads.
//!
//! Both the GPU emit (inlined into WGSL) and the CPU cascade dispatch
//! call this helper with identical inputs and get identical u32 outputs.
//! This is the load-bearing parity primitive for P11 sort-then-fold.

/// Pack the three identifiers into a u32 seq value.
///
/// Layout (MSB → LSB):
///   bits 24..32: producer_kernel_id   (8 bits, ≤ 256 emit-producer kernels)
///   bits  4..24: producer_thread_id   (20 bits, ≤ 1M threads — megaswarm headroom)
///   bits  0..4:  intra_emit_idx       (4 bits, ≤ 16 emit stmts/thread/tick)
///
/// Callers MUST pass values within the documented ranges; out-of-range
/// values silently alias other (producer_kernel_id, thread, idx) tuples
/// and break determinism.
pub fn compute_event_seq(
    producer_kernel_id: u32,
    producer_thread_id: u32,
    intra_emit_idx: u32,
) -> u32 {
    debug_assert!(producer_kernel_id < 256, "producer_kernel_id exceeds 8-bit range");
    debug_assert!(producer_thread_id < (1 << 20), "producer_thread_id exceeds 20-bit range");
    debug_assert!(intra_emit_idx < 16, "intra_emit_idx exceeds 4-bit range");

    (producer_kernel_id << 24) | (producer_thread_id << 4) | intra_emit_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_three_fields_into_distinct_u32() {
        let a = compute_event_seq(0, 0, 0);
        let b = compute_event_seq(0, 0, 1);
        let c = compute_event_seq(0, 1, 0);
        let d = compute_event_seq(1, 0, 0);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 16);  // 1 << 4
        assert_eq!(d, 1 << 24);
    }

    #[test]
    fn max_legal_values_pack_without_overlap() {
        let max = compute_event_seq(255, (1 << 20) - 1, 15);
        assert_eq!(max, u32::MAX);
    }

    #[test]
    fn sort_order_matches_kernel_then_thread_then_idx() {
        // Higher kernel_id dominates; within kernel, higher thread_id
        // dominates; within thread, higher emit_idx dominates.
        let a = compute_event_seq(0, 0, 5);
        let b = compute_event_seq(0, 1, 0);
        let c = compute_event_seq(1, 0, 0);
        assert!(a < b);
        assert!(b < c);
    }
}
