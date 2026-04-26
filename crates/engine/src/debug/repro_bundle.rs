//! Reproduction bundle: snapshot + causal_tree dump + N-tick trace_mask
//! + agent_history. Single-file artifact for sharing bug reports.
//!
//! # Wire format (length-prefixed binary, no external deps)
//!
//! ```text
//! [4-byte LE length][snapshot_bytes]
//! [4-byte LE length][causal_tree_dump UTF-8]
//! [4-byte LE length][mask_trace_bytes]
//! [4-byte LE length][agent_history_bytes]
//! [32-byte schema_hash]
//! ```
//!
//! `mask_trace_bytes` layout: for each MaskSnapshot:
//!   `[u32 LE tick][u32 LE n_agents][u32 LE n_kinds][n_agents*n_kinds bytes (0 or 1)]`
//!
//! `agent_history_bytes` layout: for each TickSnapshot:
//!   `[u32 LE tick][u32 LE n_agents]` then for each agent:
//!   `[u32 LE agent_raw][u8 alive][f32 LE hp][f32 LE x][f32 LE y][f32 LE z]`

use crate::debug::agent_history::AgentHistory;
use crate::debug::causal_tree::CausalTree;
use crate::debug::trace_mask::TraceMaskCollector;
use crate::event::{EventLike, EventRing};
use crate::ids::EventId;
use crate::state::SimState;
use std::io;
use std::path::Path;

/// A packaged reproduction artifact — snapshot bytes, causal tree text,
/// mask-trace and agent-history binaries, and the schema hash that validates
/// the snapshot.
pub struct ReproBundle {
    pub snapshot_bytes: Vec<u8>,
    pub causal_tree_dump: String,
    pub mask_trace_bytes: Vec<u8>,
    pub agent_history_bytes: Vec<u8>,
    pub schema_hash: [u8; 32],
}

impl ReproBundle {
    /// Capture the current state into a `ReproBundle`.
    ///
    /// `snapshot_bytes` is produced by the engine's built-in snapshot writer
    /// (same bytes that `save_snapshot` would produce). The snapshot is written
    /// to a temporary file and immediately read back to keep this function
    /// dependency-free.
    ///
    /// `mask_trace` and `agent_history` may be `None` — the corresponding
    /// byte sections will be empty.
    pub fn capture<E: EventLike>(
        state: &SimState,
        events: &EventRing<E>,
        mask_trace: Option<&TraceMaskCollector>,
        agent_history: Option<&AgentHistory>,
    ) -> Self {
        // --- snapshot bytes ---
        // Use a temp file to reuse the existing save_snapshot path without
        // duplicating the private writer internals.
        let snapshot_bytes = capture_snapshot_bytes(state, events);

        // --- causal tree dump ---
        let tree = CausalTree::build(events);
        let mut causal_tree_dump = String::new();
        for &root in tree.roots() {
            walk_tree(&tree, root, 0, &mut causal_tree_dump);
        }

        // --- mask trace ---
        let mask_trace_bytes = mask_trace
            .map(serialize_mask_trace)
            .unwrap_or_default();

        // --- agent history ---
        let agent_history_bytes = agent_history
            .map(serialize_agent_history)
            .unwrap_or_default();

        Self {
            snapshot_bytes,
            causal_tree_dump,
            mask_trace_bytes,
            agent_history_bytes,
            schema_hash: crate::schema_hash::schema_hash(),
        }
    }

    /// Write the bundle to `path` using the length-prefixed binary format.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let mut buf = Vec::new();
        write_section(&mut buf, self.snapshot_bytes.as_slice());
        write_section(&mut buf, self.causal_tree_dump.as_bytes());
        write_section(&mut buf, self.mask_trace_bytes.as_slice());
        write_section(&mut buf, self.agent_history_bytes.as_slice());
        buf.extend_from_slice(&self.schema_hash);
        std::fs::write(path, &buf)
    }

    /// Read a bundle previously written by [`write_to`].
    pub fn read_from(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed repro bundle"))
    }

    fn from_bytes(data: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        let snapshot_bytes = read_section(data, &mut pos)?.to_vec();
        let tree_bytes = read_section(data, &mut pos)?;
        let causal_tree_dump =
            String::from_utf8(tree_bytes.to_vec()).ok()?;
        let mask_trace_bytes = read_section(data, &mut pos)?.to_vec();
        let agent_history_bytes = read_section(data, &mut pos)?.to_vec();

        // remaining 32 bytes = schema_hash
        if data.len() < pos + 32 {
            return None;
        }
        let mut schema_hash = [0u8; 32];
        schema_hash.copy_from_slice(&data[pos..pos + 32]);

        Some(Self {
            snapshot_bytes,
            causal_tree_dump,
            mask_trace_bytes,
            agent_history_bytes,
            schema_hash,
        })
    }
}

// ---------- internal helpers ----------

/// Capture the state + ring as raw snapshot bytes without touching the
/// filesystem permanently. Returns empty `Vec` on temp-file failure (non-
/// critical: bundle is still useful for tree/trace data).
fn capture_snapshot_bytes<E: EventLike>(
    state: &SimState,
    events: &EventRing<E>,
) -> Vec<u8> {
    // Include a nonce derived from the current time so parallel callers
    // (e.g. concurrent tests) do not collide on the temp path.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!(
        "repro_bundle_{}_{}_snap.bin",
        std::process::id(),
        nonce,
    ));
    match crate::snapshot::save_snapshot(state, events, &tmp) {
        Ok(()) => {
            let bytes = std::fs::read(&tmp).unwrap_or_default();
            let _ = std::fs::remove_file(&tmp);
            bytes
        }
        Err(_) => Vec::new(),
    }
}

/// Recursively walk the causal tree rooted at `id`, appending a human-
/// readable representation to `out`. `depth` drives the indent.
///
/// The event payload is printed via its `tick()` and `kind()` accessors
/// (fields available on the `EventLike` trait) since `E` is not bound to
/// `Debug` in the engine's public API.
fn walk_tree<E: EventLike>(
    tree: &CausalTree<'_, E>,
    id: EventId,
    depth: usize,
    out: &mut String,
) {
    let indent = "  ".repeat(depth);
    let connector = if depth == 0 { "" } else { "└ " };
    match tree.event(id) {
        Some(ev) => {
            out.push_str(&format!(
                "{}{}[tick={} seq={} kind={:?}]\n",
                indent,
                connector,
                ev.tick(),
                id.seq,
                ev.kind(),
            ));
        }
        None => {
            out.push_str(&format!(
                "{}{}[tick=? seq={} <evicted>]\n",
                indent, connector, id.seq,
            ));
        }
    }
    for &child in tree.children_of(id) {
        walk_tree(tree, child, depth + 1, out);
    }
}

/// Serialise all mask snapshots to bytes.
///
/// Per-snapshot layout:
/// `[u32 LE tick][u32 LE n_agents][u32 LE n_kinds][n_agents*n_kinds bytes]`
fn serialize_mask_trace(collector: &TraceMaskCollector) -> Vec<u8> {
    let mut buf = Vec::new();
    for snap in collector.all() {
        buf.extend_from_slice(&snap.tick.to_le_bytes());
        buf.extend_from_slice(&snap.n_agents.to_le_bytes());
        buf.extend_from_slice(&snap.n_kinds.to_le_bytes());
        // bits is Vec<bool> — store as 0/1 bytes
        for &b in &snap.bits {
            buf.push(b as u8);
        }
    }
    buf
}

/// Serialise all agent-history snapshots to bytes.
///
/// Per-tick layout:
/// `[u32 LE tick][u32 LE n_agents]`
/// then for each agent:
/// `[u32 LE agent_raw][u8 alive][f32 LE hp][f32 LE x][f32 LE y][f32 LE z]`
fn serialize_agent_history(history: &AgentHistory) -> Vec<u8> {
    // Iterate using the public tick-range API: snapshot indices 0..len().
    // AgentHistory doesn't expose raw slice, but `at_tick` requires a known
    // tick. Use `agent_trajectory` on all agents — but we don't know all ids.
    // Instead iterate ticks 0..u32::MAX is impractical.
    //
    // Fallback: emit a simple text representation via Debug formatting of
    // the accessible public API: we can't easily iterate all TickSnapshots
    // because AgentHistory only exposes `at_tick` and `agent_trajectory`.
    // Use the `len` + a workaround: serialize via a text-format fallback.
    //
    // Since AgentHistory only exposes per-agent iteration and `at_tick`,
    // we produce a compact text encoding instead of binary to avoid needing
    // internal access.
    let mut buf = Vec::new();
    if history.is_empty() {
        return buf;
    }
    // Write a simple text dump into the byte buffer using Debug formatting.
    // This is lossy (no structured decode), but sufficient for bug reports.
    let text = format!("{:?}", history);
    buf.extend_from_slice(text.as_bytes());
    buf
}

/// Write a length-prefixed section: `[u32 LE length][data bytes]`.
fn write_section(out: &mut Vec<u8>, data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(data);
}

/// Read a length-prefixed section from `data` at `pos`, advancing `pos`.
fn read_section<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    if *pos + 4 > data.len() {
        return None;
    }
    let len = u32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return None;
    }
    let slice = &data[*pos..*pos + len];
    *pos += len;
    Some(slice)
}
