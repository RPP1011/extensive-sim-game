//! Snapshot migration registry. One-step hop from an older schema hash to
//! the current one; chain composition is future work.

use super::format::SnapshotError;

pub type MigrationFn =
    Box<dyn Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync + 'static>;

pub struct MigrationRegistry {
    migrations: Vec<([u8; 32], [u8; 32], MigrationFn)>,
}

impl MigrationRegistry {
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
        }
    }

    /// Register a migration from `from` schema hash to `to` schema hash.
    pub fn register<F>(&mut self, from: [u8; 32], to: [u8; 32], f: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync + 'static,
    {
        self.migrations.push((from, to, Box::new(f)));
    }

    /// Apply a one-step migration if one exists from `from` → `to`.
    ///
    /// Chain composition (from → mid → to) is intentionally not implemented;
    /// add it when multi-step migrations are needed.
    pub fn apply(
        &self,
        from: [u8; 32],
        to: [u8; 32],
        bytes: &[u8],
    ) -> Result<Vec<u8>, SnapshotError> {
        for (f_from, f_to, f) in &self.migrations {
            if *f_from == from && *f_to == to {
                return f(bytes);
            }
        }
        Err(SnapshotError::MigrationFailed(
            "no one-step migration registered for this schema-hash pair",
        ))
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}
