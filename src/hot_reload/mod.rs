//! Hot-reload infrastructure for data-driven TOML config files.
//!
//! Polls file modification timestamps at a configurable interval and notifies
//! dependent systems via `HotReloadEvent` when changes are detected. No external
//! crate dependencies — uses only `std::fs::metadata()`.

use bevy::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Reload kind — identifies which system needs to react
// ---------------------------------------------------------------------------

/// Identifies which subsystem a hot-reloaded file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Event)]
pub enum HotReloadKind {
    MoraleCultures,
    NarrativeTemplates,
    FactionPersonalities,
    TerrainVisuals,
}

// ---------------------------------------------------------------------------
// Registry entry
// ---------------------------------------------------------------------------

struct HotReloadEntry {
    path: PathBuf,
    kind: HotReloadKind,
    last_modified: Option<SystemTime>,
}

impl HotReloadEntry {
    fn new(path: PathBuf, kind: HotReloadKind) -> Self {
        let last_modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok();
        Self {
            path,
            kind,
            last_modified,
        }
    }

    /// Returns `true` if the file was modified since last check.
    fn poll(&mut self) -> bool {
        let current = std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .ok();
        if current != self.last_modified {
            self.last_modified = current;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Bevy resource that tracks config files for hot-reload.
#[derive(Resource)]
pub struct HotReloadRegistry {
    entries: Vec<HotReloadEntry>,
    poll_interval_secs: f32,
    elapsed: f32,
}

impl Default for HotReloadRegistry {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            poll_interval_secs: 1.5,
            elapsed: 0.0,
        }
    }
}

impl HotReloadRegistry {
    /// Register a file path to watch for changes.
    pub fn watch(&mut self, path: impl Into<PathBuf>, kind: HotReloadKind) {
        let path = path.into();
        // Don't double-register
        if self.entries.iter().any(|e| e.path == path) {
            return;
        }
        info!("[hot-reload] Watching {:?} for {:?}", path, kind);
        self.entries.push(HotReloadEntry::new(path, kind));
    }

    /// Set the poll interval in seconds.
    pub fn set_poll_interval(&mut self, secs: f32) {
        self.poll_interval_secs = secs.max(0.1);
    }
}

// ---------------------------------------------------------------------------
// Bevy system
// ---------------------------------------------------------------------------

/// Bevy system that polls watched files and emits `HotReloadKind` events.
pub fn hot_reload_poll_system(
    time: Res<Time>,
    mut registry: ResMut<HotReloadRegistry>,
    mut events: EventWriter<HotReloadKind>,
) {
    registry.elapsed += time.delta_seconds();
    if registry.elapsed < registry.poll_interval_secs {
        return;
    }
    registry.elapsed = 0.0;

    for entry in &mut registry.entries {
        if entry.poll() {
            info!(
                "[hot-reload] Reloaded {}",
                entry.path.file_name().unwrap_or_default().to_string_lossy()
            );
            events.send(entry.kind);
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Bevy plugin that sets up the hot-reload polling system.
pub struct HotReloadPlugin;

impl Plugin for HotReloadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HotReloadRegistry>()
            .add_event::<HotReloadKind>()
            .add_systems(Update, hot_reload_poll_system);
    }
}

// ---------------------------------------------------------------------------
// Helper: load and parse a TOML file
// ---------------------------------------------------------------------------

/// Read and parse a TOML file into a deserializable type.
/// Returns `None` on any error (missing file, parse failure), logging a warning.
pub fn load_toml_file<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    match std::fs::read_to_string(path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(val) => Some(val),
            Err(e) => {
                warn!("[hot-reload] Failed to parse {}: {}", path.display(), e);
                None
            }
        },
        Err(e) => {
            warn!("[hot-reload] Failed to read {}: {}", path.display(), e);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_entry_detects_modification() {
        let dir = std::env::temp_dir().join("hot_reload_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");

        // Create initial file
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "value = 1").unwrap();
        }

        let mut entry = HotReloadEntry::new(path.clone(), HotReloadKind::MoraleCultures);

        // First poll after creation: no change since we just captured the timestamp
        assert!(!entry.poll());

        // Modify the file (need a small delay for filesystem timestamp granularity)
        std::thread::sleep(std::time::Duration::from_millis(50));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "value = 2").unwrap();
        }

        assert!(entry.poll());
        // Second poll without change
        assert!(!entry.poll());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_registry_no_double_watch() {
        let mut reg = HotReloadRegistry::default();
        reg.watch("/tmp/test.toml", HotReloadKind::MoraleCultures);
        reg.watch("/tmp/test.toml", HotReloadKind::MoraleCultures);
        assert_eq!(reg.entries.len(), 1);
    }
}
