use super::companion::CompanionStoryState;
use super::overworld_types::*;
use super::roster_types::*;
use super::save::*;
use super::types::*;

// ── Save migration trait ──────────────────────────────────────────────────────

/// Trait that every top-level saveable type can implement to centralise
/// version-aware deserialisation.
///
/// Callers obtain a `serde_json::Value` from disk (already parsed but not yet
/// strongly typed) and hand it to [`Migrate::migrate`] together with the
/// `save_version` tag extracted from the envelope.  The implementation is
/// responsible for understanding all versions <= `CURRENT_VERSION` and
/// producing a correctly-typed value, or returning a human-readable error
/// string.
///
/// # Minimal contract
/// * `CURRENT_VERSION` -- the highest version number this type is aware of.
/// * `migrate(raw, from_version)` -- maps an arbitrary JSON value taken from a
///   save file at `from_version` into `Self`.  Implementations that do not
///   need version-specific field surgery may simply forward to
///   `serde_json::from_value(raw)`.
#[allow(dead_code)]
pub trait Migrate: Sized + serde::de::DeserializeOwned {
    /// The highest save-file schema version that this type understands.
    const CURRENT_VERSION: u32;

    /// Deserialise `raw` into `Self`, applying any structural fixes needed to
    /// bridge `from_version` -> `Self::CURRENT_VERSION`.
    ///
    /// Returns `Err(String)` when the version is too new to handle or the JSON
    /// is structurally incompatible.
    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String>;
}

// ── Blanket helper ────────────────────────────────────────────────────────────

/// Deserialise a `serde_json::Value` into `T`, converting serde errors to
/// `String` for uniform error handling.
#[allow(dead_code)]
fn from_value_erased<T: serde::de::DeserializeOwned>(v: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(v).map_err(|e| e.to_string())
}

// ── Implementations ───────────────────────────────────────────────────────────

/// `CampaignRoster` participates in the campaign save format from v1 onwards.
/// No structural field additions occurred between versions (backward
/// compatibility is handled entirely by `#[serde(default)]` attributes on the
/// struct fields), so all supported versions deserialise identically.
impl Migrate for CampaignRoster {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "CampaignRoster: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `OverworldMap` exists in all save versions.
impl Migrate for OverworldMap {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "OverworldMap: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `FlashpointState` was introduced in v2.
impl Migrate for FlashpointState {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "FlashpointState: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `CampaignParties` was added in v2.
impl Migrate for CampaignParties {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "CampaignParties: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `CampaignEventLog` was added in v2.
impl Migrate for CampaignEventLog {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "CampaignEventLog: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `CompanionStoryState` was added in v2.
impl Migrate for CompanionStoryState {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "CompanionStoryState: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `RunState` has been present and unchanged since v1.
impl Migrate for RunState {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "RunState: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `MissionSnapshot` is used in the `mission_snapshots` vec (added in v2).
impl Migrate for MissionSnapshot {
    const CURRENT_VERSION: u32 = 3;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        if from_version > Self::CURRENT_VERSION {
            return Err(format!(
                "MissionSnapshot: save version {} is newer than supported {}",
                from_version,
                Self::CURRENT_VERSION
            ));
        }
        from_value_erased(raw)
    }
}

/// `CampaignSaveData` is the top-level save envelope.
impl Migrate for CampaignSaveData {
    const CURRENT_VERSION: u32 = CURRENT_SAVE_VERSION;

    fn migrate(raw: serde_json::Value, from_version: u32) -> Result<Self, String> {
        let mut save: CampaignSaveData =
            from_value_erased(raw).map_err(|e| format!("CampaignSaveData: {e}"))?;
        save.save_version = from_version;
        migrate_campaign_save_data(save)
    }
}
