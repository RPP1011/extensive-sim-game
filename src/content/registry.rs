use std::collections::HashMap;
use bevy::prelude::*;
use super::schema::ContentData;
use super::validation::{self, ValidationError};

/// Namespace for content items.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContentNamespace {
    /// Built-in static content (hero_templates, abilities, etc.)
    Base,
    /// AOT or runtime generated content for a campaign
    Gen,
    /// Modded content from an external source
    Mod(String),
}

impl std::fmt::Display for ContentNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentNamespace::Base => write!(f, "base"),
            ContentNamespace::Gen => write!(f, "gen"),
            ContentNamespace::Mod(name) => write!(f, "mod:{name}"),
        }
    }
}

/// Kind of content item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    HeroTemplate,
    EnemyTemplate,
    Ability,
    Faction,
    Settlement,
    Npc,
    Quest,
    Dialogue,
    Encounter,
    ScenarioConfig,
    // Tier 2 (Issue #15)
    Theme,
    Region,
    Event,
    Item,
    NarrativeArc,
}

impl std::fmt::Display for ContentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::HeroTemplate => "hero",
            Self::EnemyTemplate => "enemy",
            Self::Ability => "ability",
            Self::Faction => "faction",
            Self::Settlement => "settlement",
            Self::Npc => "npc",
            Self::Quest => "quest",
            Self::Dialogue => "dialogue",
            Self::Encounter => "encounter",
            Self::ScenarioConfig => "scenario",
            Self::Theme => "theme",
            Self::Region => "region",
            Self::Event => "event",
            Self::Item => "item",
            Self::NarrativeArc => "narrative_arc",
        };
        write!(f, "{s}")
    }
}

/// Unique identifier for a content item: `namespace:kind:name`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentId {
    pub namespace: ContentNamespace,
    pub kind: ContentKind,
    pub name: String,
}

impl ContentId {
    pub fn new(namespace: ContentNamespace, kind: ContentKind, name: impl Into<String>) -> Self {
        Self { namespace, kind, name: name.into() }
    }

    pub fn base(kind: ContentKind, name: impl Into<String>) -> Self {
        Self::new(ContentNamespace::Base, kind, name)
    }

    pub fn gen(kind: ContentKind, name: impl Into<String>) -> Self {
        Self::new(ContentNamespace::Gen, kind, name)
    }
}

impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.namespace, self.kind, self.name)
    }
}

/// Content lifecycle tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentTier {
    /// Loaded from static files (TOML, DSL). Never regenerated.
    Static,
    /// Generated ahead-of-time per campaign and cached to disk.
    AotGenerated,
    /// Generated on-demand at runtime. Not cached.
    RuntimeGenerated,
}

/// A single entry in the content registry.
pub struct ContentEntry {
    pub id: ContentId,
    pub tier: ContentTier,
    pub data: ContentData,
}

/// Typed content registry — the central store for all game content.
#[derive(Resource, Default)]
pub struct ContentRegistry {
    entries: HashMap<ContentId, ContentEntry>,
}

impl ContentRegistry {
    /// Insert a content entry, validating it first.
    pub fn insert(&mut self, entry: ContentEntry) -> Result<(), ValidationError> {
        validation::validate_entry(&entry)?;
        self.entries.insert(entry.id.clone(), entry);
        Ok(())
    }

    /// Insert without validation (for trusted static content).
    pub fn insert_unchecked(&mut self, entry: ContentEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Look up a content entry by ID.
    pub fn get(&self, id: &ContentId) -> Option<&ContentEntry> {
        self.entries.get(id)
    }

    /// Look up content data by ID.
    pub fn get_data(&self, id: &ContentId) -> Option<&ContentData> {
        self.entries.get(id).map(|e| &e.data)
    }

    /// Iterate all entries of a given kind.
    pub fn iter_kind(&self, kind: ContentKind) -> impl Iterator<Item = &ContentEntry> {
        self.entries.values().filter(move |e| e.id.kind == kind)
    }

    /// Iterate all entries in a given namespace.
    pub fn iter_namespace<'a>(&'a self, ns: &'a ContentNamespace) -> impl Iterator<Item = &'a ContentEntry> {
        self.entries.values().filter(move |e| &e.id.namespace == ns)
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove an entry by ID.
    pub fn remove(&mut self, id: &ContentId) -> Option<ContentEntry> {
        self.entries.remove(id)
    }

    /// Check if an entry exists.
    pub fn contains(&self, id: &ContentId) -> bool {
        self.entries.contains_key(id)
    }
}
