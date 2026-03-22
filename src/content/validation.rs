use super::registry::{ContentEntry, ContentKind};
use super::schema::ContentData;

/// Error returned when content validation fails.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub content_id: String,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "content validation failed for '{}': {}", self.content_id, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validate a content entry for consistency.
pub fn validate_entry(entry: &ContentEntry) -> Result<(), ValidationError> {
    let id_str = entry.id.to_string();

    // Verify data variant matches declared kind
    let kind_matches = matches!(
        (&entry.id.kind, &entry.data),
        (ContentKind::HeroTemplate, ContentData::HeroTemplate(_))
            | (ContentKind::EnemyTemplate, ContentData::EnemyTemplate(_))
            | (ContentKind::Ability, ContentData::Ability(_))
            | (ContentKind::Faction, ContentData::Faction(_))
            | (ContentKind::Settlement, ContentData::Settlement(_))
            | (ContentKind::Npc, ContentData::Npc(_))
            | (ContentKind::Quest, ContentData::Quest(_))
            | (ContentKind::Dialogue, ContentData::Dialogue(_))
            | (ContentKind::Encounter, ContentData::Encounter(_))
            | (ContentKind::ScenarioConfig, ContentData::ScenarioConfig(_))
    );

    if !kind_matches {
        return Err(ValidationError {
            content_id: id_str,
            message: "content data variant does not match declared ContentKind".to_string(),
        });
    }

    // Kind-specific validation
    match &entry.data {
        ContentData::HeroTemplate(h) => {
            if h.name.is_empty() {
                return Err(ValidationError {
                    content_id: id_str,
                    message: "hero template name is empty".to_string(),
                });
            }
            if h.hp <= 0.0 {
                return Err(ValidationError {
                    content_id: id_str,
                    message: format!("hero HP must be positive, got {}", h.hp),
                });
            }
        }
        ContentData::EnemyTemplate(e) => {
            if e.name.is_empty() {
                return Err(ValidationError {
                    content_id: id_str,
                    message: "enemy template name is empty".to_string(),
                });
            }
            if e.hp <= 0.0 {
                return Err(ValidationError {
                    content_id: id_str,
                    message: format!("enemy HP must be positive, got {}", e.hp),
                });
            }
        }
        ContentData::Ability(a) => {
            if a.name.is_empty() {
                return Err(ValidationError {
                    content_id: id_str,
                    message: "ability name is empty".to_string(),
                });
            }
        }
        ContentData::Quest(q) => {
            if q.name.is_empty() {
                return Err(ValidationError {
                    content_id: id_str,
                    message: "quest name is empty".to_string(),
                });
            }
            if q.objectives.is_empty() {
                return Err(ValidationError {
                    content_id: id_str,
                    message: "quest must have at least one objective".to_string(),
                });
            }
        }
        _ => {}
    }

    Ok(())
}
