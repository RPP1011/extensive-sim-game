//! Target type for GOAP intent resolution.
//!
//! Specifies which unit a GOAP action should target at execution time.

/// A target selector resolved at execution time against the current SimState.
#[derive(Debug, Clone)]
pub enum Target {
    Self_,
    NearestEnemy,
    NearestAlly,
    LowestHpEnemy,
    LowestHpAlly,
    HighestDpsEnemy,
    HighestThreatEnemy,
    CastingEnemy,
    EnemyAttacking(Box<Target>),
    Tagged(String),
    UnitId(u32),
}

/// Parse a target from a slice of tokens (e.g. `["nearest_enemy"]`).
pub fn parse_target(tokens: &[String]) -> Result<Target, String> {
    if tokens.is_empty() {
        return Err("expected target".into());
    }
    match tokens[0].as_str() {
        "self" => Ok(Target::Self_),
        "nearest_enemy" => Ok(Target::NearestEnemy),
        "nearest_ally" => Ok(Target::NearestAlly),
        "lowest_hp_enemy" => Ok(Target::LowestHpEnemy),
        "lowest_hp_ally" => Ok(Target::LowestHpAlly),
        "highest_dps_enemy" => Ok(Target::HighestDpsEnemy),
        "highest_threat_enemy" => Ok(Target::HighestThreatEnemy),
        "casting_enemy" => Ok(Target::CastingEnemy),
        "enemy_attacking" => {
            let inner = parse_target(&tokens[1..])?;
            Ok(Target::EnemyAttacking(Box::new(inner)))
        }
        "tagged" => {
            if tokens.len() < 2 {
                return Err("tagged requires a name".into());
            }
            let name = tokens[1].trim_matches('"').to_string();
            Ok(Target::Tagged(name))
        }
        "unit" => {
            if tokens.len() < 2 {
                return Err("unit requires an ID".into());
            }
            let id: u32 = tokens[1]
                .parse()
                .map_err(|_| format!("invalid unit ID '{}'", tokens[1]))?;
            Ok(Target::UnitId(id))
        }
        other => Err(format!("unknown target '{other}'")),
    }
}
