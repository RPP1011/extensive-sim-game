use std::path::Path;
use anyhow::{bail, Context, Result};
use glam::Vec3;
use serde::Deserialize;

use engine::creature::CreatureType;

#[derive(Debug, Deserialize)]
pub struct Scenario {
    #[serde(default)] pub world: World,
    #[serde(default)] pub agent: Vec<AgentSpec>,
}

#[derive(Debug, Deserialize)]
pub struct World {
    #[serde(default = "default_seed")]      pub seed:      u64,
    #[serde(default = "default_agent_cap")] pub agent_cap: u32,
}
fn default_seed() -> u64 { 42 }
fn default_agent_cap() -> u32 { 64 }

impl Default for World {
    fn default() -> Self { Self { seed: default_seed(), agent_cap: default_agent_cap() } }
}

#[derive(Debug, Deserialize)]
pub struct AgentSpec {
    pub creature_type: String,
    pub pos: [f32; 3],
    #[serde(default = "default_hp")] pub hp: f32,
}
fn default_hp() -> f32 { 100.0 }

impl AgentSpec {
    pub fn creature(&self) -> Result<CreatureType> {
        match self.creature_type.as_str() {
            "Human"  => Ok(CreatureType::Human),
            "Wolf"   => Ok(CreatureType::Wolf),
            "Deer"   => Ok(CreatureType::Deer),
            "Dragon" => Ok(CreatureType::Dragon),
            other => bail!("unknown creature_type {:?} (expected Human/Wolf/Deer/Dragon)", other),
        }
    }
    pub fn position(&self) -> Vec3 { Vec3::new(self.pos[0], self.pos[1], self.pos[2]) }
}

pub fn load<P: AsRef<Path>>(path: P) -> Result<Scenario> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("read {:?}", path))?;
    let s: Scenario = toml::from_str(&text).with_context(|| format!("parse {:?}", path))?;
    Ok(s)
}
