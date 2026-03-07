//! AST types for the ability DSL.
//!
//! These are intermediate representations produced by the parser and consumed
//! by the lowering pass to produce `AbilityDef` / `PassiveDef`.

/// A parsed `.ability` file containing ability and passive blocks.
#[derive(Debug, Clone)]
pub struct AbilityFile {
    pub items: Vec<TopLevel>,
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Ability(AbilityNode),
    Passive(PassiveNode),
}

/// A parsed `ability Name { ... }` block.
#[derive(Debug, Clone)]
pub struct AbilityNode {
    pub name: String,
    pub props: Vec<Property>,
    pub effects: Vec<EffectNode>,
    pub delivery: Option<DeliveryNode>,
    pub morph: Option<Box<MorphNode>>,
    pub recasts: Vec<RecastNode>,
}

/// A parsed `passive Name { ... }` block.
#[derive(Debug, Clone)]
pub struct PassiveNode {
    pub name: String,
    pub props: Vec<Property>,
    pub effects: Vec<EffectNode>,
}

/// A key-value header property like `target: enemy` or `cooldown: 5s`.
#[derive(Debug, Clone)]
pub struct Property {
    pub key: String,
    pub value: PropValue,
}

#[derive(Debug, Clone)]
pub enum PropValue {
    Ident(String),
    Number(f64),
    Duration(u32),
    StringLit(String),
    Bool(bool),
}

/// A single effect line like `damage 55 in circle(3.0) [FIRE: 60]`.
#[derive(Debug, Clone)]
pub struct EffectNode {
    pub effect_type: String,
    pub args: Vec<Arg>,
    pub area: Option<AreaNode>,
    pub tags: Vec<(String, f64)>,
    pub condition: Option<ConditionNode>,
    pub else_effects: Vec<EffectNode>,
    pub stacking: Option<String>,
    pub chance: Option<f64>,
    pub scaling: Vec<ScalingNode>,
    pub duration: Option<u32>,
    pub children: Vec<EffectNode>,
}

/// A positional argument to an effect.
#[derive(Debug, Clone)]
pub enum Arg {
    Number(f64),
    Duration(u32),
    Ident(String),
    StringLit(String),
}

impl Arg {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Arg::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Arg::Number(n) => Some(*n as i32),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Arg::Number(n) => Some(*n as u32),
            Arg::Duration(ms) => Some(*ms),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Arg::Ident(s) | Arg::StringLit(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_duration_ms(&self) -> Option<u32> {
        match self {
            Arg::Duration(ms) => Some(*ms),
            Arg::Number(n) => Some(*n as u32),
            _ => None,
        }
    }
}

/// An area modifier like `in circle(3.0)`.
#[derive(Debug, Clone)]
pub struct AreaNode {
    pub shape: String,
    pub args: Vec<f64>,
}

/// A condition like `when target_hp_below(30%)`.
#[derive(Debug, Clone)]
pub enum ConditionNode {
    Simple { name: String, args: Vec<CondArg> },
    And(Vec<ConditionNode>),
    Or(Vec<ConditionNode>),
    Not(Box<ConditionNode>),
}

#[derive(Debug, Clone)]
pub enum CondArg {
    Number(f64),
    Percent(f64),
    StringLit(String),
    Ident(String),
}

/// A scaling term like `+ 10% target_max_hp`.
#[derive(Debug, Clone)]
pub struct ScalingNode {
    pub percent: f64,
    pub stat: String,
    pub consume: bool,
    pub cap: Option<i32>,
}

/// A `deliver method { params } { hooks }` block.
#[derive(Debug, Clone)]
pub struct DeliveryNode {
    pub method: String,
    pub params: Vec<(String, Arg)>,
    pub on_hit: Vec<EffectNode>,
    pub on_arrival: Vec<EffectNode>,
    pub on_complete: Vec<EffectNode>,
}

/// A `morph into { ... } for Xs` block.
#[derive(Debug, Clone)]
pub struct MorphNode {
    pub inner: AbilityNode,
    pub duration_ms: u32,
}

/// A `recast N { ... }` block.
#[derive(Debug, Clone)]
pub struct RecastNode {
    pub index: u32,
    pub effects: Vec<EffectNode>,
}
