use super::categories::AbilityCategory;

// ---------------------------------------------------------------------------
// Evaluator weights (loaded from JSON)
// ---------------------------------------------------------------------------

/// Weights for a single ability category evaluator.
#[derive(Debug, Clone)]
pub struct EvalWeights {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl EvalWeights {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let layers = v["layers"].as_array().unwrap().iter().map(|layer| {
            let w: Vec<Vec<f32>> = serde_json::from_value(layer["w"].clone()).unwrap();
            let b: Vec<f32> = serde_json::from_value(layer["b"].clone()).unwrap();
            (w, b)
        }).collect();
        EvalWeights { layers }
    }

    /// Forward pass. Returns raw output (urgency logit for single-output, or urgency + target logits).
    pub fn predict(&self, features: &[f32]) -> Vec<f32> {
        let mut activations: Vec<f32> = features.to_vec();
        for (layer_idx, (weights, biases)) in self.layers.iter().enumerate() {
            let is_last = layer_idx == self.layers.len() - 1;
            let next: Vec<f32> = biases.iter().enumerate().map(|(j, &b)| {
                let sum: f32 = activations.iter().enumerate()
                    .map(|(i, &x)| x * weights[i][j]).sum();
                if is_last { sum + b } else { (sum + b).max(0.0) }
            }).collect();
            activations = next;
        }
        activations
    }
}

/// Full set of evaluator weights for all categories.
#[derive(Debug, Clone)]
pub struct AbilityEvalWeights {
    pub evaluators: std::collections::HashMap<AbilityCategory, EvalWeights>,
}

impl AbilityEvalWeights {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let mut evaluators = std::collections::HashMap::new();
        if let Some(obj) = v.as_object() {
            for (key, val) in obj {
                if let Some(cat) = AbilityCategory::from_name(key) {
                    evaluators.insert(cat, EvalWeights::from_json(val));
                }
            }
        }
        AbilityEvalWeights { evaluators }
    }
}
