//! Actor-critic MLP for combat AI.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use rl4burn::MaskedActorCritic;

use tactical_sim::sim::ability_eval::GAME_STATE_DIM;

const OBS_DIM: usize = GAME_STATE_DIM; // 210
const HIDDEN1: usize = 128;
const HIDDEN2: usize = 64;
const NUM_ACTIONS: usize = 14;

/// Simple two-layer MLP actor-critic.
///
/// ```text
/// obs (210) -> shared1 (128, relu) -> shared2 (64, relu) -> actor (14 logits)
///                                                        -> critic (1 value)
/// ```
#[derive(Module, Debug)]
pub struct CombatPolicy<B: Backend> {
    shared1: Linear<B>,
    shared2: Linear<B>,
    actor: Linear<B>,
    critic: Linear<B>,
}

impl<B: Backend> CombatPolicy<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            shared1: LinearConfig::new(OBS_DIM, HIDDEN1).init(device),
            shared2: LinearConfig::new(HIDDEN1, HIDDEN2).init(device),
            actor: LinearConfig::new(HIDDEN2, NUM_ACTIONS).init(device),
            critic: LinearConfig::new(HIDDEN2, 1).init(device),
        }
    }
}

impl<B: Backend> MaskedActorCritic<B> for CombatPolicy<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.shared1.forward(obs);
        let h = burn::tensor::activation::relu(h);
        let h = self.shared2.forward(h);
        let h = burn::tensor::activation::relu(h);

        let logits = self.actor.forward(h.clone());
        let values = self.critic.forward(h).squeeze_dim::<1>(1);

        (logits, values)
    }
}
