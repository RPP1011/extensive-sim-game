use engine_data::entities::CreatureType;
use engine::ids::AgentId;
use engine::obs::{FeatureSource, ObsPacker};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

struct ConstantSource(f32);
impl FeatureSource for ConstantSource {
    fn dim(&self) -> usize {
        2
    }
    fn pack(&self, _state: &SimState, _agent: AgentId, out: &mut [f32]) {
        out[0] = self.0;
        out[1] = self.0 + 1.0;
    }
}

#[test]
fn packer_computes_total_feature_dim() {
    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(1.0)));
    packer.register(Box::new(ConstantSource(2.0)));
    assert_eq!(packer.feature_dim(), 4);
}

#[test]
fn pack_batch_writes_row_major_per_agent() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::X,
            hp: 80.0,
            max_hp: 100.0,
        })
        .unwrap();

    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(5.0))); // dim 2 → [5, 6]
    packer.register(Box::new(ConstantSource(10.0))); // dim 2 → [10, 11]

    let mut out = vec![0.0f32; 2 * 4];
    packer.pack_batch(&state, &[a, b], &mut out);

    // Row 0 = agent a: [5,6,10,11]. Row 1 = agent b: same.
    assert_eq!(&out[..4], &[5.0, 6.0, 10.0, 11.0]);
    assert_eq!(&out[4..], &[5.0, 6.0, 10.0, 11.0]);
}

#[test]
fn pack_batch_panics_on_wrong_output_size() {
    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(1.0)));

    let state = SimState::new(4, 42);
    let mut out = vec![0.0f32; 1]; // too small for even 1 agent × 2 dim
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        packer.pack_batch(&state, &[AgentId::new(1).unwrap()], &mut out);
    }));
    assert!(result.is_err(), "expected panic on undersized buffer");
}

#[test]
fn empty_packer_has_zero_feature_dim() {
    let packer = ObsPacker::new();
    assert_eq!(packer.feature_dim(), 0);
    // pack_batch with zero-dim and zero-len output is a no-op.
    let state = SimState::new(2, 0);
    let mut out: Vec<f32> = vec![];
    packer.pack_batch(&state, &[], &mut out);
}
