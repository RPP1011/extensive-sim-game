use voxel_engine::scene::{Scene, SceneConfig};
use voxel_engine::camera::Camera;

pub struct GameRenderer {
    scene: Scene,
}

impl GameRenderer {
    pub fn new() -> Self {
        Self {
            scene: Scene::new_headless(SceneConfig::default()),
        }
    }
}
