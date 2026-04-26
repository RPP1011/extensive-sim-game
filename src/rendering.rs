use voxel_engine::scene::{Scene, SceneConfig};
#[allow(unused_imports)]
use voxel_engine::camera::Camera;

pub struct GameRenderer {
    #[allow(dead_code)]
    scene: Scene,
}

impl GameRenderer {
    pub fn new() -> Self {
        Self {
            scene: Scene::new_headless(SceneConfig::default()),
        }
    }
}
