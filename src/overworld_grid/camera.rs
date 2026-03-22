//! Overworld camera with smooth pan/zoom interpolation.

use serde::{Deserialize, Serialize};

/// Overworld camera state with smooth interpolation toward target values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverworldCamera {
    /// Current world position (top-left, in cell coordinates).
    pub pos_x: f32,
    pub pos_y: f32,
    /// Target position for smooth interpolation.
    pub target_x: f32,
    pub target_y: f32,
    /// Current zoom level (1.0 = default).
    pub zoom: f32,
    /// Target zoom for smooth interpolation.
    pub target_zoom: f32,
    /// Grid dimensions for clamping.
    pub grid_w: f32,
    pub grid_h: f32,
}

impl Default for OverworldCamera {
    fn default() -> Self {
        Self {
            pos_x: 0.0,
            pos_y: 0.0,
            target_x: 0.0,
            target_y: 0.0,
            zoom: 1.0,
            target_zoom: 1.0,
            grid_w: 150.0,
            grid_h: 80.0,
        }
    }
}

impl OverworldCamera {
    pub fn new(grid_w: f32, grid_h: f32) -> Self {
        Self {
            grid_w,
            grid_h,
            ..Default::default()
        }
    }

    /// Update camera position and zoom with smooth interpolation.
    /// `dt` is delta time in seconds. The lerp factor of 6.0 means the camera
    /// reaches ~95% of target position within ~0.5 seconds.
    pub fn update(&mut self, dt: f32) {
        let lerp_factor = 6.0 * dt;
        let lerp_factor = lerp_factor.min(1.0); // prevent overshoot at very low fps

        self.pos_x += (self.target_x - self.pos_x) * lerp_factor;
        self.pos_y += (self.target_y - self.pos_y) * lerp_factor;
        self.zoom += (self.target_zoom - self.zoom) * lerp_factor;

        self.clamp();
    }

    /// Pan the camera by a delta in cell coordinates.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.target_x += dx;
        self.target_y += dy;
        self.clamp_target();
    }

    /// Zoom by a delta amount (positive = zoom in, negative = zoom out).
    pub fn zoom_by(&mut self, delta: f32) {
        self.target_zoom = (self.target_zoom + delta).clamp(0.5, 2.0);
    }

    /// Center the camera on a specific cell position.
    pub fn center_on(&mut self, x: f32, y: f32, viewport_w: f32, viewport_h: f32) {
        self.target_x = x - viewport_w / (2.0 * self.zoom);
        self.target_y = y - viewport_h / (2.0 * self.zoom);
        self.clamp_target();
    }

    /// Get the visible rectangle in cell coordinates: (x0, y0, x1, y1).
    pub fn visible_rect(&self, viewport_w: f32, viewport_h: f32) -> (i32, i32, i32, i32) {
        let cells_w = (viewport_w / self.zoom).ceil() as i32;
        let cells_h = (viewport_h / self.zoom).ceil() as i32;
        let x0 = self.pos_x.floor() as i32;
        let y0 = self.pos_y.floor() as i32;
        let x1 = (x0 + cells_w).min(self.grid_w as i32);
        let y1 = (y0 + cells_h).min(self.grid_h as i32);
        (x0.max(0), y0.max(0), x1, y1)
    }

    fn clamp(&mut self) {
        self.pos_x = self.pos_x.clamp(0.0, (self.grid_w - 1.0).max(0.0));
        self.pos_y = self.pos_y.clamp(0.0, (self.grid_h - 1.0).max(0.0));
        self.zoom = self.zoom.clamp(0.5, 2.0);
    }

    fn clamp_target(&mut self) {
        self.target_x = self.target_x.clamp(0.0, (self.grid_w - 1.0).max(0.0));
        self.target_y = self.target_y.clamp(0.0, (self.grid_h - 1.0).max(0.0));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_default() {
        let cam = OverworldCamera::default();
        assert_eq!(cam.zoom, 1.0);
        assert_eq!(cam.pos_x, 0.0);
    }

    #[test]
    fn test_smooth_interpolation() {
        let mut cam = OverworldCamera::new(150.0, 80.0);
        cam.target_x = 50.0;
        cam.target_y = 30.0;

        // Simulate ~0.5 seconds
        for _ in 0..30 {
            cam.update(1.0 / 60.0);
        }

        // Should be close to target (~95%)
        assert!((cam.pos_x - 50.0).abs() < 3.0);
        assert!((cam.pos_y - 30.0).abs() < 2.0);
    }

    #[test]
    fn test_zoom_clamping() {
        let mut cam = OverworldCamera::default();
        cam.zoom_by(10.0);
        cam.update(1.0);
        assert!(cam.zoom <= 2.0);

        cam.zoom_by(-10.0);
        cam.update(1.0);
        assert!(cam.zoom >= 0.5);
    }

    #[test]
    fn test_visible_rect() {
        let mut cam = OverworldCamera::new(150.0, 80.0);
        cam.pos_x = 10.0;
        cam.pos_y = 5.0;
        cam.zoom = 1.0;

        let (x0, y0, x1, y1) = cam.visible_rect(80.0, 40.0);
        assert_eq!(x0, 10);
        assert_eq!(y0, 5);
        assert!(x1 <= 150);
        assert!(y1 <= 80);
        assert!(x1 - x0 >= 70); // ~80 cells visible
    }

    #[test]
    fn test_pan() {
        let mut cam = OverworldCamera::new(150.0, 80.0);
        cam.pan(10.0, 5.0);
        assert_eq!(cam.target_x, 10.0);
        assert_eq!(cam.target_y, 5.0);
    }
}
