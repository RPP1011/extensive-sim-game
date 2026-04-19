//! Library facade. Bin (`main.rs`) keeps winit-specific modules (`app`,
//! `state`) local because tests can't drive a real window; the rest
//! lives here for test reach.

pub mod grid_paint;
pub mod overlays;
pub mod palette;
pub mod scenario;
