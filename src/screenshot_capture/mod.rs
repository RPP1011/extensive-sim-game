mod types;
mod capture;

pub use types::{
    ScreenshotCaptureConfig, ScreenshotCaptureState, ScreenshotMode,
};

pub use capture::screenshot_capture_system;
