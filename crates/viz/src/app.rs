use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::state::{AppState, WINDOW_WIDTH, WINDOW_HEIGHT};

pub struct VizApp {
    pub scenario: Option<viz::scenario::Scenario>,
    pub state:    Option<AppState>,
}

impl VizApp {
    pub fn new(scenario: viz::scenario::Scenario) -> Self {
        Self { scenario: Some(scenario), state: None }
    }
}

impl ApplicationHandler for VizApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title("World Sim Viz")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => { eprintln!("[viz] create_window failed: {}", e); event_loop.exit(); return; }
        };
        let scenario = self.scenario.take().expect("scenario set before resumed");
        match AppState::new(window, scenario) {
            Ok(app) => { eprintln!("[viz] Ready."); self.state = Some(app); }
            Err(e)  => { eprintln!("[viz] AppState::new failed: {}", e); event_loop.exit(); }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let app = match &mut self.state { Some(a) => a, None => return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    app.handle_key(key, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    app.mouse_captured = state == ElementState::Pressed;
                    if !app.mouse_captured { app.last_mouse = None; }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if app.mouse_captured {
                    if let Some((lx, ly)) = app.last_mouse {
                        use voxel_engine::camera::{CameraController, InputState};
                        app.camera.update(&InputState {
                            move_forward: 0.0, move_right: 0.0, move_up: 0.0,
                            mouse_dx: (position.x - lx) as f32,
                            mouse_dy: (position.y - ly) as f32,
                            scroll_delta: 0.0,
                        }, 0.0);
                    }
                    app.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.run_frame();
        }
    }
}

pub fn run(scenario: viz::scenario::Scenario) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = VizApp::new(scenario);
    event_loop.run_app(&mut app)?;
    Ok(())
}
