use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::state::{AppState, WINDOW_WIDTH, WINDOW_HEIGHT};

pub struct VizApp {
    pub scenario:      Option<viz::scenario::Scenario>,
    pub scenario_path: Option<std::path::PathBuf>,
    pub state:         Option<AppState>,
}

impl VizApp {
    pub fn new(scenario: viz::scenario::Scenario, scenario_path: std::path::PathBuf) -> Self {
        Self { scenario: Some(scenario), scenario_path: Some(scenario_path), state: None }
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
        let scenario_path = self.scenario_path.take().expect("scenario_path set before resumed");
        match AppState::new(window, scenario, scenario_path) {
            Ok(app) => {
                eprintln!("[viz] Ready. Controls:");
                eprintln!("       WASD        — pan camera horizontally");
                eprintln!("       Q / E       — lower / raise camera");
                eprintln!("       RMB drag    — look (rotate view around eye)");
                eprintln!("       MMB drag    — orbit around scene center");
                eprintln!("       Scroll      — zoom toward / away from scene center");
                eprintln!("       Space       — pause/resume");
                eprintln!("       .           — single step (paused only)");
                eprintln!("       R           — reload scenario from disk");
                eprintln!("       [ / ]       — halve/double sim speed");
                eprintln!("       Esc         — quit");
                self.state = Some(app);
            }
            Err(e) => { eprintln!("[viz] AppState::new failed: {}", e); event_loop.exit(); }
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
                match button {
                    MouseButton::Right => {
                        app.mouse_captured = state == ElementState::Pressed;
                        if !app.mouse_captured { app.last_mouse = None; }
                    }
                    MouseButton::Middle => {
                        app.orbit_captured = state == ElementState::Pressed;
                        if !app.orbit_captured { app.last_orbit = None; }
                    }
                    _ => {}
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
                if app.orbit_captured {
                    if let Some((lx, ly)) = app.last_orbit {
                        app.orbit_camera(
                            (position.x - lx) as f32,
                            (position.y - ly) as f32,
                        );
                    }
                    app.last_orbit = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                app.zoom_camera(scroll);
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

pub fn run(scenario: viz::scenario::Scenario, scenario_path: std::path::PathBuf) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = VizApp::new(scenario, scenario_path);
    event_loop.run_app(&mut app)?;
    Ok(())
}
