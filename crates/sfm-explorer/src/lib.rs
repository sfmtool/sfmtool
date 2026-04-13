// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SfM Explorer GUI application.
//!
//! A viewer for SfM reconstructions with 3D point cloud visualization,
//! camera frustums, and image browsing capabilities.
//!
//! Uses winit + wgpu directly with egui as an embedded UI renderer,
//! bypassing eframe. Windows DirectManipulation (precision touchpad
//! gestures) could not be made to work through eframe's event loop
//! and window management layers, but works when we own the event loop
//! and window creation directly.

mod app;
mod colormap;
mod dock;
mod image_browser;
mod image_detail;
mod platform;
mod point_track_detail;
mod scene_renderer;
mod state;
mod viewer_3d;

use std::sync::Arc;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};

use egui::ViewportId;
use egui_dock::{DockState, NodeIndex};
use image_browser::ImageBrowser;
use image_detail::ImageDetail;
use point_track_detail::PointTrackDetail;
use scene_renderer::SceneRenderer;
use state::AppState;
use viewer_3d::Viewer3D;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
#[cfg(target_os = "windows")]
use winit::event_loop::ControlFlow;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

use dock::Tab;

#[cfg(target_os = "windows")]
use platform::windows::{EarlyDmState, WinGestureHandler};

/// Interval for DirectManipulation update polling.
#[cfg(target_os = "windows")]
const DM_UPDATE_INTERVAL: Duration = Duration::from_millis(16);

/// Entry point for the SfM Explorer GUI application.
pub fn run() {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    env_logger::init();

    // Parse CLI args: sfm-explorer [path.sfmr]
    let mut state = AppState::new();
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let path = std::path::Path::new(&args[1]);
        state.load_file(path);
    }

    // Create DirectManipulation manager BEFORE the winit EventLoop so that
    // DM_POINTERHITTEST messages are generated for precision touchpad contacts.
    #[cfg(target_os = "windows")]
    let early_dm = match platform::windows::create_manager() {
        Ok(state) => Some(state),
        Err(e) => {
            log::warn!("Failed to create DirectManipulation manager: {:?}", e);
            None
        }
    };

    let event_loop = EventLoop::builder()
        .build()
        .expect("Failed to create event loop");

    // Set up the default dock layout:
    //   top-left: Viewer3D, top-right: ImageDetail, bottom: ImageBrowser
    let mut dock_state = DockState::new(vec![Tab::Viewer3D]);
    let surface = dock_state.main_surface_mut();
    let [top, _browser] = surface.split_below(NodeIndex::root(), 0.8, vec![Tab::ImageBrowser]);
    let [_viewer, _detail] =
        surface.split_right(top, 0.67, vec![Tab::ImageDetail, Tab::PointTrackDetail]);

    let mut app = App {
        egui_ctx: egui::Context::default(),
        egui_winit_state: None,
        window: None,
        // wgpu state
        wgpu_device: None,
        wgpu_queue: None,
        wgpu_surface: None,
        wgpu_surface_config: None,
        egui_renderer: None,
        // app state
        state,
        viewer_3d: Viewer3D::new(),
        image_browser: ImageBrowser::new(),
        image_detail: ImageDetail::new(),
        point_track_detail: PointTrackDetail::new(),
        scene_renderer: SceneRenderer::new(),
        dock_state,
        prev_frustum_length_scale: 0.0,
        prev_frustum_size_multiplier: 0.0,
        prev_selected_image: None,
        prev_selected_point: None,
        prev_hidden_image: None,
        #[cfg(target_os = "windows")]
        early_dm,
        #[cfg(target_os = "windows")]
        gesture_handler: None,
        #[cfg(target_os = "windows")]
        next_dm_update: None,
    };

    event_loop.run_app(&mut app).expect("Event loop failed");
}

pub(crate) struct App {
    pub(crate) egui_ctx: egui::Context,
    pub(crate) egui_winit_state: Option<egui_winit::State>,
    pub(crate) window: Option<Arc<Window>>,
    // Raw wgpu state (matching the working winit_wgpu test pattern)
    pub(crate) wgpu_device: Option<wgpu::Device>,
    pub(crate) wgpu_queue: Option<wgpu::Queue>,
    pub(crate) wgpu_surface: Option<wgpu::Surface<'static>>,
    pub(crate) wgpu_surface_config: Option<wgpu::SurfaceConfiguration>,
    pub(crate) egui_renderer: Option<eframe::egui_wgpu::Renderer>,
    // App state
    pub(crate) state: AppState,
    pub(crate) viewer_3d: Viewer3D,
    pub(crate) image_browser: ImageBrowser,
    pub(crate) image_detail: ImageDetail,
    pub(crate) point_track_detail: PointTrackDetail,
    pub(crate) scene_renderer: SceneRenderer,
    pub(crate) dock_state: DockState<Tab>,
    pub(crate) prev_frustum_length_scale: f32,
    pub(crate) prev_frustum_size_multiplier: f32,
    pub(crate) prev_selected_image: Option<usize>,
    pub(crate) prev_selected_point: Option<usize>,
    pub(crate) prev_hidden_image: Option<usize>,
    #[cfg(target_os = "windows")]
    pub(crate) early_dm: Option<EarlyDmState>,
    #[cfg(target_os = "windows")]
    pub(crate) gesture_handler: Option<WinGestureHandler>,
    #[cfg(target_os = "windows")]
    pub(crate) next_dm_update: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Step 1: Create window (raw winit, matching working test)
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("SfM Explorer")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
                        .with_min_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                )
                .expect("Failed to create window"),
        );

        self.window = Some(window.clone());

        // Step 2: Raw wgpu setup
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create wgpu surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find wgpu adapter");

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("Failed to create wgpu device");

        let size = window.inner_size();
        let surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Surface not supported by adapter");
        surface.configure(&device, &surface_config);

        // Step 3: Initialize DirectManipulation AFTER wgpu (matching working test order)
        #[cfg(target_os = "windows")]
        self.try_init_gesture_handler();

        // Step 4: Create egui renderer (uses raw wgpu device, not Painter)
        let egui_renderer = eframe::egui_wgpu::Renderer::new(
            &device,
            surface_config.format,
            eframe::egui_wgpu::RendererOptions::default(),
        );

        // Step 5: Set up egui-winit integration
        let max_texture_side = device.limits().max_texture_dimension_2d as usize;
        let egui_winit_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            event_loop.system_theme(),
            Some(max_texture_side),
        );

        // Repaint callback
        let repaint_window = window.clone();
        self.egui_ctx.set_request_repaint_callback(move |_info| {
            repaint_window.request_redraw();
        });

        self.wgpu_device = Some(device);
        self.wgpu_queue = Some(queue);
        self.wgpu_surface = Some(surface);
        self.wgpu_surface_config = Some(surface_config);
        self.egui_renderer = Some(egui_renderer);
        self.egui_winit_state = Some(egui_winit_state);

        // Schedule initial DM update and repaint
        #[cfg(target_os = "windows")]
        if let Some(next) = self.next_dm_update {
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        }

        window.request_redraw();
    }

    #[allow(unused_variables)]
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        // Drive DirectManipulation updates on a timer (matching the working test).
        #[cfg(target_os = "windows")]
        if self.gesture_handler.is_some() {
            let now = Instant::now();
            let should_update = match cause {
                winit::event::StartCause::ResumeTimeReached { .. } => true,
                winit::event::StartCause::WaitCancelled {
                    requested_resume: Some(deadline),
                    ..
                } if now >= deadline => true,
                _ => false,
            };

            if should_update {
                if let Some(handler) = self.gesture_handler.as_ref() {
                    handler.update();
                }
                self.next_dm_update = Some(now + DM_UPDATE_INTERVAL);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }

            if let Some(next) = self.next_dm_update {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(egui_winit_state) = self.egui_winit_state.as_mut() {
            if let Some(window) = self.window.as_ref() {
                let response = egui_winit_state.on_window_event(window, &event);
                if response.repaint {
                    window.request_redraw();
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let (Some(device), Some(surface), Some(config)) = (
                        self.wgpu_device.as_ref(),
                        self.wgpu_surface.as_ref(),
                        self.wgpu_surface_config.as_mut(),
                    ) {
                        config.width = size.width;
                        config.height = size.height;
                        surface.configure(device, config);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.run_ui_and_paint();
            }
            _ => {}
        }
    }

    #[allow(unused_variables)]
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(target_os = "windows")]
        if let Some(next) = self.next_dm_update {
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        }
    }
}
