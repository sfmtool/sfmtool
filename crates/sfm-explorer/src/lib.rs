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

mod action_log;
mod align;
mod app;
mod cli;
mod colormap;
mod dock;
mod goto_point;
mod image_browser;
mod image_detail;
mod intrinsics_detail;
mod layout;
#[cfg(feature = "mcp")]
mod mcp;
mod metrics;
mod platform;
mod point_track_detail;
mod resect;
mod scene;
mod scene_graph;
mod scene_renderer;
mod state;
#[cfg(test)]
mod test_support;
mod texture;
mod viewer_3d;
mod window;

use std::sync::Arc;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};

use egui::ViewportId;

use image_browser::ImageBrowser;
use image_detail::ImageDetail;
use intrinsics_detail::IntrinsicsDetail;
use point_track_detail::PointTrackDetail;
use scene::{CameraRef, ImageRef, PointRef};
use scene_graph::SceneGraphPanel;
use scene_renderer::SceneRenderer;
use state::AppState;
use viewer_3d::Viewer3D;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
#[cfg(target_os = "windows")]
use winit::event_loop::ControlFlow;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes};

#[cfg(target_os = "windows")]
use platform::windows::{EarlyDmState, WinGestureHandler};

/// Interval for DirectManipulation update polling.
#[cfg(target_os = "windows")]
const DM_UPDATE_INTERVAL: Duration = Duration::from_millis(16);

/// User event type that carries AccessKit events through the winit event loop.
#[derive(Debug)]
pub(crate) enum UserEvent {
    AccessKit(egui_winit::accesskit_winit::Event),
    /// An MCP tool call is waiting on the command channel.
    ///
    /// Carries nothing: the request itself travels over the channel, and this
    /// is only the wake. `App::user_event` requests a redraw for any user
    /// event, and `run_ui_and_paint` drains the whole queue, so one wake covers
    /// however many requests arrived alongside it.
    #[cfg(feature = "mcp")]
    McpRequest,
}

impl From<egui_winit::accesskit_winit::Event> for UserEvent {
    fn from(event: egui_winit::accesskit_winit::Event) -> Self {
        Self::AccessKit(event)
    }
}

/// Whether the UI tests asked us to render continuously. The harness sets
/// `SFMTOOL_EXPLORER_FORCE_REPAINT` on the spawned process: without it the app
/// renders a couple of frames and goes idle, and the accessibility tree can be
/// queried before egui has fully published it. Kept out of normal runs so the
/// idle viewer doesn't spin the CPU.
fn force_repaint_for_tests() -> bool {
    use std::sync::OnceLock;
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("SFMTOOL_EXPLORER_FORCE_REPAINT").is_some())
}

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

    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if args.help {
        print!("{}", cli::USAGE);
        return;
    }

    // The first log entry, before anything is loaded: a session read back later
    // — next to an agent's transcript, or in a bug report — should say what it
    // was and when it started.
    let mut state = AppState::new();
    state.action_log.record_as(
        action_log::Actor::Viewer,
        action_log::Kind::Session,
        format!("SfM Explorer {} started", env!("CARGO_PKG_VERSION")),
    );

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

    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");
    let proxy = event_loop.create_proxy();

    // The MCP endpoint, if it was asked for. Started after the event loop
    // exists, because the server's only way to reach the viewer is the proxy
    // this hands it; started before the window, so a port collision is reported
    // on a terminal rather than behind a window that came up looking fine.
    #[cfg(feature = "mcp")]
    let mcp_rx = start_mcp(&mut state, args.mcp_port, &proxy);
    #[cfg(not(feature = "mcp"))]
    start_mcp(&mut state, args.mcp_port, &proxy);

    // Every path is loaded as its own scene node, in the order given — after
    // the endpoint has been brought up, so the Action Log reads in the order
    // the session happened and a file named on the command line sits under the
    // session lines that explain where it came from.
    for path in &args.paths {
        if let Err(message) = state.load_file(path) {
            state.action_log.fail(action_log::Kind::File, message);
        }
    }

    let mut app = App {
        proxy,
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
        scene_graph: SceneGraphPanel::new(),
        image_browser: ImageBrowser::new(),
        image_detail: ImageDetail::new(),
        point_track_detail: PointTrackDetail::new(),
        intrinsics_detail: IntrinsicsDetail::new(),
        scene_renderer: SceneRenderer::new(),
        prev_frustum_length_scale: 0.0,
        prev_frustum_size_multiplier: 0.0,
        prev_selected_image: None,
        prev_selected_camera: None,
        prev_selected_point: None,
        prev_hidden_image: None,
        prev_transform_epoch: 0,
        quit_requested: false,
        applied_title: String::new(),
        no_default_layout: args.no_default_layout,
        #[cfg(feature = "mcp")]
        mcp_rx,
        #[cfg(feature = "mcp")]
        mcp_deferred: Vec::new(),
        #[cfg(feature = "mcp")]
        surface_readable: false,
        #[cfg(target_os = "windows")]
        early_dm,
        #[cfg(target_os = "windows")]
        gesture_handler: None,
        #[cfg(target_os = "windows")]
        next_dm_update: None,
    };

    event_loop.run_app(&mut app).expect("Event loop failed");
}

/// Bring the MCP endpoint up, or return `None` because it was not asked for.
///
/// **A bind failure is fatal and loud.** Two viewers on one port is the common
/// mistake, and a viewer that silently came up without the endpoint the agent
/// was told to use is worse than one that refused to start.
///
/// The endpoint line goes to stdout, because that is what a human pastes into
/// a client config; the error goes to stderr and takes the process with it.
#[cfg(feature = "mcp")]
fn start_mcp(
    state: &mut AppState,
    port: Option<u16>,
    proxy: &EventLoopProxy<UserEvent>,
) -> Option<tokio::sync::mpsc::UnboundedReceiver<mcp::Request>> {
    let port = port?;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    // The server's whole access to the viewer: this channel, and a wake. A
    // closure rather than the proxy itself, so `mcp::server` depends on no
    // winit type and can be driven by a test with no event loop at all.
    let proxy = proxy.clone();
    match mcp::serve(port, tx, move || {
        let _ = proxy.send_event(UserEvent::McpRequest);
    }) {
        Ok(address) => {
            println!("SfM Explorer MCP endpoint: http://{address}/mcp");
            state.mcp = Some(state::McpStatus::new(address.port()));
            let endpoint = state.mcp.as_ref().expect("just set").endpoint();
            state.action_log.record_as(
                action_log::Actor::Viewer,
                action_log::Kind::Session,
                format!("MCP endpoint listening on {endpoint}"),
            );
            Some(rx)
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// Reject `--mcp` in a build compiled without it, rather than ignoring the flag
/// and coming up with no endpoint.
#[cfg(not(feature = "mcp"))]
fn start_mcp(_state: &mut AppState, port: Option<u16>, _proxy: &EventLoopProxy<UserEvent>) {
    if port.is_some() {
        eprintln!(
            "This sfm-explorer was built without the \"mcp\" feature, so --mcp has nothing to \
             start. Rebuild with it (it is on by default) to use the MCP endpoint."
        );
        std::process::exit(2);
    }
}

pub(crate) struct App {
    pub(crate) proxy: EventLoopProxy<UserEvent>,
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
    pub(crate) scene_graph: SceneGraphPanel,
    pub(crate) image_browser: ImageBrowser,
    pub(crate) image_detail: ImageDetail,
    pub(crate) point_track_detail: PointTrackDetail,
    pub(crate) intrinsics_detail: IntrinsicsDetail,
    pub(crate) scene_renderer: SceneRenderer,
    pub(crate) prev_frustum_length_scale: f32,
    pub(crate) prev_frustum_size_multiplier: f32,
    pub(crate) prev_selected_image: Option<ImageRef>,
    /// `AppState::selected_camera` as of the previous frame: the sibling
    /// highlight is a per-image frustum color, so a camera selection moves it
    /// exactly as an image or point selection moves the other two.
    pub(crate) prev_selected_camera: Option<CameraRef>,
    pub(crate) prev_selected_point: Option<PointRef>,
    pub(crate) prev_hidden_image: Option<ImageRef>,
    /// `AppState::transform_epoch` as of the previous frame — how the upload
    /// phase notices that a node transform was set or reset.
    pub(crate) prev_transform_epoch: u64,
    /// Set by File > Quit and read by the event loop right after the frame it
    /// was clicked in, which then exits.
    pub(crate) quit_requested: bool,
    /// Window title as last handed to the window manager, so the per-frame
    /// sync in `run_ui_and_paint` only calls `set_title` when it changes.
    /// Starts empty so the first frame always applies the real title.
    pub(crate) applied_title: String,
    /// `--no-default-layout`: start from the stock grid whatever is saved at
    /// [`layout::default_layout_path`]. Read once, in `resumed`.
    pub(crate) no_default_layout: bool,
    /// Tool calls waiting to be applied, or `None` when no endpoint is running.
    ///
    /// Drained at the top of every frame ([`App::drain_mcp`]) and nowhere else:
    /// that one point is what makes a reply a snapshot taken with exclusive
    /// access rather than a read that could straddle a load.
    #[cfg(feature = "mcp")]
    pub(crate) mcp_rx: Option<tokio::sync::mpsc::UnboundedReceiver<mcp::Request>>,
    /// Tool calls whose answer needs this frame to have been rendered —
    /// `screenshot`, and only `screenshot`. Resolved in the readback phase,
    /// where the `wgpu::Device` already is.
    #[cfg(feature = "mcp")]
    pub(crate) mcp_deferred: Vec<(mcp::Deferred, tokio::sync::oneshot::Sender<mcp::Reply>)>,
    /// Whether the window surface was configured with `COPY_SRC`, which is what
    /// a screenshot of the window reads back from. Set once, when the surface
    /// is first configured; a platform that refuses the usage leaves it false
    /// and every window screenshot is refused rather than attempted.
    #[cfg(feature = "mcp")]
    pub(crate) surface_readable: bool,
    #[cfg(target_os = "windows")]
    pub(crate) early_dm: Option<EarlyDmState>,
    #[cfg(target_os = "windows")]
    pub(crate) gesture_handler: Option<WinGestureHandler>,
    #[cfg(target_os = "windows")]
    pub(crate) next_dm_update: Option<Instant>,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Step 1: Create window (raw winit, matching working test)
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(crate::state::WINDOW_TITLE_BASE)
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
                        .with_min_inner_size(winit::dpi::LogicalSize::new(800, 600))
                        .with_visible(false), // shown after AccessKit registers its UIAutomation provider
                )
                .expect("Failed to create window"),
        );

        self.window = Some(window.clone());

        // Step 2: Raw wgpu setup. Pick the backend per platform: DX12 on
        // Windows (pairs with the DirectManipulation integration), Metal on
        // macOS, Vulkan elsewhere. A single hardcoded backend would leave the
        // surface uncreatable on the others.
        let backends = if cfg!(target_os = "windows") {
            wgpu::Backends::DX12
        } else if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
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
        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Surface not supported by adapter");
        // `COPY_SRC` on the swapchain is what lets the MCP `screenshot` tool
        // photograph the window the human is looking at rather than only the 3D
        // render target. `get_default_config` asks for `RENDER_ATTACHMENT`
        // alone, and a usage the surface does not advertise is a validation
        // error at `configure` — so it is added only where the platform allows
        // it (DX12, Vulkan and Metal all do in practice) and the viewer
        // remembers whether it got it. The same config is what `resize`
        // reconfigures with, so the flag survives a resize.
        let surface_readable = surface
            .get_capabilities(&adapter)
            .usages
            .contains(wgpu::TextureUsages::COPY_SRC);
        if surface_readable {
            surface_config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&device, &surface_config);
        log::debug!(
            "surface {:?}, readable: {surface_readable}",
            surface_config.format
        );
        #[cfg(feature = "mcp")]
        {
            self.surface_readable = surface_readable;
        }

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
        let mut egui_winit_state = egui_winit::State::new(
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

        // Initialize AccessKit so egui's widget tree is visible to UIAutomation
        // (and screen readers). Must happen after egui_winit_state is created
        // but while the window is still hidden.
        egui_winit_state.init_accesskit::<UserEvent>(event_loop, &window, self.proxy.clone());
        self.egui_ctx.enable_accesskit();

        self.wgpu_device = Some(device);
        self.wgpu_queue = Some(queue);
        self.wgpu_surface = Some(surface);
        self.wgpu_surface_config = Some(surface_config);
        self.egui_renderer = Some(egui_renderer);
        self.egui_winit_state = Some(egui_winit_state);

        // The default layout, if the human saved one — window and panels both,
        // through the same path as Panels ▸ Load Layout…. Applied while the
        // window is still hidden, so a saved "maximized on the left monitor"
        // comes up that way rather than appearing at 1280 × 720 in the middle
        // and jumping. A file that is absent is nothing: no entry, no log line.
        if !self.no_default_layout {
            if let Some(path) = layout::default_layout_path().filter(|path| path.is_file()) {
                let mut host = window.clone();
                self.state.load_layout_file(&mut host, &path);
            }
        }

        window.set_visible(true);

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
                // On Windows a secondary or middle click reaches us as a
                // `Touch`, which egui would read as the primary button; the
                // platform layer hands back the `MouseInput` it should have
                // been. See `platform::windows::restore_mouse_button`.
                #[cfg(target_os = "windows")]
                let restored = platform::windows::restore_mouse_button(&event);
                #[cfg(not(target_os = "windows"))]
                let restored: Option<[WindowEvent; 2]> = None;

                let mut repaint = false;
                match restored.as_ref() {
                    Some(events) => {
                        for event in events {
                            repaint |= egui_winit_state.on_window_event(window, event).repaint;
                        }
                    }
                    None => {
                        repaint = egui_winit_state.on_window_event(window, &event).repaint;
                    }
                }
                if repaint {
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
                // A frame that bailed early — GPU state not up yet, or a
                // surface that could not be presented — leaves any deferred
                // screenshot unanswered. Ask for another frame rather than let
                // it sit until the caller's timeout: an idle viewer requests no
                // redraws of its own, so nothing else would come along.
                #[cfg(feature = "mcp")]
                if !self.mcp_deferred.is_empty() {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                if self.quit_requested {
                    event_loop.exit();
                }
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
        // Under UI tests, keep drawing so egui continuously republishes its
        // AccessKit tree; otherwise the tree may be queried before the idle app
        // has fully populated it.
        if force_repaint_for_tests() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::AccessKit(ak_event) => match ak_event.window_event {
                egui_winit::accesskit_winit::WindowEvent::ActionRequested(request) => {
                    if let Some(state) = self.egui_winit_state.as_mut() {
                        state.on_accesskit_action_request(request);
                    }
                }
                egui_winit::accesskit_winit::WindowEvent::InitialTreeRequested => {}
                egui_winit::accesskit_winit::WindowEvent::AccessibilityDeactivated => {}
            },
            // Nothing to do here: the redraw below is the whole handling. The
            // request is drained and applied at the top of the frame it wakes.
            #[cfg(feature = "mcp")]
            UserEvent::McpRequest => {}
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
