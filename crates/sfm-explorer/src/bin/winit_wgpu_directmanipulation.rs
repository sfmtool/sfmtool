// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! DirectManipulation + winit + wgpu test.
//!
//! Same as `winit_directmanipulation` but adds wgpu surface creation to the
//! window, to test whether wgpu/DXGI initialization breaks DM_POINTERHITTEST.
//!
//! Run with: cargo run --bin winit_wgpu_directmanipulation --features directmanipulation
//!
//! This binary only builds on Windows.

#[cfg(not(windows))]
fn main() {
    eprintln!("This binary only runs on Windows.");
    std::process::exit(1);
}

#[cfg(windows)]
fn main() -> windows::core::Result<()> {
    platform::run()
}

#[cfg(windows)]
mod platform {

    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use winit::application::ApplicationHandler;
    use winit::event::{StartCause, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopBuilder};
    use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::{Window, WindowAttributes, WindowId};

    use windows::core::implement;
    use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
    use windows::Win32::Graphics::DirectManipulation::{
        DirectManipulationManager, IDirectManipulationContent, IDirectManipulationManager,
        IDirectManipulationUpdateManager, IDirectManipulationViewport,
        IDirectManipulationViewportEventHandler, IDirectManipulationViewportEventHandler_Impl,
        DIRECTMANIPULATION_CONFIGURATION_INTERACTION, DIRECTMANIPULATION_CONFIGURATION_SCALING,
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA,
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X,
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y, DIRECTMANIPULATION_STATUS,
        DIRECTMANIPULATION_VIEWPORT_OPTIONS,
    };
    use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
    use windows::Win32::System::Ole::OleInitialize;
    use windows::Win32::UI::Input::Pointer::EnableMouseInPointer;
    use windows::Win32::UI::Shell::{DefSubclassProc, SetWindowSubclass};

    const DM_POINTERHITTEST: u32 = 0x0250;
    const UPDATE_INTERVAL: Duration = Duration::from_millis(16);
    const SUBCLASS_ID: usize = 1;

    static CONTENT_UPDATE_COUNT: AtomicU32 = AtomicU32::new(0);

    struct DmState {
        viewport: IDirectManipulationViewport,
        update_manager: IDirectManipulationUpdateManager,
        prev_x: f32,
        prev_y: f32,
        prev_scale: f32,
    }

    unsafe impl Send for DmState {}

    static GLOBAL_STATE: OnceLock<Arc<Mutex<DmState>>> = OnceLock::new();

    // --- COM event handler ---

    #[implement(IDirectManipulationViewportEventHandler)]
    struct GestureHandler;

    impl IDirectManipulationViewportEventHandler_Impl for GestureHandler_Impl {
        fn OnViewportStatusChanged(
            &self,
            _viewport: Option<&IDirectManipulationViewport>,
            current: DIRECTMANIPULATION_STATUS,
            previous: DIRECTMANIPULATION_STATUS,
        ) -> windows::core::Result<()> {
            println!("Status: {:?} -> {:?}", previous, current);
            if current.0 == 5 {
                if let Some(state) = GLOBAL_STATE.get() {
                    if let Ok(mut s) = state.lock() {
                        s.prev_x = 0.0;
                        s.prev_y = 0.0;
                        s.prev_scale = 1.0;
                        let rect = RECT {
                            left: 0,
                            top: 0,
                            right: 10000,
                            bottom: 10000,
                        };
                        let _ = unsafe { s.viewport.SetViewportRect(&rect) };
                    }
                }
            }
            Ok(())
        }

        fn OnViewportUpdated(
            &self,
            _viewport: Option<&IDirectManipulationViewport>,
        ) -> windows::core::Result<()> {
            Ok(())
        }

        fn OnContentUpdated(
            &self,
            _viewport: Option<&IDirectManipulationViewport>,
            content: Option<&IDirectManipulationContent>,
        ) -> windows::core::Result<()> {
            let count = CONTENT_UPDATE_COUNT.fetch_add(1, Ordering::Relaxed);
            let Some(content) = content else {
                return Ok(());
            };
            let mut transform = [0.0f32; 6];
            unsafe { content.GetContentTransform(&mut transform)? };
            if let Some(state) = GLOBAL_STATE.get() {
                if let Ok(mut s) = state.lock() {
                    let dx = transform[4] - s.prev_x;
                    let dy = transform[5] - s.prev_y;
                    let ds = transform[0] / s.prev_scale.max(0.001);
                    s.prev_x = transform[4];
                    s.prev_y = transform[5];
                    s.prev_scale = transform[0];
                    println!(
                        "Content[{}]: dx={:.2}, dy={:.2}, scale={:.4}",
                        count, dx, dy, ds
                    );
                }
            }
            Ok(())
        }
    }

    // --- Subclass procedure ---

    unsafe extern "system" fn dm_subclass_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
        _uid_subclass: usize,
        _dw_ref_data: usize,
    ) -> LRESULT {
        if msg == DM_POINTERHITTEST {
            let pointer_id = (wparam.0 & 0xFFFF) as u32;
            println!("DM_POINTERHITTEST id={}", pointer_id);
            if let Some(state) = GLOBAL_STATE.get() {
                if let Ok(s) = state.lock() {
                    let _ = s.viewport.SetContact(pointer_id);
                }
            }
            return LRESULT(0);
        }
        unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
    }

    // --- Winit application ---

    struct EarlyDmState {
        manager: IDirectManipulationManager,
        update_manager: IDirectManipulationUpdateManager,
    }

    unsafe impl Send for EarlyDmState {}

    struct App {
        early_dm: Option<EarlyDmState>,
        window: Option<Arc<Window>>,
        _wgpu_surface: Option<wgpu::Surface<'static>>,
        _wgpu_device: Option<wgpu::Device>,
        dm_ready: bool,
        next_update: Option<Instant>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }

            let attrs = WindowAttributes::default()
                .with_title("Winit+wgpu DirectManipulation")
                .with_inner_size(winit::dpi::LogicalSize::new(640, 480));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("Failed to create window"),
            );

            let hwnd_raw = if let Ok(handle) = HasWindowHandle::window_handle(&*window) {
                if let RawWindowHandle::Win32(win32) = handle.as_raw() {
                    Some(win32.hwnd.get())
                } else {
                    None
                }
            } else {
                None
            };

            // --- wgpu initialization ---
            println!("[wgpu] Creating instance...");
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::DX12,
                ..Default::default()
            });

            println!("[wgpu] Creating surface...");
            let surface = instance
                .create_surface(window.clone())
                .expect("create_surface");

            println!("[wgpu] Requesting adapter...");
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                }))
                .expect("request_adapter");

            println!("[wgpu] Requesting device...");
            let (device, _queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .expect("request_device");

            println!("[wgpu] Configuring surface...");
            let size = window.inner_size();
            let config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .expect("get_default_config");
            surface.configure(&device, &config);
            println!("[wgpu] Done.");

            self._wgpu_surface = Some(surface);
            self._wgpu_device = Some(device);
            self.window = Some(window);

            if let (Some(hwnd_raw), Some(early)) = (hwnd_raw, &self.early_dm) {
                let hwnd = HWND(hwnd_raw as *mut _);

                // Workaround #2: Subclass to intercept DM_POINTERHITTEST (SendMessage)
                let _ = unsafe { SetWindowSubclass(hwnd, Some(dm_subclass_proc), SUBCLASS_ID, 0) };

                unsafe {
                    let viewport: IDirectManipulationViewport = early
                        .manager
                        .CreateViewport(None, hwnd)
                        .expect("CreateViewport");
                    viewport
                        .ActivateConfiguration(
                            DIRECTMANIPULATION_CONFIGURATION_INTERACTION
                                | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X
                                | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y
                                | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA
                                | DIRECTMANIPULATION_CONFIGURATION_SCALING,
                        )
                        .expect("ActivateConfiguration");
                    viewport
                        .SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS(2))
                        .expect("SetViewportOptions"); // MANUALUPDATE

                    let handler: IDirectManipulationViewportEventHandler = GestureHandler.into();
                    viewport
                        .AddEventHandler(hwnd, &handler)
                        .expect("AddEventHandler");

                    let rect = RECT {
                        left: 0,
                        top: 0,
                        right: 10000,
                        bottom: 10000,
                    };
                    viewport.SetViewportRect(&rect).expect("SetViewportRect");
                    early.manager.Activate(hwnd).expect("Activate");
                    viewport.Enable().expect("Enable");

                    let _ = GLOBAL_STATE.set(Arc::new(Mutex::new(DmState {
                        viewport,
                        update_manager: early.update_manager.clone(),
                        prev_x: 0.0,
                        prev_y: 0.0,
                        prev_scale: 1.0,
                    })));
                }

                self.dm_ready = true;
                let next = Instant::now() + UPDATE_INTERVAL;
                self.next_update = Some(next);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));

                println!("Setup complete. Listening for gestures...");
                println!();
            }
        }

        fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
            if !self.dm_ready {
                return;
            }

            let now = Instant::now();
            let should_update = match cause {
                StartCause::ResumeTimeReached { .. } => true,
                StartCause::WaitCancelled {
                    requested_resume: Some(deadline),
                    ..
                } if now >= deadline => true,
                _ => false,
            };

            if should_update {
                // Workaround #3: Drive DM updates manually (WM_TIMER doesn't fire)
                if let Some(state) = GLOBAL_STATE.get() {
                    if let Ok(s) = state.lock() {
                        let _ = unsafe { s.update_manager.Update(None) };
                    }
                }
                self.next_update = Some(now + UPDATE_INTERVAL);
            }

            if let Some(next) = self.next_update {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            }
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            if let Some(next) = self.next_update {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            if matches!(event, WindowEvent::CloseRequested) {
                event_loop.exit();
            }
        }
    }

    // --- Entry point ---

    pub fn run() -> windows::core::Result<()> {
        println!("=== Winit+wgpu DirectManipulation Test ===");
        println!("Tests whether wgpu surface creation breaks DM_POINTERHITTEST.");
        println!("Touch the precision trackpad to see gesture callbacks.");
        println!();

        // Workaround #1: Create DM manager BEFORE winit's EventLoop
        unsafe { OleInitialize(None)? };
        unsafe {
            let _ = EnableMouseInPointer(true);
        }

        let manager: IDirectManipulationManager =
            unsafe { CoCreateInstance(&DirectManipulationManager, None, CLSCTX_INPROC_SERVER)? };
        let update_manager: IDirectManipulationUpdateManager =
            unsafe { manager.GetUpdateManager()? };

        let early_dm = EarlyDmState {
            manager,
            update_manager,
        };

        let mut event_loop = EventLoopBuilder::default()
            .build()
            .expect("Failed to create EventLoop");

        let mut app = App {
            early_dm: Some(early_dm),
            window: None,
            _wgpu_surface: None,
            _wgpu_device: None,
            dm_ready: false,
            next_update: None,
        };

        event_loop
            .run_app_on_demand(&mut app)
            .expect("Event loop failed");

        Ok(())
    }
} // mod platform
