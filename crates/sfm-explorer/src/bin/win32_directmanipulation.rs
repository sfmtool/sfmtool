// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Minimal DirectManipulation example using a pure Win32 window.
//!
//! Demonstrates the DirectManipulation gesture pipeline with no external
//! windowing library. The window procedure intercepts DM_POINTERHITTEST
//! and calls SetContact() to begin gesture tracking.
//!
//! Run with: cargo run --bin win32_directmanipulation --features directmanipulation
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
    use windows::Win32::Graphics::Gdi::{BeginPaint, EndPaint, PAINTSTRUCT};
    use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::System::Ole::OleInitialize;
    use windows::Win32::UI::Input::Pointer::{EnableMouseInPointer, GetPointerType};
    use windows::Win32::UI::WindowsAndMessaging::*;

    const DM_POINTERHITTEST: u32 = 0x0250;

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

            // Reset transform tracking when gesture ends (READY = 5)
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

    // --- Window procedure ---

    unsafe extern "system" fn wndproc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match msg {
            DM_POINTERHITTEST => {
                let pointer_id = (wparam.0 & 0xFFFF) as u32;
                let mut pt_type = POINTER_INPUT_TYPE(0);
                let _ = GetPointerType(pointer_id, &mut pt_type);
                println!("DM_POINTERHITTEST id={} type={}", pointer_id, pt_type.0);

                if let Some(state) = GLOBAL_STATE.get() {
                    if let Ok(s) = state.lock() {
                        let _ = s.viewport.SetContact(pointer_id);
                    }
                }
                return LRESULT(0);
            }
            WM_TIMER => {
                if let Some(state) = GLOBAL_STATE.get() {
                    if let Ok(s) = state.lock() {
                        let _ = s.update_manager.Update(None);
                    }
                }
                return LRESULT(0);
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                return LRESULT(0);
            }
            WM_PAINT => {
                let mut ps = PAINTSTRUCT::default();
                let _hdc = BeginPaint(hwnd, &mut ps);
                let _ = EndPaint(hwnd, &ps);
                return LRESULT(0);
            }
            _ => {}
        }
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }

    // --- Entry point ---

    pub fn run() -> windows::core::Result<()> {
        println!("=== Win32 DirectManipulation Example ===");
        println!("Touch the precision trackpad to see gesture callbacks.");
        println!();

        unsafe {
            OleInitialize(None)?;
            let _ = EnableMouseInPointer(true);

            let instance = GetModuleHandleW(None)?;
            let class_name = windows::core::w!("DmWin32Example");

            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: WNDCLASS_STYLES(CS_HREDRAW.0 | CS_VREDRAW.0),
                lpfnWndProc: Some(wndproc),
                hInstance: instance.into(),
                lpszClassName: class_name,
                ..Default::default()
            };
            RegisterClassExW(&wc);

            let hwnd = CreateWindowExW(
                WS_EX_APPWINDOW,
                class_name,
                windows::core::w!("Win32 DirectManipulation"),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                800,
                600,
                None,
                None,
                instance,
                None,
            )?;

            // Set up DirectManipulation
            let manager: IDirectManipulationManager =
                CoCreateInstance(&DirectManipulationManager, None, CLSCTX_INPROC_SERVER)?;
            let update_manager: IDirectManipulationUpdateManager = manager.GetUpdateManager()?;

            let viewport: IDirectManipulationViewport = manager.CreateViewport(None, hwnd)?;
            viewport.ActivateConfiguration(
                DIRECTMANIPULATION_CONFIGURATION_INTERACTION
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA
                    | DIRECTMANIPULATION_CONFIGURATION_SCALING,
            )?;
            viewport.SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS(2))?; // MANUALUPDATE

            let handler: IDirectManipulationViewportEventHandler = GestureHandler.into();
            viewport.AddEventHandler(hwnd, &handler)?;

            let rect = RECT {
                left: 0,
                top: 0,
                right: 10000,
                bottom: 10000,
            };
            viewport.SetViewportRect(&rect)?;
            manager.Activate(hwnd)?;
            viewport.Enable()?;

            let _ = GLOBAL_STATE.set(Arc::new(Mutex::new(DmState {
                viewport,
                update_manager,
                prev_x: 0.0,
                prev_y: 0.0,
                prev_scale: 1.0,
            })));

            // Timer drives update_manager.Update() at ~60fps
            SetTimer(hwnd, 1, 16, None);

            println!("Setup complete. Listening for gestures...");
            println!();

            // Message loop
            let mut msg = MSG::default();
            loop {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    if msg.message == WM_QUIT {
                        return Ok(());
                    }
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
} // mod platform
