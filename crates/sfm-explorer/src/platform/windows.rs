// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Windows-specific gesture handling using DirectManipulation API.
//!
//! This module provides precision touchpad gesture support on Windows by using
//! the DirectManipulation API, the same approach used by Blender for smooth,
//! hardware-accelerated gesture recognition with inertia support.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use windows::core::implement;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::DirectManipulation::{
    DirectManipulationManager, IDirectManipulationContent, IDirectManipulationManager,
    IDirectManipulationUpdateManager, IDirectManipulationViewport,
    IDirectManipulationViewportEventHandler, IDirectManipulationViewportEventHandler_Impl,
    DIRECTMANIPULATION_CONFIGURATION_INTERACTION, DIRECTMANIPULATION_CONFIGURATION_SCALING,
    DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA,
    DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X, DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y,
    DIRECTMANIPULATION_STATUS, DIRECTMANIPULATION_VIEWPORT_OPTIONS,
};
use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
use windows::Win32::System::Ole::OleInitialize;
use windows::Win32::System::Registry::{
    RegCloseKey, RegOpenKeyExW, RegQueryValueExW, HKEY_CURRENT_USER, KEY_READ, REG_DWORD,
};
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::Shell::{DefSubclassProc, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{SC_KEYMENU, WM_SYSCOMMAND};

use super::GestureEvent;

/// DM_POINTERHITTEST message ID for DirectManipulation hit testing.
const DM_POINTERHITTEST: u32 = 0x0250;

/// WM_POINTER message IDs for tracking mouse button state.
const WM_POINTERDOWN: u32 = 0x0246;
const WM_POINTERUP: u32 = 0x0247;
const WM_POINTERUPDATE: u32 = 0x0245;

/// Pointer flags for button identification (from POINTER_INFO.pointerFlags).
const POINTER_FLAG_FIRSTBUTTON: u32 = 0x00000010; // left
const POINTER_FLAG_SECONDBUTTON: u32 = 0x00000020; // right
const POINTER_FLAG_THIRDBUTTON: u32 = 0x00000040; // middle

/// Bit flags for tracked mouse button state.
pub const BUTTON_LEFT: u8 = 0x01;
pub const BUTTON_RIGHT: u8 = 0x02;
pub const BUTTON_MIDDLE: u8 = 0x04;

/// Global mouse button state, updated from the subclass window procedure.
/// Stores a bitmask of BUTTON_LEFT | BUTTON_RIGHT | BUTTON_MIDDLE.
static MOUSE_BUTTON_STATE: AtomicU8 = AtomicU8::new(0);

/// Latest pointer position in physical client coordinates, updated from WM_POINTER messages.
/// These are always up-to-date even when egui's hover state goes stale after clicks.
static POINTER_CLIENT_X: AtomicI32 = AtomicI32::new(0);
static POINTER_CLIENT_Y: AtomicI32 = AtomicI32::new(0);

/// Returns the current mouse button state as a bitmask.
pub fn mouse_button_state() -> u8 {
    MOUSE_BUTTON_STATE.load(Ordering::Relaxed)
}

/// Returns the latest pointer position in physical client coordinates.
pub fn pointer_client_pos() -> (i32, i32) {
    (
        POINTER_CLIENT_X.load(Ordering::Relaxed),
        POINTER_CLIENT_Y.load(Ordering::Relaxed),
    )
}

// Helper to extract pointer ID from WPARAM
fn get_pointer_id(wparam: WPARAM) -> u32 {
    (wparam.0 & 0xFFFF) as u32
}

/// Checks the Windows registry for precision touchpad scroll direction.
fn get_inverted_scroll() -> bool {
    let mut hkey = windows::Win32::System::Registry::HKEY::default();
    let subkey =
        windows::core::w!("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\PrecisionTouchPad");
    unsafe {
        if RegOpenKeyExW(HKEY_CURRENT_USER, subkey, 0, KEY_READ, &mut hkey).is_ok() {
            let mut value = 0u32;
            let mut size = std::mem::size_of::<u32>() as u32;
            let mut type_ = REG_DWORD;
            let res = RegQueryValueExW(
                hkey,
                windows::core::w!("ScrollDirection"),
                None,
                Some(&mut type_),
                Some(&mut value as *mut u32 as *mut u8),
                Some(&mut size),
            );
            let _ = RegCloseKey(hkey);
            if res.is_ok() {
                // 0 means "natural scrolling" (which is actually inverted deltas), 1 means traditional.
                // In Blender's implementation, value == 0 signifies natural scrolling.
                return value == 0;
            }
        }
    }
    false
}

/// Early DirectManipulation state created before the winit EventLoop.
///
/// The DM manager and update manager must be created before the EventLoop
/// so that `DM_POINTERHITTEST` messages are generated. This struct holds
/// them until the window is created and `WinGestureHandler::new()` can
/// attach a viewport.
pub struct EarlyDmState {
    manager: IDirectManipulationManager,
    update_manager: IDirectManipulationUpdateManager,
}

// Safety: COM pointers are Send when used from the STA thread they were created on.
// We create on the main thread and only use on the main thread.
unsafe impl Send for EarlyDmState {}

/// Creates the DirectManipulation manager and update manager early, before
/// the winit EventLoop is created. This is required so that Windows generates
/// `DM_POINTERHITTEST` messages for precision touchpad contacts.
pub fn create_manager() -> Result<EarlyDmState, windows::core::Error> {
    let ole_result = unsafe { OleInitialize(None) };
    if let Err(ref e) = ole_result {
        log::warn!("OleInitialize failed: {:?}", e);
    }

    unsafe {
        use windows::Win32::UI::Input::Pointer::EnableMouseInPointer;
        let _ = EnableMouseInPointer(true);
    }

    let manager: IDirectManipulationManager =
        unsafe { CoCreateInstance(&DirectManipulationManager, None, CLSCTX_INPROC_SERVER) }?;

    let update_manager: IDirectManipulationUpdateManager = unsafe { manager.GetUpdateManager() }?;

    log::info!("DirectManipulation manager and update manager created (early init)");

    Ok(EarlyDmState {
        manager,
        update_manager,
    })
}

/// Gesture handler for Windows using DirectManipulation API.
///
/// This handler uses the same approach as Blender for precision touchpad
/// navigation, providing hardware-accelerated gesture recognition with
/// automatic pan/pinch detection, inertia, and smoothing.
pub struct WinGestureHandler {
    /// Receiver for gesture events produced by the callback.
    rx: Receiver<GestureEvent>,
    /// Shared state that needs to be kept alive.
    state: Arc<Mutex<DirectManipulationState>>,
    /// Update manager for driving gesture updates each frame.
    update_manager: IDirectManipulationUpdateManager,
    /// The DM manager must be kept alive for the lifetime of the handler.
    /// Dropping it causes COM to Release the manager, which deactivates DM
    /// for the HWND and stops DM_POINTERHITTEST from being generated.
    _manager: IDirectManipulationManager,
}

/// State for DirectManipulation gesture tracking.
struct DirectManipulationState {
    /// Sender for gesture events.
    tx: Sender<GestureEvent>,
    /// DirectManipulation viewport.
    viewport: IDirectManipulationViewport,
    /// Previous transform values for computing deltas.
    prev_translate_x: f32,
    prev_translate_y: f32,
    prev_scale: f32,
    /// Track current gesture type: None, Pan, or Pinch.
    gesture_type: GestureType,
    /// DPI scale factor for proper coordinate handling.
    dpi_scale: f32,
    /// Whether scroll direction is inverted (Natural Scrolling).
    inverted_scroll: bool,
    /// Diagnostic: number of hit tests received.
    pub hit_test_count: u32,
    /// Diagnostic: number of touchpad contacts registered.
    pub contact_count: u32,
    /// Diagnostic: number of content updates received.
    pub update_count: u32,
    /// Diagnostic: current viewport status.
    pub status: String,
}

// Safety: State is protected by Mutex
unsafe impl Send for DirectManipulationState {}

/// Current gesture type for state machine.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GestureType {
    None,
    Pan,
    Pinch,
}

/// Type alias for the subclass data stored per window.
type SubclassEntry = Arc<Mutex<DirectManipulationState>>;

/// Global storage for window procedure state.
static GLOBAL_SUBCLASS_DATA: OnceLock<RwLock<HashMap<isize, SubclassEntry>>> = OnceLock::new();

fn get_subclass_data() -> &'static RwLock<HashMap<isize, SubclassEntry>> {
    GLOBAL_SUBCLASS_DATA.get_or_init(|| RwLock::new(HashMap::new()))
}

/// COM implementation of IDirectManipulationViewportEventHandler.
///
/// This receives callbacks from DirectManipulation when gesture state changes.
#[implement(IDirectManipulationViewportEventHandler)]
struct ViewportEventHandler {
    state: Arc<Mutex<DirectManipulationState>>,
}

impl IDirectManipulationViewportEventHandler_Impl for ViewportEventHandler_Impl {
    fn OnViewportStatusChanged(
        &self,
        _viewport: Option<&IDirectManipulationViewport>,
        current: DIRECTMANIPULATION_STATUS,
        previous: DIRECTMANIPULATION_STATUS,
    ) -> windows::core::Result<()> {
        log::debug!("DirectManipulation status: {:?} -> {:?}", previous, current);

        if let Ok(mut state) = self.state.lock() {
            state.status = format!("{:?}", current);

            // Reset gesture state on transition to READY
            // DIRECTMANIPULATION_READY is usually 5
            if current.0 == 5 {
                reset_gesture_state(&mut state);
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
        if let Ok(mut state) = self.state.lock() {
            state.update_count += 1;
        }

        let Some(content) = content else {
            log::warn!("OnContentUpdated called with no content");
            return Ok(());
        };

        // Get the output transform matrix (6 floats: 2x3 affine transform)
        // [0]: scale X
        // [1]: rotation
        // [2]: rotation
        // [3]: scale Y
        // [4]: translate X
        // [5]: translate Y
        let mut transform = [0.0f32; 6];
        unsafe {
            content.GetContentTransform(&mut transform)?;
        }

        log::trace!(
            "OnContentUpdated: transform=[{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            transform[0],
            transform[1],
            transform[2],
            transform[3],
            transform[4],
            transform[5],
        );

        if let Ok(mut state) = self.state.lock() {
            process_transform(&mut state, &transform);
        }

        Ok(())
    }
}

/// Process the transform matrix from DirectManipulation and emit gesture events.
fn process_transform(state: &mut DirectManipulationState, transform: &[f32; 6]) {
    const EPSILON: f32 = 0.00003;
    const PINCH_SCALE_FACTOR: f32 = 125.0;

    let scale = transform[0];
    let translate_x = transform[4] / state.dpi_scale;
    let translate_y = transform[5] / state.dpi_scale;

    log::trace!(
        "process_transform translate=({:.2}, {:.2}) scale={:.4}",
        translate_x,
        translate_y,
        scale
    );

    // Check for significant scale change (pinch gesture)
    let scale_delta = scale / state.prev_scale.max(0.001);
    let has_scale_change = (scale_delta - 1.0).abs() > EPSILON;

    // Transition from Pan to Pinch if scaling detected
    if has_scale_change && state.gesture_type == GestureType::Pan {
        state.gesture_type = GestureType::Pinch;
        log::debug!("Gesture transitioned from Pan to Pinch");
    }

    // Start a new gesture if none active — capture baseline and skip first event
    if state.gesture_type == GestureType::None {
        if has_scale_change {
            state.gesture_type = GestureType::Pinch;
        } else {
            state.gesture_type = GestureType::Pan;
        }
        // Capture current transform as baseline so the first real delta is small
        state.prev_translate_x = translate_x;
        state.prev_translate_y = translate_y;
        state.prev_scale = scale;
        return;
    }

    // Compute float deltas
    let delta_x = translate_x - state.prev_translate_x;
    let delta_y = translate_y - state.prev_translate_y;

    // Update previous values
    state.prev_translate_x = translate_x;
    state.prev_translate_y = translate_y;
    state.prev_scale = scale;

    // Apply scroll inversion if set (Natural Scrolling)
    let inv = if state.inverted_scroll { -1.0 } else { 1.0 };
    let final_dx = delta_x * inv;
    let final_dy = delta_y * inv;

    // Filter out tiny movements (noise)
    let has_translation = final_dx.abs() > 0.0 || final_dy.abs() > 0.0;

    match state.gesture_type {
        GestureType::Pan => {
            if has_translation {
                let _ = state.tx.send(GestureEvent::Pan {
                    dx: final_dx as f64,
                    dy: -final_dy as f64, // Flip Y for screen coordinates
                });
            }
        }
        GestureType::Pinch => {
            if has_scale_change {
                // Normalize the scale for smooth zoom feel
                let normalized_scale = 1.0 + (scale_delta - 1.0) * PINCH_SCALE_FACTOR / 100.0;
                let _ = state.tx.send(GestureEvent::Zoom {
                    scale: normalized_scale as f64,
                });
                // Suppress pan while actively scaling to avoid unwanted orbit
            } else if has_translation {
                // No active scaling this frame — allow pan through
                let _ = state.tx.send(GestureEvent::Pan {
                    dx: final_dx as f64,
                    dy: -final_dy as f64,
                });
            }
        }
        GestureType::None => {}
    }
}

/// Reset gesture state when a gesture ends.
fn reset_gesture_state(state: &mut DirectManipulationState) {
    log::debug!("Resetting gesture state (was {:?})", state.gesture_type);
    state.gesture_type = GestureType::None;
    state.prev_translate_x = 0.0;
    state.prev_translate_y = 0.0;
    state.prev_scale = 1.0;

    // Reset the viewport to initial state
    let rect = RECT {
        left: 0,
        top: 0,
        right: 10000,
        bottom: 10000,
    };
    unsafe {
        let _ = state.viewport.SetViewportRect(&rect);
    }

    log::debug!("Gesture state reset");
}

impl WinGestureHandler {
    /// Creates a new gesture handler for the given window, using a
    /// pre-created `EarlyDmState` (manager + update manager).
    ///
    /// The `EarlyDmState` must have been created via [`create_manager()`]
    /// **before** the winit EventLoop was started. This function attaches a
    /// viewport to the HWND and subclasses the window to intercept
    /// `DM_POINTERHITTEST` messages.
    pub fn new(hwnd: HWND, early: &EarlyDmState) -> Result<Self, windows::core::Error> {
        let manager = early.manager.clone();
        let update_manager = early.update_manager.clone();

        // Subclass the window early to intercept DM_POINTERHITTEST messages.
        let subclass_ok =
            unsafe { SetWindowSubclass(hwnd, Some(subclass_wndproc), 1, 0) }.as_bool();
        if !subclass_ok {
            log::warn!("Failed to install window subclass for DirectManipulation");
        }

        let (tx, rx) = mpsc::channel();

        // Create a viewport for gesture tracking
        let viewport: IDirectManipulationViewport = unsafe { manager.CreateViewport(None, hwnd) }?;

        // Configure the viewport for pan and pinch gestures with inertia
        unsafe {
            viewport.ActivateConfiguration(
                DIRECTMANIPULATION_CONFIGURATION_INTERACTION
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA
                    | DIRECTMANIPULATION_CONFIGURATION_SCALING,
            )
        }?;

        // Set viewport options to MANUALUPDATE mode (like Blender)
        // MANUALUPDATE = 2
        unsafe { viewport.SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS(2)) }?;

        // Get DPI scale for the window
        let dpi = unsafe { GetDpiForWindow(hwnd) };
        let dpi_scale = dpi as f32 / 96.0;
        log::debug!("Window DPI: {}, scale factor: {:.2}", dpi, dpi_scale);

        let inverted_scroll = get_inverted_scroll();
        log::debug!(
            "System scroll inversion (Natural Scrolling): {}",
            inverted_scroll
        );

        let state = Arc::new(Mutex::new(DirectManipulationState {
            tx,
            viewport: viewport.clone(),
            prev_translate_x: 0.0,
            prev_translate_y: 0.0,
            prev_scale: 1.0,
            gesture_type: GestureType::None,
            dpi_scale,
            inverted_scroll,
            hit_test_count: 0,
            contact_count: 0,
            update_count: 0,
            status: "READY".to_string(),
        }));

        // Add event handler BEFORE SetViewportRect (Blender order)
        let event_handler = ViewportEventHandler {
            state: Arc::clone(&state),
        };
        let handler_interface: IDirectManipulationViewportEventHandler = event_handler.into();
        let _cookie = unsafe { viewport.AddEventHandler(hwnd, &handler_interface) }?;

        // Set viewport rect AFTER AddEventHandler
        let viewport_rect = RECT {
            left: 0,
            top: 0,
            right: 10000,
            bottom: 10000,
        };
        unsafe { viewport.SetViewportRect(&viewport_rect) }?;

        // Activate the manager for this HWND
        unsafe { manager.Activate(hwnd) }?;

        // Enable the viewport
        unsafe { viewport.Enable() }?;

        // Store state in global map for the window procedure (after DM is ready)
        if let Ok(mut data) = get_subclass_data().write() {
            data.insert(hwnd.0 as isize, Arc::clone(&state));
        }

        log::info!("DirectManipulation initialized for HWND={:?}", hwnd);

        Ok(Self {
            rx,
            state,
            update_manager,
            _manager: manager,
        })
    }

    /// Drives DirectManipulation updates. Call this once per frame.
    ///
    /// This processes any pending gesture state and triggers callbacks.
    pub fn update(&self) {
        unsafe {
            let _ = self.update_manager.Update(None);
        }
    }

    /// Polls all pending gesture events, coalescing multiple updates into a single frame delta.
    pub fn poll_events(&self) -> Vec<GestureEvent> {
        // Drive update before polling
        self.update();

        let mut pan_dx = 0.0;
        let mut pan_dy = 0.0;
        let mut zoom_scale = 1.0;
        let mut has_pan = false;
        let mut has_zoom = false;

        for event in self.rx.try_iter() {
            match event {
                GestureEvent::Pan { dx, dy } => {
                    pan_dx += dx;
                    pan_dy += dy;
                    has_pan = true;
                }
                GestureEvent::Zoom { scale } => {
                    zoom_scale *= scale;
                    has_zoom = true;
                }
            }
        }

        let mut events = Vec::new();
        if has_pan {
            log::trace!("Coalesced Pan (dx={}, dy={})", pan_dx, pan_dy);
            events.push(GestureEvent::Pan {
                dx: pan_dx,
                dy: pan_dy,
            });
        }
        if has_zoom {
            log::trace!("Coalesced Zoom (scale={})", zoom_scale);
            events.push(GestureEvent::Zoom { scale: zoom_scale });
        }
        events
    }

    /// Returns diagnostic information about the handler state.
    pub fn get_diagnostics(&self) -> (u32, u32, u32, u32) {
        if let Ok(state) = self.state.lock() {
            (
                state.hit_test_count,
                state.contact_count,
                state.update_count,
                0,
            )
        } else {
            (0, 0, 0, 0)
        }
    }
}

impl super::GestureHandler for WinGestureHandler {
    fn poll_events(&self) -> Vec<GestureEvent> {
        self.poll_events()
    }
}

/// Window procedure that intercepts DM_POINTERHITTEST messages.
///
/// When a precision touchpad contact is detected, this registers the pointer
/// with DirectManipulation's viewport for gesture tracking.
unsafe extern "system" fn subclass_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _uid_subclass: usize,
    _dw_ref_data: usize,
) -> LRESULT {
    // Suppress Alt key menu activation so Alt can be used as a navigation modifier.
    // This prevents WM_SYSCOMMAND/SC_KEYMENU from activating the system menu when
    // the user holds Alt for target control. Alt+F4 and other system keys are unaffected.
    if msg == WM_SYSCOMMAND && (wparam.0 & 0xFFF0) == SC_KEYMENU as usize {
        return LRESULT(0);
    }

    // Track mouse button state and pointer position from WM_POINTER messages.
    // EnableMouseInPointer(true) converts all mouse input to WM_POINTER*,
    // but winit/egui_winit only reports Primary. We extract the real button
    // from the pointer flags and store it in an atomic for the viewer to read.
    // We also track the client-area pointer position so gesture dispatch can
    // determine which panel the pointer is over, even when egui's hover state
    // goes stale (e.g. after a double-click with no subsequent mouse movement).
    if msg == WM_POINTERDOWN || msg == WM_POINTERUP || msg == WM_POINTERUPDATE {
        let pointer_id = get_pointer_id(wparam);
        let mut pointer_info =
            std::mem::zeroed::<windows::Win32::UI::Input::Pointer::POINTER_INFO>();
        let ok = windows::Win32::UI::Input::Pointer::GetPointerInfo(pointer_id, &mut pointer_info);
        if ok.is_ok() {
            let flags = pointer_info.pointerFlags.0;
            let mut state = 0u8;
            if flags & POINTER_FLAG_FIRSTBUTTON != 0 {
                state |= BUTTON_LEFT;
            }
            if flags & POINTER_FLAG_SECONDBUTTON != 0 {
                state |= BUTTON_RIGHT;
            }
            if flags & POINTER_FLAG_THIRDBUTTON != 0 {
                state |= BUTTON_MIDDLE;
            }
            MOUSE_BUTTON_STATE.store(state, Ordering::Relaxed);

            // Store client-area pointer position.
            let mut pt = pointer_info.ptPixelLocation;
            let _ = windows::Win32::Graphics::Gdi::ScreenToClient(hwnd, &mut pt);
            POINTER_CLIENT_X.store(pt.x, Ordering::Relaxed);
            POINTER_CLIENT_Y.store(pt.y, Ordering::Relaxed);
        }
    }

    // Handle DM_POINTERHITTEST synchronously and return immediately.
    // Only register non-mouse pointers with DirectManipulation. With
    // EnableMouseInPointer(true), mouse clicks become WM_POINTER events that
    // can trigger DM_POINTERHITTEST. Registering a mouse pointer as a DM
    // contact puts the viewport into RUNNING state, blocking touchpad gestures
    // until the next mouse move resolves it.
    if msg == DM_POINTERHITTEST {
        let pointer_id = get_pointer_id(wparam);
        log::trace!("DM_POINTERHITTEST id={}", pointer_id);

        let mut pointer_type =
            windows::Win32::UI::WindowsAndMessaging::POINTER_INPUT_TYPE::default();
        let is_mouse = unsafe {
            windows::Win32::UI::Input::Pointer::GetPointerType(pointer_id, &mut pointer_type)
                .is_ok()
        } && pointer_type.0 == 4; // PT_MOUSE

        if !is_mouse {
            let data = get_subclass_data()
                .read()
                .ok()
                .and_then(|d| d.get(&(hwnd.0 as isize)).cloned());

            if let Some(state) = data {
                if let Ok(mut state_guard) = state.lock() {
                    state_guard.hit_test_count += 1;
                    state_guard.contact_count += 1;
                    let _ = unsafe { state_guard.viewport.SetContact(pointer_id) };
                }
            }
        }
        return LRESULT(0);
    }

    DefSubclassProc(hwnd, msg, wparam, lparam)
}
