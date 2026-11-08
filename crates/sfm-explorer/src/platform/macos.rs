// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The application menu's Quit item (Cmd+Q), made to ask before quitting.
//!
//! winit installs a default application menu whose Quit item sends
//! `terminate:` to the application. That ends the process through
//! `applicationWillTerminate:` without the window ever receiving a close
//! request, so the close prompt would never be asked. [`route_quit_to_window`]
//! points that item at the window's `performClose:` instead, which is what the
//! window's close button sends: AppKit calls the window delegate's
//! `windowShouldClose:`, winit reports it as `WindowEvent::CloseRequested`
//! and refuses the close, and the viewer then asks `AppState::request_quit`
//! as it does for the close button.

use objc2::runtime::AnyObject;
use objc2::sel;
use objc2_app_kit::{NSApplication, NSMenu, NSView, NSWindow};
use objc2_foundation::MainThreadMarker;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

/// Point every menu item that would send `terminate:` at `window`'s
/// `performClose:`, keeping its title and its Cmd+Q key equivalent.
///
/// Called once the window exists, which is after winit has installed its
/// default menu (it does so in `applicationDidFinishLaunching:`, before the
/// first `resumed`). A menu item holds its target weakly, so the item is
/// left with no target rather than a dangling one if the window goes away
/// first. Does nothing off the main thread or without a menu.
pub(crate) fn route_quit_to_window(window: &Window) {
    if MainThreadMarker::new().is_none() {
        return;
    }
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };
    // SAFETY: winit hands out the window's own `NSView`, which lives as long
    // as the window, and this runs on the main thread (checked above).
    let view: &NSView = unsafe { handle.ns_view.cast().as_ref() };
    let Some(ns_window) = view.window() else {
        return;
    };
    let app = NSApplication::sharedApplication(MainThreadMarker::from(view));
    // SAFETY: a plain property read on the main thread.
    if let Some(menu) = unsafe { app.mainMenu() } {
        retarget_terminate(&menu, &ns_window);
    }
}

/// Retarget the `terminate:` items of `menu` and of its submenus.
fn retarget_terminate(menu: &NSMenu, window: &NSWindow) {
    // SAFETY: plain menu reads and writes on the main thread; `performClose:`
    // is a method `NSWindow` has, so the new target answers the new action.
    unsafe {
        for item in menu.itemArray().iter() {
            if item.action() == Some(sel!(terminate:)) {
                item.setAction(Some(sel!(performClose:)));
                let target: &AnyObject = window;
                item.setTarget(Some(target));
            }
            if let Some(submenu) = item.submenu() {
                retarget_terminate(&submenu, window);
            }
        }
    }
}
