// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The application icon, as the viewer hands it to the window system.
//!
//! The PNG files are rendered from `assets/icon/sfm-explorer-icon.svg` by
//! `scripts/render_explorer_icon.py` and embedded at compile time, so the icon
//! is there however the viewer is started: as `sfm-explorer`, or as
//! `sfm explorer` through the `sfmtool._sfmtool` extension, whose process is
//! Python's. Each platform embeds only the sizes it uses. See
//! `specs/gui/application-icon.md`.

#[cfg(not(target_os = "macos"))]
use winit::window::Icon;

/// The window's icon: on Windows the small icon of the title bar, elsewhere
/// the one icon winit gives the window (`_NET_WM_ICON` on X11).
#[cfg(target_os = "windows")]
const WINDOW_ICON_PNG: &[u8] = include_bytes!("../assets/icon/sfm-explorer-icon-32.png");
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const WINDOW_ICON_PNG: &[u8] = include_bytes!("../assets/icon/sfm-explorer-icon-256.png");

/// The window's big icon on Windows, which the taskbar and Alt-Tab show.
#[cfg(target_os = "windows")]
const TASKBAR_ICON_PNG: &[u8] = include_bytes!("../assets/icon/sfm-explorer-icon-64.png");

/// The application's Dock icon on macOS, where winit ignores a window icon.
#[cfg(target_os = "macos")]
pub(crate) const DOCK_ICON_PNG: &[u8] = include_bytes!("../assets/icon/sfm-explorer-icon-512.png");

/// The window icon, or `None` with a logged warning if the embedded PNG does
/// not decode. A test decodes every embedded size, so `None` is not expected.
#[cfg(not(target_os = "macos"))]
pub(crate) fn window_icon() -> Option<Icon> {
    icon_from_png(WINDOW_ICON_PNG)
}

/// The window's big icon on Windows, as [`window_icon`] gives the small one.
#[cfg(target_os = "windows")]
pub(crate) fn taskbar_icon() -> Option<Icon> {
    icon_from_png(TASKBAR_ICON_PNG)
}

/// Decode `png` into a winit icon, logging a warning and returning `None` if
/// it is not a PNG winit can use.
#[cfg(not(target_os = "macos"))]
fn icon_from_png(png: &[u8]) -> Option<Icon> {
    match decode_icon_png(png) {
        Ok(icon) => Some(icon),
        Err(message) => {
            log::warn!("The viewer's window icon could not be read: {message}");
            None
        }
    }
}

/// Decode `png` to RGBA and build a winit icon from it.
#[cfg(not(target_os = "macos"))]
fn decode_icon_png(png: &[u8]) -> Result<Icon, String> {
    let rgba = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    let (width, height) = rgba.dimensions();
    Icon::from_rgba(rgba.into_raw(), width, height).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
