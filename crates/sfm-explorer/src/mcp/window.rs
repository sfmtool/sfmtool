// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The window block: what the wire says about the window the human is looking
//! at.
//!
//! Only the rendering lives here. What the window *is*, how a change to it is
//! applied, how a placement is spelled in a layout document, and what a window
//! read says when there is no window ([`crate::window::NO_WINDOW`]) are
//! [`crate::window`]'s — unconditional, because Panels ▸ Save Layout… carries
//! the window's placement in every build and not only where this feature is
//! compiled in.

use serde_json::{json, Value};

use crate::window::{MonitorInfo, WindowInfo};

/// The `window` block: what `get_scene` embeds, and — with `monitors`
/// alongside — what `get_window_layout` returns.
pub(super) fn block(info: &WindowInfo, monitors: Option<&[MonitorInfo]>) -> Value {
    // A scale factor of zero cannot come from winit, but the logical size is a
    // division and a reply is not the place to find that out.
    let scale = if info.scale_factor > 0.0 {
        info.scale_factor
    } else {
        1.0
    };
    let [outer_width, outer_height] = info.outer_size;
    let fraction = info.monitor.as_ref().map(|monitor| {
        let [monitor_width, monitor_height] = monitor.size;
        [
            f64::from(outer_width) / f64::from(monitor_width.max(1)),
            f64::from(outer_height) / f64::from(monitor_height.max(1)),
        ]
    });
    let mut value = json!({
        "state": info.state.wire_name(),
        "focused": info.focused,
        "scale_factor": info.scale_factor,
        "outer_position": info.outer_position,
        "outer_size": info.outer_size,
        "inner_size": info.inner_size,
        "monitor": info.monitor.as_ref().map(monitor_block),
        "derived": {
            "inner_size_logical": [
                f64::from(info.inner_size[0]) / scale,
                f64::from(info.inner_size[1]) / scale,
            ],
            // How much of the desktop this window is, per axis and by area —
            // the question a size alone cannot answer, since it depends on the
            // monitor the window is on.
            "monitor_fraction": fraction,
            "monitor_area_fraction": fraction.map(|[x, y]| x * y),
        },
    });
    if let Some(monitors) = monitors {
        value
            .as_object_mut()
            .expect("a window block is an object")
            .insert(
                "monitors".into(),
                Value::Array(monitors.iter().map(monitor_block).collect()),
            );
    }
    value
}

fn monitor_block(monitor: &MonitorInfo) -> Value {
    json!({
        "name": monitor.name,
        "position": monitor.position,
        "size": monitor.size,
        "scale_factor": monitor.scale_factor,
    })
}
