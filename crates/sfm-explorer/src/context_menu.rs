// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The context menu this window opens, on a secondary click and on nothing
//! else.
//!
//! `egui::Popup::context_menu` opens on `egui::Response::secondary_clicked`,
//! which is a real right click *or* a long touch: egui implements
//! press-and-hold for a context menu on touch screens by reading a primary
//! contact held still for longer than `max_click_duration` (0.8 s) as a
//! secondary click.
//!
//! This window is not on the path that assumption was written for. On Windows
//! `platform::windows::create_manager` turns on `EnableMouseInPointer`, which
//! DirectManipulation needs in order to see precision-touchpad contacts at
//! all, and the side effect is that every mouse button arrives as a
//! `WM_POINTER*` message, which winit renders as a `Touch` event.
//! `platform::windows::restore_mouse_button` rewrites the secondary and middle
//! buttons back into real `MouseInput`, but leaves the left button on the touch
//! path, which the 3D viewport's drag handling is built against. So to egui the
//! left mouse button **is** a finger: rest a left press for 0.8 s, which is
//! what a slow pan in Image Detail does before the drag begins and what any
//! pause before letting go does, and egui reports a secondary click and the
//! context menu comes up under a left button.
//!
//! [`on_secondary_click`] is `Popup::context_menu` with the long touch taken
//! out of the open condition. The cost is that press-and-hold on a real touch
//! screen opens none of this window's menus; that is accepted, because the
//! window is driven by a mouse or a touchpad and every menu it has is reachable
//! by a right click.

/// A context menu on `response`, opened by a secondary click and by nothing
/// else.
///
/// A drop-in replacement for `egui::Popup::context_menu`: the same menu kind,
/// layout, style and pointer anchoring, and the same explicit close when the
/// widget under the menu is clicked. Only the open condition differs, for the
/// reason the module documentation gives. The returned `Popup` is still open to
/// the builder methods a caller needs, `close_behavior` above all:
///
/// ```ignore
/// context_menu::on_secondary_click(&row)
///     .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
///     .show(|ui| node_context_menu(ui, node, out));
/// ```
pub(crate) fn on_secondary_click(response: &egui::Response) -> egui::Popup<'static> {
    on_secondary_click_where(response, true)
}

/// [`on_secondary_click`] for a widget only part of which carries the menu: a
/// secondary click opens it when `here` is true and closes it otherwise.
///
/// For a widget that paints several things and hit-tests them itself, such as
/// the Image Browser strip, whose thumbnails carry the image menu and whose
/// gaps and minibar carry none.
pub(crate) fn on_secondary_click_where(
    response: &egui::Response,
    here: bool,
) -> egui::Popup<'static> {
    let open = if response.clicked_by(egui::PointerButton::Secondary) {
        Some(egui::SetOpenCommand::Bool(here))
    } else if response.clicked() {
        // Without this an open menu would stay up when the widget under it is
        // clicked. It is also what keeps a long touch from leaving one open:
        // egui reports the long touch as an ordinary click as well, so the
        // gesture that used to open the menu now closes it.
        Some(egui::SetOpenCommand::Bool(false))
    } else {
        None
    };
    egui::Popup::menu(response)
        .open_memory(open)
        .at_pointer_fixed()
}

/// [`on_secondary_click`] for a menu whose target is decided after the click
/// lands: the 3D view's, whose point is the GPU pick at the click's pixel,
/// read back a frame or more later.
///
/// `open` is true on the frame the target is decided, and the menu then stands
/// at `at`, the place of the click, wherever the pointer has gone since. A
/// secondary click closes a menu that is up, since the target of the new one
/// is not known yet, and so does a primary click on the widget, as
/// [`on_secondary_click`]'s does. The caller opens it only on a secondary
/// click egui counted as a click, so a long press and a right drag open none.
///
/// egui keeps the place of a menu opened at the pointer in its memory, and
/// offers no way to set it, so the place is kept beside it, where
/// [`position_of_id`] reads it.
pub(crate) fn once_decided(
    response: &egui::Response,
    open: bool,
    at: egui::Pos2,
) -> egui::Popup<'static> {
    let set = if open {
        Some(egui::SetOpenCommand::Bool(true))
    } else if response.clicked_by(egui::PointerButton::Secondary) || response.clicked() {
        Some(egui::SetOpenCommand::Bool(false))
    } else {
        None
    };
    let id = egui::Popup::default_response_id(response);
    response
        .ctx
        .data_mut(|data| data.insert_temp(placed_id(id), at));
    egui::Popup::menu(response).open_memory(set).at_position(at)
}

/// Where the context menu `popup_id` was opened, while it is open: egui's
/// record for a menu opened at the pointer, or [`once_decided`]'s for one
/// opened at the click once its target was decided. `None` for a popup
/// opened against its button.
pub(crate) fn position_of_id(ctx: &egui::Context, popup_id: egui::Id) -> Option<egui::Pos2> {
    egui::Popup::position_of_id(ctx, popup_id).or_else(|| {
        egui::Popup::is_id_open(ctx, popup_id)
            .then(|| ctx.data(|data| data.get_temp(placed_id(popup_id))))
            .flatten()
    })
}

/// Where [`once_decided`] keeps the place of its menu.
fn placed_id(popup_id: egui::Id) -> egui::Id {
    popup_id.with("placed at")
}

#[cfg(test)]
mod tests;
