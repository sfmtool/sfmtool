// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Windows backend's own logic — touch-phase bookkeeping and the
//! DirectManipulation state machine — without a real window.

use winit::dpi::PhysicalPosition;
use winit::event::{DeviceId, Touch};

use super::*;

fn touch(phase: TouchPhase) -> WindowEvent {
    WindowEvent::Touch(Touch {
        device_id: DeviceId::dummy(),
        phase,
        location: PhysicalPosition::new(120.0, 48.0),
        force: None,
        id: 1,
    })
}

/// The bug this exists for: with `EnableMouseInPointer` on, a right-click
/// reaches winit as a touch contact, and egui would read it as the primary
/// button — so nothing in the app could ever be `secondary_clicked` and no
/// context menu could open.
#[test]
fn a_secondary_click_is_restored_as_a_real_right_button_press_and_release() {
    for (phase, expected) in [
        (TouchPhase::Started, ElementState::Pressed),
        (TouchPhase::Ended, ElementState::Released),
    ] {
        let [moved, click] = restore_mouse_button_from(&touch(phase), BUTTON_RIGHT)
            .expect("a mouse right-button contact is rewritten");
        assert!(
            matches!(
                moved,
                WindowEvent::CursorMoved { position, .. }
                    if position == PhysicalPosition::new(120.0, 48.0)
            ),
            "the click was not positioned at the contact: {moved:?}"
        );
        assert!(
            matches!(
                click,
                WindowEvent::MouseInput { state, button, .. }
                    if state == expected && button == MouseButton::Right
            ),
            "{phase:?} did not become a {expected:?} right button: {click:?}"
        );
    }
}

#[test]
fn a_middle_click_is_restored_too() {
    let [_, click] = restore_mouse_button_from(&touch(TouchPhase::Started), BUTTON_MIDDLE)
        .expect("a mouse middle-button contact is rewritten");
    assert!(matches!(
        click,
        WindowEvent::MouseInput {
            button: MouseButton::Middle,
            ..
        }
    ));
}

/// Left clicks and real touch contacts (which record no mouse button) keep
/// the path they already travel: egui's touch emulation reports them as the
/// primary button, which is what they are.
#[test]
fn a_left_click_and_a_real_touch_contact_are_left_alone() {
    for down in [BUTTON_LEFT, 0] {
        assert!(restore_mouse_button_from(&touch(TouchPhase::Started), down).is_none());
        assert!(restore_mouse_button_from(&touch(TouchPhase::Ended), down).is_none());
    }
}

/// A move carries no button, and a cancelled contact has no click to
/// deliver; both still have to reach egui as they are so the pointer
/// position keeps up.
#[test]
fn only_the_press_and_release_of_a_contact_are_rewritten() {
    for phase in [TouchPhase::Moved, TouchPhase::Cancelled] {
        assert!(restore_mouse_button_from(&touch(phase), BUTTON_RIGHT).is_none());
    }
}

#[test]
fn events_that_are_not_touches_pass_through() {
    assert!(restore_mouse_button_from(&WindowEvent::CloseRequested, BUTTON_RIGHT).is_none());
}

/// `PT_TOUCH` and `PT_TOUCHPAD` — the two non-mouse types that reach this
/// window on this hardware.
const PT_TOUCH: i32 = 2;
const PT_TOUCHPAD: i32 = 5;

#[test]
fn a_mouse_message_names_every_button_it_carries() {
    assert_eq!(mouse_buttons_from(PT_MOUSE, 0), Some(0));
    assert_eq!(
        mouse_buttons_from(PT_MOUSE, POINTER_FLAG_SECONDBUTTON),
        Some(BUTTON_RIGHT),
    );
    assert_eq!(
        mouse_buttons_from(
            PT_MOUSE,
            POINTER_FLAG_FIRSTBUTTON | POINTER_FLAG_THIRDBUTTON,
        ),
        Some(BUTTON_LEFT | BUTTON_MIDDLE),
    );
}

/// The bug this exists for: a precision-touchpad contact is in contact with
/// the pad, so it sets `POINTER_FLAG_FIRSTBUTTON` and carries a
/// `ptPixelLocation` of its own. Read as a mouse, a two-finger scroll
/// becomes a phantom cursor walking across the window — and every DM-driven
/// panel decides whether a gesture is for it by asking where the cursor is,
/// so the gesture lands on whatever panel the phantom is over rather than
/// the one under the real cursor.
#[test]
fn a_touch_or_touchpad_contact_is_not_the_mouse() {
    for pointer_type in [PT_TOUCH, PT_TOUCHPAD] {
        assert_eq!(
            mouse_buttons_from(pointer_type, POINTER_FLAG_FIRSTBUTTON),
            None
        );
    }
}

/// The bug this exists for: on Windows every click arrives as a contact, and
/// `egui-winit` ends a contact by forgetting where the pointer is
/// (`Event::PointerGone`), which costs `egui::ScrollArea` the `interact_pos`
/// it takes a scroll by. A click on a Track Edit row or a Scene tree camera
/// left that list dead to the wheel and to a two-finger pan until the mouse was
/// moved. The release is followed back into egui by the cursor position the
/// window procedure tracks, so the pointer never goes anywhere.
#[test]
fn a_contact_that_ends_puts_the_pointer_back_where_the_cursor_is() {
    let cursor = PhysicalPosition::new(640.0, 360.0);
    for phase in [TouchPhase::Ended, TouchPhase::Cancelled] {
        let moved = restore_pointer_after_click_at(&touch(phase), Some(cursor))
            .expect("a contact that ended restores the pointer");
        assert!(
            matches!(
                moved,
                WindowEvent::CursorMoved { position, .. } if position == cursor
            ),
            "{phase:?} did not restore the cursor position: {moved:?}"
        );
    }
}

/// The contact's own location is not the cursor: for a touchpad contact it is
/// the finger on the pad, which is why the tracked mouse position is what the
/// pointer is put back to. A press or a move needs no restoring: `egui-winit`
/// positions the pointer from both of those itself.
#[test]
fn only_the_end_of_a_contact_restores_the_pointer() {
    for phase in [TouchPhase::Started, TouchPhase::Moved] {
        assert!(restore_pointer_after_click_at(
            &touch(phase),
            Some(PhysicalPosition::new(1.0, 2.0))
        )
        .is_none());
    }
    assert!(restore_pointer_after_click_at(
        &WindowEvent::CloseRequested,
        Some(PhysicalPosition::new(1.0, 2.0)),
    )
    .is_none());
}

/// With no mouse position on record there is nothing to restore, and the
/// contact keeps the path it travels today, which is what a machine whose only
/// pointer is a touch screen wants.
#[test]
fn a_contact_that_ends_with_no_cursor_on_record_is_left_alone() {
    assert!(restore_pointer_after_click_at(&touch(TouchPhase::Ended), None).is_none());
}
