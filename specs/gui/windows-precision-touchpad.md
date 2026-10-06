# Windows Precision Touchpad Support

On Windows, SfM Explorer reads two-finger pans and pinches on a precision
touchpad through Microsoft's DirectManipulation API, which recognizes the
gestures and smooths them, instead of through the windowing library, which does
not report them well. The 3D viewport turns these gestures into orbit, pan and
zoom, the Image Detail panel into panning and zooming its image, and the other
panels into scrolling. This spec covers why the API
is needed, the order in which it has to be set up beside the winit window and
event loop, how a gesture is routed to the panel under the cursor, and the
examples that reproduce the setup on its own. What each gesture does in the 3D
viewport is in [viewport-navigation.md](viewport-navigation.md#trackpad-controls).

## Where the code is

The integration lives in
[platform/windows.rs](../../crates/sfm-explorer/src/platform/windows.rs): the
manager created before the event loop (`create_manager`), the gesture handler
attached to the window (`WinGestureHandler`), the window subclass procedure, and
the pointer filtering described below. The platform-neutral side that the panels
call, `platform::pointer_in_rect` and `platform::gesture_scroll_events`, is in
[platform/mod.rs](../../crates/sfm-explorer/src/platform/mod.rs). The manager is
created in `run_with_args` in [lib.rs](../../crates/sfm-explorer/src/lib.rs),
which also drives the per-frame update, and the handler is attached by
`try_init_gesture_handler` in [app.rs](../../crates/sfm-explorer/src/app.rs).
None of this is behind a Cargo feature: it is compiled on every Windows build.
The `directmanipulation` feature in
[Cargo.toml](../../crates/sfm-explorer/Cargo.toml) gates only the three
standalone examples listed under [Reference Implementation](#reference-implementation).

## Problem Statement

On Windows, trackpad viewport navigation works poorly compared to Blender. The winit
library does not properly expose precision touchpad gestures to applications.

## Solution: DirectManipulation API

Microsoft's DirectManipulation API provides hardware-accelerated gesture recognition for
precision touchpads. This is the same approach used by Blender and Firefox. How Blender
wires it up, which is what this design was drawn from, is written up in
[research/blender-viewport-navigation-implementation-overview.md](../research/blender-viewport-navigation-implementation-overview.md).

**Key benefits:**
- Automatic pan vs. pinch gesture detection
- Noise filtering and smoothing
- Works with all Windows Precision Touchpads

## Implementation

The implementation is in
[platform/windows.rs](../../crates/sfm-explorer/src/platform/windows.rs):

1. Create DirectManipulation manager via `CoCreateInstance`
2. Create a viewport configured for pan/pinch gestures with inertia
3. Set viewport to `MANUALUPDATE` mode (we drive updates each frame)
4. Implement `IDirectManipulationViewportEventHandler` COM interface
5. Subclass the window to intercept `DM_POINTERHITTEST` messages
6. On touchpad contact, call `viewport.SetContact(pointerId)`
7. Each frame, call `update_manager.Update()` to process gesture state
8. Extract pan/scale deltas from the transform matrix in `OnContentUpdated`

## Using DirectManipulation with Winit

Winit's window creation and event loop interact poorly with DirectManipulation out of the box.
Three issues must be worked around:

1. **`DM_POINTERHITTEST` is never generated** if the DM manager is created after winit's
   `EventLoop` initialization. DirectManipulation installs internal hooks when the manager is
   activated; something about winit's prior initialization prevents those hooks from observing
   precision touchpad input.

2. **`DM_POINTERHITTEST` is delivered via `SendMessage`**, which bypasses winit's message queue
   and `msg_hook`. Winit's wndproc receives the message but has no knowledge of
   DirectManipulation, so `SetContact()` is never called and gestures never begin.

3. **`WM_TIMER` is not delivered** when a winit-created window exists, so the standard pattern
   of calling `update_manager.Update()` from a timer callback does not work. Updates must be
   driven manually.

### Workaround

The following initialization order resolves all three issues:

```
OleInitialize()
EnableMouseInPointer(true)
CoCreateInstance(DirectManipulationManager)   ← BEFORE winit
manager.GetUpdateManager()                    ← BEFORE winit

EventLoopBuilder::default().build()           ← winit EventLoop
event_loop.create_window(...)                 ← winit window

SetWindowSubclass(hwnd, dm_subclass_proc)     ← intercept DM_POINTERHITTEST
manager.CreateViewport(hwnd)                  ← attach DM to the winit HWND
viewport.ActivateConfiguration(...)
viewport.SetViewportOptions(MANUALUPDATE)
viewport.AddEventHandler(hwnd, handler)
viewport.SetViewportRect(...)
manager.Activate(hwnd)
viewport.Enable()

// In the winit event loop (ControlFlow::WaitUntil at ~16ms):
update_manager.Update(None)                   ← drive DM manually each frame
```

### Subclass Procedure

The subclass procedure intercepts `DM_POINTERHITTEST` (0x0250) and calls `SetContact()`
synchronously before returning. All other messages are forwarded to winit via `DefSubclassProc`.

```rust
unsafe extern "system" fn dm_subclass_proc(
    hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM,
    _uid_subclass: usize, _dw_ref_data: usize,
) -> LRESULT {
    if msg == 0x0250 {  // DM_POINTERHITTEST
        let pointer_id = (wparam.0 & 0xFFFF) as u32;
        viewport.SetContact(pointer_id);  // must be called synchronously
        return LRESULT(0);
    }
    DefSubclassProc(hwnd, msg, wparam, lparam)
}
```

(In practice, the viewport is accessed via a global or passed through `dw_ref_data`.
See `winit_directmanipulation.rs` for the full implementation.)

`SetContact()` **must** be called synchronously inside the subclass proc. DirectManipulation
uses `SendMessage` precisely because it expects an immediate response. Deferring the call
(e.g., posting a message) causes DM to assume the app declined to track that contact.

### Which panel a gesture is addressed to

DM gesture events are polled once per frame in `app.rs` and handed to every
panel at once; each decides whether a gesture is for it by asking
`platform::pointer_in_rect(ctx, panel_rect)`. That is a raw geometric
containment test against the OS cursor position rather than egui's hover state,
which goes stale after a click with no movement and which knows nothing about
layers (the HUD has to correct for the latter — see
[viewport-hud.md](viewport-hud.md)).

On Windows the cursor position comes from the `WM_POINTERDOWN` /
`WM_POINTERUP` / `WM_POINTERUPDATE` messages the subclass procedure already
decodes for the button state, and **only from mouse ones**
(`POINTER_INFO.pointerType == PT_MOUSE`). `EnableMouseInPointer(true)` routes
the mouse through `WM_POINTER*`, but so does every touch, pen and
precision-touchpad contact — and a touchpad contact carries a `ptPixelLocation`
of its own: the finger on the pad, mapped onto the screen. It has nothing to do
with where the cursor is, and it walks across the window as the fingers move.
Admitted, it turns the tracked cursor into a phantom that follows the gesture
around, and the gesture lands on whichever panel the phantom is over rather than
the one under the real cursor — a two-finger scroll aimed at the Camera
Intrinsics panel panning the Image Detail image beside it. The same rule covers
the button state: a contact resting on the pad sets `POINTER_FLAG_FIRSTBUTTON`,
which is not a mouse button being held.

The other half of the rule is that the rect a panel tests must be the panel's
own. `ui.available_rect_before_wrap()` is only that if nothing drawn above it in
the same `Ui` has overflowed, since egui grows a `Ui`'s `max_rect` to include a
widget that did — see
[multi-panel-image-browser.md](multi-panel-image-browser.md#the-toolbar-may-not-widen-the-panel),
where the Image Detail overlay toolbar overflowing a narrow dock cell handed
that panel gestures aimed at the one beside it.

`platform::windows::mouse_buttons_from` is the single decision point. It returns
`None` for a non-mouse pointer, and both statics — the button mask and the
cursor position — are written only when it returns `Some`. The one thing a
non-mouse press still updates is `LAST_MOUSE_DOWN_BUTTON`, which it *clears*, so
that a touch never inherits the button of the last real click.

### A click must not take the pointer away from egui

The panels that have no gesture handling of their own are scrolled by the wheel
event `platform::gesture_scroll_events` synthesizes, and egui delivers a wheel
to whichever `ScrollArea` its own pointer position is inside: `interact_pos`,
not the tracked cursor above. That position does not survive a click. With
`EnableMouseInPointer(true)` a click reaches winit as a `Touch`, and `egui-winit`
ends a touch the way a finger leaving a touch screen ends one: it forgets its
pointer position and pushes `Event::PointerGone`, so that nothing is left
hovered. egui clears `interact_pos` on the frame after that, and until the
pointer moves again no scroll area in the app takes a scroll: neither a wheel
notch nor a two-finger pan. The panels that read DM events themselves are
unaffected, since they route by `platform::pointer_in_rect`.

`platform::windows::restore_pointer_after_click` closes that gap: a contact that
ends is followed into egui by a `CursorMoved` carrying the tracked cursor
position, which puts the pointer back where it never stopped being. The tracked
position rather than the contact's own location, for the reason above (a
touchpad contact's location is the finger on the pad), and so egui's hover and
the position gestures are routed by stay the same one. Before any mouse pointer
message has placed the cursor there is nothing to restore and the contact is
left alone, which is also what a machine whose only pointer is a touch screen
wants. The secondary and middle contacts need none of this: rewriting them into
`MouseInput` drops the `Touch`, and the `PointerGone` with it.

### Why Early DM Manager Creation Matters

When `CoCreateInstance(DirectManipulationManager)` is called **before** winit's `EventLoop`
initialization, `DM_POINTERHITTEST` is generated with the correct pointer type (type=5,
PT_TOUCHPAD). When created **after** winit initialization, the message is never generated —
DirectManipulation's internal hooks fail to observe precision touchpad input.

The exact mechanism is not fully understood, but a series of tests working to isolate the problem
narrowed it down: winit's `EventLoop::new()` or window creation changes something about how
Windows routes pointer input such that DirectManipulation's observation hooks no longer fire.
Creating the DM manager first avoids this interaction.

### Why Manual Update Calls Are Needed

DirectManipulation in `MANUALUPDATE` mode expects the application to call
`update_manager.Update()` periodically so it can process accumulated gesture state and fire
`OnContentUpdated` callbacks. The standard approach uses `SetTimer` / `WM_TIMER`, but
`WM_TIMER` messages are not delivered when a winit-created window exists (the cause is
unknown but was consistently observed across tests of winit).

The workaround is to call `Update()` from winit's event loop using `ControlFlow::WaitUntil`
with a ~16ms interval, triggered from `new_events` when `StartCause::ResumeTimeReached` fires.

### Reference Implementation

See [examples/winit_directmanipulation.rs](../../crates/sfm-explorer/examples/winit_directmanipulation.rs)
for a minimal working example of all three workarounds combined. Compare with
[examples/win32_directmanipulation.rs](../../crates/sfm-explorer/examples/win32_directmanipulation.rs)
for a minimal working example directly using Win32.

Additional test example:

- [examples/winit_wgpu_directmanipulation.rs](../../crates/sfm-explorer/examples/winit_wgpu_directmanipulation.rs) — winit + wgpu + DM (working). Tests that wgpu/DXGI
  surface creation does not interfere with DirectManipulation.

All examples require the `directmanipulation` Cargo feature:

```
cargo run --example win32_directmanipulation --features directmanipulation
cargo run --example winit_directmanipulation --features directmanipulation
cargo run --example winit_wgpu_directmanipulation --features directmanipulation
```

## Known Limitation: Eframe Window Creation

Extensive investigation confirmed that DirectManipulation does not work on windows created through
eframe's `WgpuWinitApp::resumed()` code path. The symptom is that `DM_POINTERHITTEST` is never
generated — all pointer events arrive as `PT_MOUSE` (4) instead of `PT_TOUCHPAD` (5), meaning
DM's internal observation hooks fail to install.

The root cause was never fully identified. Every individual component of eframe's initialization
(window creation, wgpu setup, egui context, event dispatch wrapper, etc.) works correctly when
tested in isolation. The failure only occurs when running through eframe's actual
`WinitAppWrapper<WgpuWinitApp>` as the `ApplicationHandler`. The process is not globally
poisoned — a second window created after eframe initialization receives DM events correctly.

The working solution bypasses this issue by setting up DirectManipulation directly on the winit
window handle, using the three workarounds documented above. Since sfm-explorer creates its own
winit event loop and window (not using `eframe::run_native`), this limitation does not apply.
