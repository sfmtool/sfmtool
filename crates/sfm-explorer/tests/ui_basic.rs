// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos"))]

use std::process::{Child, Command};
use std::sync::Once;
use std::time::Duration;

use xa11y::{App, AppExt, Toggled};

/// xa11y 0.9 no longer hardcodes a 5s default; an unset default means
/// single-attempt, no-polling. The polling locator ops here (`exists`,
/// `press`, `toggle`) rely on a non-zero default, so set one process-wide
/// before any of them run.
///
/// macOS gets a much larger budget: a freshly launched app's deep widget
/// subtree (menu buttons, checkboxes, labels) isn't queryable over the AX API
/// for several seconds after launch, even though the app/window nodes register
/// quickly. The read-only checks poll the default timeout, so it must outlast
/// that registration lag.
fn init() {
    static SET_TIMEOUT: Once = Once::new();
    #[cfg(target_os = "macos")]
    let default = Duration::from_secs(60);
    #[cfg(not(target_os = "macos"))]
    let default = Duration::from_secs(5);
    SET_TIMEOUT.call_once(|| xa11y::set_default_timeout(default));
}

fn launch() -> Child {
    #[allow(unused_mut)] // `cmd` is only mutated on macOS (see below)
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sfm-explorer"));
    // Keep egui rendering so its AccessKit tree stays fresh for queries — an
    // idle window can be inspected before the tree is fully published. Only
    // needed on macOS; Windows attaches to a window that already repaints
    // enough, and forcing ControlFlow::Poll there would disturb its
    // DirectManipulation timer.
    #[cfg(target_os = "macos")]
    cmd.env("SFM_EXPLORER_FORCE_REPAINT", "1");
    cmd.spawn().expect("failed to spawn sfm-explorer")
}

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

// Generous timeout: the first launch on a cold CI runner pays wgpu
// adapter/shader init (and, on Windows, AV scanning of the fresh binary),
// which has been observed to exceed 15s. Healthy launches attach in ~1s.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(60);

/// Budget for a widget to appear in (or update within) the tree. macOS needs a
/// much larger window: a freshly launched app's deep widget subtree isn't
/// queryable over the AX API for several seconds after launch. Polling lookups
/// return as soon as the element appears, so healthy cases stay fast.
#[cfg(target_os = "macos")]
const CONTENT_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(not(target_os = "macos"))]
const CONTENT_TIMEOUT: Duration = Duration::from_secs(5);

fn attach(child: &Child) -> App {
    init();
    attach_app(child)
}

/// On Windows, xa11y 0.9's `by_pid` roots at the first top-level window for the
/// pid, which is one of winit's helper windows (a 16px-wide "group") rather than
/// our UI, so locator queries and bounds resolve against the wrong element.
/// Select our window by its title instead.
#[cfg(windows)]
fn attach_app(_child: &Child) -> App {
    App::find(ATTACH_TIMEOUT, |d| {
        d.name.as_deref() == Some("SfM Explorer")
    })
    .expect("sfm-explorer window did not appear")
}

/// On macOS a process is a single AXApplication whose name is the executable,
/// not the window title, so `by_pid` resolves the right root directly — and a
/// title-based `find` would just burn the full timeout before any fallback.
#[cfg(target_os = "macos")]
fn attach_app(child: &Child) -> App {
    App::by_pid(child.id(), ATTACH_TIMEOUT).expect("sfm-explorer did not appear")
}

// --- Window-level tests ---

/// App process appears in the accessibility tree.
#[test]
fn window_appears() {
    let _guard = Guard(launch());
    attach(&_guard.0);
}

/// The window respects the 800×600 minimum size constraint.
#[test]
fn window_min_size() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);
    // The attached root is the window itself on Windows but the AXApplication on
    // macOS, whose own bounds are unset — fall back to the window element there.
    let b = app
        .as_element()
        .data()
        .bounds
        .or_else(|| {
            app.locator(r#"window"#)
                .wait_attached(Duration::from_secs(5))
                .ok()
                .and_then(|w| w.data().bounds)
        })
        .expect("window has no bounds");
    assert!(b.width >= 800, "width {} < 800", b.width);
    assert!(b.height >= 600, "height {} < 600", b.height);
}

// --- Menu bar tests (AccessKit) ---

/// Both top-level menu buttons are exposed in the accessibility tree.
#[test]
fn menu_bar_buttons_present() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);
    for name in ["File", "View"] {
        app.locator(&format!(r#"button[name="{name}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("'{name}' menu button not found"));
    }
}

/// The empty-state placeholder text is shown before any file is loaded.
#[test]
fn empty_state_placeholder_text() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);
    app.locator(r#"static_text[name="No reconstruction loaded."]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("placeholder text 'No reconstruction loaded.' not found");
}

/// Opening the File menu exposes all three items in the accessibility tree.
#[test]
fn file_menu_items() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);

    app.locator(r#"button[name="File"]"#)
        .press()
        .expect("press File menu button");

    for item in ["Open...", "Load Demo Data...", "Quit"] {
        app.locator(&format!(r#"button[name="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("File menu item '{item}' did not appear"));
    }
}

/// Opening the View menu shows Show Points, Show Camera Images, and Show Grid
/// checkboxes all checked by default.
#[test]
fn view_checkboxes_checked_by_default() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);

    app.locator(r#"button[name="View"]"#)
        .press()
        .expect("press View menu button");

    for name in ["Show Points", "Show Camera Images", "Show Grid"] {
        let el = app
            .locator(&format!(r#"check_box[name="{name}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("View checkbox '{name}' did not appear"));
        assert!(
            matches!(el.data().states.checked, Some(Toggled::On)),
            "'{name}' should be checked by default (got {:?})",
            el.data().states.checked,
        );
    }
}

/// Toggling the Show Points checkbox via accessibility updates its checked state.
#[test]
fn toggle_show_points() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);

    app.locator(r#"button[name="View"]"#)
        .press()
        .expect("open View menu");

    // Verify initial checked state
    let el = app
        .locator(r#"check_box[name="Show Points"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("Show Points checkbox not found");
    assert!(
        matches!(el.data().states.checked, Some(Toggled::On)),
        "Show Points should start checked",
    );

    // Toggle it off
    app.locator(r#"check_box[name="Show Points"]"#)
        .toggle()
        .expect("toggle Show Points");

    // Wait for egui to process the action and update the tree
    app.locator(r#"check_box[name="Show Points"]"#)
        .wait_until(
            |data| data.is_some_and(|d| matches!(d.states.checked, Some(Toggled::Off))),
            CONTENT_TIMEOUT,
        )
        .expect("Show Points should be unchecked after toggle");
}

/// Diagnostic: dump the tree after pressing View (run with -- --ignored --nocapture).
#[test]
#[ignore]
fn dump_tree_after_view() {
    init();
    let _guard = Guard(launch());
    let pid = _guard.0.id();
    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    app.locator(r#"button[name="View"]"#)
        .press()
        .expect("press View");
    std::thread::sleep(Duration::from_secs(1));
    println!(
        "{}",
        app.dump(Some(6))
            .unwrap_or_else(|e| format!("dump error: {e}"))
    );
}

/// Diagnostic: dump the accessibility tree (run with -- --ignored --nocapture).
#[test]
#[ignore]
fn dump_tree() {
    let child = launch();
    let pid = child.id();
    let _guard = Guard(child);
    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    println!(
        "{}",
        app.dump(Some(5))
            .unwrap_or_else(|e| format!("dump error: {e}"))
    );
}
