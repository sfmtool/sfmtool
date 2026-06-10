// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(windows)]

use std::process::{Child, Command};
use std::time::Duration;

use xa11y::{App, AppExt, Toggled};

fn launch() -> Child {
    Command::new(env!("CARGO_BIN_EXE_sfm-explorer"))
        .spawn()
        .expect("failed to spawn sfm-explorer")
}

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

fn attach(child: &Child) -> App {
    // Generous timeout: the first launch on a cold CI runner pays wgpu
    // adapter/shader init and AV scanning of the fresh binary, which has
    // been observed to exceed 15s. Healthy launches attach in ~1s; this
    // only delays failure reporting.
    App::by_pid(child.id(), Duration::from_secs(60)).expect("sfm-explorer did not appear")
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
    let b = app
        .as_element()
        .data()
        .bounds
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
        assert!(
            app.locator(&format!(r#"button[name="{name}"]"#))
                .exists()
                .unwrap_or(false),
            "'{name}' menu button not found",
        );
    }
}

/// The empty-state placeholder text is shown before any file is loaded.
#[test]
fn empty_state_placeholder_text() {
    let _guard = Guard(launch());
    let app = attach(&_guard.0);
    assert!(
        app.locator(r#"static_text[name="No reconstruction loaded."]"#)
            .exists()
            .unwrap_or(false),
        "placeholder text 'No reconstruction loaded.' not found",
    );
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
            .wait_attached(Duration::from_secs(3))
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
            .wait_attached(Duration::from_secs(3))
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
        .wait_attached(Duration::from_secs(3))
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
            Duration::from_secs(3),
        )
        .expect("Show Points should be unchecked after toggle");
}

/// Diagnostic: dump the tree after pressing View (run with -- --ignored --nocapture).
#[test]
#[ignore]
fn dump_tree_after_view() {
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
