// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(windows)]

use std::process::{Child, Command};
use std::time::Duration;

use xa11y::{App, AppExt};

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

/// App process appears in the accessibility tree within the timeout.
#[test]
fn window_appears() {
    let child = launch();
    let pid = child.id();
    let _guard = Guard(child);
    App::by_pid(pid, Duration::from_secs(15)).expect("sfm-explorer did not appear within 15s");
}

/// The window carries the expected title.
#[test]
fn window_title() {
    let child = launch();
    let pid = child.id();
    let _guard = Guard(child);

    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    let name = app.as_element().data().name.clone().unwrap_or_default();
    assert!(
        name.contains("SfM Explorer"),
        "expected window name to contain 'SfM Explorer', got: {name:?}",
    );
}

/// The window respects the 800×600 minimum size constraint.
#[test]
fn window_min_size() {
    let child = launch();
    let pid = child.id();
    let _guard = Guard(child);

    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    let b = app
        .as_element()
        .data()
        .bounds
        .expect("window element has no bounds");
    assert!(b.width >= 800, "width {} < 800", b.width);
    assert!(b.height >= 600, "height {} < 600", b.height);
}

/// The standard window control buttons are present and enabled.
#[test]
fn window_controls_exist() {
    let child = launch();
    let pid = child.id();
    let _guard = Guard(child);

    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    for name in ["Minimize", "Maximize", "Close"] {
        let exists = app
            .locator(&format!(r#"button[name="{name}"]"#))
            .exists()
            .unwrap_or(false);
        assert!(exists, "window control button '{name}' not found");
    }
}
