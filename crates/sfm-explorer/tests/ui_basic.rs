// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos"))]

use std::process::{Child, Command};
use std::sync::{Mutex, MutexGuard, Once};
use std::time::{Duration, Instant};

use xa11y::{App, AppExt, Toggled};

/// Serializes the UI tests so at most one `sfm-explorer` window is alive at a
/// time. `cargo test` runs tests on multiple threads by default, and several
/// identically-titled "SfM Explorer" windows (plus concurrent accessibility
/// tree walks) make the Windows UI Automation backend fail with
/// `E_UNEXPECTED` (0x8000FFFF, "Catastrophic failure"). Each test holds this
/// lock for its whole body, so a plain `cargo test` behaves the same as
/// `--test-threads=1` without the caller having to remember the flag.
static UI_TEST_LOCK: Mutex<()> = Mutex::new(());

fn ui_test_lock() -> MutexGuard<'static, ()> {
    // A panicking test (every assertion failure here panics) would otherwise
    // poison the mutex and turn every later test into a spurious failure;
    // recover the guard instead so the suite keeps running serially.
    UI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// xa11y (since 0.9) no longer hardcodes a 5s default; an unset default means
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
    cmd.env("SFMTOOL_EXPLORER_FORCE_REPAINT", "1");
    cmd.spawn().expect("failed to spawn sfm-explorer")
}

/// Owns a launched `sfm-explorer` process and the serialization lock for the
/// test that spawned it. Fields drop in declaration order, so `child` is killed
/// (its window torn down) *before* `_lock` is released and the next test may
/// launch — keeping windows strictly non-overlapping.
struct Guard {
    child: Child,
    _lock: MutexGuard<'static, ()>,
}

impl Guard {
    /// Acquire the serialization lock, then launch the app under it.
    fn new() -> Self {
        let _lock = ui_test_lock();
        Guard {
            child: launch(),
            _lock,
        }
    }

    fn child(&self) -> &Child {
        &self.child
    }

    /// Wait up to `budget` for the app to exit on its own. Returns whether it
    /// did — `false` means it was still running when the budget ran out.
    fn wait_for_exit(&mut self, budget: Duration) -> bool {
        let deadline = Instant::now() + budget;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return true,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                Ok(None) => return false,
                Err(e) => panic!("failed to poll the app process: {e}"),
            }
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
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
///
/// Windows needs more than a nominal budget too: the HUD tests' first lookup
/// comes right after loading the demo scene, which pays the scene renderer's
/// wgpu pipeline init on a cold CI runner before the next AccessKit push can
/// publish the HUD. At 5s that lookup timed out flakily with the checkbox
/// present in the post-timeout diagnosis snapshot — it had appeared just past
/// the budget. Only a genuine failure ever waits the full budget.
#[cfg(target_os = "macos")]
const CONTENT_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(not(target_os = "macos"))]
const CONTENT_TIMEOUT: Duration = Duration::from_secs(30);

fn attach(child: &Child) -> App {
    init();
    attach_app(child)
}

/// On Windows, xa11y's `by_pid` roots at the first top-level window for the
/// pid (still true in 0.12: on Windows an "app" *is* a top-level window, so one
/// process can own several), which is one of winit's helper windows (a
/// 16px-wide "group") rather than our UI, so locator queries and bounds resolve
/// against the wrong element. Select our window by its title instead.
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
    let _guard = Guard::new();
    attach(_guard.child());
}

/// The window respects the 800×600 minimum size constraint.
#[test]
fn window_min_size() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
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

/// File is the only top-level menu. The display controls that made a View menu
/// worth having moved into the viewport HUD, and the dock panels are permanent,
/// so nothing app-global is left for a second menu — see
/// `specs/gui/gui-viewport-hud.md`.
#[test]
fn file_is_the_only_menu() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    app.locator(r#"button[name="File"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("'File' menu button not found");
    assert!(
        app.locator(r#"button[name="View"]"#)
            .wait_attached(Duration::from_millis(500))
            .is_err(),
        "the View menu is still in the menu bar"
    );
}

/// The empty-state placeholder text is shown before any file is loaded.
#[test]
fn empty_state_placeholder_text() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    app.locator(r#"static_text[name="No reconstruction loaded."]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("placeholder text 'No reconstruction loaded.' not found");
}

/// Opening the File menu exposes all three items in the accessibility tree.
#[test]
fn file_menu_items() {
    let _guard = Guard::new();
    let app = attach(_guard.child());

    app.locator(r#"button[name="File"]"#)
        .press()
        .expect("press File menu button");

    for item in ["Open...", "Close All", "Load Demo Data...", "Quit"] {
        app.locator(&format!(r#"button[name="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("File menu item '{item}' did not appear"));
    }
}

/// File > Quit exits the process.
///
/// It used to send `ViewportCommand::Close`, which this app's own winit loop
/// never reads, so the menu item did nothing at all. Asserting on the child
/// process rather than on the window is the point: only a real exit proves it.
#[test]
fn quit_menu_item_exits_the_process() {
    let mut guard = Guard::new();
    let app = attach(guard.child());

    app.locator(r#"button[name="File"]"#)
        .press()
        .expect("press File menu button");
    app.locator(r#"button[name="Quit"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("Quit item did not appear")
        .press()
        .expect("press Quit");

    assert!(
        guard.wait_for_exit(Duration::from_secs(10)),
        "File > Quit did not exit the process"
    );
}

/// Load demo data, so there is a reconstruction for the 3D viewer — and so the
/// HUD, which the dock only builds once one is loaded, exists to be found.
fn load_demo_data(app: &App) {
    app.locator(r#"button[name="File"]"#)
        .press()
        .expect("press File menu button");
    app.locator(r#"button[name="Load Demo Data..."]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("Load Demo Data item did not appear")
        .press()
        .expect("press Load Demo Data");
    app.locator(r#"button[name="Load"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("demo dialog's Load button did not appear")
        .press()
        .expect("press Load");
}

/// With a reconstruction loaded the viewport HUD is open, so its layer toggles
/// are reachable in the accessibility tree without opening anything — the point
/// of moving them out of a menu. Also the end-to-end check that the HUD reaches
/// a real window; everything else about it is exercised headlessly in
/// `viewer_3d/hud/tests.rs`.
#[test]
fn hud_layer_toggles_are_present_and_checked_once_a_scene_is_loaded() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    for name in ["Points", "Cameras", "Grid"] {
        let el = app
            .locator(&format!(r#"check_box[name="{name}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("HUD checkbox '{name}' did not appear"));
        assert!(
            matches!(el.data().states.checked, Some(Toggled::On)),
            "'{name}' should be checked by default (got {:?})",
            el.data().states.checked,
        );
    }
}

/// The Scene Graph panel reaches a real window: once the demo node is loaded,
/// its reconstruction row and the Cameras group beneath it are in the
/// accessibility tree. Everything else about the tree is exercised headlessly
/// in `scene_graph/tests.rs`.
#[test]
fn the_scene_panel_lists_the_loaded_reconstruction() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    // The demo reconstruction is labeled "demo" and rings the scene with 8
    // cameras, so both strings are fixed by the fixture.
    for text in ["demo", "Cameras (8)"] {
        app.locator(&format!(r#"static_text[name="{text}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Scene panel row '{text}' did not appear"));
    }
}

/// Toggling a HUD checkbox via accessibility updates its checked state.
#[test]
fn toggle_hud_layer_checkbox() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    let el = app
        .locator(r#"check_box[name="Grid"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("Grid checkbox not found");
    assert!(
        matches!(el.data().states.checked, Some(Toggled::On)),
        "Grid should start checked",
    );

    app.locator(r#"check_box[name="Grid"]"#)
        .toggle()
        .expect("toggle Grid");

    // Wait for egui to process the action and update the tree
    app.locator(r#"check_box[name="Grid"]"#)
        .wait_until(
            |data| data.is_some_and(|d| matches!(d.states.checked, Some(Toggled::Off))),
            CONTENT_TIMEOUT,
        )
        .expect("Grid should be unchecked after toggle");
}

/// A real right-click on the Scene panel's reconstruction row opens its context
/// menu.
///
/// Windows only, and driven by synthetic mouse input rather than the
/// accessibility API, because the defect this guards lives *below* egui and
/// nothing above the window can see it: `platform::windows::create_manager`
/// turns on `EnableMouseInPointer` for DirectManipulation, which routes every
/// mouse button through `WM_POINTER` — and winit 0.30 renders those as `Touch`
/// events that egui's touch emulation reads as the *primary* button. With that
/// unhandled, no `secondary_clicked` ever fires anywhere in the app: no context
/// menu can open, and a right-click on a tree row selects it like a left-click.
/// The whole panel behaves correctly under `Context::run_ui`, so only a real
/// window can catch it.
#[cfg(windows)]
#[test]
fn a_real_right_click_opens_the_reconstruction_rows_context_menu() {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
        MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEINPUT, MOUSE_EVENT_FLAGS,
    };
    use windows::Win32::UI::WindowsAndMessaging::SetCursorPos;

    fn mouse_event(flags: MOUSE_EVENT_FLAGS) {
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        unsafe { SendInput(&[input], std::mem::size_of::<INPUT>() as i32) };
    }

    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    // The demo node's row is labelled "demo"; its bounds are screen pixels.
    let row = app
        .locator(r#"static_text[name="demo"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the demo reconstruction row did not appear");
    let bounds = row.data().bounds.expect("the row has no bounds");
    let x = bounds.x + (bounds.width / 2) as i32;
    let y = bounds.y + (bounds.height / 2) as i32;

    // Two moves with a pause: the app repaints on demand, and egui resolves a
    // click against the widget rects of the frame before it.
    for _ in 0..2 {
        unsafe { SetCursorPos(x, y).ok() };
        std::thread::sleep(Duration::from_millis(250));
    }
    // A left click first, which both activates the window (a right-press on an
    // inactive one is swallowed by the activation) and puts the row through the
    // selection change that used to take its menu's identity with it.
    mouse_event(MOUSEEVENTF_LEFTDOWN);
    std::thread::sleep(Duration::from_millis(120));
    mouse_event(MOUSEEVENTF_LEFTUP);
    std::thread::sleep(Duration::from_millis(400));

    mouse_event(MOUSEEVENTF_RIGHTDOWN);
    std::thread::sleep(Duration::from_millis(120));
    mouse_event(MOUSEEVENTF_RIGHTUP);

    // "Close" is deliberately not checked: the window's own title-bar close
    // button carries that name too, so it would pass with no menu at all.
    for item in ["Select", "Zoom to Fit", "Reload from Disk"] {
        app.locator(&format!(r#"button[name="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| {
                panic!("context menu item '{item}' did not appear after a right-click")
            });
    }
}

/// Diagnostic: dump the accessibility tree (run with -- --ignored --nocapture).
#[test]
#[ignore]
fn dump_tree() {
    let _guard = Guard::new();
    let pid = _guard.child().id();
    let app = App::by_pid(pid, Duration::from_secs(15)).expect("app not found");
    println!(
        "{}",
        app.dump(Some(5))
            .unwrap_or_else(|e| format!("dump error: {e}"))
    );
}
