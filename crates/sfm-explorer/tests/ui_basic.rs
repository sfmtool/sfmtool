// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos", target_os = "linux"))]

use std::cell::RefCell;
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

/// Launch the viewer with the given arguments.
///
/// Every test but the startup-load one passes `--no-default-layout`: a
/// developer who has saved a layout of their own to
/// `~/.sfm-explorer-default-layout.json` must not have this suite's panel
/// assertions fail on their machine.
fn launch_with(args: &[&str]) -> Child {
    #[allow(unused_mut)] // `cmd` is only mutated on macOS (see below)
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sfm-explorer"));
    cmd.args(args);
    // Keep egui rendering so its AccessKit tree stays fresh for queries — an
    // idle window can be inspected before the tree is fully published. Only
    // needed on macOS; Windows attaches to a window that already repaints
    // enough, and forcing ControlFlow::Poll there would disturb its
    // DirectManipulation timer. Linux needs it as little as Windows does, for
    // a different reason: AccessKit's Unix adapter pushes the tree onto the
    // AT-SPI bus, where it stays readable after the viewer goes idle, and an
    // action arriving back over that bus wakes the loop for the frame that
    // answers it — the suite passes there with the viewer idling.
    #[cfg(target_os = "macos")]
    cmd.env("SFMTOOL_EXPLORER_FORCE_REPAINT", "1");
    cmd.spawn().expect("failed to spawn sfm-explorer")
}

/// Owns a launched `sfm-explorer` process and the serialization lock for the
/// test that spawned it. Fields drop in declaration order, so `child` is killed
/// (its window torn down) *before* `_lock` is released and the next test may
/// launch — keeping windows strictly non-overlapping.
///
/// `child` sits behind a `RefCell` because [`attach`] replaces it: a launch
/// that never becomes discoverable is retried once, in place, so the guard
/// still owns (and on drop still kills) whichever process is current.
struct Guard {
    child: RefCell<Child>,
    /// The viewer's command line, kept so a stuck launch can be respawned the
    /// same way. `None` marks a guard whose process cannot simply be
    /// re-spawned — the MCP viewer, whose endpoint line has already been read
    /// off its stdout — and [`ChildHandle::relaunch`] declines to retry it.
    args: Option<Vec<String>>,
    _lock: MutexGuard<'static, ()>,
}

impl Guard {
    /// Acquire the serialization lock, then launch the app under it.
    fn new() -> Self {
        Guard::with_args(&["--no-default-layout"])
    }

    /// The same, with the viewer's command line spelled out.
    fn with_args(args: &[&str]) -> Self {
        Guard::with_args_under(ui_test_lock(), args)
    }

    /// The same again, under a lock the caller already holds.
    ///
    /// A test that has to put something in place *before* the viewer starts,
    /// such as a default layout file in the home directory, has to do that
    /// under the serialization lock too: two tests writing that one file would
    /// otherwise each read the other's. So the lock is taken first, the file is
    /// written, and the lock is handed on to the guard that owns it for the
    /// rest of the test.
    fn with_args_under(lock: MutexGuard<'static, ()>, args: &[&str]) -> Self {
        Guard {
            child: RefCell::new(launch_with(args)),
            args: Some(args.iter().map(|a| (*a).to_string()).collect()),
            _lock: lock,
        }
    }

    fn child(&self) -> ChildHandle<'_> {
        ChildHandle { guard: self }
    }

    /// Wait up to `budget` for the app to exit on its own. Returns whether it
    /// did — `false` means it was still running when the budget ran out.
    fn wait_for_exit(&mut self, budget: Duration) -> bool {
        let deadline = Instant::now() + budget;
        let child = self.child.get_mut();
        loop {
            match child.try_wait() {
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
        let child = self.child.get_mut();
        child.kill().ok();
        child.wait().ok();
    }
}

/// A borrow of the process a [`Guard`] currently owns: its pid, and the one
/// operation [`attach`] needs beyond reading that — replacing it.
#[derive(Clone, Copy)]
struct ChildHandle<'a> {
    guard: &'a Guard,
}

impl ChildHandle<'_> {
    /// The pid of the process the guard owns *now* — re-read after a relaunch.
    fn id(&self) -> u32 {
        self.guard.child.borrow().id()
    }

    /// Kill and reap the current process, then spawn a replacement with the
    /// same command line, leaving the guard owning the new one. Returns
    /// whether a replacement was launched; `false` for a guard that recorded
    /// no command line, whose caller must report the original failure.
    ///
    /// Reaping matters as much as killing: the tests are serialized on
    /// `UI_TEST_LOCK` and match the Windows window by title, so an abandoned
    /// viewer would be found by the next `attach` instead of the fresh one.
    fn relaunch(&self) -> bool {
        let Some(args) = self.guard.args.as_deref() else {
            return false;
        };
        {
            let mut child = self.guard.child.borrow_mut();
            child.kill().ok();
            child.wait().ok();
        }
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        *self.guard.child.borrow_mut() = launch_with(&args);
        true
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

/// Attach to the launched viewer, relaunching it once if it never becomes
/// discoverable, and panicking with both failures if the relaunch is no better.
///
/// The retry is for the runner, not the product: over 400 CI runs, three
/// `ui-test-windows` launches (2026-08-08, 08-09, 08-24) never registered with
/// UI Automation inside the full [`ATTACH_TIMEOUT`], failing as
/// `SelectorNotMatched` from application discovery, or as
/// `Platform { code: -2146233083 }` (HRESULT 0x80131505 out of the automation
/// client) — while every other test in the same job attached normally, and a
/// re-run of the same commit passed. A second process is the cheapest way past a launch
/// the runner lost; one retry only, so a real regression still fails fast and
/// reports what it saw both times.
fn attach(child: ChildHandle<'_>) -> App {
    init();
    let first = match try_attach_app(child) {
        Ok(app) => return app,
        Err(e) => e,
    };
    assert!(
        child.relaunch(),
        "sfm-explorer window did not appear: {first}"
    );
    match try_attach_app(child) {
        Ok(app) => app,
        Err(second) => panic!(
            "sfm-explorer window did not appear, on the original launch or on \
             one relaunch: {first}; after relaunching: {second}"
        ),
    }
}

/// On Windows, xa11y's `by_pid` roots at the first top-level window for the
/// pid (still true in 0.12: on Windows an "app" *is* a top-level window, so one
/// process can own several), which is one of winit's helper windows (a
/// 16px-wide "group") rather than our UI, so locator queries and bounds resolve
/// against the wrong element. Select our window by its title instead.
#[cfg(windows)]
fn try_attach_app(_child: ChildHandle<'_>) -> Result<App, String> {
    App::find(ATTACH_TIMEOUT, |d| {
        d.name.as_deref() == Some("SfM Explorer")
    })
    .map_err(|e| format!("{e:?}"))
}

/// Everywhere else a process has exactly one accessibility root and the pid
/// resolves it directly, so `by_pid` is both correct and cheap — a title-based
/// `find` would just burn the full timeout before any fallback. On macOS that
/// root is the AXApplication, named after the executable rather than the
/// window; on Linux it is the AT-SPI `application` the AccessKit adapter
/// registers for the process. Being pid-addressed also makes the MCP tests'
/// `[MCP :port]` title suffix a non-issue on both.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn try_attach_app(child: ChildHandle<'_>) -> Result<App, String> {
    App::by_pid(child.id(), ATTACH_TIMEOUT).map_err(|e| format!("{e:?}"))
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

/// The menu bar's top-level menus, and the one that is deliberately absent.
///
/// The display controls that made a View menu worth having moved into the
/// viewport HUD — see `specs/gui/viewport-hud.md` — so what is left is what is
/// app-global: what is loaded, what has been done to it, where to go, and which
/// panels are up.
#[test]
fn the_menu_bar_holds_file_edit_go_and_panels() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    for menu in ["File", "Edit", "Go", "Panels"] {
        app.locator(&format!(r#"button[name="{menu}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("'{menu}' menu button not found"));
    }
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

/// The Edit menu opens, and its items are in the tree even when every one of
/// them is greyed.
///
/// With nothing loaded there is no node to undo in and nothing selected to
/// delete or adjust, so every item is disabled -- which is the state this asserts
/// they are nonetheless *present* in: an action that vanishes when it does not
/// apply reads as unimplemented.
///
/// Only the two items with no keyboard shortcut are asserted by name. The
/// others carry their shortcut in the button's text, and the shortcut is
/// spelled by the platform (`Ctrl+Z` against `⌘Z`), so matching them on an
/// exact accessible name would be asserting egui's formatting rather than the
/// menu. What they do, and when they are enabled, is covered headlessly in
/// `state/edits/tests.rs`.
#[test]
fn edit_menu_items() {
    let _guard = Guard::new();
    let app = attach(_guard.child());

    app.locator(r#"button[name="Edit"]"#)
        .press()
        .expect("press Edit menu button");

    // Named without their shortcuts, for the reason `file_menu_items` gives:
    // the shortcut is in the button's text and is spelled by the platform.
    for item in ["Delete Image", "Cancel Camera Move", "Bundle Adjust..."] {
        app.locator(&format!(r#"button[name="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Edit menu item '{item}' did not appear"));
    }
}

/// The File menu's two save items, and which of them applies to demo data.
///
/// Both are matched on a name *prefix*: each carries its keyboard shortcut in
/// the button's text, and the shortcut is spelled by the platform (`Ctrl+S`
/// against `⌘S`), so an exact name would be asserting egui's formatting rather
/// than the menu -- the same reason `edit_menu_items` above names only the one
/// item with no shortcut.
///
/// Demo data came from no file, so there is nothing for Save to write over and
/// Save As is the only way out. Both items are nonetheless present: an action
/// that vanishes when it does not apply reads as unimplemented. Presence is
/// all this asserts: which of the two is enabled is not read here, because the
/// accessibility tree does not report egui's disabled state the same way on
/// every platform (Linux never matched an `enabled="false"` selector), and the
/// enabled logic, like what each item does, is covered headlessly in
/// `state/save/tests.rs`.
#[test]
fn file_menu_save_items_apply_to_a_node_that_came_from_no_file() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    app.locator(r#"button[name="File"]"#)
        .press()
        .expect("press File menu button");

    app.locator(r#"button[name^="Save As..."]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("File menu item 'Save As...' did not appear");
    app.locator(r#"button[name^="Save "]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("File menu item 'Save' did not appear");
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

    for name in ["Points", "Camera Images", "Grid"] {
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
/// its reconstruction row and the Camera Images group beneath it are in the
/// accessibility tree. Everything else about the tree is exercised headlessly
/// in `scene_graph/tests.rs`.
#[test]
fn the_scene_panel_lists_the_loaded_reconstruction() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    // The demo reconstruction is labeled "demo" and rings the scene with 8
    // images, so both strings are fixed by the fixture.
    for text in ["demo", "Camera Images (8)"] {
        app.locator(&format!(r#"static_text[name="{text}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Scene panel row '{text}' did not appear"));
    }

    // The row's solo toggle. Its own behaviour is headless (`scene_graph`
    // tests); what only a real window can show is that a *third* glyph button
    // squeezed onto the row is still laid out and still reachable.
    app.locator(r#"button[name="S"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the reconstruction row's solo toggle did not appear");
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
    // Neither are the two submenus, `Align to ▸` and `Tint ▸`: a menu button
    // does not surface under the `button` role here, and their contents only
    // exist once the submenu is opened — both are covered headlessly instead.
    for item in ["Select", "Zoom to Fit"] {
        app.locator(&format!(r#"button[name="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| {
                panic!("context menu item '{item}' did not appear after a right-click")
            });
    }
}

/// The default layout file, moved aside for the length of a test and put back
/// afterwards.
///
/// The file is a real one in the developer's home directory — the whole point
/// of the feature is that the viewer reads it at startup — so a test that
/// writes one has to give theirs back, whatever the test does.
struct DefaultLayoutFile {
    path: std::path::PathBuf,
    saved: Option<std::path::PathBuf>,
}

impl DefaultLayoutFile {
    /// Take the serialization lock, put `contents` at
    /// `~/.sfm-explorer-default-layout.json` preserving whatever was there, and
    /// hand the lock back for [`Guard::with_args_under`].
    ///
    /// The lock comes first because the file is one file: two tests writing it
    /// at once would each launch a viewer on the other's layout. Nothing here
    /// launches anything, so the caller passes the lock straight on to the
    /// guard that owns the viewer.
    fn written(contents: &str) -> (Self, MutexGuard<'static, ()>) {
        let lock = ui_test_lock();
        #[allow(deprecated)] // Un-deprecated in 1.85, below the workspace MSRV.
        let home = std::env::home_dir().expect("a home directory");
        let path = home.join(".sfm-explorer-default-layout.json");
        let saved = path.exists().then(|| {
            let saved = home.join(".sfm-explorer-default-layout.json.ui-test-backup");
            std::fs::rename(&path, &saved).expect("move the developer's layout aside");
            saved
        });
        std::fs::write(&path, contents).expect("write a default layout file");
        (DefaultLayoutFile { path, saved }, lock)
    }
}

impl Drop for DefaultLayoutFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
        if let Some(saved) = &self.saved {
            std::fs::rename(saved, &self.path).ok();
        }
    }
}

/// A layout saved to the default file comes back at the next start.
///
/// End to end, and only a real window can show it: the file is read in
/// `resumed`, between the window's creation and its first appearance. The
/// layout names the Action Log alone, so the panel's own toolbar — which the
/// stock grid keeps behind the Image Browser and never draws — is the evidence
/// that the file was read.
#[test]
fn a_saved_default_layout_is_loaded_at_startup() {
    let (_file, lock) = DefaultLayoutFile::written(
        r#"{
  "sfm_explorer_layout": 2,
  "layout": {
    "main": {
      "tabs": ["action_log"],
      "active": "action_log"
    },
    "windows": []
  }
}
"#,
    );
    // Launched *without* `--no-default-layout`, unlike every other test here.
    let guard = Guard::with_args_under(lock, &[]);
    let app = attach(guard.child());

    app.locator(r#"button[name="Latest"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the Action Log toolbar did not appear, so the layout was not loaded");
    assert!(
        app.locator(r#"static_text[name="No reconstruction loaded."]"#)
            .wait_attached(Duration::from_millis(500))
            .is_err(),
        "the 3D viewer is still docked, so the stock grid was used"
    );
}

/// The Edit History panel reaches a real window: with a node loaded it lists that
/// node's one version, the file as it was opened.
///
/// The panel is put in front by a default layout file naming it alone, as the
/// startup-load test above puts the Action Log there: the stock grid keeps the
/// Edit History tab behind the Image Browser, and `egui_dock`'s tab bar is painted
/// rather than built out of widgets, so there is no tab in the accessibility
/// tree to press. A layout file is the deterministic way in, since the panel is
/// in front from the viewer's first frame. Everything the panel decides is
/// exercised headlessly in `edit_history_panel/tests.rs`.
#[test]
fn the_edit_history_panel_lists_the_loaded_version() {
    let (_file, lock) = DefaultLayoutFile::written(
        r#"{
  "sfm_explorer_layout": 2,
  "layout": {
    "main": {
      "tabs": ["edit_history"],
      "active": "edit_history"
    },
    "windows": []
  }
}
"#,
    );
    // Launched *without* `--no-default-layout`, so the file above is read.
    let guard = Guard::with_args_under(lock, &[]);
    let app = attach(guard.child());
    load_demo_data(&app);

    // The demo node is labeled "demo" and has been through no edit, so both
    // strings are fixed by the fixture.
    for text in [
        "1 version",
        "demo has not been edited; its one version is the file as it was opened.",
    ] {
        app.locator(&format!(r#"static_text[name="{text}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Edit History panel text '{text}' did not appear"));
    }
}

// --- The MCP screenshot, against a real frame ---
//
// Everything else the MCP surface does is under headless test in
// `mcp::tests`, which is where it belongs: the command vocabulary takes no
// GPU and no window. `screenshot` is the exception — it is a picture of a
// frame that has actually been rendered and presented — so it is here, and
// what these assert is the size and decodability of the PNG rather than its
// pixels, since what the frame *looks* like is not a stable thing to assert.

/// A viewer with its MCP endpoint live, and the address it printed.
struct McpViewer {
    /// Held for its `Drop`, which kills the viewer and releases the
    /// serialization lock, and read for the pid the accessibility root is
    /// found by.
    guard: Guard,
    address: String,
}

impl McpViewer {
    /// Launch a viewer on an ephemeral port and wait for it to say where it is.
    ///
    /// `--mcp 0` rather than a fixed port, because a developer running this
    /// suite very likely has a viewer of their own on 8787 and a port
    /// collision is a fatal startup error by design.
    fn launch() -> McpViewer {
        let _lock = ui_test_lock();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sfm-explorer"));
        cmd.args(["--mcp", "0", "--no-default-layout"]);
        cmd.stdout(std::process::Stdio::piped());
        #[cfg(target_os = "macos")]
        cmd.env("SFMTOOL_EXPLORER_FORCE_REPAINT", "1");
        let mut child = cmd.spawn().expect("failed to spawn sfm-explorer");
        let stdout = child.stdout.take().expect("stdout was piped");
        // Read the endpoint line off a thread: a viewer that died before
        // printing it must fail this test rather than block it forever.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut line = String::new();
            let _ = std::io::BufReader::new(stdout).read_line(&mut line);
            let _ = tx.send(line);
        });
        let line = rx
            .recv_timeout(ATTACH_TIMEOUT)
            .expect("the viewer never printed its MCP endpoint");
        let address = line
            .trim()
            .rsplit_once("http://")
            .and_then(|(_, rest)| rest.strip_suffix("/mcp"))
            .unwrap_or_else(|| panic!("no endpoint in {line:?}"))
            .to_string();
        McpViewer {
            // No recorded command line: this viewer's endpoint was read off
            // the stdout of *this* process, so a respawn would be a viewer on
            // a different port that nothing is listening to.
            guard: Guard {
                child: RefCell::new(child),
                args: None,
                _lock,
            },
            address,
        }
    }

    /// Wait for the window to exist, since MCP commands are applied inside a
    /// frame and a viewer with no window yet renders none.
    ///
    /// Not [`attach`]: while the endpoint is live the title carries an
    /// `[MCP :port]` suffix, which an exact-name match does not find.
    ///
    /// **The pid is part of the match on every platform.** A developer running
    /// this suite very likely has a viewer of their own open, which is the
    /// whole point of the surface being tested; matching on the title alone
    /// would drive theirs while the assertions read this one, and every
    /// interaction would silently land in the wrong window. On Windows the
    /// title still has to come into it, since `by_pid` roots at the first
    /// top-level window for the process and that is one of winit's helper
    /// windows rather than the UI (see [`try_attach_app`]).
    fn wait_for_window(&self) -> App {
        init();
        let pid = self.guard.child().id();
        #[cfg(windows)]
        {
            App::find(ATTACH_TIMEOUT, |d| {
                d.pid == Some(pid)
                    && d.name
                        .as_deref()
                        .is_some_and(|name| name.starts_with("SfM Explorer"))
            })
            .expect("sfm-explorer window did not appear")
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            App::by_pid(pid, ATTACH_TIMEOUT).expect("sfm-explorer did not appear")
        }
    }

    /// POST one JSON-RPC body and return the `result` object.
    ///
    /// Hand-written HTTP/1.1 for the reason `mcp::tests` writes its own: a POST
    /// with a JSON body is a dozen lines, and an HTTP client dev-dependency
    /// would buy nothing this needs.
    fn rpc(&self, body: &str) -> serde_json::Value {
        use std::io::{Read as _, Write as _};
        let mut stream =
            std::net::TcpStream::connect(&self.address).expect("the endpoint is listening");
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .expect("a read timeout is settable");
        let request = format!(
            "POST /mcp HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            self.address,
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .expect("the request is writable");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .expect("the response is readable");
        let response = String::from_utf8_lossy(&response).into_owned();
        let json = response
            .lines()
            .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
            .find(|line| line.starts_with('{'))
            .unwrap_or_else(|| panic!("no JSON in {response:?}"));
        let parsed: serde_json::Value =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("{e} in {json:?}"));
        assert_eq!(parsed["error"], serde_json::Value::Null, "{parsed}");
        parsed["result"].clone()
    }

    /// Complete the handshake, which a client does once before anything else.
    fn initialize(&self) {
        self.rpc(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"ui-test","version":"0"}}}"#,
        );
    }

    /// Call one tool and return its result.
    fn call(&self, name: &str, arguments: serde_json::Value) -> serde_json::Value {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": name, "arguments": arguments },
        });
        self.rpc(&body.to_string())
    }

    /// Call `screenshot` and decode the PNG it handed back.
    fn screenshot(&self, arguments: serde_json::Value) -> image::RgbaImage {
        use base64::Engine as _;
        let result = self.call("screenshot", arguments.clone());
        assert_ne!(
            result["isError"],
            serde_json::Value::Bool(true),
            "{arguments}: {result}"
        );
        let encoded = result["content"]
            .as_array()
            .expect("a content array")
            .iter()
            .find_map(|block| block["data"].as_str())
            .unwrap_or_else(|| panic!("no image block in {result}"));
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("the image block is base64");
        image::load_from_memory(&bytes)
            .expect("the image block is a decodable PNG")
            .to_rgba8()
    }

    /// The window's drawable area in physical pixels, as the viewer reports it.
    fn inner_size(&self) -> [u32; 2] {
        let layout = self.call("get_window_layout", serde_json::json!({}));
        let size = &layout["structuredContent"]["window"]["inner_size"];
        [
            size[0].as_u64().expect("a width") as u32,
            size[1].as_u64().expect("a height") as u32,
        ]
    }
}

/// The default screenshot is the window itself, read back off the presented
/// surface — which is what `COPY_SRC` on the swapchain buys.
#[test]
fn a_screenshot_is_the_whole_window() {
    let viewer = McpViewer::launch();
    // The window has to exist before it can be photographed.
    viewer.wait_for_window();
    viewer.initialize();

    let [width, height] = viewer.inner_size();
    let window = viewer.screenshot(serde_json::json!({}));
    assert_eq!(
        (window.width(), window.height()),
        (width, height),
        "the picture is not the window's drawable area"
    );

    // A panel is a crop of that same frame, so it is smaller in both axes.
    let scene = viewer.screenshot(serde_json::json!({ "panel_name": "scene" }));
    assert!(
        scene.width() < window.width() && scene.height() < window.height(),
        "the Scene panel's crop ({} × {}) is not inside the window ({} × {})",
        scene.width(),
        scene.height(),
        window.width(),
        window.height()
    );

    // `max_dimension` bounds the longer side, after the crop.
    let bounded = viewer.screenshot(serde_json::json!({ "max_dimension": 320 }));
    assert_eq!(bounded.width().max(bounded.height()), 320);
}

/// The two pictures of the 3D viewport: the crop of the presented frame, with
/// the HUD over it, and the render target it was drawn from.
///
/// They are the same view and very nearly the same size — the crop is the tab
/// *body*, which egui_dock insets by its own margin before the viewport
/// allocates what is left — so what this asserts is that the render is inside
/// the crop and close to it, not that the two are identical.
#[test]
fn the_viewport_can_be_photographed_with_and_without_its_hud() {
    let viewer = McpViewer::launch();
    let app = viewer.wait_for_window();
    viewer.initialize();
    // The render target only exists once the viewport has something to draw:
    // with nothing loaded the panel shows its empty state and never sizes one.
    load_demo_data(&app);

    let with_hud = viewer.screenshot(serde_json::json!({ "panel_name": "viewer_3d" }));
    let without_hud =
        viewer.screenshot(serde_json::json!({ "panel_name": "viewer_3d", "hud": false }));
    assert!(
        without_hud.width() <= with_hud.width() && without_hud.height() <= with_hud.height(),
        "the render ({} × {}) is not inside its panel's body ({} × {})",
        without_hud.width(),
        without_hud.height(),
        with_hud.width(),
        with_hud.height()
    );
    assert!(
        with_hud.width() - without_hud.width() < 64
            && with_hud.height() - without_hud.height() < 64,
        "the two pictures of the viewport are further apart than a body margin"
    );
}

/// A panel that is not drawn is refused rather than photographed, and the
/// refusal names the call that fixes it.
#[test]
fn a_panel_that_is_not_drawn_is_refused_by_a_real_viewer() {
    let viewer = McpViewer::launch();
    viewer.wait_for_window();
    viewer.initialize();

    // Behind Image Detail in the stock grid.
    let behind = viewer.call(
        "screenshot",
        serde_json::json!({ "panel_name": "point_track" }),
    );
    assert_eq!(behind["isError"], serde_json::Value::Bool(true), "{behind}");
    let message = behind["content"][0]["text"].as_str().expect("a refusal");
    assert!(
        message.contains("Image Detail") && message.contains("show_panel"),
        "{message}"
    );

    // Closed.
    viewer.call(
        "hide_panel",
        serde_json::json!({ "panel_name": "action_log" }),
    );
    let closed = viewer.call(
        "screenshot",
        serde_json::json!({ "panel_name": "action_log" }),
    );
    assert_eq!(closed["isError"], serde_json::Value::Bool(true), "{closed}");
    let message = closed["content"][0]["text"].as_str().expect("a refusal");
    assert!(
        message.contains("closed") && message.contains("show_panel"),
        "{message}"
    );
}

/// The editing surface against a real viewer: an edit over the wire, the
/// version it made, the undo that takes it back, and the Action Log the human
/// beside the window is reading.
///
/// Here rather than only in `mcp::tests` for the one thing the headless tests
/// cannot show: that an edit applied inside a real frame reaches the window the
/// human is looking at, attributed to the agent, and that the history and the
/// log both know about it afterwards. What each edit *does* to a reconstruction
/// is asserted headlessly, over the same `AppState` calls.
#[test]
fn a_point_can_be_deleted_over_the_wire_and_undone() {
    let viewer = McpViewer::launch();
    let app = viewer.wait_for_window();
    viewer.initialize();
    load_demo_data(&app);
    // The node arrives in a frame of its own, after the button press returns,
    // so the first call waits for it rather than racing it.
    let deadline = std::time::Instant::now() + CONTENT_TIMEOUT;
    let label = loop {
        let scene = viewer.call("get_scene", serde_json::json!({}));
        if let Some(label) = scene["structuredContent"]["scene"][0]["label"].as_str() {
            break label.to_string();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the demo node never appeared: {scene}"
        );
        std::thread::sleep(Duration::from_millis(100));
    };

    let deleted = viewer.call(
        "delete_point",
        serde_json::json!({ "reconstruction_label": label, "point": 3 }),
    );
    assert_ne!(
        deleted["isError"],
        serde_json::Value::Bool(true),
        "{deleted}"
    );
    let made = deleted["structuredContent"].clone();
    assert_eq!(made["label"], format!("Deleted point 3 in {label}"));
    let serial = made["serial"]
        .as_str()
        .expect("a version serial")
        .to_string();

    let history = viewer.call(
        "get_history",
        serde_json::json!({ "reconstruction_label": label }),
    )["structuredContent"]
        .clone();
    let versions = history["versions"].as_array().expect("a version list");
    assert_eq!(versions.len(), 2, "{history}");
    assert_eq!(history["cursor"], serde_json::Value::String(serial.clone()));
    assert_eq!(history["dirty"], true);
    assert_eq!(history["can_undo"], true);
    // The node came from no file, so nothing of it is on disk.
    assert_eq!(history["disk_serial"], serde_json::Value::Null);

    let undone = viewer.call("undo", serde_json::json!({ "reconstruction_label": label }))
        ["structuredContent"]
        .clone();
    assert_eq!(undone["cursor"], versions[0]["serial"]);
    assert_eq!(undone["dirty"], false);

    // Both rows are the agent's, in the words the human's own Edit menu would
    // have written.
    let log = viewer.call("get_action_log", serde_json::json!({ "actors": ["mcp"] }))
        ["structuredContent"]
        .clone();
    let texts: Vec<String> = log["entries"]
        .as_array()
        .expect("a list of entries")
        .iter()
        .map(|entry| {
            assert_eq!(entry["actor"], "mcp", "{entry}");
            entry["text"].as_str().unwrap_or_default().to_string()
        })
        .collect();
    assert!(
        texts
            .iter()
            .any(|text| text.starts_with(&format!("Deleted point 3 in {label}"))),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|text| text.starts_with("Undo: ")),
        "{texts:?}"
    );
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

/// Taking a camera in hand, in a real window: the Edit menu and the lock's
/// release.
///
/// The one part of the Move Camera family a headless frame cannot reach. Every
/// decision it makes -- the snap, the pending pose, the dead band, the commit --
/// is asserted in `camera_lock/tests.rs`; what this asks is whether the menu
/// entry and the release reach a live viewport at all. The banner is painted
/// rather than built of widgets, so it is not in the accessibility tree and is
/// covered headlessly instead.
///
/// The pose is deliberately not moved here. Synthetic mouse input on Windows
/// arrives through `WM_POINTER` (`EnableMouseInPointer`, see the right-click
/// test), and winit renders a moving contact as a touch rather than a drag, so
/// an injected drag moves no camera in this app -- with a lock held or without
/// one. What a real window can be asked, and is asked below, is whether the
/// menu takes the camera in hand and gives it back.
///
/// Windows-only for the reason the right-click test is: injecting real input is
/// platform code, and one platform proves the wiring.
#[cfg(windows)]
#[test]
fn taking_a_camera_in_hand_reaches_a_real_window() {
    let viewer = McpViewer::launch();
    let app = viewer.wait_for_window();
    viewer.initialize();
    load_demo_data(&app);

    // The 3D viewer alone in the dock, so nothing but the viewport is under
    // the menu the test drives.
    viewer.call(
        "set_window_layout",
        serde_json::json!({
            "layout": { "main": { "tabs": ["viewer_3d"], "active": "viewer_3d" }, "windows": [] },
        }),
    );
    viewer.call(
        "set_view",
        serde_json::json!({ "look_through": { "camera_image": 0 } }),
    );

    // The menu button toggles, so one opening is one place to look: reopening
    // it to find a second item would close it instead.
    let open_edit_menu = || {
        app.locator(r#"button[name="Edit"]"#)
            .press()
            .expect("press Edit menu button");
    };
    // Matched on a name *prefix*, for the reason `edit_menu_items` gives: both
    // spellings of the Move Camera item carry the `M` shortcut in the button's
    // text, and the shortcut is spelled by the platform.
    let edit_item = |item: &str| {
        app.locator(&format!(r#"button[name^="{item}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Edit menu item '{item}' did not appear"))
    };
    let log_texts = || -> Vec<String> {
        let log = viewer.call(
            "get_action_log",
            serde_json::json!({ "since_revision": 0, "limit": 200 }),
        );
        log["structuredContent"]["entries"]
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|e| e["text"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let wait_for_line = |prefix: &str| {
        let deadline = Instant::now() + CONTENT_TIMEOUT;
        loop {
            let texts = log_texts();
            if texts.iter().any(|text| text.starts_with(prefix)) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "no '{prefix}' line; the log holds {texts:?}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    };

    open_edit_menu();
    edit_item("Move Camera")
        .press()
        .expect("press Edit menu item 'Move Camera'");
    wait_for_line("Moving the camera of image_000.jpg");

    // One opening for both: with a lock held the same entry commits rather
    // than takes, and the entry beside it gives the camera back.
    open_edit_menu();
    edit_item("Commit Camera Move");
    edit_item("Cancel Camera Move")
        .press()
        .expect("press Edit menu item 'Cancel Camera Move'");
    wait_for_line("Cancelled the camera move of image_000.jpg");
}
