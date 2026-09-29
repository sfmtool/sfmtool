// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos", target_os = "linux"))]

//! The viewer's windowed tests: what only a real window, on a real desktop,
//! can be asked.
//!
//! **The tests read the window through the viewer's own MCP endpoint.** A
//! viewer launched with `--mcp 0` answers `get_widgets` with every widget egui
//! drew in a frame (role, name, rectangle, enabled and toggled state) and the
//! dialogs and menus drawn above the dock, and `click` presses a widget the way
//! a person's mouse would. That is the same information the platform's
//! accessibility tree carries, read in the viewer's process instead of across
//! the operating system's accessibility bridge, which is the expensive part: on
//! a GitHub-hosted Windows runner one walk of the viewer's tree through that
//! bridge took about 18.6s. See `specs/gui/architecture.md` § "Testing" for how
//! the suite reads the window, and `specs/gui/mcp-server.md` § "`get_widgets`",
//! § "`click` / `hover`" and § "What synthetic input does not exercise" for the
//! tools.
//!
//! **One smoke test reads the tree through the platform, on Windows only**:
//! `window_appears` asks UI Automation for the viewer's window subtree and
//! checks that the menu bar's buttons are in it. What it checks is outside the
//! viewer's process, so MCP cannot show it. Its doc says why it runs on Windows
//! alone.
//!
//! **One test sends real OS input**:
//! `a_real_right_click_opens_the_reconstruction_rows_context_menu` (Windows
//! only) presses the right mouse button with `SendInput` and checks that it
//! reaches egui. A synthetic `click` over MCP goes into egui's input directly
//! and cannot see a defect in how the operating system's input gets there. It
//! finds where to press, and reads the menu that opened, over MCP.
//!
//! The MCP tests wait on the endpoint, polling `get_widgets` until the window's
//! menu bar is listed (see [`McpViewer::launched`]), and anything else that has
//! to wait for state polls an MCP read with a deadline. ([`dump_widgets`],
//! `#[ignore]`d, prints the whole window's listing for a person debugging one.)
//!
//! Setup goes through the **command line** where it can: `--demo` in place of
//! driving File > Load Demo Data… and its dialog.
//! [`the_scene_panel_lists_the_loaded_reconstruction`] is the one test that
//! drives that route, and says so.
//!
//! **The suite reports its own cost**: every test prints a `UIPROBE` line as
//! its [`Guard`] drops, which is how a reader of one CI log tells a slow runner
//! apart from an expensive suite. See [`Guard::report`] for the fields.

use std::cell::Cell;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// Serializes the UI tests so at most one `sfm-explorer` window is alive at a
/// time. `cargo test` runs tests on multiple threads by default, and several
/// viewers at once compete for one desktop and one GPU. Two tests also share
/// one on-disk file, the default layout (see [`DefaultLayoutFile`]), and the
/// Windows-only right-click test drives the real cursor, which belongs to
/// whichever window is in front. Each test holds this lock for its whole body,
/// so a plain `cargo test` behaves the same as `--test-threads=1` without the
/// caller having to remember the flag.
static UI_TEST_LOCK: Mutex<()> = Mutex::new(());

fn ui_test_lock() -> MutexGuard<'static, ()> {
    // A panicking test (every assertion failure here panics) would otherwise
    // poison the mutex and turn every later test into a spurious failure;
    // recover the guard instead so the suite keeps running serially.
    UI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// --- What the suite costs, as the suite sees it ---
//
// Costs with different causes are added together in a job's wall clock:
// launching a viewer (process spawn, GPU init, the first frame), querying the
// platform's accessibility tree (a cross-process snapshot of the window's
// subtree), and the MCP calls the tests make. A log that reports only the total
// cannot tell them apart, nor tell a change that made the *suite* cheaper from
// a run that happened to land on a faster machine. So each is counted and
// printed separately; see `Guard::report`.

/// Waits on the accessibility tree run since the current [`Guard`] was made.
static OPS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds spent inside those waits.
static OP_NANOS: AtomicU64 = AtomicU64::new(0);
/// UI Automation subtree queries run since then, and the nanoseconds spent
/// inside them — the *platform's* share of `OP_NANOS`. See `walked`.
static WALKS: AtomicU64 = AtomicU64::new(0);
static WALK_NANOS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds spent getting a UI Automation element for the viewer's window,
/// which is neither of the above and is reported on its own. See `UiaWindow`.
static WINDOW_NANOS: AtomicU64 = AtomicU64::new(0);
/// MCP tool calls made since then, polls included, and the nanoseconds spent
/// waiting for their replies. See [`McpViewer::call`].
static CALLS: AtomicU64 = AtomicU64::new(0);
static CALL_NANOS: AtomicU64 = AtomicU64::new(0);
/// The same, plus launch time and wall time, accumulated over every test this
/// process has run, for the `UIPROBE TOTAL` line.
static TOTAL_TESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LAUNCH_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OPS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OP_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WALKS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WALK_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WINDOW_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_CALL_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_NANOS: AtomicU64 = AtomicU64::new(0);

/// Zero the per-test counters and start the clock a `UIPROBE` line is measured
/// from. Every path that makes a [`Guard`] calls this immediately before
/// spawning the viewer, and nothing else does.
///
/// **Process-wide counters are sound here only because of [`UI_TEST_LOCK`].**
/// The instrumented calls go through UI Automation and the MCP endpoint, which
/// know nothing about the guard, so the counters cannot hang off one — but
/// every test holds that lock for its whole body, so exactly one guard is ever
/// alive and "the operations since the last reset" and "this test's
/// operations" are the same set.
fn begin_accounting() -> Instant {
    for counter in [
        &OPS,
        &OP_NANOS,
        &WALKS,
        &WALK_NANOS,
        &WINDOW_NANOS,
        &CALLS,
        &CALL_NANOS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    Instant::now()
}

/// Run one wait on the accessibility tree, counting it and timing it.
///
/// **One call here is one `op`, however many subtree queries happen inside
/// it.** That keeps `ops` a deterministic fingerprint of suite *shape* rather
/// than of how a particular run went: a run that had to poll three times
/// reports the same `ops` as one that did not, and pays for it in `op_ms`
/// where the time actually went.
#[cfg(windows)]
fn measured<T>(op: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = op();
    OPS.fetch_add(1, Ordering::Relaxed);
    OP_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

/// Run one UI Automation subtree query, counting it and timing it.
///
/// **A different counter from `measured`, asking a different question.** An
/// `op` is what a *test* asked for; a `walk` is what the *platform* was asked
/// to do to service it, and moves with the run. One op is one or many walks —
/// a wait that polls three times for the menu bar walks three times — so
/// `walks` is never a second spelling of `ops`.
///
/// Bracketing only the query, never the `sleep` between two of them, makes
/// `walk_ms` the platform's share and `op_ms − walk_ms` the suite's own
/// waiting — the difference between a tree query that is expensive and an app
/// that is slow to draw.
#[cfg(windows)]
fn walked<T>(query: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = query();
    WALKS.fetch_add(1, Ordering::Relaxed);
    WALK_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

/// The viewer's command line with the given arguments, not yet spawned.
///
/// Every test but the two that start on a saved layout passes
/// `--no-default-layout`: a developer who has saved a layout of their own to
/// `~/.sfm-explorer-default-layout.json` must not have this suite's panel
/// assertions fail on their machine.
fn command(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sfm-explorer"));
    cmd.args(args);
    cmd
}

/// Owns a launched `sfm-explorer` process, anything the test put on disk
/// outside its own directory, and the serialization lock for the test that
/// spawned it.
///
/// Fields drop in declaration order, and the order is the point: `child` is
/// killed (its window torn down), then `_layout_file` is put back, and only
/// then is `_lock` released and the next test free to launch. Both of the
/// things before the lock are shared with every other test — one desktop, and
/// one path in the developer's home directory — so releasing the lock while
/// either is still in flight hands the next test a machine that is not yet the
/// one it asked for.
struct Guard {
    /// When the viewer was spawned, and how long it took to become usable —
    /// the two halves of the `UIPROBE` line's `launch_ms`, filled in by
    /// [`McpViewer::launched`] or by `window_appears`. Neither owns anything,
    /// so neither takes part in the drop order described above.
    started: Instant,
    launch: Cell<Option<Duration>>,
    child: Child,
    /// The default layout file this test wrote, for the two tests that start
    /// the viewer on one. Held here so that it is restored *under* the lock:
    /// restored after the lock is released, it would race the next test's own
    /// `rename` of that one path, which fails with "Access is denied".
    _layout_file: Option<DefaultLayoutFile>,
    _lock: MutexGuard<'static, ()>,
}

impl Guard {
    /// Acquire the serialization lock, then launch the viewer under it with no
    /// MCP endpoint, for the smoke test that reads it through the platform.
    #[cfg(windows)]
    fn new() -> Self {
        let lock = ui_test_lock();
        let started = begin_accounting();
        Guard {
            started,
            launch: Cell::new(None),
            child: command(&["--no-default-layout"])
                .spawn()
                .expect("failed to spawn sfm-explorer"),
            _layout_file: None,
            _lock: lock,
        }
    }

    /// The viewer's process id.
    #[cfg(windows)]
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Print this test's `UIPROBE` line, and the running `UIPROBE TOTAL`.
    ///
    /// ```text
    /// UIPROBE test=file_menu_items launch_ms=369 window_ms=0 ops=0 op_ms=0 walks=0 walk_ms=0 calls=2 call_ms=12 total_ms=431
    /// UIPROBE TOTAL tests=20 launch_ms=10438 window_ms=16 ops=1 op_ms=35 walks=2 walk_ms=67 calls=54 call_ms=660 total_ms=13672 mean_launch_ms=521 mean_op_ms=35 mean_walk_ms=33 mean_call_ms=12
    /// ```
    ///
    /// `launch_ms` is the runner-speed yardstick: spawning a process, waiting
    /// for the GPU and for the viewer to become usable is work the suite cannot
    /// make cheaper, so `mean_launch_ms` moving between two runs means the
    /// *machine* moved. For the smoke test "usable" is the viewer's window
    /// becoming visible, which it does only once its UI Automation provider is
    /// registered; for an MCP test it is `get_widgets` listing the menu bar.
    ///
    /// `window_ms`, `ops`, `op_ms`, `walks` and `walk_ms` are the
    /// accessibility bridge's share, and are zero for every test but the
    /// Windows smoke test. `window_ms` is getting a UI Automation element for
    /// the viewer's window (see `UiaWindow::of`). `ops` is how many *waits* on
    /// the tree the tests made, which only a change to the tests moves; `walks`
    /// is how many UI Automation subtree queries serviced them, and `walk_ms`
    /// the time inside those calls — so `walk_ms` near `op_ms` says the
    /// platform's query is what costs, and a gap says the suite was waiting for
    /// the app to draw. `mean_walk_ms` is the price of one tree query, which
    /// against a tree size (see `window_appears`, which prints one) gives a
    /// per-node figure. See `walked`.
    ///
    /// `calls` and `call_ms` are the MCP tool calls a test made after the
    /// launch, polls included, and the time spent waiting for their replies.
    /// Each reply comes after at least one frame the viewer drew, so
    /// `mean_call_ms` is close to the price of a few frames.
    ///
    /// libtest offers no end-of-suite hook, so the `TOTAL` line is cumulative
    /// and re-printed after every test; the last one is the run's. That also
    /// means a run that aborts halfway still reports what it spent.
    ///
    /// Emitted from `Drop`, so a **panicking** test — every assertion failure
    /// here panics — reports too, which is when the numbers are most wanted.
    /// The test's name comes from its thread, which libtest names after it
    /// even under `--test-threads=1`. A guard whose viewer never became usable
    /// reports `launch_ms=0`.
    ///
    /// None of this reaches a green CI log without `--nocapture`: libtest
    /// captures a passing test's stdout and discards it. Both invocations of
    /// this suite pass the flag — the `ui-test` task in `pixi.toml` and its
    /// Linux override, which every `ui-test-*` CI job runs.
    fn report(&self) {
        let total = self.started.elapsed();
        let launch = self.launch.get().unwrap_or_default();
        let ops = OPS.load(Ordering::Relaxed);
        let op_nanos = OP_NANOS.load(Ordering::Relaxed);
        let walks = WALKS.load(Ordering::Relaxed);
        let walk_nanos = WALK_NANOS.load(Ordering::Relaxed);
        let window_nanos = WINDOW_NANOS.load(Ordering::Relaxed);
        let calls = CALLS.load(Ordering::Relaxed);
        let call_nanos = CALL_NANOS.load(Ordering::Relaxed);

        let add =
            |counter: &AtomicU64, value: u64| counter.fetch_add(value, Ordering::Relaxed) + value;
        let tests = add(&TOTAL_TESTS, 1);
        let launch_total = add(&TOTAL_LAUNCH_NANOS, launch.as_nanos() as u64);
        let ops_total = add(&TOTAL_OPS, ops);
        let op_nanos_total = add(&TOTAL_OP_NANOS, op_nanos);
        let walks_total = add(&TOTAL_WALKS, walks);
        let walk_nanos_total = add(&TOTAL_WALK_NANOS, walk_nanos);
        let window_nanos_total = add(&TOTAL_WINDOW_NANOS, window_nanos);
        let calls_total = add(&TOTAL_CALLS, calls);
        let call_nanos_total = add(&TOTAL_CALL_NANOS, call_nanos);
        let nanos_total = add(&TOTAL_NANOS, total.as_nanos() as u64);

        let ms = |nanos: u64| nanos / 1_000_000;
        let name = std::thread::current()
            .name()
            .unwrap_or("<unnamed>")
            .to_string();
        // Leading newline: under `--nocapture` libtest has already written
        // `test <name> ... ` and is waiting to finish that line with the
        // verdict, so without it both lines below start mid-line and `^UIPROBE`
        // matches only the `TOTAL` one.
        println!(
            "\nUIPROBE test={name} launch_ms={} window_ms={} ops={ops} op_ms={} walks={walks} \
             walk_ms={} calls={calls} call_ms={} total_ms={}",
            launch.as_millis(),
            ms(window_nanos),
            ms(op_nanos),
            ms(walk_nanos),
            ms(call_nanos),
            total.as_millis(),
        );
        // `checked_div`: a suite filtered down to tests that make no call of
        // one kind reports no mean for it rather than dividing by zero.
        println!(
            "UIPROBE TOTAL tests={tests} launch_ms={} window_ms={} ops={ops_total} op_ms={} \
             walks={walks_total} walk_ms={} calls={calls_total} call_ms={} total_ms={} \
             mean_launch_ms={} mean_op_ms={} mean_walk_ms={} mean_call_ms={}",
            ms(launch_total),
            ms(window_nanos_total),
            ms(op_nanos_total),
            ms(walk_nanos_total),
            ms(call_nanos_total),
            ms(nanos_total),
            ms(launch_total) / tests,
            ms(op_nanos_total).checked_div(ops_total).unwrap_or(0),
            ms(walk_nanos_total).checked_div(walks_total).unwrap_or(0),
            ms(call_nanos_total).checked_div(calls_total).unwrap_or(0),
        );
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
        // Last, so `total_ms` covers the teardown the test also pays for.
        self.report();
    }
}

// Generous timeout: the first launch on a cold CI runner pays wgpu
// adapter/shader init (and, on Windows, AV scanning of the fresh binary),
// which can exceed 15s. Healthy launches are ready in ~1s.
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Budget for a widget to appear in, or change within, a listing or the
/// accessibility tree. Polls return as soon as the condition holds, so healthy
/// cases stay fast; only a genuine failure waits the full budget.
///
/// More than a nominal budget: a lookup straight after loading the demo scene
/// pays the scene renderer's wgpu pipeline init, which on a cold Windows CI
/// runner can take more than 5s.
const CONTENT_TIMEOUT: Duration = Duration::from_secs(30);

// --- The smoke test: the accessibility tree, read through UI Automation ---

/// The viewer's window, as UI Automation sees it, and what a subtree query
/// against it needs.
///
/// The query is rooted at the window's own element, from `ElementFromHandle`,
/// so a `FindAllBuildCache(TreeScope_Subtree)` fetches the window and every
/// node beneath it in one cross-process call, with the two properties the test
/// reads (name and control type) cached on each node. The viewer runs a single
/// egui viewport, so its menus and popups are inside that one window.
#[cfg(windows)]
struct UiaWindow {
    root: windows::Win32::UI::Accessibility::IUIAutomationElement,
    everything: windows::Win32::UI::Accessibility::IUIAutomationCondition,
    request: windows::Win32::UI::Accessibility::IUIAutomationCacheRequest,
}

/// One node of a subtree query: its UI Automation control type and its name.
#[cfg(windows)]
type UiaNode = (
    windows::Win32::UI::Accessibility::UIA_CONTROLTYPE_ID,
    String,
);

#[cfg(windows)]
impl UiaWindow {
    /// The UI Automation element for `hwnd`, with a true condition and a cache
    /// request for name and control type ready for [`Self::walk`].
    ///
    /// Joins this thread to the multithreaded apartment, which is what UI
    /// Automation recommends for a client. libtest runs each test on a thread
    /// of its own, so the thread has no apartment yet; `S_FALSE`, for a thread
    /// already in the MTA, is a success too.
    fn of(hwnd: windows::Win32::Foundation::HWND) -> windows::core::Result<UiaWindow> {
        use windows::Win32::System::Com::{
            CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
        };
        use windows::Win32::UI::Accessibility::{
            CUIAutomation, IUIAutomation, UIA_ControlTypePropertyId, UIA_NamePropertyId,
        };
        let started = Instant::now();
        let window = unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
            let automation: IUIAutomation =
                CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)?;
            let request = automation.CreateCacheRequest()?;
            request.AddProperty(UIA_NamePropertyId)?;
            request.AddProperty(UIA_ControlTypePropertyId)?;
            UiaWindow {
                root: automation.ElementFromHandle(hwnd)?,
                everything: automation.CreateTrueCondition()?,
                request,
            }
        };
        WINDOW_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(window)
    }

    /// Every node in the window's subtree, the window itself included. One
    /// cross-process call, counted as one walk.
    fn walk(&self) -> windows::core::Result<Vec<UiaNode>> {
        use windows::Win32::UI::Accessibility::TreeScope_Subtree;
        walked(|| unsafe {
            let found =
                self.root
                    .FindAllBuildCache(TreeScope_Subtree, &self.everything, &self.request)?;
            (0..found.Length()?)
                .map(|i| {
                    let node = found.GetElement(i)?;
                    Ok((node.CachedControlType()?, node.CachedName()?.to_string()))
                })
                .collect()
        })
    }
}

/// Whether `nodes` holds a button named exactly `name`.
#[cfg(windows)]
fn has_button(nodes: &[UiaNode], name: &str) -> bool {
    use windows::Win32::UI::Accessibility::UIA_ButtonControlTypeId;
    nodes
        .iter()
        .any(|(kind, text)| *kind == UIA_ButtonControlTypeId && text == name)
}

/// The viewer's accessibility tree reaches UI Automation, with content in it.
///
/// The one test that checks the platform can read the viewer at all, which is
/// what a screen reader needs and what nothing inside the process can see. It
/// waits for the viewer's window, then polls a UI Automation subtree query of
/// that window until the menu bar's four buttons are in one snapshot, so a tree
/// that is published but empty fails here; the same snapshot must hold no
/// `View` button.
///
/// **Windows only, because the viewer's own accessibility logic is.** The
/// viewer creates its window hidden and shows it only after AccessKit has
/// registered its UI Automation provider (`sfm_explorer::run`), and that
/// ordering is what this checks. On macOS and Linux the AccessKit adapters are
/// egui-winit's, with no code of the viewer's around them. On every platform
/// the MCP tests already show that the tree has content, because
/// `get_widgets` reads the same AccessKit update the adapters publish.
#[cfg(windows)]
#[test]
fn window_appears() {
    let guard = Guard::new();
    let pid = guard.pid();

    // The window becomes visible only once the UI Automation provider is
    // registered, so that is where the launch ends.
    let deadline = Instant::now() + LAUNCH_TIMEOUT;
    let hwnd = loop {
        if let Some(&hwnd) = top_level_windows(pid).first() {
            break hwnd;
        }
        assert!(
            Instant::now() < deadline,
            "sfm-explorer never showed a window"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    guard.launch.set(Some(guard.started.elapsed()));

    let window = UiaWindow::of(hwnd)
        .unwrap_or_else(|e| panic!("no UI Automation element for the viewer's window: {e}"));
    let menu_bar = ["File", "Edit", "Go", "Panels"];
    let snapshot = measured(|| {
        let deadline = Instant::now() + CONTENT_TIMEOUT;
        loop {
            let last = match window.walk() {
                Ok(nodes) => {
                    let missing: Vec<&str> = menu_bar
                        .iter()
                        .copied()
                        .filter(|name| !has_button(&nodes, name))
                        .collect();
                    if missing.is_empty() {
                        break nodes;
                    }
                    format!("no {missing:?} button among {} nodes", nodes.len())
                }
                Err(e) => format!("the subtree query failed: {e}"),
            };
            assert!(
                Instant::now() < deadline,
                "the menu bar's buttons did not reach UI Automation: {last}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    assert!(
        !has_button(&snapshot, "View"),
        "UI Automation lists a View button in the menu bar"
    );
    report_tree_size(&window);
}

/// Print the `UIPROBE TREE` line: how big the tree a walk crosses actually is.
///
/// ```text
/// UIPROBE TREE nodes=52 walk_ms=31
/// ```
///
/// `mean_walk_ms` says what one tree query costs and says nothing about why,
/// because the two candidate whys — a tree with a lot of nodes in it, and a
/// platform that is slow per node — are indistinguishable without the
/// denominator. This is the denominator: `walk_ms / nodes` is the per-node
/// price.
///
/// Taken after `window_appears` has seen the menu bar, so the count is of a
/// published tree, and against the viewer's empty state. It is one more
/// [`UiaWindow::walk`], so it prices the same work the wait did, and counts the
/// window itself among the nodes.
#[cfg(windows)]
fn report_tree_size(window: &UiaWindow) {
    let started = Instant::now();
    let nodes = window.walk().map(|found| found.len());
    let walk = started.elapsed();
    match nodes {
        Ok(nodes) => println!("\nUIPROBE TREE nodes={nodes} walk_ms={}", walk.as_millis()),
        // Not an assertion: this line is a measurement the suite reports, and
        // a test named for whether the window appears must not start failing
        // over a second query of the tree inside it.
        Err(e) => println!(
            "\nUIPROBE TREE unavailable: {e} walk_ms={}",
            walk.as_millis()
        ),
    }
}

// --- Real OS input, aimed and read back over MCP (Windows) ---

/// Every visible, titled top-level window belonging to `pid`.
///
/// winit keeps helper windows of its own beside the UI — invisible and
/// untitled — so the filter is what picks out the window the human sees.
#[cfg(windows)]
fn top_level_windows(pid: u32) -> Vec<windows::Win32::Foundation::HWND> {
    use windows::core::BOOL;
    use windows::Win32::Foundation::{HWND, LPARAM, TRUE};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowTextLengthW, GetWindowThreadProcessId, IsWindowVisible,
    };

    struct Wanted {
        pid: u32,
        found: Vec<HWND>,
    }

    unsafe extern "system" fn visit(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // Safe: `EnumWindows` below passes a pointer to a live `Wanted`, and
        // the enumeration finishes before that borrow ends.
        let wanted = unsafe { &mut *(lparam.0 as *mut Wanted) };
        let mut owner = 0u32;
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut owner)) };
        if owner == wanted.pid
            && unsafe { IsWindowVisible(hwnd) }.as_bool()
            && unsafe { GetWindowTextLengthW(hwnd) } > 0
        {
            wanted.found.push(hwnd);
        }
        TRUE
    }

    let mut wanted = Wanted {
        pid,
        found: Vec::new(),
    };
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut wanted as isize)) };
    wanted.found
}

/// Put the cursor on `(x, y)` and establish that a button pressed there will
/// reach the viewer, or panic saying what is in the way.
///
/// `SendInput` presses wherever the cursor is, on whatever window is under it,
/// and neither of those belongs to this process. The viewer has just launched,
/// something else on the desktop can be in front of it, the point can be
/// off-screen, and `SetCursorPos` can be clamped or refused outright — and from
/// inside the test all of those look the same as a broken context menu: the
/// widget lookup afterwards burns its full [`CONTENT_TIMEOUT`] and fails as if
/// the product had regressed.
///
/// So the aim is established before anything is pressed, and it is the *aim*
/// that is retried, never the assertion: a click that lands on the row and
/// opens no menu still fails, immediately and for the right reason.
#[cfg(windows)]
fn aim_at(pid: u32, x: i32, y: i32) {
    use windows::Win32::Foundation::POINT;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetAncestor, GetCursorPos, GetWindowThreadProcessId, SetCursorPos, SetForegroundWindow,
        SetWindowPos, WindowFromPoint, GA_ROOT, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE,
        SWP_NOSIZE,
    };

    let mut last = "the viewer has no visible top-level window".to_string();
    for attempt in 0..20 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(150));
        }
        let Some(&hwnd) = top_level_windows(pid).first() else {
            continue;
        };
        // A window that is behind another still reports these screen
        // coordinates over MCP, so raising it is part of aiming rather
        // than a courtesy. Two calls, because the obvious one is not reliable:
        // `SetForegroundWindow` is refused whenever the caller is not already
        // the foreground process — which a test runner launched from a terminal
        // is not — and then the viewer stays behind it. `SetWindowPos` to
        // `HWND_TOPMOST` carries no such restriction: it changes Z order
        // without activating anything, which is all `WindowFromPoint` below
        // reads, and the left click the caller sends does the activating. The
        // check below is what decides, not either call.
        let _ = unsafe { SetForegroundWindow(hwnd) };
        let _ = unsafe {
            SetWindowPos(
                hwnd,
                Some(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        };
        if let Err(e) = unsafe { SetCursorPos(x, y) } {
            last = format!("SetCursorPos({x}, {y}) failed: {e}");
            continue;
        }
        let mut at = POINT::default();
        if let Err(e) = unsafe { GetCursorPos(&mut at) } {
            last = format!("GetCursorPos failed: {e}");
            continue;
        }
        if (at.x, at.y) != (x, y) {
            last = format!(
                "the cursor went to ({}, {}) rather than ({x}, {y}); the point may be off-screen",
                at.x, at.y
            );
            continue;
        }
        let under = unsafe { GetAncestor(WindowFromPoint(POINT { x, y }), GA_ROOT) };
        let mut owner = 0u32;
        unsafe { GetWindowThreadProcessId(under, Some(&mut owner)) };
        if owner != pid {
            last = format!("({x}, {y}) is over process {owner}, not the viewer ({pid})");
            continue;
        }
        return;
    }
    panic!("could not aim at the viewer's own window: {last}");
}

/// Make this process per-monitor DPI aware, once, and panic if it is not.
///
/// `SetCursorPos`, `GetCursorPos` and `WindowFromPoint` take and return
/// coordinates in the calling process's DPI context. The viewer is per-monitor
/// aware (its manifest and `sfm_explorer::run` both say so), so the window
/// block's `inner_position` and a listing's `rect_px` are physical pixels. A
/// process that is not DPI aware sees a scaled display in logical pixels, and
/// its cursor would land at a fraction of the point it was given. Setting the
/// same awareness here makes the two processes' pixels the same pixels.
///
/// The call fails when the awareness is already set (by a manifest, or by an
/// earlier call), so its result is not what decides; the awareness read back
/// afterwards is.
#[cfg(windows)]
fn per_monitor_dpi_aware() {
    use windows::Win32::UI::HiDpi::{
        GetAwarenessFromDpiAwarenessContext, GetThreadDpiAwarenessContext,
        SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        DPI_AWARENESS_PER_MONITOR_AWARE,
    };
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        let _ =
            unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    });
    let awareness = unsafe { GetAwarenessFromDpiAwarenessContext(GetThreadDpiAwarenessContext()) };
    assert_eq!(
        awareness, DPI_AWARENESS_PER_MONITOR_AWARE,
        "the test process is not per-monitor DPI aware, so its cursor coordinates \
         are not the viewer's physical pixels"
    );
}

/// A real right-click on the Scene panel's reconstruction row opens its context
/// menu.
///
/// Windows only, and driven by synthetic OS mouse input, because the defect
/// this guards lives *below* egui and nothing inside the viewer can see it:
/// `platform::windows::create_manager` turns on `EnableMouseInPointer` for
/// DirectManipulation, which routes every mouse button through `WM_POINTER` —
/// and winit 0.30 renders those as `Touch` events that egui's touch emulation
/// reads as the *primary* button. With that unhandled, no `secondary_clicked`
/// ever fires anywhere in the app: no context menu can open, and a right-click
/// on a tree row selects it like a left-click. A `click` over MCP goes into
/// egui's input directly and would pass;
/// [`the_reconstruction_rows_context_menu_lists_every_entry`] is that check,
/// and asserts on the menu's contents in detail.
///
/// The input is real and everything else goes over MCP. The point to press is
/// the window block's `inner_position` (the drawable area's corner on the
/// desktop) plus the centre of the `demo` label's `rect_px` in a listing of the
/// whole window, which is measured from that corner. Both are physical pixels,
/// and [`per_monitor_dpi_aware`] makes them the pixels `SetCursorPos` takes.
/// The menu is then read back with `get_widgets`, whose `owner.at_px` is the
/// point egui received the right button at, so the test also shows that the
/// press landed on the row it aimed at.
#[cfg(windows)]
#[test]
fn a_real_right_click_opens_the_reconstruction_rows_context_menu() {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
        MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEINPUT, MOUSE_EVENT_FLAGS,
    };

    // The return value is checked because `SendInput` is refused silently: UIPI
    // blocks injection into the session whenever the foreground window belongs
    // to a more privileged process, and the call then inserts nothing and
    // returns 0. Unchecked, the test would wait out its budget for a menu that
    // was never asked for and report it as a product failure.
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
        let sent = unsafe { SendInput(&[input], std::mem::size_of::<INPUT>() as i32) };
        assert_eq!(
            sent,
            1,
            "SendInput inserted nothing: {}",
            windows::core::Error::from_thread()
        );
    }

    per_monitor_dpi_aware();
    let viewer = McpViewer::launch_demo();
    let pid = viewer.guard.pid();

    // The demo node's row is labelled "demo". Found in the Scene panel, so a
    // "demo" label elsewhere cannot be taken for it, and then read from a
    // listing of the whole window, whose rectangles are measured from the
    // drawable area's corner rather than from the panel's.
    let id = viewer.wait_for(
        json!({ "panel_name": "scene" }),
        "the demo row never appeared",
        |listing| find_widget(listing, "label", "demo").map(|label| label["widget"].clone()),
    );
    let listing = viewer.get_widgets(json!({}));
    let label = entries(&listing)
        .iter()
        .find(|widget| widget["widget"] == id)
        .unwrap_or_else(|| panic!("the demo label {id} is not in the window's listing"))
        .clone();
    let rect = label["rect_px"].clone();
    let window = viewer.ok("get_window_layout", json!({}))["window"].clone();
    let n = |value: &Value| value.as_i64().expect("a whole number of pixels") as i32;
    let origin = &window["inner_position"];
    assert!(
        origin.is_array(),
        "the window block has no inner_position to aim from: {window}"
    );
    let x = n(&origin[0]) + n(&rect[0]) + n(&rect[2]) / 2;
    let y = n(&origin[1]) + n(&rect[1]) + n(&rect[3]) / 2;
    // Where the aim came from, for a failure on a scaled or multi-monitor
    // desktop.
    println!(
        "\nright-click aim: scale_factor={} inner_position={origin} rect_px={rect} at=({x}, {y})",
        window["scale_factor"]
    );

    // Two moves with a pause: the app repaints on demand, and egui resolves a
    // click against the widget rects of the frame before it.
    for _ in 0..2 {
        aim_at(pid, x, y);
        std::thread::sleep(Duration::from_millis(250));
    }
    // A left click first, which both activates the window (a right-press on an
    // inactive one is swallowed by the activation) and selects the row, so the
    // menu has to open on a row whose selection just changed.
    mouse_event(MOUSEEVENTF_LEFTDOWN);
    std::thread::sleep(Duration::from_millis(120));
    mouse_event(MOUSEEVENTF_LEFTUP);
    std::thread::sleep(Duration::from_millis(400));

    // Aimed again: the pause above is long enough for the desktop to have
    // moved on, and the right-click is the one the assertion rests on.
    aim_at(pid, x, y);
    mouse_event(MOUSEEVENTF_RIGHTDOWN);
    std::thread::sleep(Duration::from_millis(120));
    mouse_event(MOUSEEVENTF_RIGHTUP);

    // Polled: the right button reaches the viewer through the OS, so no reply
    // waits for the frame that opens the menu.
    let menu = viewer.wait_for(
        json!({}),
        &format!(
            "the reconstruction row's context menu did not open on a right-click at ({x}, {y})"
        ),
        |listing| {
            listing["menus"].as_array()?.iter().find_map(|menu| {
                (menu["kind"] == "context_menu" && menu["owner"]["panel_name"] == "scene")
                    .then(|| menu.clone())
            })
        },
    );
    assert!(
        contains(&rect, &menu["owner"]["at_px"]),
        "the menu was opened at {} rather than on the demo label at {rect}",
        menu["owner"]["at_px"]
    );
    // A few entries, enough to say this is the reconstruction row's menu. The
    // MCP twin checks every entry and its state.
    for label in [
        "Select",
        "Zoom to Fit",
        "Bundle Adjust...",
        "Bake Transform",
        "Close",
    ] {
        assert!(
            menu_item(&menu, label).is_some(),
            "no {label:?} in the context menu: {menu}"
        );
    }
}

// --- Diagnostics ---

/// Diagnostic: print the whole window's `get_widgets` listing at the empty
/// state, as the MCP tests see it (run with `-- --ignored --nocapture`).
#[test]
#[ignore]
fn dump_widgets() {
    let viewer = McpViewer::launch();
    let listing = viewer.get_widgets(json!({}));
    println!(
        "{}",
        serde_json::to_string_pretty(&listing).expect("a listing is JSON")
    );
}

// --- Everything else: the viewer's own MCP endpoint ---

/// A viewer with its MCP endpoint live, and the address it printed.
struct McpViewer {
    /// Held for its `Drop`, which kills the viewer and releases the
    /// serialization lock.
    guard: Guard,
    address: String,
}

impl McpViewer {
    /// Launch a viewer on an ephemeral port and wait until it lists its menu
    /// bar.
    ///
    /// `--mcp 0` rather than a fixed port, because a developer running this
    /// suite very likely has a viewer of their own on 8787 and a port
    /// collision is a fatal startup error by design.
    fn launch() -> McpViewer {
        McpViewer::launched(ui_test_lock(), None, &["--no-default-layout"])
    }

    /// The same, with the demo reconstruction already loaded: the command-line
    /// form of File > Load Demo Data…, which appends the same node at the
    /// dialog's default point count.
    fn launch_demo() -> McpViewer {
        McpViewer::launched(ui_test_lock(), None, &["--no-default-layout", "--demo"])
    }

    /// Put `contents` at `~/.sfm-explorer-default-layout.json` and launch the
    /// viewer on it, giving the file back when the test ends.
    ///
    /// Launched *without* `--no-default-layout`, so the file is read. Writing
    /// the file has to happen under the serialization lock — it is one file,
    /// and two tests writing it would each launch a viewer on the other's
    /// layout — and so does giving it back, which is why the guard owns it.
    fn launch_with_default_layout(contents: &str, args: &[&str]) -> McpViewer {
        let lock = ui_test_lock();
        let file = DefaultLayoutFile::written(contents);
        McpViewer::launched(lock, Some(file), args)
    }

    /// Spawn the viewer under a lock the caller has already taken, read the
    /// endpoint it prints, and wait until it is ready.
    ///
    /// Ready is `get_widgets` listing the menu bar's `File` button: the call
    /// is answered after a frame has been laid out, so a listing with the menu
    /// bar in it says the window exists and draws. No accessibility API is
    /// involved, so the launch costs a spawn and a first frame and nothing
    /// else. That point stops the `launch_ms` clock.
    fn launched(
        lock: MutexGuard<'static, ()>,
        layout_file: Option<DefaultLayoutFile>,
        args: &[&str],
    ) -> McpViewer {
        let started = begin_accounting();
        let mut cmd = command(&["--mcp", "0"]);
        cmd.args(args);
        cmd.stdout(std::process::Stdio::piped());
        let mut child = cmd.spawn().expect("failed to spawn sfm-explorer");
        let stdout = child.stdout.take().expect("stdout was piped");
        // Owned by a guard before anything below can panic, so a viewer that
        // never becomes ready is still killed.
        let guard = Guard {
            started,
            launch: Cell::new(None),
            child,
            _layout_file: layout_file,
            _lock: lock,
        };
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
            .recv_timeout(LAUNCH_TIMEOUT)
            .expect("the viewer never printed its MCP endpoint");
        let address = line
            .trim()
            .rsplit_once("http://")
            .and_then(|(_, rest)| rest.strip_suffix("/mcp"))
            .unwrap_or_else(|| panic!("no endpoint in {line:?}"))
            .to_string();
        let viewer = McpViewer { guard, address };

        viewer.rpc(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"ui-test","version":"0"}}}"#,
        );
        // Not counted as calls: this is the launch. A request that times out
        // (a first frame slower than the read timeout) is asked again.
        let body = tool_body("get_widgets", json!({}));
        let deadline = Instant::now() + LAUNCH_TIMEOUT;
        loop {
            let last = match try_rpc(&viewer.address, &body) {
                Ok(response) => {
                    let listing = &response["result"]["structuredContent"];
                    if menu_bar_button(listing, "File").is_some() {
                        break;
                    }
                    response.to_string()
                }
                Err(error) => error,
            };
            assert!(
                Instant::now() < deadline,
                "the viewer never listed its menu bar; the last reply was {last}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        viewer
            .guard
            .launch
            .set(Some(viewer.guard.started.elapsed()));
        viewer
    }

    /// POST one JSON-RPC body and return the `result` object, panicking on a
    /// protocol error.
    fn rpc(&self, body: &str) -> Value {
        let parsed = try_rpc(&self.address, body).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(parsed["error"], Value::Null, "{parsed}");
        parsed["result"].clone()
    }

    /// Call one tool and return its result, refusal or not. Counted and timed
    /// for the `UIPROBE` line.
    fn call(&self, name: &str, arguments: Value) -> Value {
        let started = Instant::now();
        let result = self.rpc(&tool_body(name, arguments));
        CALLS.fetch_add(1, Ordering::Relaxed);
        CALL_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    /// Call one tool that must succeed, and return its `structuredContent`.
    fn ok(&self, name: &str, arguments: Value) -> Value {
        let result = self.call(name, arguments.clone());
        assert_ne!(
            result["isError"],
            Value::Bool(true),
            "{name} {arguments}: {result}"
        );
        result["structuredContent"].clone()
    }

    fn get_widgets(&self, arguments: Value) -> Value {
        self.ok("get_widgets", arguments)
    }

    fn click(&self, arguments: Value) -> Value {
        self.ok("click", arguments)
    }

    /// Poll `get_widgets` with `arguments` until `found` returns something,
    /// and return that; panic with the last listing after [`CONTENT_TIMEOUT`].
    ///
    /// A refused call (a panel not laid out yet) counts as "not yet", so a
    /// test can wait for a panel as well as for something in it.
    fn wait_for<T>(
        &self,
        arguments: Value,
        what: &str,
        mut found: impl FnMut(&Value) -> Option<T>,
    ) -> T {
        let deadline = Instant::now() + CONTENT_TIMEOUT;
        loop {
            let result = self.call("get_widgets", arguments.clone());
            let last = if result["isError"] == Value::Bool(true) {
                result.to_string()
            } else {
                let listing = &result["structuredContent"];
                if let Some(value) = found(listing) {
                    return value;
                }
                listing.to_string()
            };
            assert!(
                Instant::now() < deadline,
                "{what}; the last get_widgets {arguments} replied {last}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Click a menu bar button and return the menu it opened, from the
    /// click's reply.
    ///
    /// The reply is built after the frame in which a menu opened by the click
    /// appears, so the menu is asserted there rather than polled for.
    fn open_menu(&self, name: &str) -> Value {
        let bar = self.get_widgets(json!({}));
        let button = menu_bar_button(&bar, name)
            .unwrap_or_else(|| panic!("no {name} button in the menu bar: {bar}"));
        let reply = self.click(json!({ "widget": button["widget"] }));
        menu_owned_by(&reply, name)
            .cloned()
            .unwrap_or_else(|| panic!("clicking {name} opened no {name} menu: {reply}"))
    }

    /// Call `screenshot` and return its content blocks.
    fn screenshot_blocks(&self, arguments: Value) -> Vec<Value> {
        let result = self.call("screenshot", arguments.clone());
        assert_ne!(
            result["isError"],
            Value::Bool(true),
            "{arguments}: {result}"
        );
        result["content"]
            .as_array()
            .expect("a content array")
            .clone()
    }

    /// Call `screenshot` and decode the PNG it handed back.
    fn screenshot(&self, arguments: Value) -> image::RgbaImage {
        decode_png(&self.screenshot_blocks(arguments))
    }

    /// The window's drawable area in physical pixels, as the viewer reports it.
    fn inner_size(&self) -> [u32; 2] {
        let layout = self.ok("get_window_layout", json!({}));
        let size = &layout["window"]["inner_size"];
        [
            size[0].as_u64().expect("a width") as u32,
            size[1].as_u64().expect("a height") as u32,
        ]
    }
}

/// A `tools/call` request body.
fn tool_body(name: &str, arguments: Value) -> String {
    json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": { "name": name, "arguments": arguments },
    })
    .to_string()
}

/// POST one JSON-RPC body to `address` and return the whole parsed response.
///
/// Hand-written HTTP/1.1 for the reason `mcp::tests` writes its own: a POST
/// with a JSON body is a dozen lines, and an HTTP client dev-dependency would
/// buy nothing this needs. A free function rather than a method so a request
/// can be sent from a thread that does not wait for its reply.
fn try_rpc(address: &str, body: &str) -> Result<Value, String> {
    use std::io::{Read as _, Write as _};
    let mut stream = std::net::TcpStream::connect(address)
        .map_err(|e| format!("the endpoint is not listening: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("a read timeout is not settable: {e}"))?;
    let request = format!(
        "POST /mcp HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("the request is not writable: {e}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("the response is not readable: {e}"))?;
    let response = String::from_utf8_lossy(&response).into_owned();
    let json = response
        .lines()
        .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
        .find(|line| line.starts_with('{'))
        .ok_or_else(|| format!("no JSON in {response:?}"))?;
    serde_json::from_str(json).map_err(|e| format!("{e} in {json:?}"))
}

/// Decode the PNG in a screenshot's content blocks.
fn decode_png(blocks: &[Value]) -> image::RgbaImage {
    use base64::Engine as _;
    let encoded = blocks
        .iter()
        .find_map(|block| block["data"].as_str())
        .unwrap_or_else(|| panic!("no image block in {blocks:?}"));
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("the image block is base64");
    image::load_from_memory(&bytes)
        .expect("the image block is a decodable PNG")
        .to_rgba8()
}

// --- Reading a listing ---
//
// Small lookups over the JSON `get_widgets`, `click` and `screenshot` return.
// A block is anything with a `widgets` array: a listing, a dialog or a menu.

/// The `widgets` array of a listing, a dialog or a menu.
fn entries(block: &Value) -> &[Value] {
    block["widgets"].as_array().map_or(&[], Vec::as_slice)
}

/// The first widget of `role` named exactly `name`.
///
/// The role is part of the question because names repeat across roles: the
/// HUD has a `Points` check box, a `Points` slider and a `Points` label, and
/// Track View has an `Edit` check box beside the menu bar's `Edit` button.
fn find_widget<'a>(block: &'a Value, role: &str, name: &str) -> Option<&'a Value> {
    entries(block)
        .iter()
        .find(|widget| widget["role"] == role && widget["name"] == name)
}

/// A menu bar button: a button whose `path` is its name alone.
fn menu_bar_button<'a>(listing: &'a Value, name: &str) -> Option<&'a Value> {
    find_widget(listing, "button", name).filter(|widget| widget["path"] == json!([name]))
}

/// The item of `menu` labelled `label`.
///
/// An item drawn with a keyboard shortcut carries it in its name after a space
/// (`Save Ctrl+S`, `Save ⌘S` on macOS), because AccessKit reads the button's
/// text and its shortcut text into one label. A shortcut has no space in it,
/// so the name is the label alone or the label, a space and one word — which
/// keeps `Save` from matching `Save As... Ctrl+Shift+S` or `Save As
/// Minimal...` without the test spelling a platform's shortcut.
fn menu_item<'a>(menu: &'a Value, label: &str) -> Option<&'a Value> {
    entries(menu).iter().find(|widget| {
        widget["name"].as_str().is_some_and(|name| {
            name == label
                || name
                    .strip_prefix(label)
                    .and_then(|rest| rest.strip_prefix(' '))
                    .is_some_and(|shortcut| !shortcut.is_empty() && !shortcut.contains(' '))
        })
    })
}

/// The open menu whose owner is named `owner`.
fn menu_owned_by<'a>(reply: &'a Value, owner: &str) -> Option<&'a Value> {
    reply["menus"]
        .as_array()?
        .iter()
        .find(|menu| menu["owner"]["name"] == owner)
}

/// The open dialog titled `title`.
fn dialog<'a>(reply: &'a Value, title: &str) -> Option<&'a Value> {
    reply["dialogs"]
        .as_array()?
        .iter()
        .find(|dialog| dialog["title"] == title)
}

/// Assert that `menu` holds exactly `items`, in that order, each enabled or
/// greyed as given.
///
/// Every item is listed whether it applies or not, which is what this checks
/// alongside the state: an action that vanishes when it does not apply reads
/// as unimplemented.
fn assert_menu(menu: &Value, items: &[(&str, bool)]) {
    let mut previous = None;
    for (label, enabled) in items {
        let item = menu_item(menu, label).unwrap_or_else(|| panic!("no {label:?} in {menu}"));
        assert_eq!(
            item["enabled"],
            *enabled,
            "{label:?} should be {}: {item}",
            if *enabled { "enabled" } else { "greyed" }
        );
        let index = entries(menu)
            .iter()
            .position(|widget| widget == item)
            .expect("the item is in the menu it was found in");
        assert!(
            previous.is_none_or(|previous| previous < index),
            "{label:?} is out of order in {menu}"
        );
        previous = Some(index);
    }
    assert_eq!(
        entries(menu).len(),
        items.len(),
        "the menu holds items this test does not name: {menu}"
    );
}

/// Whether `[x, y]` lies inside the rectangle `[left, top, width, height]`.
fn contains(rect: &Value, point: &Value) -> bool {
    let n = |value: &Value| value.as_i64().expect("a whole number of pixels");
    let [left, top, width, height] = [0, 1, 2, 3].map(|i| n(&rect[i]));
    let [x, y] = [0, 1].map(|i| n(&point[i]));
    x >= left && x < left + width && y >= top && y < top + height
}

// --- Window and menu bar ---

/// The window opens at no less than its 800×600 logical minimum.
///
/// The minimum is `with_min_inner_size`, which the platform enforces on a
/// person's drag. A programmatic resize does not respect it on Windows (a
/// `set_window_layout` to 400×300 gets 400×300), so this cannot ask the viewer
/// to shrink and read back a clamp; it checks the size the window opens at, in
/// logical points so a scaled display does not change the answer.
#[test]
fn window_min_size() {
    let viewer = McpViewer::launch();
    let layout = viewer.ok("get_window_layout", json!({}));
    let logical = &layout["window"]["derived"]["inner_size_logical"];
    let width = logical[0].as_f64().expect("a width");
    let height = logical[1].as_f64().expect("a height");
    assert!(width >= 800.0, "width {width} < 800: {layout}");
    assert!(height >= 600.0, "height {height} < 600: {layout}");
}

/// The menu bar's top-level menus, and the one that is deliberately absent.
///
/// The display controls that made a View menu worth having moved into the
/// viewport HUD — see `specs/gui/viewport-hud.md` — so what is left is what is
/// app-global: what is loaded, what has been done to it, where to go, and which
/// panels are up.
#[test]
fn the_menu_bar_holds_file_edit_go_and_panels() {
    let viewer = McpViewer::launch();
    // One listing answers all five, the absence included: the listing that
    // finds no View button is the one that found the other four.
    let bar = viewer.get_widgets(json!({}));
    let tops: Vec<i64> = ["File", "Edit", "Go", "Panels"]
        .iter()
        .map(|name| {
            let button = menu_bar_button(&bar, name)
                .unwrap_or_else(|| panic!("no {name} button in the menu bar: {bar}"));
            button["rect_px"][1].as_i64().expect("a top edge")
        })
        .collect();
    assert!(
        tops.windows(2).all(|pair| pair[0] == pair[1]),
        "the four buttons are not in one row: tops {tops:?}"
    );
    assert!(
        find_widget(&bar, "button", "View").is_none(),
        "there is a View button: {bar}"
    );
}

/// The empty-state placeholder text is shown before any file is loaded.
#[test]
fn empty_state_placeholder_text() {
    let viewer = McpViewer::launch();
    viewer.wait_for(
        json!({ "panel_name": "viewer_3d" }),
        "the 3D viewer never showed 'No reconstruction loaded.'",
        |listing| find_widget(listing, "label", "No reconstruction loaded.").map(|_| ()),
    );
}

/// The File menu opens and lists every item, greyed where nothing is loaded.
///
/// With nothing loaded there is no node to save or close, so those items are
/// greyed, and they are still listed: an action that vanishes when it does not
/// apply reads as unimplemented. What each item does is covered headlessly.
#[test]
fn file_menu_items() {
    let viewer = McpViewer::launch();
    let menu = viewer.open_menu("File");
    assert_menu(
        &menu,
        &[
            ("Open...", true),
            ("Save", false),
            ("Save As...", false),
            ("Save As Minimal...", false),
            ("Close All", false),
            ("Load Demo Data...", true),
            ("Quit", true),
        ],
    );
}

/// The Edit menu opens, and its items are listed even when every one of them
/// is greyed.
///
/// With nothing loaded there is no node to undo in and nothing selected to
/// delete or move, so every item is disabled — which is the state this asserts
/// they are nonetheless *present* in. What they do, and when they are enabled,
/// is covered headlessly in `state/edits/tests.rs`.
#[test]
fn edit_menu_items() {
    let viewer = McpViewer::launch();
    let menu = viewer.open_menu("Edit");
    assert_menu(
        &menu,
        &[
            ("Undo", false),
            ("Redo", false),
            ("Delete Point", false),
            ("Delete Image", false),
            ("Move Camera", false),
            ("Cancel Camera Move", false),
        ],
    );
}

/// The File menu's three save items, and which of them apply to demo data.
///
/// Demo data came from no file — `--demo` appends a generated node, whose path
/// is `None` exactly as the menu's is — so there is nothing for Save to write
/// over, and it is greyed; Save As is the way out, and Save As Minimal writes a
/// copy of any node, from a file or not. The demo node is selected when it is
/// loaded, which is what those two act on. What each item does is covered
/// headlessly in `state/save/tests.rs`.
#[test]
fn file_menu_save_items_apply_to_a_node_that_came_from_no_file() {
    let viewer = McpViewer::launch_demo();
    let menu = viewer.open_menu("File");
    for (label, enabled) in [
        ("Save", false),
        ("Save As...", true),
        ("Save As Minimal...", true),
    ] {
        let item = menu_item(&menu, label).unwrap_or_else(|| panic!("no {label:?} in {menu}"));
        assert_eq!(item["enabled"], enabled, "{label:?}: {item}");
    }
}

/// File > Quit exits the process.
///
/// Asserted on the child process rather than on the window: a menu item wired
/// to a command the event loop never reads leaves the window up, and only the
/// process exiting shows the item works.
#[test]
fn quit_menu_item_exits_the_process() {
    let mut viewer = McpViewer::launch();
    let menu = viewer.open_menu("File");
    let quit = menu_item(&menu, "Quit").unwrap_or_else(|| panic!("no Quit in {menu}"));
    // The viewer exits in the frame that delivers the click, so the reply may
    // never be written: the click is sent from a thread nobody waits on.
    let address = viewer.address.clone();
    let body = tool_body("click", json!({ "widget": quit["widget"] }));
    std::thread::spawn(move || try_rpc(&address, &body));

    assert!(
        viewer.guard.wait_for_exit(Duration::from_secs(10)),
        "File > Quit did not exit the process"
    );
}

// --- Panels ---

/// With a reconstruction loaded the viewport HUD is open, so its layer toggles
/// are there without opening anything — the point of moving them out of a
/// menu. Also the end-to-end check that the HUD reaches a real window;
/// everything else about it is exercised headlessly in `viewer_3d/hud/tests.rs`.
#[test]
fn hud_layer_toggles_are_present_and_checked_once_a_scene_is_loaded() {
    let viewer = McpViewer::launch_demo();
    let layers = ["Points", "Camera Images", "Grid"];
    let hud = viewer.wait_for(
        json!({ "panel_name": "viewer_3d" }),
        "a HUD layer check box never appeared",
        |listing| {
            layers
                .iter()
                .all(|name| find_widget(listing, "check_box", name).is_some())
                .then(|| listing.clone())
        },
    );
    for name in layers {
        let layer = find_widget(&hud, "check_box", name).expect("found above");
        assert_eq!(
            layer["toggled"], true,
            "'{name}' should be checked by default: {layer}"
        );
    }
}

/// Clicking a HUD check box turns its layer off.
#[test]
fn toggle_hud_layer_checkbox() {
    let viewer = McpViewer::launch_demo();
    let panel = json!({ "panel_name": "viewer_3d" });
    let grid = viewer.wait_for(
        panel.clone(),
        "the Grid check box never appeared",
        |listing| find_widget(listing, "check_box", "Grid").cloned(),
    );
    assert_eq!(grid["toggled"], true, "Grid should start checked: {grid}");

    let reply = viewer.click(json!({ "widget": grid["widget"] }));
    assert_eq!(reply["hit"]["name"], "Grid", "{reply}");

    viewer.wait_for(panel, "Grid is still checked after the click", |listing| {
        find_widget(listing, "check_box", "Grid")
            .filter(|grid| grid["toggled"] == false)
            .map(|_| ())
    });
}

/// The Scene panel reaches a real window: once the demo node is loaded
/// through File > Load Demo Data…, its reconstruction row and the Camera Images
/// group beneath it are listed. Everything else about the tree is exercised
/// headlessly in `scene_graph/tests.rs`.
///
/// **This is the one test that loads demo data through the File menu and its
/// dialog**, and it keeps that route rather than `--demo` on purpose: what it
/// asserts — that the node the load made, and its rows, appear in the Scene
/// panel — is what that menu route is for, so driving it here is what keeps
/// the route covered. Every other test asks for the node on the command line.
/// (`file_menu_items` asserts the item is *in* the menu, not that it works.)
#[test]
fn the_scene_panel_lists_the_loaded_reconstruction() {
    let viewer = McpViewer::launch();
    let menu = viewer.open_menu("File");
    let item = menu_item(&menu, "Load Demo Data...")
        .unwrap_or_else(|| panic!("no Load Demo Data... in {menu}"));
    let reply = viewer.click(json!({ "widget": item["widget"] }));
    let load_dialog =
        dialog(&reply, "Load Demo Data").unwrap_or_else(|| panic!("no demo dialog in {reply}"));
    let load = find_widget(load_dialog, "button", "Load")
        .unwrap_or_else(|| panic!("the demo dialog has no Load button: {load_dialog}"));
    let reply = viewer.click(json!({ "widget": load["widget"] }));
    assert!(
        dialog(&reply, "Load Demo Data").is_none(),
        "the demo dialog is still open after Load: {reply}"
    );

    // The demo reconstruction is labeled "demo" and rings the scene with 8
    // images, so both strings are fixed by the fixture. The third is the row's
    // solo toggle: its own behaviour is headless (`scene_graph` tests), and what
    // only a real window can show is that a *third* glyph button squeezed onto
    // the row is still laid out. The node may take more frames to arrive than
    // the click's reply waited for, so this polls.
    viewer.wait_for(
        json!({ "panel_name": "scene" }),
        "the Scene panel never listed the loaded reconstruction",
        |listing| {
            (find_widget(listing, "label", "demo").is_some()
                && find_widget(listing, "label", "Camera Images (8)").is_some()
                && find_widget(listing, "button", "S").is_some())
            .then_some(())
        },
    );
}

/// A right click on the Scene panel's reconstruction row opens its context
/// menu, owned by that row, with every entry in it.
///
/// The MCP twin of `a_real_right_click_opens_the_reconstruction_rows_context_menu`,
/// on all three platforms. That test proves a real right button reaches egui on
/// Windows; this one asserts on everything the menu holds: which row owns it,
/// every entry in order, which entries open a submenu, and which are greyed.
/// With one reconstruction loaded there is nothing to
/// align to, so `Align to` is a greyed entry rather than a submenu; the demo is
/// drawn in its own frame, so there is no transform to reset or bake. When each
/// entry is enabled is covered headlessly in `scene_graph/tests.rs`; this test
/// pins a few of them to show the listing reports egui's own state.
#[test]
fn the_reconstruction_rows_context_menu_lists_every_entry() {
    let viewer = McpViewer::launch_demo();
    let panel = json!({ "panel_name": "scene" });
    let label = viewer.wait_for(panel, "the demo row never appeared", |listing| {
        find_widget(listing, "label", "demo").cloned()
    });
    let rect = &label["rect_px"];
    let n = |i: usize| rect[i].as_i64().expect("a whole number of pixels");
    let at = json!([n(0) + n(2) / 2, n(1) + n(3) / 2]);

    let reply = viewer.click(json!({
        "panel_name": "scene", "at_px": at, "mouse_button": "right",
    }));
    // The row's click target is an unnamed rectangle under its label.
    assert_eq!(reply["hit"]["panel_name"], "scene", "{reply}");
    assert_eq!(reply["hit"]["name"], Value::Null, "{reply}");

    let menus = reply["menus"].as_array().expect("a menus array");
    assert_eq!(menus.len(), 1, "{reply}");
    let menu = &menus[0];
    assert_eq!(menu["kind"], "context_menu", "{menu}");
    let owner = &menu["owner"];
    assert_eq!(owner["panel_name"], "scene", "{owner}");
    assert_eq!(owner["widget"], reply["hit"]["widget"], "{owner}");
    assert!(
        contains(rect, &owner["at_px"]),
        "the menu was opened at {} rather than on the demo label at {rect}",
        owner["at_px"]
    );

    assert_menu(
        menu,
        &[
            ("Select", true),
            ("Zoom to Fit", true),
            ("Align to", false),
            ("Reset Transform", false),
            ("Bake Transform", false),
            ("Tint", true),
            ("Bundle Adjust...", false),
            ("Retriangulate All Points", false),
            ("Prune Covered Observations", false),
            ("Build Index Files", false),
            ("Convert to Embedded Patches", true),
            ("Close", true),
        ],
    );
    let submenus: Vec<&str> = entries(menu)
        .iter()
        .filter(|item| item["submenu"] == true)
        .filter_map(|item| item["name"].as_str())
        .collect();
    assert_eq!(submenus, ["Tint"], "{menu}");
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
    /// Put `contents` at `~/.sfm-explorer-default-layout.json`, preserving
    /// whatever was there.
    ///
    /// **The caller must already hold [`UI_TEST_LOCK`]**, and must keep holding
    /// it until this value is dropped — that is
    /// [`McpViewer::launch_with_default_layout`], which is the only caller. The
    /// file is one file: two tests writing it at once would each launch a
    /// viewer on the other's layout, and a restore outside the lock races the
    /// next test's own `rename` of it.
    fn written(contents: &str) -> Self {
        let home = std::env::home_dir().expect("a home directory");
        let path = home.join(".sfm-explorer-default-layout.json");
        let saved = path.exists().then(|| {
            let saved = home.join(".sfm-explorer-default-layout.json.ui-test-backup");
            std::fs::rename(&path, &saved).expect("move the developer's layout aside");
            saved
        });
        std::fs::write(&path, contents).expect("write a default layout file");
        DefaultLayoutFile { path, saved }
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
/// that the file was read, and the 3D viewer, which the stock grid always
/// shows, is closed.
#[test]
fn a_saved_default_layout_is_loaded_at_startup() {
    let viewer = McpViewer::launch_with_default_layout(
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
        &[],
    );

    viewer.wait_for(
        json!({ "panel_name": "action_log" }),
        "the Action Log's toolbar never appeared, so the saved layout was not loaded",
        |listing| find_widget(listing, "button", "Latest").map(|_| ()),
    );
    let layout = viewer.ok("get_window_layout", json!({}));
    assert_eq!(
        layout["panels"]["viewer_3d"]["open"], false,
        "the 3D viewer is still docked, so the saved layout was not loaded: {layout}"
    );
}

/// The Edit History panel reaches a real window: with a node loaded it lists
/// that node's one version, the file as it was opened.
///
/// The panel is put in front by a default layout file naming it alone, as the
/// startup-load test above puts the Action Log there: the stock grid keeps the
/// Edit History tab behind the Image Browser, and a layout file puts the panel
/// in front from the viewer's first frame. Everything the panel decides is
/// exercised headlessly in `edit_history_panel/tests.rs`.
#[test]
fn the_edit_history_panel_lists_the_loaded_version() {
    // With `--demo`, so the node is there before the first query.
    let viewer = McpViewer::launch_with_default_layout(
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
        &["--demo"],
    );

    // The demo node is labeled "demo" and has been through no edit, so both
    // strings are fixed by the fixture.
    viewer.wait_for(
        json!({ "panel_name": "edit_history" }),
        "the Edit History panel never listed the loaded version",
        |listing| {
            (find_widget(listing, "label", "1 version").is_some()
                && find_widget(
                    listing,
                    "label",
                    "demo has not been edited; its one version is the file as it was opened.",
                )
                .is_some())
            .then_some(())
        },
    );
}

// --- The MCP screenshot, against a real frame ---
//
// Everything else the MCP surface does is under headless test in
// `mcp::tests`, which is where it belongs: the command vocabulary takes no
// GPU and no window. `screenshot` is the exception — it is a picture of a
// frame that has actually been rendered and presented — so it is here, and
// what these assert is the size and decodability of the PNG rather than its
// pixels, since what the frame *looks* like is not a stable thing to assert.

/// The default screenshot is the window itself, read back off the presented
/// surface — which is what `COPY_SRC` on the swapchain buys — and a panel's
/// screenshot can carry the listing of the frame it is a picture of.
#[test]
fn a_screenshot_is_the_whole_window() {
    // With the demo loaded, so the Scene panel has a row to find in the
    // listing below.
    let viewer = McpViewer::launch_demo();

    let [width, height] = viewer.inner_size();
    let window = viewer.screenshot(json!({}));
    assert_eq!(
        (window.width(), window.height()),
        (width, height),
        "the picture is not the window's drawable area"
    );

    // A panel is a crop of that same frame, so it is smaller in both axes.
    let scene = viewer.screenshot(json!({ "panel_name": "scene" }));
    assert!(
        scene.width() < window.width() && scene.height() < window.height(),
        "the Scene panel's crop ({} × {}) is not inside the window ({} × {})",
        scene.width(),
        scene.height(),
        window.width(),
        window.height()
    );

    // `max_dimension` bounds the longer side, after the crop.
    let bounded = viewer.screenshot(json!({ "max_dimension": 320 }));
    assert_eq!(bounded.width().max(bounded.height()), 320);

    // `widgets: true` adds the listing of the same frame after the picture,
    // in the picture's pixels: the demo row's label is inside it.
    let blocks = viewer.screenshot_blocks(json!({ "panel_name": "scene", "widgets": true }));
    let picture = decode_png(&blocks);
    let listing: Value = blocks
        .iter()
        .skip_while(|block| block["type"] != "image")
        .find_map(|block| block["text"].as_str())
        .map(|text| serde_json::from_str(text).expect("the listing is JSON"))
        .unwrap_or_else(|| panic!("no listing after the picture: {blocks:?}"));
    let label = find_widget(&listing, "label", "demo")
        .unwrap_or_else(|| panic!("no demo label in the listing: {listing}"));
    let rect: Vec<i64> = (0..4)
        .map(|i| {
            label["rect_px"][i]
                .as_i64()
                .expect("a whole number of pixels")
        })
        .collect();
    assert!(
        rect[0] >= 0
            && rect[1] >= 0
            && rect[0] + rect[2] <= i64::from(picture.width())
            && rect[1] + rect[3] <= i64::from(picture.height()),
        "the demo label at {rect:?} is not inside the {} × {} picture",
        picture.width(),
        picture.height()
    );
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
    // The render target only exists once the viewport has something to draw:
    // with nothing loaded the panel shows its empty state and never sizes one.
    let viewer = McpViewer::launch_demo();

    let with_hud = viewer.screenshot(json!({ "panel_name": "viewer_3d" }));
    let without_hud = viewer.screenshot(json!({ "panel_name": "viewer_3d", "hud": false }));
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

    // Behind Track View in the stock grid.
    let behind = viewer.call("screenshot", json!({ "panel_name": "camera_intrinsics" }));
    assert_eq!(behind["isError"], Value::Bool(true), "{behind}");
    let message = behind["content"][0]["text"].as_str().expect("a refusal");
    assert!(
        message.contains("Track View") && message.contains("show_panel"),
        "{message}"
    );

    // Closed.
    viewer.call("hide_panel", json!({ "panel_name": "action_log" }));
    let closed = viewer.call("screenshot", json!({ "panel_name": "action_log" }));
    assert_eq!(closed["isError"], Value::Bool(true), "{closed}");
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
    let viewer = McpViewer::launch_demo();
    // The node is appended before the window opens, but `get_scene` is answered
    // inside a frame, so the first call waits for it rather than racing it.
    let deadline = Instant::now() + CONTENT_TIMEOUT;
    let label = loop {
        let scene = viewer.call("get_scene", json!({}));
        if let Some(label) = scene["structuredContent"]["scene"][0]["label"].as_str() {
            break label.to_string();
        }
        assert!(
            Instant::now() < deadline,
            "the demo node never appeared: {scene}"
        );
        std::thread::sleep(Duration::from_millis(100));
    };

    let made = viewer.ok(
        "delete_point",
        json!({ "reconstruction_label": label, "point": 3 }),
    );
    assert_eq!(made["label"], format!("Deleted point 3 in {label}"));
    let serial = made["serial"]
        .as_str()
        .expect("a version serial")
        .to_string();

    let history = viewer.ok("get_history", json!({ "reconstruction_label": label }));
    let versions = history["versions"].as_array().expect("a version list");
    assert_eq!(versions.len(), 2, "{history}");
    assert_eq!(history["cursor"], Value::String(serial.clone()));
    assert_eq!(history["dirty"], true);
    assert_eq!(history["can_undo"], true);
    // The node came from no file, so nothing of it is on disk.
    assert_eq!(history["disk_serial"], Value::Null);

    let undone = viewer.ok("undo", json!({ "reconstruction_label": label }));
    assert_eq!(undone["cursor"], versions[0]["serial"]);
    assert_eq!(undone["dirty"], false);

    // Both rows are the agent's, in the words the human's own Edit menu would
    // have written.
    let log = viewer.ok("get_action_log", json!({ "actors": ["mcp"] }));
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

/// Taking a camera in hand, in a real window: the Edit menu and the lock's
/// release.
///
/// The one part of the Move Camera family a headless frame cannot reach. Every
/// decision it makes -- the snap, the pending pose, the dead band, the commit --
/// is asserted in `camera_lock/tests.rs`; what this asks is whether the menu
/// entry and the release reach a live viewport at all. The banner is painted
/// rather than built of widgets, so it is not in a widget listing and is
/// covered headlessly instead.
///
/// The pose is deliberately not moved here: the MCP input tools have no drag,
/// and moving the camera is what the headless tests cover. What a real window
/// can be asked, and is asked below, is whether the menu takes the camera in
/// hand and gives it back.
#[test]
fn taking_a_camera_in_hand_reaches_a_real_window() {
    let viewer = McpViewer::launch_demo();

    // The 3D viewer alone in the dock, so nothing but the viewport is under
    // the menu the test drives.
    viewer.ok(
        "set_window_layout",
        json!({
            "layout": { "main": { "tabs": ["viewer_3d"], "active": "viewer_3d" }, "windows": [] },
        }),
    );
    viewer.ok("set_view", json!({ "look_through": { "camera_image": 0 } }));

    let log_texts = || -> Vec<String> {
        let log = viewer.ok(
            "get_action_log",
            json!({ "since_revision": 0, "limit": 200 }),
        );
        log["entries"]
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
    let click_item = |menu: &Value, label: &str| {
        let item = menu_item(menu, label).unwrap_or_else(|| panic!("no {label:?} in {menu}"));
        assert_eq!(item["enabled"], true, "{label:?} is greyed: {item}");
        viewer.click(json!({ "widget": item["widget"] }));
    };

    click_item(&viewer.open_menu("Edit"), "Move Camera");
    wait_for_line("Moving the camera of image_000.jpg");

    // With a lock held the same entry commits rather than takes, and the entry
    // beside it gives the camera back.
    let menu = viewer.open_menu("Edit");
    assert!(
        menu_item(&menu, "Commit Camera Move").is_some(),
        "the held lock did not change Move Camera to Commit Camera Move: {menu}"
    );
    click_item(&menu, "Cancel Camera Move");
    wait_for_line("Cancelled the camera move of image_000.jpg");
}
