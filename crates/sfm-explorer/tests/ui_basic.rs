// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos", target_os = "linux"))]

//! The viewer's windowed tests: what only a real window, on a real desktop,
//! can be asked.
//!
//! **One locator resolution is one full snapshot of the app's accessibility
//! subtree**, and that is what shapes this file. `wait_attached`, `press`,
//! `toggle`, `elements` and `count` each walk the whole tree — on Windows a
//! single `FindAllBuildCache(TreeScope_Subtree)` — so their cost is per
//! *operation*, not per launch, and it is the platform's, not the viewer's:
//! around 0.5s on a developer's machine and around 17s on a GitHub-hosted
//! Windows runner, against 3.8s for a launch, attach and teardown there.
//!
//! Two habits follow. Setup goes through the **command line** rather than the
//! accessibility API — `--demo` in place of driving File > Load Demo Data… and
//! its dialog, which is three `Locator` calls and more snapshots than that,
//! since the menu item and the dialog's button each appear a poll or two after
//! the press that makes them. And an assertion that something is *absent* uses
//! `Locator::count`, exactly one resolution, rather than a short-budget
//! `wait_attached`, which polls a whole snapshot every 100ms for its budget.
//! (Only a `Locator` method resolves; a `press` on an `Element` a lookup
//! already handed back invokes what it holds.) Exactly one test still drives
//! each route a shortcut replaces, and says so.
//!
//! Because that is where the cost is, **the suite reports its own**: every
//! test prints a `UIPROBE` line as its [`Guard`] drops, which is how a reader
//! of one CI log tells a slow runner apart from an expensive suite. See
//! [`Guard::report`] for the fields and [`Probed::probe`] for what is counted.

use std::cell::{Cell, RefCell};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, Once};
use std::time::{Duration, Instant};

use xa11y::{App, AppExt, Element, ElementData, Locator, Toggled};

/// Serializes the UI tests so at most one `sfm-explorer` window is alive at a
/// time. `cargo test` runs tests on multiple threads by default, and several
/// viewers plus concurrent accessibility tree walks make the Windows UI
/// Automation backend fail with `E_UNEXPECTED` (0x8000FFFF, "Catastrophic
/// failure"). Two tests also share one on-disk file, the default layout (see
/// [`DefaultLayoutFile`]), and the Windows-only input tests drive the real
/// cursor, which belongs to whichever window is in front. Each test holds this
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

// --- What the suite costs, as the suite sees it ---
//
// Two costs with nothing in common are added together in a job's wall clock:
// launching a viewer (process spawn, GPU init, the OS registering the window)
// and resolving a locator (a cross-process snapshot of the whole accessibility
// subtree). They have different causes and different fixes, and a log that
// reports only the total cannot tell them apart — nor tell a change that made
// the *suite* cheaper from a run that happened to land on a faster machine.
// So each is counted and printed separately; see `Guard::report`.

/// Locator resolutions run since the current [`Guard`] was made.
static OPS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds spent inside those resolutions.
static OP_NANOS: AtomicU64 = AtomicU64::new(0);
/// The same two, plus launch time and wall time, accumulated over every test
/// this process has run, for the `UIPROBE TOTAL` line.
static TOTAL_TESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LAUNCH_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OPS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OP_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_NANOS: AtomicU64 = AtomicU64::new(0);

/// Zero the per-test counters and start the clock a `UIPROBE` line is measured
/// from. Every path that makes a [`Guard`] calls this immediately before
/// spawning the viewer, and nothing else does.
///
/// **Process-wide counters are sound here only because of [`UI_TEST_LOCK`].**
/// The instrumented calls go through [`App`], which knows nothing about the
/// guard, so the counters cannot hang off one — but every test holds that lock
/// for its whole body, so exactly one guard is ever alive and "the operations
/// since the last reset" and "this test's operations" are the same set.
fn begin_accounting() -> Instant {
    OPS.store(0, Ordering::Relaxed);
    OP_NANOS.store(0, Ordering::Relaxed);
    Instant::now()
}

/// Run one locator resolution, counting it and timing it.
///
/// **One call here is one `op`, however many platform attempts happen inside
/// it.** That is what keeps `ops` a deterministic fingerprint of suite *shape*
/// rather than of how a particular run went: a run that needed a retry reports
/// the same `ops` as one that did not, and pays for it in `op_ms` where the
/// time actually went. A retry that incremented `ops` would make the two
/// indistinguishable from a real change to the tests — and it was exactly a
/// stable `ops` that localized the `taking_a_camera_in_hand_reaches_a_real_window`
/// failure to its third operation. Retries announce themselves on stdout
/// instead; see [`retrying_transient`].
fn measured<T>(op: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = op();
    OPS.fetch_add(1, Ordering::Relaxed);
    OP_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

// --- Transient platform failures ---
//
// The accessibility APIs are cross-process, and a call can fail because the
// tree was being rebuilt underneath it rather than because the suite asked for
// the wrong thing. Those two look identical to `expect`, so the ones that are
// recoverable are named here and nothing else is retried: a genuine selector
// mistake must still fail on its first attempt, loudly, instead of spending
// three budgets discovering the same absence.

/// `UIA_E_TIMEOUT` — the UI Automation layer gave up on a cross-process call.
/// Says nothing about the app; the call can simply be made again.
const UIA_E_TIMEOUT: u32 = 0x8013_1505;

/// `UIA_E_ELEMENTNOTAVAILABLE` — the element went away mid-call, which for an
/// egui app means the frame that owned that node has been replaced. The same
/// "the tree moved under me" story as a timeout.
const UIA_E_ELEMENTNOTAVAILABLE: u32 = 0x8004_0201;

/// How many extra attempts a read-only probe gets. Bounded deliberately: a
/// condition that survives three snapshots is not transient.
const TRANSIENT_ATTEMPTS: u32 = 3;

/// Whether an error is the platform losing its footing rather than the suite
/// being wrong.
///
/// Compared on the low 32 bits because `code` is an `i64` carrying an HRESULT,
/// and which of the two spellings of a high-bit-set HRESULT arrives -- the
/// sign-extended `-2146233083` the failing run printed, or a raw
/// `0x0000_0000_8013_1505` -- is a detail of how the backend widened it. Both
/// truncate to the same `u32`.
fn is_transient(error: &xa11y::Error) -> bool {
    matches!(
        error,
        xa11y::Error::Platform { code, .. }
            if matches!(*code as u32, UIA_E_TIMEOUT | UIA_E_ELEMENTNOTAVAILABLE)
    )
}

/// Re-run a **side-effect-free** probe when the platform reports a transient
/// failure.
///
/// Safe here precisely because the caller has none: resolving a locator twice
/// costs two snapshots and changes nothing, so an attempt that died mid-call
/// can simply be made again.
///
/// **This must not be extended to a press or a toggle,** and the asymmetry is
/// the point rather than an oversight — see the note on
/// [`Probe::press_revealing`]. A retry is announced on stdout so a run that
/// needed one says so, and is *not* counted as a second `op`; see [`measured`].
fn retrying_transient<T>(
    what: &str,
    mut attempt: impl FnMut() -> xa11y::Result<T>,
) -> xa11y::Result<T> {
    for tries in 1..TRANSIENT_ATTEMPTS {
        match attempt() {
            Err(error) if is_transient(&error) => {
                println!(
                    "UIPROBE RETRY op={what} attempt={} of {TRANSIENT_ATTEMPTS} after {error}",
                    tries + 1
                );
                // The tree is mid-rebuild by assumption, so give the next frame
                // a chance to land rather than racing the same one again.
                std::thread::sleep(Duration::from_millis(100));
            }
            outcome => return outcome,
        }
    }
    attempt()
}

/// A [`Locator`] that reports what it costs, wrapping only the methods this
/// suite calls.
///
/// One method call here is one resolution — one whole-subtree snapshot, per
/// this file's module comment — so this is the single place the suite's
/// dominant cost can be counted without deriving it by reading the test
/// bodies, loops and all. Each method forwards its arguments unchanged and
/// returns what the inner call returned: nothing a test waits for or asserts
/// passes through this differently.
///
/// `Element` methods are deliberately *not* wrapped. A `press` on an element a
/// lookup already handed back invokes what it holds and queries nothing, so
/// counting one would overstate the tree traffic — which is exactly the
/// miscount hand-derived figures used to make.
struct Probe(Locator);

impl Probe {
    fn wait_attached(&self, timeout: Duration) -> xa11y::Result<Element> {
        measured(|| retrying_transient("wait_attached", || self.0.wait_attached(timeout)))
    }

    fn wait_until(
        &self,
        predicate: impl Fn(Option<&ElementData>) -> bool,
        timeout: Duration,
    ) -> xa11y::Result<Option<Element>> {
        measured(|| retrying_transient("wait_until", || self.0.wait_until(&predicate, timeout)))
    }

    /// Press this element and confirm it revealed `revealed`, pressing once
    /// more if it did not.
    ///
    /// **The only press this type offers, and the asymmetry with the read-only
    /// probes above is the point rather than an oversight.** Those are wrapped
    /// in [`retrying_transient`]; a press is not, and must not be. A transient
    /// error says the *call* did not complete, not that the press did not land,
    /// and every press this suite makes is a toggle: a menu button that did
    /// open its menu closes it again on a second press. A blind retry therefore
    /// turns a recovered timeout into a shut menu, and the test fails later and
    /// less legibly than it would have here.
    ///
    /// So reliability comes from being *idempotent by observation* instead —
    /// press, look for what the press was supposed to reveal, press again only
    /// if it is absent. Because the outcome is read off the app rather than off
    /// the return value, both halves of the ambiguity end the same way: if the
    /// menu is open, this succeeded, whatever the press reported.
    ///
    /// That is what `taking_a_camera_in_hand_reaches_a_real_window` needed on
    /// run 35477064462, where `UIA_E_TIMEOUT` came back from opening the Edit
    /// menu a second time while the camera lock was held and the move banner
    /// was repainting the tree every frame.
    ///
    /// Counted as **one** op, including any second press — see [`measured`].
    ///
    /// **Returns the element it confirmed, and callers are expected to use
    /// it.** The confirmation is a full subtree snapshot — around 25 seconds on
    /// the Windows runner — so handing it back is what keeps this guard free:
    /// a caller that re-resolved the same selector afterwards would pay for the
    /// same snapshot twice, which on the job this suite is trying to shrink is
    /// the wrong trade. Every site that wants the revealed item immediately
    /// therefore takes it from here rather than looking it up again, and the
    /// guard costs those sites nothing at all.
    fn press_revealing(&self, revealed: &Probe, timeout: Duration) -> xa11y::Result<Element> {
        measured(|| {
            let first = self.0.press();
            if let Ok(element) = retrying_transient("press_revealing/confirm", || {
                revealed.0.wait_attached(timeout)
            }) {
                // It opened. A transient error from the press was a report
                // about the call, not about the app.
                return Ok(element);
            }
            // It did not open, so a press that also failed is the better
            // diagnosis than anything a second attempt would produce.
            first?;
            println!("UIPROBE RETRY op=press_revealing attempt=2 of 2 (nothing was revealed)");
            self.0.press()?;
            // The second press is not re-confirmed here: the caller's own next
            // lookup is the confirmation, and it fails with the selector it
            // actually wanted rather than with this one.
            revealed.0.wait_attached(timeout)
        })
    }

    /// Flip this element's checked state.
    ///
    /// Not retried, for the reason spelled out on [`Self::press_revealing`] —
    /// doubly so here, where a second toggle is *guaranteed* to undo the first
    /// rather than merely likely to. There is no `press_revealing` equivalent
    /// because a checkbox reveals nothing: callers confirm the new state with a
    /// `wait_until` on `states.checked` instead, which already fails loudly if
    /// the toggle did not land.
    fn toggle(&self) -> xa11y::Result<()> {
        measured(|| self.0.toggle())
    }

    fn count(&self) -> xa11y::Result<usize> {
        measured(|| retrying_transient("count", || self.0.count()))
    }
}

/// `app.probe(selector)` in place of `app.locator(selector)`: the same lookup,
/// counted and timed.
///
/// The method is named `probe` rather than `locator` because `App::locator` is
/// an *inherent* method, and an inherent method wins over a trait one of the
/// same name — a `locator` here would compile and silently never be called, so
/// the suite would report zero operations while running the usual number.
trait Probed {
    fn probe(&self, selector: &str) -> Probe;
}

impl Probed for App {
    fn probe(&self, selector: &str) -> Probe {
        Probe(self.locator(selector))
    }
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
///
/// `child` sits behind a `RefCell` because [`attach`] replaces it: a launch
/// that never becomes discoverable is retried once, in place, so the guard
/// still owns (and on drop still kills) whichever process is current.
struct Guard {
    /// When the viewer was spawned, and how long it took to become
    /// attachable — the two halves of the `UIPROBE` line's `launch_ms`, filled
    /// in by [`ChildHandle::attached`]. Neither owns anything, so neither
    /// takes part in the drop order described above.
    started: Instant,
    launch: Cell<Option<Duration>>,
    child: RefCell<Child>,
    /// The viewer's command line, kept so a stuck launch can be respawned the
    /// same way. `None` marks a guard whose process cannot simply be
    /// re-spawned — the MCP viewer, whose endpoint line has already been read
    /// off its stdout — and [`ChildHandle::relaunch`] declines to retry it.
    args: Option<Vec<String>>,
    /// The default layout file this test wrote, for the two tests that start
    /// the viewer on one. Held here rather than as a local of the test so that
    /// it is restored *under* the lock: it used to be a local declared before
    /// the guard, so it dropped last, after the lock was released — and under a
    /// plain multi-threaded `cargo test` the next test's own `rename` of that
    /// one path raced this restore and failed with "Access is denied".
    _layout_file: Option<DefaultLayoutFile>,
    _lock: MutexGuard<'static, ()>,
}

impl Guard {
    /// Acquire the serialization lock, then launch the app under it.
    fn new() -> Self {
        Guard::with_args(&["--no-default-layout"])
    }

    /// The same, with the demo reconstruction already loaded.
    ///
    /// `--demo` makes the node the File menu's Load Demo Data… dialog makes,
    /// at the same default point count, so a test that only wants a scene in
    /// front of it gets one for no accessibility operations at all — see
    /// [`load_demo_data`], which is what the one test that still drives the
    /// menu path calls.
    fn demo() -> Self {
        Guard::with_args(&["--no-default-layout", "--demo"])
    }

    /// The same, with the viewer's command line spelled out.
    fn with_args(args: &[&str]) -> Self {
        Guard::launched(ui_test_lock(), None, args)
    }

    /// Put `contents` at `~/.sfm-explorer-default-layout.json` and launch the
    /// viewer on it, giving the file back when the test ends.
    ///
    /// Writing that file has to happen under the serialization lock — it is one
    /// file, and two tests writing it would each launch a viewer on the other's
    /// layout — and so does giving it back, which is why the guard owns it.
    fn with_default_layout(contents: &str, args: &[&str]) -> Self {
        let lock = ui_test_lock();
        let file = DefaultLayoutFile::written(contents);
        Guard::launched(lock, Some(file), args)
    }

    /// Launch the viewer under a lock the caller has already taken.
    fn launched(
        lock: MutexGuard<'static, ()>,
        layout_file: Option<DefaultLayoutFile>,
        args: &[&str],
    ) -> Self {
        let started = begin_accounting();
        Guard {
            started,
            launch: Cell::new(None),
            child: RefCell::new(launch_with(args)),
            args: Some(args.iter().map(|a| (*a).to_string()).collect()),
            _layout_file: layout_file,
            _lock: lock,
        }
    }

    /// Print this test's `UIPROBE` line, and the running `UIPROBE TOTAL`.
    ///
    /// ```text
    /// UIPROBE test=file_menu_items launch_ms=886 ops=5 op_ms=3732 total_ms=4733
    /// UIPROBE TOTAL tests=19 launch_ms=19559 ops=45 op_ms=29234 total_ms=53012 mean_launch_ms=1029 mean_op_ms=649
    /// ```
    ///
    /// `launch_ms` is the runner-speed yardstick: spawning a process, waiting
    /// for the GPU and for the OS to register the window is work the suite
    /// cannot make cheaper, so `mean_launch_ms` moving between two runs means
    /// the *machine* moved. `ops` and `mean_op_ms` are the suite's own cost:
    /// `ops` is how many whole-subtree snapshots the tests asked for, which
    /// only a change to the tests moves, and `mean_op_ms` is what the platform
    /// charges for one. Comparing two logs, then: `ops` down is a cheaper
    /// suite, `mean_launch_ms` and `mean_op_ms` down together is a faster
    /// machine, and `total_ms` alone says nothing about which happened.
    ///
    /// libtest offers no end-of-suite hook, so the `TOTAL` line is cumulative
    /// and re-printed after every test; the last one is the run's. That also
    /// means a run that aborts halfway still reports what it spent.
    ///
    /// Emitted from `Drop`, so a **panicking** test — every assertion failure
    /// here panics — reports too, which is when the numbers are most wanted.
    /// The test's name comes from its thread, which libtest names after it
    /// even under `--test-threads=1` (it still spawns a thread per test and
    /// joins it at once). A guard whose test never attaches reports
    /// `launch_ms=0`; the only one is the `#[ignore]`d [`dump_tree`], which
    /// resolves the app itself rather than through [`attach`].
    ///
    /// None of this reaches a green CI log without `--nocapture`: libtest
    /// captures a passing test's stdout and discards it. All three invocations
    /// of this suite pass the flag — the `ui-test` task in `pixi.toml`, its
    /// Linux override, and the `ui-test-macos` job, which runs the built
    /// binary directly.
    fn report(&self) {
        let total = self.started.elapsed();
        let launch = self.launch.get().unwrap_or_default();
        let ops = OPS.load(Ordering::Relaxed);
        let op_nanos = OP_NANOS.load(Ordering::Relaxed);

        let tests = TOTAL_TESTS.fetch_add(1, Ordering::Relaxed) + 1;
        let launch_total = TOTAL_LAUNCH_NANOS
            .fetch_add(launch.as_nanos() as u64, Ordering::Relaxed)
            + launch.as_nanos() as u64;
        let ops_total = TOTAL_OPS.fetch_add(ops, Ordering::Relaxed) + ops;
        let op_nanos_total = TOTAL_OP_NANOS.fetch_add(op_nanos, Ordering::Relaxed) + op_nanos;
        let nanos_total = TOTAL_NANOS.fetch_add(total.as_nanos() as u64, Ordering::Relaxed)
            + total.as_nanos() as u64;

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
            "\nUIPROBE test={name} launch_ms={} ops={ops} op_ms={} total_ms={}",
            launch.as_millis(),
            ms(op_nanos),
            total.as_millis(),
        );
        println!(
            "UIPROBE TOTAL tests={tests} launch_ms={} ops={ops_total} op_ms={} total_ms={} \
             mean_launch_ms={} mean_op_ms={}",
            ms(launch_total),
            ms(op_nanos_total),
            ms(nanos_total),
            ms(launch_total) / tests,
            // `checked_div`: a suite filtered down to tests that resolve
            // nothing reports no mean rather than dividing by zero.
            ms(op_nanos_total).checked_div(ops_total).unwrap_or(0),
        );
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
        // Last, so `total_ms` covers the teardown the test also pays for.
        self.report();
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

    /// Stop the launch clock: the app is discoverable, so everything from the
    /// spawn to here is launch cost and everything after it is the test's.
    ///
    /// Only the first attach counts. A relaunch happens *inside* that first
    /// one and is honestly part of what the launch cost; a second `attach`
    /// later in the same test is not a launch at all.
    fn attached(&self) {
        if self.guard.launch.get().is_none() {
            self.guard.launch.set(Some(self.guard.started.elapsed()));
        }
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
///
/// Those three predate [`try_attach_app`]'s move to pid addressing, and the
/// first of the two shapes is the one xa11y 0.15 says pid attachment closes:
/// enumerating applications skips a window that has not named itself yet, which
/// is exactly a viewer mid-launch. The second comes out of the automation
/// client rather than out of discovery, so it is not addressed, and the retry
/// stays until a few hundred more runs say it can go.
fn attach(child: ChildHandle<'_>) -> App {
    init();
    let first = match try_attach_app(child) {
        Ok(app) => {
            child.attached();
            return app;
        }
        Err(e) => e,
    };
    assert!(
        child.relaunch(),
        "sfm-explorer window did not appear: {first}"
    );
    match try_attach_app(child) {
        Ok(app) => {
            child.attached();
            app
        }
        Err(second) => panic!(
            "sfm-explorer window did not appear, on the original launch or on \
             one relaunch: {first}; after relaunching: {second}"
        ),
    }
}

/// A process has one accessibility root on all three platforms, and the pid
/// resolves it directly: the AXApplication on macOS, the `application` node
/// AccessKit's Unix adapter registers on Linux, and — since xa11y 0.15 — a
/// synthesized per-process `application` node on Windows, whose children are
/// the process's top-level windows. Before that, a Windows "app" *was* a
/// top-level window and `by_pid` landed on whichever came first, which for this
/// viewer is one of winit's helper windows rather than the UI; the suite
/// matched the window by title to get around it. That is no longer merely
/// unnecessary but wrong — the synthesized node is named after the executable
/// (`sfm-explorer`), not the window (`SfM Explorer`).
///
/// Addressing by pid rather than title also makes the MCP tests' `[MCP :port]`
/// title suffix a non-issue, and picks out *this* viewer when the developer
/// running the suite has one of their own open.
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
    // The attached root is the process, which has no geometry of its own on any
    // of the three platforms — a process is not a rectangle. The window under it
    // is what carries bounds.
    let b = app
        .probe(r#"window"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the viewer's window did not appear")
        .data()
        .bounds
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
        app.probe(&format!(r#"button[name="{menu}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("'{menu}' menu button not found"));
    }
    // `count`, not a short-budget `wait_attached`: one instantaneous resolution
    // rather than a poll loop that snapshots the whole subtree every 100ms for
    // its budget (see this file's module comment).
    //
    // **This depends on the loop above running first.** A single check for
    // absence only means anything once the tree is known to be published, and
    // the four menu buttons — painted in the same menu bar as a View menu would
    // be — are what establishes that. Do not reorder these two.
    assert_eq!(
        app.probe(r#"button[name="View"]"#)
            .count()
            .expect("the tree is queryable"),
        0,
        "the View menu is still in the menu bar"
    );
}

/// The empty-state placeholder text is shown before any file is loaded.
#[test]
fn empty_state_placeholder_text() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    app.probe(r#"static_text[name="No reconstruction loaded."]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("placeholder text 'No reconstruction loaded.' not found");
}

/// Opening the File menu exposes all three items in the accessibility tree.
#[test]
fn file_menu_items() {
    let _guard = Guard::new();
    let app = attach(_guard.child());

    // Opening the menu confirms `Open...` on the way, so that item is already
    // proven present and the loop covers the rest.
    app.probe(r#"button[name="File"]"#)
        .press_revealing(&app.probe(r#"button[name="Open..."]"#), CONTENT_TIMEOUT)
        .expect("File menu item 'Open...' did not appear");

    for item in ["Close All", "Load Demo Data...", "Quit"] {
        app.probe(&format!(r#"button[name="{item}"]"#))
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

    // Opening the menu confirms `Delete Image`, so the loop covers the rest.
    app.probe(r#"button[name="Edit"]"#)
        .press_revealing(
            &app.probe(r#"button[name="Delete Image"]"#),
            CONTENT_TIMEOUT,
        )
        .expect("Edit menu item 'Delete Image' did not appear");

    // Named without their shortcuts, for the reason `file_menu_items` gives:
    // the shortcut is in the button's text and is spelled by the platform.
    for item in ["Cancel Camera Move", "Bundle Adjust..."] {
        app.probe(&format!(r#"button[name="{item}"]"#))
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
/// Demo data came from no file — `--demo` appends a generated node, whose path
/// is `None` exactly as the menu's is — so there is nothing for Save to write
/// over and Save As is the only way out. Both items are nonetheless present:
/// an action that vanishes when it does not apply reads as unimplemented.
/// Presence is
/// all this asserts: which of the two is enabled is not read here, because the
/// accessibility tree does not report egui's disabled state the same way on
/// every platform (Linux never matched an `enabled="false"` selector), and the
/// enabled logic, like what each item does, is covered headlessly in
/// `state/save/tests.rs`.
#[test]
fn file_menu_save_items_apply_to_a_node_that_came_from_no_file() {
    let _guard = Guard::demo();
    let app = attach(_guard.child());

    // The menu opening confirms `Save As...`, which is this test's first
    // assertion; only `Save` is left to look up.
    app.probe(r#"button[name="File"]"#)
        .press_revealing(&app.probe(r#"button[name^="Save As..."]"#), CONTENT_TIMEOUT)
        .expect("File menu item 'Save As...' did not appear");
    app.probe(r#"button[name^="Save "]"#)
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

    app.probe(r#"button[name="File"]"#)
        .press_revealing(&app.probe(r#"button[name="Quit"]"#), CONTENT_TIMEOUT)
        .expect("Quit item did not appear")
        .press()
        .expect("press Quit");

    assert!(
        guard.wait_for_exit(Duration::from_secs(10)),
        "File > Quit did not exit the process"
    );
}

/// Drive File > Load Demo Data… and its dialog to completion.
///
/// Three `Locator` calls and a handful of whole-tree snapshots, so it is
/// **not** how a test gets a scene in front of it — [`Guard::demo`] and
/// [`McpViewer::launch_demo`] ask for the same node on the command line for
/// none at all. This exists for
/// [`the_scene_panel_lists_the_loaded_reconstruction`], the one test that
/// asserts on what the menu route itself produces.
fn load_demo_data(app: &App) {
    app.probe(r#"button[name="File"]"#)
        .press_revealing(
            &app.probe(r#"button[name="Load Demo Data..."]"#),
            CONTENT_TIMEOUT,
        )
        .expect("Load Demo Data item did not appear")
        .press()
        .expect("press Load Demo Data");
    app.probe(r#"button[name="Load"]"#)
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
    let _guard = Guard::demo();
    let app = attach(_guard.child());

    for name in ["Points", "Camera Images", "Grid"] {
        let el = app
            .probe(&format!(r#"check_box[name="{name}"]"#))
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
///
/// **This is the one test that loads demo data through the File menu**, and it
/// keeps [`load_demo_data`] rather than `--demo` on purpose: what it asserts —
/// that the node the load made, and its rows, appear in the Scene panel — is
/// precisely what that menu route is for, so driving it here is what keeps the
/// route covered. Every other test asks for the node on the command line, for
/// the reason this file's module comment gives. Do not "optimize" this one too;
/// nothing else presses Load Demo Data… and then looks at what arrived.
/// (`file_menu_items` asserts the item is *in* the menu, not that it works.)
#[test]
fn the_scene_panel_lists_the_loaded_reconstruction() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    load_demo_data(&app);

    // The demo reconstruction is labeled "demo" and rings the scene with 8
    // images, so both strings are fixed by the fixture.
    for text in ["demo", "Camera Images (8)"] {
        app.probe(&format!(r#"static_text[name="{text}"]"#))
            .wait_attached(CONTENT_TIMEOUT)
            .unwrap_or_else(|_| panic!("Scene panel row '{text}' did not appear"));
    }

    // The row's solo toggle. Its own behaviour is headless (`scene_graph`
    // tests); what only a real window can show is that a *third* glyph button
    // squeezed onto the row is still laid out and still reachable.
    app.probe(r#"button[name="S"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the reconstruction row's solo toggle did not appear");
}

/// Toggling a HUD checkbox via accessibility updates its checked state.
#[test]
fn toggle_hud_layer_checkbox() {
    let _guard = Guard::demo();
    let app = attach(_guard.child());

    let el = app
        .probe(r#"check_box[name="Grid"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("Grid checkbox not found");
    assert!(
        matches!(el.data().states.checked, Some(Toggled::On)),
        "Grid should start checked",
    );

    app.probe(r#"check_box[name="Grid"]"#)
        .toggle()
        .expect("toggle Grid");

    // Wait for egui to process the action and update the tree
    app.probe(r#"check_box[name="Grid"]"#)
        .wait_until(
            |data| data.is_some_and(|d| matches!(d.states.checked, Some(Toggled::Off))),
            CONTENT_TIMEOUT,
        )
        .expect("Grid should be unchecked after toggle");
}

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
        // A window that is behind another is still at these screen coordinates
        // in the accessibility tree, so raising it is part of aiming rather
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

    // The return value is checked because `SendInput` is refused silently: UIPI
    // blocks injection into the session whenever the foreground window belongs
    // to a more privileged process, and the call then inserts nothing and
    // returns 0. Ignoring that turned a machine-state problem into a
    // thirty-second wait for a menu that was never asked for.
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

    let _guard = Guard::demo();
    let pid = _guard.child().id();
    let app = attach(_guard.child());

    // The demo node's row is labelled "demo"; its bounds are screen pixels.
    let row = app
        .probe(r#"static_text[name="demo"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the demo reconstruction row did not appear");
    let bounds = row.data().bounds.expect("the row has no bounds");
    let x = bounds.x + (bounds.width / 2) as i32;
    let y = bounds.y + (bounds.height / 2) as i32;

    // Two moves with a pause: the app repaints on demand, and egui resolves a
    // click against the widget rects of the frame before it.
    for _ in 0..2 {
        aim_at(pid, x, y);
        std::thread::sleep(Duration::from_millis(250));
    }
    // A left click first, which both activates the window (a right-press on an
    // inactive one is swallowed by the activation) and puts the row through the
    // selection change that used to take its menu's identity with it.
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

    // "Close" is deliberately not checked: the window's own title-bar close
    // button carries that name too, so it would pass with no menu at all.
    // Neither are the two submenus, `Align to ▸` and `Tint ▸`: a menu button
    // does not surface under the `button` role here, and their contents only
    // exist once the submenu is opened — both are covered headlessly instead.
    for item in ["Select", "Zoom to Fit"] {
        app.probe(&format!(r#"button[name="{item}"]"#))
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
    /// Put `contents` at `~/.sfm-explorer-default-layout.json`, preserving
    /// whatever was there.
    ///
    /// **The caller must already hold [`UI_TEST_LOCK`]**, and must keep holding
    /// it until this value is dropped — that is [`Guard::with_default_layout`],
    /// which is the only caller. The file is one file: two tests writing it at
    /// once would each launch a viewer on the other's layout, and a restore
    /// outside the lock races the next test's own `rename` of it.
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
/// that the file was read.
#[test]
fn a_saved_default_layout_is_loaded_at_startup() {
    // Launched *without* `--no-default-layout`, unlike every other test here.
    let guard = Guard::with_default_layout(
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
    let app = attach(guard.child());

    app.probe(r#"button[name="Latest"]"#)
        .wait_attached(CONTENT_TIMEOUT)
        .expect("the Action Log toolbar did not appear, so the layout was not loaded");
    // `count`, not a short-budget `wait_attached`, for the reason
    // `the_menu_bar_holds_file_edit_go_and_panels` gives: one resolution rather
    // than a poll loop of whole-subtree snapshots.
    //
    // **This depends on the "Latest" lookup above running first.** That lookup
    // is what establishes the dock has been built and its tree published; only
    // then does finding no placeholder mean the 3D viewer is not docked, rather
    // than that nothing is in the tree yet. Do not reorder these two.
    assert_eq!(
        app.probe(r#"static_text[name="No reconstruction loaded."]"#)
            .count()
            .expect("the tree is queryable"),
        0,
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
    // Launched *without* `--no-default-layout`, so the file above is read, and
    // with `--demo`, so the node is there before the first query.
    let guard = Guard::with_default_layout(
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
    let app = attach(guard.child());

    // The demo node is labeled "demo" and has been through no edit, so both
    // strings are fixed by the fixture.
    for text in [
        "1 version",
        "demo has not been edited; its one version is the file as it was opened.",
    ] {
        app.probe(&format!(r#"static_text[name="{text}"]"#))
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
        McpViewer::launched(&[])
    }

    /// The same, with the demo reconstruction already loaded — the command-line
    /// form of [`load_demo_data`], for the reason this file's module comment
    /// gives.
    fn launch_demo() -> McpViewer {
        McpViewer::launched(&["--demo"])
    }

    fn launched(extra: &[&str]) -> McpViewer {
        let _lock = ui_test_lock();
        // This path spawns the viewer itself rather than going through
        // `Guard::launched`, so it starts the same accounting by hand — and
        // before the spawn, so the wait for the endpoint line below is
        // charged to the launch, which is what it is.
        let started = begin_accounting();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sfm-explorer"));
        cmd.args(["--mcp", "0", "--no-default-layout"]);
        cmd.args(extra);
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
                started,
                launch: Cell::new(None),
                child: RefCell::new(child),
                args: None,
                _layout_file: None,
                _lock,
            },
            address,
        }
    }

    /// Wait for the window to exist, since MCP commands are applied inside a
    /// frame and a viewer with no window yet renders none.
    ///
    /// The same pid-addressed attach every other test makes, which is what
    /// keeps the assertions on *this* viewer: a developer running this suite
    /// very likely has one of their own open, that being the whole point of the
    /// surface under test, and every interaction would otherwise land silently
    /// in the wrong window. [`attach`]'s relaunch declines here — this viewer's
    /// [`Guard::args`] is `None`, a respawn being a viewer on a port nothing is
    /// listening to — so a stuck launch fails with the original diagnosis.
    fn wait_for_window(&self) -> App {
        attach(self.guard.child())
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
    // The render target only exists once the viewport has something to draw:
    // with nothing loaded the panel shows its empty state and never sizes one.
    let viewer = McpViewer::launch_demo();
    viewer.wait_for_window();
    viewer.initialize();

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

    // Behind Point Track in the stock grid.
    let behind = viewer.call(
        "screenshot",
        serde_json::json!({ "panel_name": "camera_intrinsics" }),
    );
    assert_eq!(behind["isError"], serde_json::Value::Bool(true), "{behind}");
    let message = behind["content"][0]["text"].as_str().expect("a refusal");
    assert!(
        message.contains("Point Track") && message.contains("show_panel"),
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
    let viewer = McpViewer::launch_demo();
    viewer.wait_for_window();
    viewer.initialize();
    // The node is appended before the window opens, but `get_scene` is answered
    // inside a frame, so the first call waits for it rather than racing it.
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
    let viewer = McpViewer::launch_demo();
    let app = viewer.wait_for_window();
    viewer.initialize();

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
    // it to find a second item would close it instead. That is also why the
    // press is confirmed against the item the caller is about to want rather
    // than retried on failure -- `Probe::press_revealing` carries the whole
    // argument, and this is the call site that made it necessary.
    // Hands back the item it confirmed, so the caller does not resolve the same
    // selector a second time -- one snapshot per menu opening rather than two.
    let open_edit_menu = |expected: &str| {
        app.probe(r#"button[name="Edit"]"#)
            .press_revealing(
                &app.probe(&format!(r#"button[name^="{expected}"]"#)),
                CONTENT_TIMEOUT,
            )
            .unwrap_or_else(|_| panic!("Edit menu item '{expected}' did not appear"))
    };
    // Matched on a name *prefix*, for the reason `edit_menu_items` gives: both
    // spellings of the Move Camera item carry the `M` shortcut in the button's
    // text, and the shortcut is spelled by the platform.
    let edit_item = |item: &str| {
        app.probe(&format!(r#"button[name^="{item}"]"#))
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

    open_edit_menu("Move Camera")
        .press()
        .expect("press Edit menu item 'Move Camera'");
    wait_for_line("Moving the camera of image_000.jpg");

    // One opening for both: with a lock held the same entry commits rather
    // than takes, and the entry beside it gives the camera back. Opening on
    // `Commit Camera Move` is therefore this half's first assertion -- that the
    // held lock changed what the entry says -- and the element it hands back is
    // deliberately dropped, because the item this test goes on to press is the
    // one beside it.
    open_edit_menu("Commit Camera Move");
    edit_item("Cancel Camera Move")
        .press()
        .expect("press Edit menu item 'Cancel Camera Move'");
    wait_for_line("Cancelled the camera move of image_000.jpg");
}
