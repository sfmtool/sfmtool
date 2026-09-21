// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(windows, target_os = "macos", target_os = "linux"))]

//! The viewer's windowed tests: what only a real window, on a real desktop,
//! can be asked.
//!
//! **One locator resolution is one full snapshot of the viewer's accessibility
//! subtree**, and that is what shapes this file. `wait_attached`, `press`,
//! `toggle` and `elements` each walk the whole tree, so the cost is per
//! *operation the tests ask for*, not per launch, and it is the platform's,
//! not the viewer's, and it differs between them by orders of magnitude:
//! around 0.1s on a developer's Windows machine, against 3.8s for
//! a launch, attach and teardown on a GitHub-hosted Windows runner. An
//! operation is that request, not a walk: one that polls for its condition or
//! retries a transient failure spends several walks and is still counted once,
//! which is what keeps `ops` a fingerprint of the tests rather than of the run
//! (see [`measured`]).
//!
//! Three habits follow. Every locator is rooted at the viewer's **window**
//! rather than at the process — see [`Attached`], which resolves that window
//! once per launch and is what every test holds. Setup goes through the
//! **command line** rather than the
//! accessibility API — `--demo` in place of driving File > Load Demo Data… and
//! its dialog, which is three `Locator` calls and more snapshots than that,
//! since the menu item and the dialog's button each appear a poll or two after
//! the press that makes them. And a run of consecutive read-only assertions is
//! asked as *one* snapshot rather than one apiece: [`Attached::wait_all`] polls
//! a comma-separated selector group, and then answers every one of the caller's
//! expectations off the elements it was handed — in memory, for nothing.
//! Assertions of *absence* ride
//! along:
//! something the group names and the snapshot does not hold is absent in the
//! same observation that proved the rest present. (Only a `Locator` method
//! resolves; a `press` on an `Element` a lookup already handed back invokes
//! what it holds.) What a snapshot must never outlive is an **interaction**:
//! egui rebuilds its accessibility tree every frame, so a press or a toggle
//! leaves every handle in one stale, and the assertions after it need a fresh
//! one. Exactly one test still drives each route a shortcut replaces, and says
//! so.
//!
//! Because that is where the cost is, **the suite reports its own**: every
//! test prints a `UIPROBE` line as its [`Guard`] drops, which is how a reader
//! of one CI log tells a slow runner apart from an expensive suite. See
//! [`Guard::report`] for the fields and [`Attached::probe`] for what is
//! counted.

use std::cell::{Cell, OnceCell, RefCell};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, Once};
use std::time::{Duration, Instant};

use xa11y::{App, AppExt, Element, ElementData, Locator, Toggled, TreeNode};

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
/// Calls into the accessibility API run since then, and the nanoseconds spent
/// inside them — the *platform's* share of `OP_NANOS`. See [`walked`].
static WALKS: AtomicU64 = AtomicU64::new(0);
static WALK_NANOS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds spent finding the window every locator is rooted at, which is
/// neither of the above and is reported on its own. See [`Attached::window`].
static WINDOW_NANOS: AtomicU64 = AtomicU64::new(0);
/// The same five, plus launch time and wall time, accumulated over every test
/// this process has run, for the `UIPROBE TOTAL` line.
static TOTAL_TESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LAUNCH_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OPS: AtomicU64 = AtomicU64::new(0);
static TOTAL_OP_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WALKS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WALK_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WINDOW_NANOS: AtomicU64 = AtomicU64::new(0);
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
    WALKS.store(0, Ordering::Relaxed);
    WALK_NANOS.store(0, Ordering::Relaxed);
    WINDOW_NANOS.store(0, Ordering::Relaxed);
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

/// Run one call into the accessibility API, counting it and timing it.
///
/// **A different counter from [`measured`], asking a different question, and
/// the two are deliberately not nested one-to-one.** An `op` is what a *test*
/// asked for and must stay a fingerprint of the suite's shape; a `walk` is what
/// the *platform* was actually asked to do to service it, and moves with the
/// run. One op is one or many walks — a poll that waits three ticks for a
/// widget walks three times, a [`Probe::press_revealing`] that has to press
/// twice walks four — so `walks` is never a second spelling of `ops`, and a
/// reader who wants suite shape reads `ops` and a reader who wants platform
/// traffic reads `walks`.
///
/// What this buys is the one thing `op_ms` cannot say: where an operation's
/// time went. Bracketing only the call, never the `sleep` between two of them,
/// makes `walk_ms` the platform's share and `op_ms − walk_ms` the suite's own
/// waiting — which is the difference between a tree query that is expensive and
/// an app that is slow to draw, two problems with nothing in common and no
/// shared fix.
///
/// **The split is only exact where this file owns the loop**, which is
/// [`Probed::wait_all`]. `Locator::wait_attached` and `Locator::wait_until`
/// poll inside xa11y, so a walk around one of those is a whole wait, its
/// sleeps included, and reads as platform time that partly is not. They are
/// still counted, because a call that is not counted at all is worse, but the
/// clean read of `op_ms − walk_ms` is the one taken over a `wait_all`.
fn walked<T>(query: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = query();
    WALKS.fetch_add(1, Ordering::Relaxed);
    WALK_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
/// One method call here is one resolution *request* — the unit of work a test
/// asks for, which the platform may service with more than one whole-subtree
/// walk when it polls or retries — so this is the single place the suite's
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
        measured(|| {
            retrying_transient("wait_attached", || walked(|| self.0.wait_attached(timeout)))
        })
    }

    fn wait_until(
        &self,
        predicate: impl Fn(Option<&ElementData>) -> bool,
        timeout: Duration,
    ) -> xa11y::Result<Option<Element>> {
        measured(|| {
            retrying_transient("wait_until", || {
                walked(|| self.0.wait_until(&predicate, timeout))
            })
        })
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
    /// it.** The confirmation is a full subtree snapshot — the dearest thing
    /// this suite does on the Windows runner — so handing it back is what keeps
    /// this guard free:
    /// a caller that re-resolved the same selector afterwards would pay for the
    /// same snapshot twice, which on the job this suite is trying to shrink is
    /// the wrong trade. Every site that wants the revealed item immediately
    /// therefore takes it from here rather than looking it up again, and the
    /// guard costs those sites nothing at all.
    fn press_revealing(&self, revealed: &Probe, timeout: Duration) -> xa11y::Result<Element> {
        measured(|| {
            let first = walked(|| self.0.press());
            if let Ok(element) = retrying_transient("press_revealing/confirm", || {
                walked(|| revealed.0.wait_attached(timeout))
            }) {
                // It opened. A transient error from the press was a report
                // about the call, not about the app.
                return Ok(element);
            }
            // It did not open, so a press that also failed is the better
            // diagnosis than anything a second attempt would produce.
            first?;
            println!("UIPROBE RETRY op=press_revealing attempt=2 of 2 (nothing was revealed)");
            walked(|| self.0.press())?;
            // The second press is not re-confirmed here: the caller's own next
            // lookup is the confirmation, and it fails with the selector it
            // actually wanted rather than with this one.
            walked(|| revealed.0.wait_attached(timeout))
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
        measured(|| walked(|| self.0.toggle()))
    }
}

/// One thing a snapshot is asked to say: that a node of this role, under this
/// exact name, is in the tree — or, for [`Self::absent`], that none is.
///
/// Role and name rather than a free-form selector because this type has to do
/// two things with the same statement and they must not drift apart: produce
/// the clause that goes into the group [`Probed::wait_all`] resolves, and
/// decide, in memory, whether an element that came back satisfies it. Writing
/// the match by hand against a selector written separately would be two
/// spellings of one rule. Every selector the collapsed clusters use is
/// `role[name="…"]`, so that is the whole shape offered; a site that needs
/// more — a prefix match, a state filter — keeps its own [`Probe`] call.
#[derive(Clone, Copy)]
struct Expect<'a> {
    role: &'a str,
    name: &'a str,
    /// What the snapshot must say: `true` that the node is there, `false` that
    /// it is not.
    present: bool,
}

impl<'a> Expect<'a> {
    /// A node that must be in the tree.
    fn present(role: &'a str, name: &'a str) -> Self {
        Expect {
            role,
            name,
            present: true,
        }
    }

    /// A node that must *not* be in the tree.
    ///
    /// Sound in a [`Probed::wait_all`] group and nowhere cheaper, because an
    /// absence only means something once the tree is known to be published: an
    /// empty tree satisfies every absence trivially. Grouped with the
    /// expectations that prove the tree is there, it is read off the very
    /// snapshot that proved it, so there is no ordering left to respect.
    fn absent(role: &'a str, name: &'a str) -> Self {
        Expect {
            role,
            name,
            present: false,
        }
    }

    /// This expectation as one clause of a selector group.
    fn clause(&self) -> String {
        format!(r#"{}[name="{}"]"#, self.role, self.name)
    }

    /// Whether `data` is the node this names — the in-memory twin of what the
    /// clause above asks the platform, which for a `role[name="…"]` selector is
    /// an exact, case-sensitive name against a normalized role.
    fn matches(&self, data: &ElementData) -> bool {
        data.role.to_snake_case() == self.role && data.name.as_deref() == Some(self.name)
    }

    /// How a failure names this expectation, given what the snapshot said.
    fn unmet(&self) -> String {
        let clause = self.clause();
        if self.present {
            format!("{clause} never appeared")
        } else {
            format!("{clause} is still there")
        }
    }
}

/// The elements one whole-subtree snapshot handed back, kept so a caller can
/// read an element it has already paid for.
///
/// Every `Element` in here carries its own already-populated `ElementData`, so
/// [`Self::first`] and the `data()` behind it are field reads rather than
/// cross-process calls: a snapshot answers arbitrarily many questions for the
/// price of the one resolution that made it.
///
/// **It goes stale the moment the app is touched.** egui republishes its
/// accessibility tree every frame, so a press, a toggle or a synthetic click
/// invalidates every handle here — see [`Probe::press_revealing`], which is
/// where that failure mode is spelled out. Read a snapshot before the next
/// interaction or take a fresh one after it.
struct Snapshot(Vec<Element>);

impl Snapshot {
    /// The first element matching `expect`.
    ///
    /// Infallible for an expectation the [`Probed::wait_all`] that produced
    /// this snapshot returned successfully on — that is exactly what it waited
    /// to be true — so a miss here is a caller asking about something it never
    /// asked for, and panics saying so.
    fn first(&self, expect: &Expect<'_>) -> &Element {
        self.0
            .iter()
            .find(|element| expect.matches(element.data()))
            .unwrap_or_else(|| panic!("{} is not in this snapshot", expect.clause()))
    }
}

/// The attached viewer: its process root, and — once something asks for it —
/// its window.
///
/// [`attach`] hands one of these back and every test holds one, because the
/// window is what the suite's locators are rooted at.
///
/// **Rooting at the window instead of the process is this suite's largest
/// lever on Windows**, and the reason is a guard inside xa11y. `App::by_pid`
/// there hands back a *synthesized* per-process `application` node, which has
/// no live UI Automation element behind it — so a search scoped to it cannot be
/// answered by UIA's own subtree query and falls back to a generic descent that
/// fetches every node's properties one cross-process call at a time, **once per
/// clause of the selector**. A top-level window is a real HWND-backed element,
/// so the same search becomes one `FindAllBuildCache(TreeScope_Subtree)` that
/// fetches the whole subtree in one COM call and evaluates every clause against
/// it in a single pass. Measured on Windows 11 against the empty-state tree (56
/// nodes): process-rooted, one clause ~0.22s and a five-clause group ~0.88s, a
/// 4.0x multiplier; window-rooted, ~0.10s and ~0.10s, a multiplier of 1.0. The
/// clause count stops mattering, and what remains is roughly halved.
///
/// Scoping to the window loses nothing to look at. The viewer runs a single
/// egui viewport, so its menus and popups are painted inside that one HWND
/// rather than in native popup windows, and winit's helper windows never
/// register with AccessKit. The synthesized application node's only child is
/// the window.
///
/// macOS and Linux answer through the generic descent whatever the root is, so
/// there the change is simply a smaller subtree — with one consequence the
/// suite has to respect: that descent matches only *descendants* of its root,
/// never the root itself. A window-rooted `window` selector therefore matches
/// on Windows (`TreeScope_Subtree` includes the element it is scoped to) and
/// not on the other two, so the one test that asks about the window node uses
/// [`Self::app_probe`].
struct Attached {
    app: App,
    /// The viewer's window, resolved by [`Self::window`] the first time
    /// anything roots a search at it, and held for the life of the test.
    ///
    /// **A snapshot that deliberately outlives the interactions the tests
    /// make**, which every other snapshot in this file must not — and it is
    /// sound for a reason none of those share. What goes stale across an egui
    /// frame is a *widget* node, republished by AccessKit every frame; a
    /// top-level window is not one. On Windows this handle resolves to an
    /// element acquired from the HWND, which outlives any number of tree
    /// rebuilds; on Linux it is the AT-SPI object path AccessKit's adapter
    /// registers for the window, and on macOS the AXWindow. All three last as
    /// long as the window does, which is as long as the [`Guard`] that owns the
    /// process.
    window: OnceCell<Element>,
}

impl Attached {
    /// The window every [`Self::probe`] is rooted at, found on first use.
    ///
    /// **Deferred rather than resolved at attach, because finding it is not
    /// cheap and five of this suite's nineteen launches never need it.** Four
    /// of the five MCP tests speak only HTTP to the viewer's own endpoint, and
    /// [`window_min_size`] asks the *process* about the window node; none of
    /// those roots a search anywhere. (The fifth MCP test,
    /// `taking_a_camera_in_hand_reaches_a_real_window` — a code span rather
    /// than a link because it is Windows-only — drives the Edit menu through
    /// the tree and does.) Resolving eagerly charged all nineteen —
    /// ~0.2s each on a developer's Windows machine and, on a GitHub-hosted
    /// Windows runner, around 10s.
    ///
    /// That is not a walk and no limit reaches it. `App::windows` is one
    /// provider call that materializes *every* top-level window of the process
    /// before anything can look at one, so the cheaper-looking alternatives are
    /// the same call underneath: measured on Windows 11, `App::windows`,
    /// `App::children`, and `app.locator("window").first().element()` — which
    /// does push a limit down into the walk — cost 68, 69 and 67ms, within
    /// noise of each other. What makes it dear on a runner is what it *is*: a
    /// desktop-wide `FindAllBuildCache` filtered to this pid, then, per window,
    /// re-acquiring it from its HWND (which is where AccessKit's UIA provider
    /// gets activated), a cache build and a property read — every one of them a
    /// cross-process call on a machine that charges hundreds of milliseconds
    /// for one. `App::by_pid`, which answers with `FindFirstBuildCache` and no
    /// re-acquisition, costs 10ms against those 68.
    ///
    /// It is reported as its own `window_ms` rather than folded into
    /// `launch_ms` or `op_ms`: it is neither the machine's speed nor what a
    /// test asked to look at, and a run where it dominates should say so in
    /// the field that names it. See [`Guard::report`].
    ///
    /// Panics if the window never appears, which for a viewer whose process is
    /// already attached is a viewer that never drew one.
    fn window(&self) -> &Element {
        self.window.get_or_init(|| {
            let started = Instant::now();
            let found = window_of(&self.app, ATTACH_TIMEOUT);
            WINDOW_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            found.unwrap_or_else(|e| panic!("{e}"))
        })
    }

    /// `attached.probe(selector)` in place of `app.locator(selector)`: the same
    /// lookup, rooted at the window, counted and timed.
    ///
    /// The method is named `probe` rather than `locator` so that it cannot be
    /// confused with `App::locator`, which is the *process*-rooted form and the
    /// one this suite is trying not to use — see the note on [`Attached`] for
    /// what that costs. The only deliberate use of it is [`Self::app_probe`].
    fn probe(&self, selector: &str) -> Probe {
        let window = self.window();
        Probe(Locator::new(
            std::sync::Arc::clone(window.provider()),
            Some(window.data().clone()),
            selector,
        ))
    }

    /// The same, rooted at the **process** rather than at the window.
    ///
    /// For the one question a window-rooted search cannot answer portably: one
    /// about the window node itself. The generic descent macOS and Linux take
    /// matches only descendants of its root, so a window-rooted `window`
    /// selector finds nothing there — while on Windows the subtree query does
    /// include its own root and it finds the window. Asking the process instead
    /// gives one answer on all three.
    fn app_probe(&self, selector: &str) -> Probe {
        Probe(self.app.locator(selector))
    }

    /// Wait until every one of `expectations` holds in a **single** snapshot,
    /// and hand that snapshot back.
    ///
    /// This is the suite's main lever on its own cost. A cluster of consecutive
    /// read-only assertions used to be one resolution each — one whole-subtree
    /// walk per assertion, around 25s apiece on the Windows runner — even
    /// though the tree they were asking about was the same tree. Here the
    /// clauses are joined into one selector *group*, which `Locator::elements`
    /// resolves together, and each expectation is then decided against the
    /// `ElementData` already in hand. A five-assertion cluster costs one
    /// operation instead of five.
    ///
    /// **One operation is not the same thing as one walk**: a tick that does
    /// not yet satisfy every expectation sleeps and resolves the group again,
    /// and the `walks` counter is what makes that visible in a log; see
    /// [`walked`]. What one walk costs no longer depends on how many clauses
    /// are in the group, which it did for as long as the search was rooted at
    /// the process — see [`Attached`] for the measurements and the mechanism.
    ///
    /// The waiting is what makes the substitution honest. `elements` on its own
    /// resolves once and returns, so swapping it in for a `wait_attached` would
    /// trade the cost for flakiness on a slow runner, where a widget routinely
    /// lands a poll or two after the query that wants it. So this polls — one
    /// whole snapshot per tick, every expectation checked against it — and
    /// returns the first tick on which they all hold, which for a healthy app
    /// is the first one.
    ///
    /// Counted as **one** op however many ticks that takes, and however many
    /// times [`retrying_transient`] re-runs it: see [`measured`]. Retrying is
    /// safe here for the reason it is safe for the other read-only probes and
    /// for no press — resolving a group twice changes nothing.
    ///
    /// Panics rather than erroring on a list that names nothing to *find* —
    /// an empty one, which would build an unparsable empty selector, or one
    /// made only of [`Expect::absent`], which every tick of an empty tree
    /// satisfies and which would therefore return before the app had drawn
    /// anything. An absence is only an assertion in the company of the
    /// presences that prove the tree is published.
    fn wait_all(&self, expectations: &[Expect<'_>], timeout: Duration) -> xa11y::Result<Snapshot> {
        assert!(
            expectations.iter().any(|expect| expect.present),
            "wait_all needs at least one node to wait *for*"
        );
        // One group, so one tick: `Locator::elements` parses the commas into
        // clauses and resolves them together, returning every match in document
        // order. The absent clauses belong in it as much as the present ones —
        // they are how the snapshot is asked about the node that must not be
        // there.
        let group = expectations
            .iter()
            .map(|expect| expect.clause())
            .collect::<Vec<_>>()
            .join(", ");
        let locator = self.probe(&group).0;
        measured(|| {
            retrying_transient("wait_all", || {
                let started = Instant::now();
                loop {
                    let found = walked(|| locator.elements())?;
                    let unmet: Vec<String> = expectations
                        .iter()
                        .filter(|expect| {
                            expect.present
                                != found.iter().any(|element| expect.matches(element.data()))
                        })
                        .map(Expect::unmet)
                        .collect();
                    if unmet.is_empty() {
                        return Ok(Snapshot(found));
                    }
                    let elapsed = started.elapsed();
                    if elapsed >= timeout {
                        // The diagnosis carries both halves of the answer: what
                        // the snapshot was missing, and what it did hold — the
                        // near-miss list a single-selector timeout would have
                        // had to go and build separately.
                        return Err(xa11y::Error::Timeout {
                            elapsed,
                            diagnosis: Some(Box::new(
                                xa11y::Diagnosis::new()
                                    .condition("every expectation in one snapshot")
                                    .selector(group.clone())
                                    .last_observed(unmet.join("; "))
                                    .candidates(found.iter().map(|element| {
                                        let data = element.data();
                                        format!("{} {:?}", data.role, data.name)
                                    })),
                            )),
                        });
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
        })
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
    /// UIPROBE test=file_menu_items launch_ms=675 window_ms=88 ops=2 op_ms=2176 walks=4 walk_ms=2088 total_ms=2951
    /// UIPROBE TOTAL tests=19 launch_ms=16960 window_ms=1320 ops=24 op_ms=22964 walks=61 walk_ms=21730 total_ms=44699 mean_launch_ms=892 mean_op_ms=956 mean_walk_ms=356
    /// ```
    ///
    /// `launch_ms` is the runner-speed yardstick: spawning a process, waiting
    /// for the GPU and for the OS to publish an accessibility root is work the
    /// suite cannot make cheaper, so `mean_launch_ms` moving between two runs
    /// means the *machine* moved.
    ///
    /// `window_ms` is finding the window the locators are rooted at, which is
    /// none of the other three: not the machine's speed, not a tree walk, and
    /// not something a test asked to look at. It is its own field because it
    /// is paid at most once per launch and only by a test that roots a search
    /// — five of nineteen do not — so folding it into `launch_ms` would make
    /// the yardstick move for a reason the machine did not, and folding it into
    /// `op_ms` would charge one test for what every later question reuses. See
    /// [`Attached::window`].
    ///
    /// `ops` and `mean_op_ms` are the suite's own cost:
    /// `ops` is how many *resolution requests* the tests made — units of work
    /// asked for, which only a change to the tests moves — and `mean_op_ms` is
    /// what the platform charged for one. It is not a snapshot count: servicing
    /// one request may take the platform several whole-subtree walks, when a
    /// wait polls for its condition or a transient failure is retried.
    ///
    /// `walks` is that snapshot count, and `walk_ms` the time inside those
    /// calls and nothing else — so it is what splits a slow `op_ms` into the
    /// two unrelated things it adds together. `walk_ms` near `op_ms` says the
    /// platform's tree query is what costs; a gap says the suite was asleep in
    /// a poll loop waiting for the app to draw, and no amount of asking for
    /// fewer operations would have helped. `mean_walk_ms` is the price of one
    /// tree query, which against a tree size (see [`window_appears`], which
    /// prints one) gives a per-node figure comparable across platforms. See
    /// [`walked`] for which calls are counted and where the split is exact.
    ///
    /// Comparing two logs, then: `ops` down is a cheaper suite,
    /// `mean_launch_ms` and `mean_walk_ms` down together is a faster machine,
    /// and `total_ms` alone says nothing about which happened.
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
        let walks = WALKS.load(Ordering::Relaxed);
        let walk_nanos = WALK_NANOS.load(Ordering::Relaxed);
        let window_nanos = WINDOW_NANOS.load(Ordering::Relaxed);

        let tests = TOTAL_TESTS.fetch_add(1, Ordering::Relaxed) + 1;
        let launch_total = TOTAL_LAUNCH_NANOS
            .fetch_add(launch.as_nanos() as u64, Ordering::Relaxed)
            + launch.as_nanos() as u64;
        let ops_total = TOTAL_OPS.fetch_add(ops, Ordering::Relaxed) + ops;
        let op_nanos_total = TOTAL_OP_NANOS.fetch_add(op_nanos, Ordering::Relaxed) + op_nanos;
        let walks_total = TOTAL_WALKS.fetch_add(walks, Ordering::Relaxed) + walks;
        let walk_nanos_total =
            TOTAL_WALK_NANOS.fetch_add(walk_nanos, Ordering::Relaxed) + walk_nanos;
        let window_nanos_total =
            TOTAL_WINDOW_NANOS.fetch_add(window_nanos, Ordering::Relaxed) + window_nanos;
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
            "\nUIPROBE test={name} launch_ms={} window_ms={} ops={ops} op_ms={} walks={walks} \
             walk_ms={} total_ms={}",
            launch.as_millis(),
            ms(window_nanos),
            ms(op_nanos),
            ms(walk_nanos),
            total.as_millis(),
        );
        println!(
            "UIPROBE TOTAL tests={tests} launch_ms={} window_ms={} ops={ops_total} op_ms={} \
             walks={walks_total} walk_ms={} total_ms={} mean_launch_ms={} mean_op_ms={} \
             mean_walk_ms={}",
            ms(launch_total),
            ms(window_nanos_total),
            ms(op_nanos_total),
            ms(walk_nanos_total),
            ms(nanos_total),
            ms(launch_total) / tests,
            // `checked_div`: a suite filtered down to tests that resolve
            // nothing reports no mean rather than dividing by zero.
            ms(op_nanos_total).checked_div(ops_total).unwrap_or(0),
            ms(walk_nanos_total).checked_div(walks_total).unwrap_or(0),
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
/// Those three predate [`try_attach`]'s move to pid addressing, and the
/// first of the two shapes is the one xa11y 0.15 says pid attachment closes:
/// enumerating applications skips a window that has not named itself yet, which
/// is exactly a viewer mid-launch. The second comes out of the automation
/// client rather than out of discovery, so it is not addressed, and the retry
/// stays until a few hundred more runs say it can go.
///
/// **This resolves the process and nothing else.** The window the suite's
/// locators are rooted at is found on first use instead, by
/// [`Attached::window`], which says why. So `launch_ms` measures what it always
/// did — a spawn, a GPU, and the OS publishing an accessibility root — and a
/// viewer that attaches but never draws a window fails in the test's first
/// probe rather than in a relaunch here.
fn attach(child: ChildHandle<'_>) -> Attached {
    init();
    let first = match try_attach(child) {
        Ok(attached) => {
            child.attached();
            return attached;
        }
        Err(e) => e,
    };
    assert!(
        child.relaunch(),
        "sfm-explorer window did not appear: {first}"
    );
    match try_attach(child) {
        Ok(attached) => {
            child.attached();
            attached
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
fn try_attach(child: ChildHandle<'_>) -> Result<Attached, String> {
    let app = App::by_pid(child.id(), ATTACH_TIMEOUT).map_err(|e| format!("{e:?}"))?;
    Ok(Attached {
        app,
        window: OnceCell::new(),
    })
}

/// The viewer's window, waiting for the OS to register it.
///
/// `App::windows` is a question about the process's *top-level* windows rather
/// than a tree walk — on Windows one enumeration of the desktop's windows
/// filtered to this pid, on the other two the application node's children
/// filtered to windows and dialogs. What it costs, and why nothing cheaper
/// answers the same question, is on [`Attached::window`], which is this
/// function's only caller and where the result is cached.
///
/// The first window is the one, and there is only ever one: the viewer runs a
/// single egui viewport, so the provider reports exactly one child here
/// (verified against the live tree — winit's helper windows are not desktop
/// children of window control type and do not appear). A `rfd` file dialog
/// would be a second top-level window, but no test opens one, and
/// `press_revealing` presses the File menu rather than the `Open...` in it.
///
/// Polling rather than asking once, because the application node can exist
/// before its window does — most visibly on macOS, where the AXApplication
/// registers at launch and the window follows. On Windows it never waits:
/// `App::by_pid` resolves through a desktop child of window control type, so
/// [`attach`] returning at all means a window is already there.
fn window_of(app: &App, budget: Duration) -> Result<Element, String> {
    let deadline = Instant::now() + budget;
    loop {
        let last = match app.windows() {
            Ok(mut windows) if !windows.is_empty() => return Ok(windows.remove(0)),
            Ok(_) => "the process has no top-level window yet".to_string(),
            Err(e) => format!("{e:?}"),
        };
        if Instant::now() >= deadline {
            return Err(format!("no window under the sfm-explorer app node: {last}"));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

// --- Window-level tests ---

/// App process appears in the accessibility tree.
#[test]
fn window_appears() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    report_tree_size(&app);
}

/// Print the `UIPROBE TREE` line: how big the tree a walk crosses actually is.
///
/// ```text
/// UIPROBE TREE nodes=214 depth=11 walk_ms=478
/// ```
///
/// `mean_walk_ms` says what one tree query costs and says nothing about why,
/// because the two candidate whys — a tree with a lot of nodes in it, and a
/// platform that is slow per node — are indistinguishable without the
/// denominator. This is the denominator: `walk_ms / nodes` is the per-node
/// price, and that is the number that compares across three platforms whose
/// trees are the same shape and whose walk costs differ by orders of
/// magnitude.
///
/// Measured here, in the one test that otherwise resolves nothing, so the
/// figure costs the suite one walk per run rather than one per test — and so
/// that it is taken against the viewer's *empty* state, which every test that
/// does not load a scene is also walking. It is deliberately not a test of its
/// own: another `Guard` is another launch, which is the expensive half of this
/// suite.
///
/// `nodes` and `walk_ms` come from resolving the universal selector, which is
/// the same `Locator::elements` call [`Attached::wait_all`] makes and therefore
/// prices the same work; `depth` comes from a second, recursive descent, since
/// a flat match list has no shape. The depth walk is not what `walk_ms`
/// reports, and is not counted as a [`walked`] call, because per-node
/// `get_children` and a single group resolution are not the same query and
/// averaging them together would blur exactly the number this line exists to
/// sharpen.
///
/// **Rooted at the window, like every other resolution the suite makes** — see
/// [`Attached`]. A denominator taken against the process root would price a
/// query no test makes any more, and on Windows an order of magnitude dearer
/// one, so `walk_ms / nodes` would not be the per-node cost of anything. The
/// node count is the window's subtree; Windows includes the window itself in
/// that (its subtree query is scoped inclusively) and the other two do not, a
/// one-node difference that does not move a per-node price.
fn report_tree_size(app: &Attached) {
    let started = Instant::now();
    let nodes = walked(|| app.probe("*").0.elements()).map(|found| found.len());
    let walk = started.elapsed();
    let depth = app.window().tree(None).map(|root| node_depth(&root));
    match (nodes, depth) {
        (Ok(nodes), Ok(depth)) => println!(
            "\nUIPROBE TREE nodes={nodes} depth={depth} walk_ms={}",
            walk.as_millis()
        ),
        // Not an assertion: this line is a measurement the suite reports, and
        // a test named for whether the window appears must not start failing
        // over the shape of the tree inside it.
        (nodes, depth) => println!(
            "\nUIPROBE TREE unavailable: nodes={nodes:?} depth={depth:?} walk_ms={}",
            walk.as_millis()
        ),
    }
}

/// How many levels `node` spans, counting itself as one.
fn node_depth(node: &TreeNode) -> usize {
    1 + node.children.iter().map(node_depth).max().unwrap_or(0)
}

/// The window respects the 800×600 minimum size constraint.
#[test]
fn window_min_size() {
    let _guard = Guard::new();
    let app = attach(_guard.child());
    // The attached root is the process, which has no geometry of its own on any
    // of the three platforms — a process is not a rectangle. The window under it
    // is what carries bounds.
    //
    // The one process-rooted lookup left in the suite, and it has to be: a
    // search scoped to the window matches the window's descendants, and on two
    // of the three platforms not the window. See `Attached::app_probe`.
    let b = app
        .app_probe(r#"window"#)
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
    // Five assertions, one snapshot. The absence of a View menu is the reason
    // they are grouped rather than merely the thing that makes it cheap: an
    // absence means nothing against a tree that has not been published yet, so
    // it needs corroboration that the menu bar is really there — and the four
    // buttons painted in that same bar are exactly that corroboration. Read off
    // one observation, the corroboration is structural: the snapshot that finds
    // no View menu is by construction the snapshot that found the other four.
    app.wait_all(
        &[
            Expect::present("button", "File"),
            Expect::present("button", "Edit"),
            Expect::present("button", "Go"),
            Expect::present("button", "Panels"),
            Expect::absent("button", "View"),
        ],
        CONTENT_TIMEOUT,
    )
    .expect("the menu bar does not hold exactly File, Edit, Go and Panels");
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

    // The rest in one snapshot: the menu is open and nothing below touches it,
    // so all three are questions about the same tree.
    app.wait_all(
        &[
            Expect::present("button", "Close All"),
            Expect::present("button", "Load Demo Data..."),
            Expect::present("button", "Quit"),
        ],
        CONTENT_TIMEOUT,
    )
    .expect("the File menu is missing an item");
}

/// The Edit menu opens, and its items are in the tree even when every one of
/// them is greyed.
///
/// With nothing loaded there is no node to undo in and nothing selected to
/// delete or move, so every item is disabled -- which is the state this asserts
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

    // Named without its shortcut, for the reason `file_menu_items` gives: the
    // shortcut is in the button's text and is spelled by the platform.
    app.wait_all(
        &[Expect::present("button", "Cancel Camera Move")],
        CONTENT_TIMEOUT,
    )
    .expect("the Edit menu is missing an item");
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
fn load_demo_data(app: &Attached) {
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

    // One snapshot for all three, presence and checked state alike: nothing
    // here presses anything, so the states are read off the same observation
    // that found the boxes.
    let layers = ["Points", "Camera Images", "Grid"].map(|name| Expect::present("check_box", name));
    let hud = app
        .wait_all(&layers, CONTENT_TIMEOUT)
        .expect("a HUD layer checkbox did not appear");
    for layer in &layers {
        let checked = hud.first(layer).data().states.checked;
        assert!(
            matches!(checked, Some(Toggled::On)),
            "'{}' should be checked by default (got {checked:?})",
            layer.name,
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

    // One snapshot of the panel the load produced. The demo reconstruction is
    // labeled "demo" and rings the scene with 8 images, so both strings are
    // fixed by the fixture. The third is the row's solo toggle: its own
    // behaviour is headless (`scene_graph` tests), and what only a real window
    // can show is that a *third* glyph button squeezed onto the row is still
    // laid out and still reachable.
    app.wait_all(
        &[
            Expect::present("static_text", "demo"),
            Expect::present("static_text", "Camera Images (8)"),
            Expect::present("button", "S"),
        ],
        CONTENT_TIMEOUT,
    )
    .expect("the Scene panel does not list the loaded reconstruction");
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
    //
    // One snapshot, taken after the last click: the row lookup above cannot be
    // folded in with these, because the clicks in between republish the tree
    // and the menu does not exist until they have landed.
    app.wait_all(
        &[
            Expect::present("button", "Select"),
            Expect::present("button", "Zoom to Fit"),
            Expect::present("button", "Bundle Adjust..."),
        ],
        CONTENT_TIMEOUT,
    )
    .expect("the reconstruction row's context menu did not open on a right-click");
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

    // Both halves of the verdict in one snapshot, for the reason
    // `the_menu_bar_holds_file_edit_go_and_panels` gives at more length: the
    // Action Log's toolbar is what establishes the dock has been built and its
    // tree published, and only against a tree that is there does finding no
    // placeholder mean the 3D viewer is not docked, rather than that nothing is
    // in the tree yet. Asked together, the two cannot come apart.
    app.wait_all(
        &[
            Expect::present("button", "Latest"),
            Expect::absent("static_text", "No reconstruction loaded."),
        ],
        CONTENT_TIMEOUT,
    )
    .expect(
        "the Action Log toolbar is absent or the 3D viewer is still docked, \
         so the saved layout was not loaded",
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
    // strings are fixed by the fixture — and both are questions about the same
    // untouched panel, so one snapshot answers them.
    app.wait_all(
        &[
            Expect::present("static_text", "1 version"),
            Expect::present(
                "static_text",
                "demo has not been edited; its one version is the file as it was opened.",
            ),
        ],
        CONTENT_TIMEOUT,
    )
    .expect("the Edit History panel does not list the loaded version");
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
    fn wait_for_window(&self) -> Attached {
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
