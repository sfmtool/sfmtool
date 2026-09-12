# Operation progress: what a long computation tells its caller

**Status:** Draft

Amends [gui/action-log.md](../gui/action-log.md) § "What an action cost" and
[gui/mcp-server.md](../gui/mcp-server.md) § "get_action_log" in the viewer, and
in `sfmtool-core` adds one parameter to the long-running functions specified by
[core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md),
[core/reconstruction/add-observation.md](../core/reconstruction/add-observation.md)
and the other kernels under [core/patch/](../core/patch/) that a viewer
operation reaches.

## What a long computation cannot say

A reconstruction kernel runs for somewhere between a millisecond and several
minutes, and for that whole time it says nothing. It cannot report which stage
it is in, cannot say how far along it is, cannot pass on a remark worth reading
("three views dropped below the ZNCC floor"), and cannot be told to stop. When
it returns, the Action Log gets one number for the whole thing: `took`, the wall
time between the action being recorded and the frame that showed its result.

That number is the wait the person at the window actually sat through, and it is
enough to notice that deleting a point costs six milliseconds while undoing back
across a bulk edit costs two and a half seconds. It is not enough to say *where*
those two and a half seconds went, which is the next question every time the
number is surprising. Answering it today means a stopwatch outside the process,
a guess about which of half a dozen candidate costs dominates, and no way to
check the guess against what the viewer did.

Five things are missing, and they are one problem:

| | |
|---|---|
| which stage it is in, and what each stage cost | **phases** |
| a remark worth keeping | **messages** |
| what it is doing at this instant, replaced as it changes | **status** |
| how far along it is | **counts** |
| and, in the other direction: stop | **cancellation** |

All five are a channel between a long computation and whoever asked for it. This
proposes one parameter carrying all five, a `Progress` that every long function
in `sfmtool-core` accepts, and then what the viewer does with what comes back:
the Action Log entry grows an expandable breakdown, and the Background panel
shows the same thing live
([background-process-panel.md](background-process-panel.md)).

## The parameter

```rust
// sfmtool-core::progress

/// What a long computation tells its caller, and how it is told to stop.
///
/// `Copy`: a shared reference to the sink, a shared reference to the cancel
/// flag, the detail bit, the phase depth, and the `(offset, extent)` of the
/// range this call owns. It borrows nothing from the `Progress` it came from,
/// which is what lets a child outlive the expression that made it.
#[derive(Clone, Copy)]
pub struct Progress<'a> { /* sink, cancel, detail, depth, offset, extent */ }

impl Progress<'static> {
    /// Reports nothing, cancels never, detail off. What every call site that
    /// does not care passes, and what every existing call site becomes.
    pub const fn none() -> Self;
}

impl<'a> Progress<'a> {
    pub fn to(sink: &'a (dyn Fn(Event<'_>) + Sync)) -> Self;
    pub fn cancelled_by(self, flag: &'a AtomicBool) -> Self;
    pub fn detailed(self, on: bool) -> Self;

    /// Has the caller asked this to stop? One relaxed atomic load.
    ///
    /// For a loop that wants to stop and hand back what it has.
    pub fn is_cancelled(&self) -> bool;
    /// The same question in `?` form, which is how most callers should ask it:
    /// `progress.check_cancel()?;` propagates without anything to remember.
    pub fn check_cancel(&self) -> Result<(), Cancelled>;
    /// Whether detailed phases are being recorded, so a caller can skip
    /// building a note nobody will read.
    pub fn is_detailed(&self) -> bool;

    /// Whether anything is listening at this level, so a caller can skip work
    /// nobody will read. The `progress_info!` family checks this first.
    pub fn wants(&self, level: Level) -> bool;
    /// Append a line. Prefer the macros, which do not evaluate their arguments
    /// when nothing is listening.
    pub fn message(&self, level: Level, text: fmt::Arguments<'_>);
    /// Replace the one-line statement of what this call is doing right now.
    /// Not the viewport status line, which is the Action Log's newest entry.
    pub fn set_status_message(&self, text: fmt::Arguments<'_>);
    /// How far along, in this call's own unit. A known total also moves the
    /// bar, within this call's range (§ "Nesting").
    pub fn count(&self, done: u64, total: Option<u64>, unit: &'static str);
    /// Where this call has got to within its own range, 0..=1.
    pub fn set_fraction(&self, f: f32);

    /// Carve this call's range into N pieces, one per nested call, in
    /// proportion to `weights`. They are relative: normalised to their sum, so
    /// `[1.0, 9.0]` and `[0.1, 0.9]` are the same split (§ "Nesting").
    pub fn split<const N: usize>(&self, weights: [f32; N]) -> [Progress<'a>; N];
    /// N equal pieces, for a loop with a known trip count.
    pub fn split_evenly(&self, n: usize) -> impl Iterator<Item = Progress<'a>>;

    /// Open a phase. Always recorded; closes when the guard drops.
    pub fn phase(&self, name: &'static str) -> Phase<'a>;
    /// The same, recorded only when detail is on; inert otherwise.
    pub fn detail_phase(&self, name: &'static str) -> Phase<'a>;
}

/// An open phase, and the `Progress` for whatever runs inside it.
///
/// Derefs to a `Progress` one depth deeper, over the same range and the same
/// sink, so a nested call is `inner(&p)?` and its phases nest under this one
/// without anything being passed but the guard.
pub struct Phase<'a> { /* the child Progress, the name, the start */ }

impl<'a> std::ops::Deref for Phase<'a> {
    type Target = Progress<'a>;
}

impl Phase<'_> {
    /// Say what this phase did, for the column beside its time: `reused`,
    /// `46 231 points`. Prefer `progress_note!`, which does not evaluate its
    /// arguments for a phase that is not recording.
    pub fn note(&mut self, note: fmt::Arguments<'_>);
    /// Whether this guard will record anything at all.
    pub fn is_recording(&self) -> bool;
    /// Drop the guard without recording: the phase did not run.
    pub fn cancel(self);
}
```

`Progress` is `Sync` and `Copy`, and is taken by `&`, never by `&mut`. All
three are load-bearing: `Sync` so a rayon closure can capture it (§ "Why a
parameter"), `Copy` so a child costs nothing and borrows nothing (§ "Nesting"),
and `&` so it never conflicts with another borrow (§ "In the viewer").

```rust
pub enum Event<'a> {
    /// A phase opened. What a live panel needs to mark the stage that is
    /// running before it is over.
    Enter { phase: &'static str, depth: u8 },
    /// A phase closed, with what it cost.
    Leave { phase: &'static str, depth: u8, took: Duration, note: Option<&'a str> },
    /// Something worth reading, in words. Appends.
    Message { level: Level, depth: u8, text: &'a str },
    /// What the operation is doing right now. Replaces whatever it last said,
    /// and is never kept in the Action Log entry.
    Status { text: &'a str },
    /// How far along, in the reporting call's own unit: what the panel prints
    /// in words, `(7, Some(20), "iteration")`.
    Count { done: u64, total: Option<u64>, unit: &'static str },
    /// Where the bar sits, already mapped out of the reporting call's range
    /// into the whole operation's, and already clamped monotone.
    Fraction { of_whole: f32 },
}

pub enum Level { Info, Warn }

/// The caller asked the operation to stop. Each kernel error type carries it,
/// as a variant or a `From`, so `?` propagates a cancellation like any failure.
pub struct Cancelled;
```

A caller that wants nothing passes `&Progress::none()`, whose every method is a
branch on a null sink. That is what makes this affordable to put on functions
that are sometimes called in a loop by other kernels, and it is what every
existing call site becomes: one token, no behaviour change.

### Why a parameter

The alternative is an ambient collector, a thread-local the way `tracing` spans
work, with nothing threaded through any signature. It does not survive contact
with this codebase:

- **rayon.** These kernels are rayon-parallel, and a rayon worker has its own
  thread-locals. Work-stealing means the calling thread runs some of the work
  and the pool runs the rest, so an ambient collector would capture a
  nondeterministic subset, which is worse than capturing nothing. This is
  exactly why the existing `prof` modules accumulate into process-global atomics
  and report thread-summed CPU time: they had this problem and solved it by
  giving up per-call attribution. A `Sync` parameter captured by the closure
  does not have the problem at all.
- **Per-call attribution.** A process-global counter cannot say what *one*
  `add_observation` cost. A parameter is per call by construction, and the
  question the Action Log asks is always about one call.
- **Worker threads.** Long operations move off the GUI thread
  ([background-process-panel.md](background-process-panel.md)). An ambient
  collector would leave the worker's phases in the worker's thread-local, to be
  drained and shipped back over a channel. A shared sink is written to directly,
  and there is nothing to ship.
- **Tests.** Rust runs tests across threads. A thread-local collector is global
  state that mostly works, until a test spawns a thread.
- **`sfmtool-core` has no logging dependency**, deliberately: no `log`, no
  `tracing`, no terminal crate. `Progress` keeps it that way, because it is not
  a logging facade that decides where output goes. It is a parameter, and the
  caller decides. The viewer routes it to a panel, a command-line run routes it
  to stderr, a test routes it to a `Vec`.

The cost is real and worth stating plainly: **it is viral.** Seven `prof`
modules and some twenty-five gate sites convert, and every function in each call
chain grows a parameter. It goes in as a parameter of its own rather than as a
field of the options struct a top-level entry point already takes, for two
reasons. An options struct says what a call is allowed to move and how hard it
should try, which a reporting channel is not; and a `BundleAdjustOptions<'a>`
would carry that lifetime into `Default`, into every construction site, into the
viewer signatures that name it, and across the binding boundary. One rule for
every function in the chain is also easier to follow than two.

### Through the Python bindings

`sfmtool-py` re-exports several of the functions that grow this parameter.
**The bindings pass
`Progress::none()`** and gain nothing: a Python-visible progress callback would
have to be `Sync` and callable from inside a rayon region, which means
reacquiring the GIL per report from arbitrary worker threads, and the Python
layer has asked for none of it. So the binding signatures do not change, and a
Python caller sees exactly what it sees today. Giving Python a sink is a
non-goal below rather than a gap.

### Precedent

This is the shape numerical libraries have converged on. Ceres Solver takes an
`IterationCallback` returning `SOLVER_CONTINUE`, `SOLVER_ABORT` or
`SOLVER_TERMINATE_SUCCESSFULLY`, which is how COLMAP drives progress and abort
through a bundle adjustment. libgit2 takes `transfer_progress`, a callback that
returns `false` to cancel. FFmpeg takes `AVIOInterruptCB`, which aborts on a
nonzero return. In Rust the two halves are usually separate crates, none
dominant: `cancellation` and `tokio_util::sync::CancellationToken` for the stop
signal, `indicatif` for progress, which is a terminal renderer and so belongs
nowhere near a kernel. One small local type costs less than two dependencies
that each solve half of it.

The range-splitting in § "Nesting" has its own precedent, and an older one.
Eclipse's `SubMonitor` is the refined version: a parent allocates ticks, a
child consumes them through `split`, and a child cannot overrun its allocation,
which is what makes the top-level bar monotone no matter how the nested calls
behave. Rust's `yield-progress` carries the same idea for async tasks. The
design below is that idea with the async removed.

The closest thing to the whole of this design is `frantic::logging` in
[aws/thinkbox-library](https://github.com/aws/thinkbox-library), which carries a
`progress_logger` passed to the work, a `null_progress_logger` for callers that
do not care, `push_progress` / `pop_progress` with an RAII tracker for nesting,
a `set_title` line separate from the log streams, cancellation that cannot be
forgotten, and level-filtered macros that decline to evaluate their arguments.
Three of those shaped what is written here. Its one piece not adopted is the
ambient `FF_GLOBAL_PROGRESS()`, offered for code too deep to thread the
parameter through: its own `isThreadSafe` flag is the argument against it here,
since the state that flag guards is a push/pop stack and every kernel below this
is rayon-parallel.

## Nesting

An algorithm is a sequence of steps, each a different nested call, and the two
things a `Progress` carries nest differently.

**A phase is about time already spent**, so it composes by addition: a child's
cost is part of its parent's, to any depth. What has to be arranged is that a
phase knows it is inside another one, and the guard is what arranges it:

```rust
let mut p = progress.phase("solve");
linearise(&p)?;                  // its phases open at depth + 1
p.note(format_args!("{iters} iterations"));
```

`Phase` derefs to a `Progress` one depth deeper over the same range, so a nested
call takes `&p` and reports underneath without being told anything. **The depth
is carried in the value, never counted at the sink**, and that is not a detail:
a collector counting `Enter` and `Leave` would be wrong the moment two rayon
threads interleave, which is the same failure the parameter exists to avoid, on
the other axis.

A phase and a split are independent. A phase narrows the depth and keeps the
range; a split narrows the range and keeps the depth. Either may contain the
other.

**A fraction is about work remaining, and it does not compose by addition.** If
every nested call reported 0 to 1 on its own terms, a bundle adjustment's bar
would run to full during `materialise`, snap back to zero for `solve`, and do it
again for the row map. So a caller **carves a share of its own range** for each
child, and the child reports within that share without ever learning it is a
child:

```rust
fn bundle_adjust(recon: &R, opts: &O, progress: &Progress) -> Result<…> {
    let [p_mat, p_solve, p_map] = progress.split([0.05, 0.90, 0.05]);
    let value = materialize(recon, &p_mat);
    let solved = solve(&value, opts, &p_solve)?;   // its 0..1 lands in 0.05..0.95
    let map = RowMap::by_scan(&value, &solved, &p_map)?;
    …
}
```

`Progress<'a>` is `Copy` -- two shared references, a bool and two `f32` -- so a
child is a value carrying the same sink with a narrowed `(offset, extent)`. It
borrows nothing from its parent, which is what keeps the lifetimes trivial and
lets a child outlive the expression that made it.

Splitting up front, rather than allocating incrementally as work is finished,
puts the weights on one line where they can be read and argued with, and removes
the mutable allocation cursor that would otherwise fight `&self`.

**Monotonicity has two guarantees, and the second does not trust the first.** A
child's range is a subinterval of its parent's, so a child cannot move the
global fraction outside its own slice however wrong its arithmetic is; and the
collector clamps what it reports to the greatest fraction seen so far in this
operation. A kernel reporting nonsense can stall the bar; it cannot run it
backwards.

`count(done, Some(total))` sets the fraction within the reporting call's range
as well as emitting the words, so the deepest call that knows a total is what
drives the bar, and everything above it only decides how much of the bar that
call owns. `count(done, None)` reports a number and moves nothing.

### What the weights are, and are not

They are constants, chosen by whoever wrote the call, and they are an estimate
of how the time divides. That is in tension with the rule that the viewer never
shows a synthesised percentage
([background-process-panel.md](background-process-panel.md)), so the bar is held
to what it can honestly claim:

- **A stage that reports nothing still advances the bar when it ends.** The bar
  steps at every phase boundary, because something did finish.
- **Smooth movement happens only where a call actually reports counts.** Nothing
  interpolates across a stage that is silent.
- **If nothing underneath reports at all, there is no bar**, only a spinner and
  the name of the open phase.

So the bar is a statement about how much of the work is behind you, never a
prediction of when it will end, and a weight that is wrong makes it uneven
rather than untrue.

### Two rules of scale

**Split for stages, not for items.** A handful of distinct steps get a `split`;
a loop over 105 000 clusters reports `count(i, Some(n))` from inside the loop.
Splitting per item would mean a range per item and a report per item, which is
the wrong shape and the wrong cost.

**Fraction reports coalesce.** These loops run to the order of 10^8 objective
evaluations, and the bar has a few hundred pixels. The collector drops a
`Fraction` that arrives within `FRACTION_INTERVAL` of the last one it kept.
Phase boundaries and messages are never dropped, since those are events rather
than samples.

## Two levels

**Overview is always on and is a coverage requirement.** Every operation the
Action Log can record that can exceed a frame's budget names its stages: a file
opening, every bulk edit, every history step, the uploads, the render, the
resection, the alignment, the decode the Image Detail panel does. The point of
the level is that the first question is never unanswerable. When a row says
2.4 s, expanding it says which stage that was, without anybody having had to
predict in advance that this would be the slow row.

The cost is two clock reads per phase and a handful of phases per operation,
which is why it can be a requirement rather than a budget. Instrumentation
anyone has to remember to switch on is instrumentation that is off when the
surprising thing happens, and the surprising thing is rarely reproducible on
demand.

**Detail is a switch, off by default**, thrown from the window or over MCP
(§ "Turning on detail"). It adds the phases too fine to carry always: the stages
inside a kernel, the sub-steps of an upload, the per-stage counts. A
`detail_phase` on a `Progress` with detail off is inert, which is what makes it
affordable to leave the call sites in place.

**The level rides on the `Progress`**, not in a global. The viewer sets it from
the checkbox when it builds the `Progress` for an operation, so two callers in
one process can legitimately differ and nothing has to be reset. It is not read
from the environment: `SFMTOOL_PROFILE` still means what it has always meant,
which is the `prof` modules below.

The existing `prof` modules and their stderr batch summaries are untouched and
keep their own `SFMTOOL_PROFILE` gate. They answer a different question, over a
whole batch rather than one call, they are the only reader of that variable, and
converting them is § "Non-goals".

## Messages are the logging

A kernel that wants to say something says it through a macro:

```rust
progress_warn!(progress, "{dropped} views below the ZNCC floor");
progress_info!(progress, "{images} images, {points} points, {obs} observations");
progress_status!(progress, "{}", image.name);      // replaces
progress_note!(phase, "{packed} tiles");           // onto an open phase
```

**Macros rather than plain calls, because a method cannot decline to evaluate
its arguments.** `message(level, format_args!(…))` defers the *formatting*, but
`format_args!` still evaluates what it interpolates, so a message built from a
function call pays for that call even when nobody is listening. Each macro
expands to a `progress.wants(level)` test around the call, which is the same
reason `log::info!` and frantic's `FF_LOG` are macros. The method stays public
for a caller that has already checked.

A message carries the depth of the phase it was emitted inside, so it nests
under that phase when displayed.

### Status is not a message

`progress_status!(progress, "{}", name)` sets a single line saying what the call
is doing **now**, replacing whatever it last said. That is a different thing
from a message, which appends and is kept:

- A loop over 85 images wants the panel to read `dino_dog_toy_42.jpg` and then
  `dino_dog_toy_43.jpg`. As messages that is 85 lines of scrollback nobody
  wants; as a status it is one line that changes.
- `count` gives the panel the numbers, `42 of 85`; the status gives it the name.
  Neither is derivable from the other.

A status is live state, so it reaches the Background panel and nothing else: it
is **not** kept in the Action Log entry, because by the time the entry exists
the answer is "finished". A synchronous operation therefore has nowhere to put
one, which costs nothing: it holds the GUI thread, so nothing could have drawn
it. Status is worth setting in the kernels that a background operation reaches,
and harmless everywhere else. It is also not the viewport status line, which stays
what it is, the newest Action Log entry
([action-log.md](../gui/action-log.md)).

### Where a message ends up

The caller's choice, and the viewer makes three:

- **The Background panel** shows the newest messages of the operation that is
  running, under its phase table, which is the whole point of being able to say
  something mid-computation.
- **The Action Log entry** keeps them as part of its detail, interleaved with
  the phases in the order they happened, so an expanded entry reads as a short
  transcript of the operation rather than a table of numbers.
- **The `log` crate**, mirrored as `log::info!` or `log::warn!` under the target
  `sfm_explorer::progress`, exactly as Action Log entries are already mirrored
  under `sfm_explorer::action_log`. A `RUST_LOG` capture of a session therefore
  carries the same stream, and this is the piece that makes the messages useful
  outside the window.

Two rules keep this from becoming a firehose. **A message is never per item**: a
message per point or per observation is a counter, and counters are what `count`
is for, or a status if it is a name rather than a number. And an entry keeps at
most `DETAIL_EVENTS` events in total, phases and messages together, past which
the remainder is dropped and a final line says how many.

## What the user sees

### The toggle column

The Action Log row gains a leading column one glyph wide, before the time:

```
  14:09:19  User      6.2 ms  Deleted point 29429 in guard (v2 → v3)
+ 14:09:22  MCP       2.36 s  Undo: Resected dino_dog_toy_09.jpg in place (v3 → v2)
  14:09:24  MCP      <1 ms    get_history guard
```

`+` marks an entry that carries detail, `-` one that is expanded, and a blank
one that carries none. Clicking the toggle, or the row's time, expands it in
place:

```
- 14:09:22  MCP       2.36 s  Undo: Resected dino_dog_toy_09.jpg in place (v3 → v2)
                              undo                    4.1 ms
                                history step          0.3 ms
                                selection follow      3.8 ms
                              uploads              1874.2 ms
                                points             1203.4 ms
                                patch atlas         502.1 ms
                                thumbnails               --  reused
                                deleted mask          1.1 ms
                                track rays          167.6 ms
                              scene render          412.0 ms
                              egui pass              71.3 ms
                              elsewhere               2.1 ms
```

and with detail on, a bundle adjustment reads as a transcript:

```
- 14:12:03  User      41.7 s  Bundle adjusted guard: 85 images, 44 912 points, …
                              materialise             412.0 ms
                              gather arrays           208.4 ms
                              • 85 images, 44 912 points, 198 331 observations
                              residuals before        301.7 ms
                              solve                    40.3 s
                                round x3               40.2 s
                                  linearise x180       12.1 s   cpu 94.4 s
                                  normal equations x180  26.4 s   cpu 201.7 s
                                  damping ladder x180    1.7 s
                              ! 3 points left unsupported and were dropped
                              write back               71.3 ms
                              push version            302.1 ms
                              elsewhere                 8.4 ms
```

A detail line is a **row of the same height as any other**, indented two spaces
per level, with a phase's cost in the same right-aligned column the entry's own
cost sits in. A message is marked `•` for `Info` and `!` for `Warn`, the latter
in `error_fg_color`. Expansion inserts rows rather than making one row tall,
which is what keeps the list virtualized on a uniform row height and ten
thousand entries free to scroll.

A phase that ran and cost nothing worth printing shows `--` and says why in its
note. `reused` is the common case and the useful one: it is how the reader tells
a phase that was skipped from one that was merely fast.

Expansion is per entry and survives new entries arriving and the panel being
docked elsewhere. **Clear** collapses everything, because the entries it was
holding are gone. **Copy** writes the detail of expanded entries, indented under
them, and nothing for collapsed ones.

### The line that makes it add up

`elsewhere` is the entry's `took` minus the sum of its top-level wall-clock
phases. It is always last, never nested, and it is the reason the breakdown can
be trusted: an expanded entry reconciles with the number in its own cost column
by construction, so a phase nobody has instrumented shows up as a gap rather
than as silence. When `elsewhere` dominates, the interesting work has no name
yet, and the panel says so instead of implying the named phases are the whole
story.

`cpu` columns are excluded from that arithmetic. A kernel that reports
thread-summed CPU time alongside its wall time can show eight seconds of CPU
inside one second of wall, and folding that into a wall-clock total would be
lying about the sum.

### Turning on detail

A **Detailed timing** checkbox in the Action Log toolbar, beside **Latest**,
**Copy** and **Clear**: it is where the detail is read, which is where somebody
decides they want more of it. Ticking it records a `Display` entry like any
other control, so the log says when the level changed and who changed it.

It takes effect on the next operation. Nothing already recorded is re-timed, and
an entry keeps the detail it was recorded with: turning detail on does not
retroactively fill a row in, and turning it off does not strip one.

## In the viewer

The sink is a **collector**, and the viewer has two.

```rust
/// Somewhere for events to land. Shared, never borrowed mutably.
pub(crate) struct Collector { /* Mutex<Vec<Detail>>, the open phase, counts */ }

impl Collector {
    pub(crate) fn progress(&self) -> Progress<'_>;
    /// Time a phase of the viewer's own work.
    pub(crate) fn phase(&self, name: &'static str) -> Phase<'_>;
    pub(crate) fn take(&self) -> Vec<Detail>;
    /// What the Background panel draws: the open phase and what is done so far.
    pub(crate) fn live(&self) -> Live<'_>;
}
```

**An operation's collector** is made when the operation starts and handed to the
entry when it finishes. There is nothing to attribute: the collector *is* that
entry's detail. This is the whole of the attribution story for an operation's
own work, and it is why the parameter is worth its virality.

**The frame's collector** lives on `App` and holds what is not any operation's:
`uploads`, `scene render`, `egui pass`, `present`. Those genuinely belong to
whatever entries the frame settles, because one upload and one draw showed all
of them, so `ActionLog::settle` appends the frame's events to every entry it
stamps and says `frame shared with 2 other entries` when there was more than
one. A frame that stamps nothing discards its events.

One consequence is worth stating because it looks like a bug otherwise. An entry
written during the egui pass, which is every click, key and menu item, does not
get the tail of its own frame: it is not stamped until the *next* frame, and
that next frame is the one that shows its result. The millisecond or two lands
in `elsewhere`.

**A collector is shared by `&`, with interior mutability**, and that is not an
implementation detail. `TabContext` hands out seven simultaneous `&mut` borrows,
and `AppState::scene_and_log` exists only because two things needed borrowing at
once; a collector reached through `&mut AppState` would conflict with every
panel closure that already holds one. Taking it by `&` also makes the worker
case free: the GUI thread and the worker hold the same `Arc<Collector>`, the
panel reads it each frame under the lock, and no events cross a channel. A
`Mutex<Vec<Detail>>` is ample: a phase costs a lock and a comparison, contention
arises only while a worker runs, and the folding below keeps what is kept small.

### Repeated phases fold

A guard goes where the code already has a boundary, and for a loop body that
means once per trip. A bundle adjustment of three rounds at sixty iterations
opens `linearise`, `normal equations` and the damping ladder five hundred and
forty times between them. Drawn one row each, that is not a breakdown of where
the time went, it is a transcript of the solve, and it would exhaust
`DETAIL_EVENTS` before the first round finished.

So the collector folds them. **Within one enclosing phase, every child phase
sharing a name is one row**, whose time is the sum and which says how many times
it ran:

```
solve                      40.9 s
  round x3                 40.8 s
    linearise x180         12.1 s
    normal equations x180  26.4 s
```

Ordering is by first appearance. A phase that ran once is drawn as it would have
been anyway, with no count, and a message keeps its own place.

Folding happens as the events arrive, so `DETAIL_EVENTS` counts rows kept rather
than phases opened, and a kernel in a long loop cannot push the rest of the
operation out of its own entry.

Two consequences are worth stating. A folded row's time is a sum of wall times,
so for phases that ran on different rayon threads at once it exceeds the span
they actually covered, exactly as a `cpu` figure does; `elsewhere` is clamped at
zero for the reason it always was. And a phase name is now a key rather than a
label, so two unrelated stages under one parent must not share one, which is
what anybody reading the expanded entry would have assumed regardless.

## What carries phases

Overview coverage is a requirement, so this is what must be instrumented rather
than what happens to be. Every row is an operation the log records that has been
seen, or can be expected, to cost more than a frame.

The frame, which every settled entry inherits:

| Phase | Where | Note it carries |
|-------|-------|-----------------|
| `mcp drain` | `App::drain_mcp` | the number of commands, when more than one |
| `uploads` | `App::prepare_uploads` | |
| `points` | `upload::points` | the instance count, when it rebuilt |
| `patch atlas` | `upload::patches` | `reused`, or the tile count it packed |
| `thumbnails` | `upload::thumbnails` | `reused`, or the image count |
| `deleted mask` | `upload::overlay` | the number of entries written |
| `track rays` | the CPU-space rebuild | |
| `scene render` | Phase 2 | |
| `egui pass` | Phase 3 | |
| `present` | the queue submit and the present | |

The operations:

| Phase | Where | Seen at |
|-------|-------|---------|
| `open`, with `read`, `decode` and `derive` under it | `AppState::load_file` | 1.84 s for the dino set |
| `save`, with `materialise`, `stamp` and `write` under it | `state::save` | |
| `undo` / `redo` / `go to`, with `history step` and `selection follow` | `state::edits` | 447 ms to 2.36 s across a bulk edit |
| `materialise` | wherever an edit folds an overlay before a kernel call | |
| the `sfmtool_core` call's own stages, which it reports itself | the kernel a bulk edit runs | 838 ms for a resection in place |
| `row map` | `RowMap::by_scan` | |
| `push version` | `History::push`, where the budget accounting runs | |
| `localize` and `refine` | `add_observation`'s two kernel calls | |
| `decode views` | the full-resolution decode an edit needs | |
| `sift cache` | the Image Detail overlay's feature load | |

A row of that table that expands to nothing but `elsewhere` is a gap in the
coverage rather than a curiosity, and the way to find one is to read the log
after using the viewer normally.

Detail adds, under those: the stages inside each kernel, the per-buffer steps
inside `uploads`, and the per-pass steps inside `scene render`.

## Cancellation

A kernel asks whether it should stop wherever it can stop without leaving a
half-built answer: between iterations, between members, at the top of a batch.
There are two ways to ask, and the `?` one is the default:

```rust
progress.check_cancel()?;              // propagates, nothing to remember
if progress.is_cancelled() { break }   // for a loop handing back what it has
```

**`check_cancel` exists because a polled bool is a thing you can forget.**
`is_cancelled` has to be checked *and* turned into an early return at every
level of the call stack, and a level that forgets makes the whole operation
uncancellable with no compile error and no test failure. frantic reaches for the
same guarantee by throwing `progress_cancel_exception` out of
`check_for_abort()`, which cannot be forgotten because it unwinds; `?` is the
Rust spelling of the same idea, without using unwinding as control flow.

The cost is that `Cancelled` has to reach each kernel's error type, as a
variant or a `From`. That is mechanical, and it buys a property worth having:
cancellation propagates by the same mechanism as every other failure, so a
kernel that already returns `Result` gets it for nothing.

Rayon has no cancellation of its own, so a parallel loop that wants to stop uses
`try_for_each` and short-circuits on the `Err`.

A kernel that never asks is simply not cancellable, and says so by never
touching the flag. Which operations are cancellable, what the button does, and
what a cancelled operation writes to the log are
[background-process-panel.md](background-process-panel.md)'s, since they are
about the worker rather than about the parameter.

## The Action Log's side

```rust
pub(crate) enum Detail {
    Phase { name: &'static str, depth: u8, took: Duration, cpu: Option<Duration>,
            note: Option<String>, runs: u32 },
    Message { level: Level, depth: u8, text: String },
}

pub(crate) struct Entry {
    // … revision, at, actor, kind, run, failed, took, text …
    /// What the operation reported, in the order it reported it.
    pub detail: Vec<Detail>,
}

impl ActionLog {
    pub(crate) const DETAIL_EVENTS: usize = 128;

    pub(crate) fn is_expanded(&self, revision: u64) -> bool;
    pub(crate) fn toggle_expanded(&mut self, revision: u64);

    /// `took` minus the entry's top-level wall-clock phases, or `None` when it
    /// has no cost yet or no phases.
    pub(crate) fn elsewhere(entry: &Entry) -> Option<Duration>;

    /// Record an entry for work that has already happened: `started` is when
    /// that work began, `detail` what it reported.
    ///
    /// The one call a long operation uses. `started` is what makes a
    /// two-minute solve report two minutes rather than the six milliseconds of
    /// the frame that installed it; the entry still settles on the frame that
    /// draws the result, like every other.
    pub(crate) fn record_done(&mut self, kind: Kind, started: Instant,
                              text: impl Into<String>, detail: Vec<Detail>);
}
```

**The expansion set lives on the log** rather than in egui's memory, because it
is the one piece of panel state a headless test needs to drive and the panel
already takes `&mut ActionLog`. It is keyed on the revision, so an expansion
survives entries dropping at `CAPACITY` and means nothing once its entry is
gone. `clear` empties it along with the buffer.

**The panel's rows.** `ScrollArea::show_rows` needs a row count and a row
mapping, and expansion changes both. The panel builds a small table once per
frame, one entry per expanded revision still held, giving where it expands and
how many rows it adds; a row lookup binary-searches that table and falls through
to the entry at the remaining offset. Expansions are a handful at most, so this
is cheaper than the layout it avoids, and the row height stays uniform.

A fold takes the new value's detail along with its text and its time, because a
folded row is the newest value of the run.

## On the wire

`get_action_log` gains one argument:

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `detail` | boolean | `false` | Include each row's phases and messages |

With it set, a row carries `detail`, an array of `{ "kind": "phase", "name",
"depth", "ms" }` with `"cpu_ms"`, `"note"` and `"runs"` when they apply, and
`{ "kind": "message", "level", "depth", "text" }`, in the order the panel draws
them, plus `"elsewhere_ms"` beside `took_ms`. Off by default, because the detail
is several times the size of the row it hangs off and an agent reading the log
to find out what happened does not want it. An agent that has found a slow row
asks again with `detail` set and `since_revision` just below it.

That argument asks for what was *recorded*. The level, which decides what gets
recorded, has its own pair, in the shape of the existing display pair:
**`get_timing_detail`** reports it and **`set_timing_detail { enabled }`** sets
it, recording the same `Display` entry the checkbox does, so a human at the
window can see that an agent raised the level. An agent investigating a slow
operation turns detail on, runs the operation, reads the log with `detail`, and
turns it off, with no restart and no environment variable. That is the case the
pair exists for.

## Testing

`sfmtool-core`, in `progress/tests.rs`:

- `Progress::none()` formats no message, builds no note, and reports no phase;
  a kernel called with it produces byte-identical output to one called before
  this existed.
- A sink receives `Enter` and `Leave` in order, nested phases at the right
  depths, and a message at the depth of the phase it was emitted in.
- **Depth comes from the value, not from the sink.** Two threads each opening a
  phase inside one of their own, reporting into one collector concurrently,
  produce depths that are right for every event; a sink counting `Enter` and
  `Leave` would not. This is the parallel case, and it is the reason the depth
  is a field.
- `Phase::note` reaches the `Leave` for that phase and nowhere else, and
  `progress_note!` does not evaluate its arguments for a phase that is not
  recording. A cancelled `Phase` emits no `Leave`.
- A `detail_phase` reports nothing when detail is off and its note closure is
  not run.
- `check_cancel` returns `Err(Cancelled)` exactly when `is_cancelled` is true,
  and a kernel three calls deep propagates it to the top with no level having
  written anything but `?`.
- A `progress_info!` whose argument calls a function does not call it when
  nothing is listening. The test counts the calls, since this is the one
  property a plain method cannot have.
- A status replaces rather than appends: three `progress_status!` calls leave
  one status and no entry detail, while three `progress_info!` calls leave three
  lines.
- **Nesting.** A child from `split` maps its own 0..1 onto its share of the
  parent's range, two children in sequence cover the parent's range without
  overlap, and `split_evenly(n)` gives n equal shares. A child that reports
  outside 0..1 is clamped into its own share and cannot reach its sibling's.
  Three levels of splitting compose, so the deepest call's fraction lands where
  the arithmetic says.
- **Monotonicity**, asserted against a deliberately misbehaving kernel: one that
  reports its fraction descending, and one that reports 1.0 and then 0.3, both
  leave the global fraction non-decreasing.
- `count(done, Some(total))` emits both the words and a mapped fraction;
  `count(done, None)` emits the words and moves nothing.
- `Fraction` reports closer together than `FRACTION_INTERVAL` are dropped, while
  a phase boundary or a message between them is kept.
- `is_cancelled` is observed by a kernel inside a rayon loop, which stops and
  returns what it had. **The point of this test is the parallel case**, since a
  thread-local collector is what this design exists to avoid.
- Two `Progress` values with different levels used concurrently on two threads
  each see their own level.

`crates/sfm-explorer/src/action_log/tests.rs`, headless:

- An operation's collector becomes exactly that entry's detail, and the next
  entry gets none of it.
- `settle` appends the frame's events to the entry it stamps; two entries
  stamped by one frame carry the same frame events and both say the frame was
  shared; a frame that stamps nothing discards them.
- A fold carries the new value's detail and drops the replaced value's.
- **Folding.** Child phases sharing a name under one parent become one row
  carrying the summed time and the count, ordered by first appearance, while the
  same name under two different parents stays two rows. A phase that ran once
  carries no count. A message between two foldable phases keeps its place.
- Folding happens before the cap: an operation opening one phase six hundred
  times leaves one row, not a truncated entry.
- Past `DETAIL_EVENTS` the entry keeps the first `DETAIL_EVENTS` and reports the
  number dropped.
- `elsewhere` is `took` minus the top-level wall-clock phases, ignores `cpu`,
  is `None` before the entry settles, and is never negative: a sum exceeding
  `took` reports zero, since the phases and the settle read the clock at
  different points and a negative duration would be a panic rather than a
  reading.
- Messages are mirrored to `log` at the matching level under
  `sfm_explorer::progress`.

Coverage, which is the overview level's own requirement rather than a mechanism
test: every operation in § "What carries phases" records at least one phase,
asserted by driving it headless and reading the entry back. That is the test
that fails when someone adds a slow operation and forgets to name its stages.

Panel, through `test_support::run_frame_headless` and `painted_texts`:

- An entry with detail paints a `+`; one without paints neither `+` nor `-`.
- Toggling paints one row per event, indented, in order, and the row count grows
  by exactly the event count; toggling again restores the original rows.
- A `Warn` message paints its marker in `error_fg_color`.
- A `cpu` figure paints in its own column and is absent from `elsewhere`.
- **Clear** empties the expansion set; **Copy** carries an expanded entry's
  detail and not a collapsed one's.
- The **Detailed timing** checkbox records one `Display` entry when it changes
  and none when clicked to the value it already has.

MCP, in `mcp/tests.rs`: `get_action_log` omits `detail` by default and carries
it when asked, with the depths and order the panel draws;
`set_timing_detail` changes what the next operation records and
`get_timing_detail` reads it back.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| detail level | off | Whether `detail_phase` records. Viewer state, set by the toolbar checkbox or `set_timing_detail`; not read from the environment |
| `FRACTION_INTERVAL` | `30 ms` | Minimum gap between kept `Fraction` reports, in `sfm-explorer`'s `progress::Collector`; phases and messages are never dropped |
| `ActionLog::DETAIL_EVENTS` | `128` | Events kept per entry; the rest are dropped and counted |
| detail indent | 2 spaces per level | Panel and clipboard alike |

## Non-goals

- **A logging framework in `sfmtool-core`.** `Progress` is a parameter, not a
  facade: it has no global state, no filtering language and no appenders, and a
  caller that wants those routes its sink into `log` as the viewer does.
- **Converting the `prof` modules' batch summaries.** Their stderr report
  answers a whole-run question and keeps its own `SFMTOOL_PROFILE` gate. A
  kernel can report phases to a `Progress` and to its own counters at once, and
  the two are read for different reasons.
- **A profiler.** This times named stages of one operation. Sampling, per-thread
  timelines and flame graphs are what a profiler is for.
- **Per-subsystem levels.** One switch. Somebody who wants only the patch
  kernels' detail reads the rows they want.
- **Timing or messaging per item**, at either level. A per-item figure is a
  count, and `count` is what it is for.
- **Learning the split weights from previous runs.** The Action Log records what
  every phase cost, so a later run could in principle weight its bar from an
  earlier one. It is not worth what it costs: the weights would have to be keyed
  on something that says two runs are comparable, kept somewhere across
  sessions, and reasoned about when they are stale or absent. Declared constants
  are wrong by a little and understandable by anyone reading the call.
- **A Python-visible sink.** The bindings pass `Progress::none()`
  (§ "Through the Python bindings"); a callback across the GIL from inside a
  rayon region is a different problem, and nothing has asked for it.
- **Persisting the detail or the level.** Both live as long as the session.

## Open questions

- **Whether the message levels and the detail switch are one dial.** frantic has
  one, running `LOG_NONE` to `LOG_DEBUG`, where we have two message levels plus
  a separate switch for detailed phases: two dials for what may be one question.
  Folding them would make `Level::Debug` messages and `detail_phase` obey the
  same rule, and would turn the **Detailed timing** checkbox into something
  closer to a log level. It removes a concept, and it changes a user-facing
  control into a more technical one, which is why it is here rather than above.
- Whether `Progress` should be a trait rather than a struct with a `dyn Fn`
  sink. A trait would let a caller avoid the indirect call; the struct keeps one
  type in every signature, which is what makes `Progress::none()` a drop-in. The
  struct, until a profile says the indirect call matters.
- Whether the level should persist in the layout file, so somebody debugging
  across restarts does not re-tick it. Against: a viewer that quietly starts up
  profiling is slower than the last person left it, for a reason nobody
  remembers.
- Whether an entry should expand on a click anywhere in the row rather than only
  on the toggle and the time. A row click does nothing today, so there is no
  conflict, but a row that expands on any click is a row a reader cannot select
  text from.
- Whether a kernel should be able to report a `count` whose `total` it revises
  downward mid-run. The bundle adjustment knows its iteration budget but may
  converge early, so its bar would jump to full rather than creep. Allowed, and
  the panel draws whatever it was last told.
