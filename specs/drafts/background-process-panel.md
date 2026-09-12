# The Background panel: an operation that outlives a frame

**Status:** Draft

Amends [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md) § "Non-goals",
[gui/document-model.md](../gui/document-model.md) § "Two kinds of edit",
[gui/panel-layout.md](../gui/panel-layout.md) § "Home positions",
[gui/action-log.md](../gui/action-log.md),
[gui/mcp-server.md](../gui/mcp-server.md) and
[gui/operation-progress.md](../gui/operation-progress.md), whose collector's
status, count and fraction this panel is the first reader of, and which gained
`Collector::live` for it.

The worker and the panel are built. What is left is the read surface an agent
sees: `get_background_process`, the `background` block in `get_scene`, and any
operation other than the bundle adjustment.

Its case is measured. A bundle adjustment of `dino_dog_toy-embedded` (85 images,
21 009 points, 392 489 observations) holds the GUI thread for 95 seconds: the
window is frozen for all of it, and every MCP call in that window fails on the
10 second apply timeout with "The viewer did not answer within 10 seconds. It
may be showing a modal dialog, or be mid-drag", which is the wrong reason. The
breakdown the frozen operation recorded is readable afterwards and says the
damping ladder was 87 of those 95 seconds.

## The problem

One operation dominates, and the estimate this draft opened with was about
right: a bundle adjustment of `dino_dog_toy-embedded` (85 images, 21 009 points,
392 489 observations) holds the GUI thread for **95 seconds**, and that is a
small real reconstruction. A resection in place is 838 ms. Every one of them
runs inside the frame that asked for it, so for its whole duration the window
does not redraw, does not orbit, does not answer a keystroke, and does not
answer an agent's call. The person at the window cannot tell a solve that is
working from one that has hung.

An agent fares worse than "no answer". Every call made during those 95 seconds
fails on the 10 second apply timeout
([../gui/mcp-server.md](../gui/mcp-server.md)), which this solve exceeds by a
factor of nine, with

> The viewer did not answer within 10 seconds. It may be showing a modal dialog,
> or be mid-drag.

which names two things that are not what happened. The agent cannot distinguish
a solve in progress from a hung window, and neither can the human.

This proposes running those operations on a worker thread, and a **Background**
panel under the Scene tree that says what is running, on which node, how far
along it is, and what it has spent its time on so far. The panel shows the phase
table live while the operation runs, and that same table is what the Action Log
entry carries once it is done, so watching a long operation and reading about it
afterwards are the same view of the same data
([../gui/operation-progress.md](../gui/operation-progress.md)).

What that table already says about the 95 seconds, read back over MCP after the
window unfroze: 87 of them are the damping ladder, 4.4 are linearisation and 2.9
are the normal equations. So the panel has something worth drawing from the
first commit, and the operation it draws has somewhere useful to point.

## Why this is safe here

The document model is value semantics
([gui/document-model.md](../gui/document-model.md)). A version is an immutable
reconstruction behind an `Arc` plus an overlay that only that version owns; the
base is never written through, `Arc::make_mut` appears nowhere, and there is no
interior mutability in the document. A bulk edit is already a **pure function
from the value at the cursor to the next value**.

That is exactly the precondition for moving one off the GUI thread. The worker
takes a clone of the `Arc` and reads it; the GUI thread keeps drawing the same
allocation and cannot disturb it; and **the document itself needs no lock,
because no part of it is shared mutably**. No part of this proposal wraps
`AppState` in a lock or hands a worker a reference into the scene, and the
reason it does not have to is that the value model was built to make this
possible.

Two small things are shared mutably, deliberately and under a lock, and neither
is the document: the collector the worker reports into and the panel reads
(§ "Reporting progress"), and the cancel flag. Both are a few words of state
that exist to be written by one thread and read by another, which is a different
thing from sharing a reconstruction.

The same argument says what is **not** safe: the GPU upload that follows a new
base. Uploads run against the device and the queue on the GUI thread, and moving
them is a different problem with a different answer. So this proposal shortens
the freeze to the upload, and makes what remains of it legible rather than
invisible.

How much remains is now measured rather than assumed, and for the first adopter
it is almost nothing. The frame that installed the 95 second dino adjustment
spent **6 ms** on uploads: the point positions moved, so their buffer was
rewritten, while the patch atlas and the thumbnails were reused untouched. So
backgrounding that solve turns a ninety-five second freeze into a few
milliseconds of one, and the earlier claim here that what remains is mostly
upload was true of an undo or an open and not of this.

An undo across a bulk edit is the case where the upload dominates, because it
installs a different base and the atlas has to be repacked. That is still worth
shortening and is still not this proposal's to shorten; what has changed is that
the Action Log can now say which of the two any given row was
([../gui/operation-progress.md](../gui/operation-progress.md)), so the question
is answerable before anybody writes code for it.

## What the user sees

### Placement

The left column splits top to bottom. Scene keeps the top of it and a ninth tab,
`Tab::Background`, titled **Background**, takes the bottom at a default share of
0.28:

```
┌────────┬──────────────────┬───────────────┐
│ Scene  │    3D Viewer     │ Image Detail  │
├────────┼──────────────────┴───────────────┤
│Backgr. │  Image Browser │ Action Log      │
└────────┴──────────────────────────────────┘
```

It is a panel like any other: draggable, closeable, ticked in **Panels ▸
Background**, and given a home position of the left edge at 0.18 with Scene as
its default group-mate, so re-opening it from the menu puts it back beside the
tree.

### Idle

The panel is not blank when nothing is running. It shows the last operation of
the session, greyed: its name, its node, what it cost, and its phases, collapsed
under a toggle that works as the Action Log's does. Those phases are the
transcript the panel drew while it ran, not the entry's folded breakdown: an
operation that collapsed into a summary at the instant it finished would be a
panel that changed its mind about what the reader had just watched. A session
that has run nothing says `Nothing running` and no more.

### Running

```
┌ Background ──────────────────┐
│ Bundle adjust                │
│ dino_dog_toy-embedded        │
│ ██████░░░░░░░░░░░  round 2/3 │
│ 42.7 s elapsed       [Cancel]│
│                              │
│     damping ladder    2.0 s  │
│     linearise         51 ms  │
│     normal equations  46 ms  │
│     damping ladder   ▶ 4.7 s │
└──────────────────────────────┘
```

The table is scrolled to its end, because that is where the operation is.
Earlier rows are above it, and none of them is a summary of another:

```
│ gather arrays          2 ms  │
│ • 85 images, 21009 points, … │
│ residuals before      10 ms  │
│   • median 1.014 px over 39… │
│ solve              ▶ 42.6 s  │
│   • 3 rounds, trim 50/12/4 px│
│   round            ▶ 21.4 s  │
│     linearise         48 ms  │
│     normal equations  44 ms  │
│     damping ladder    1.8 s  │
│     linearise         51 ms  │
└──────────────────────────────┘
```

- **The name and the node** are the operation and the label it runs on, so a
  glance says what the window is busy with.
- **Progress** is a bar with the kernel's own count and unit when anything
  underneath reports one, and a spinner with the open phase's name when nothing
  does. The bar is a statement about how much of the work is behind you, never a
  prediction of when it will end: it moves only where a stage reports a count,
  and a stage that reports none moves it not at all. The step at a boundary is
  the next stage's range beginning rather than anything the panel adds ([../gui/operation-progress.md](../gui/operation-progress.md) § "Nesting").
  Nothing interpolates across a silent stage, and there is never a synthesised
  percentage, because a bar moving at a rate nobody measured makes a promise
  about the finish.
- **The status line** is the one thing the operation says it is doing right now,
  replaced as often as it likes: a file name inside a loop over images, the
  member being refined. It is live state and is never kept in the Action Log
  entry, because once the entry exists the answer is "finished"
  ([../gui/operation-progress.md](../gui/operation-progress.md) § "Status is not a message").
  It sits under the bar, because it is the words for the same thing the numbers
  beside the bar count. The collector already keeps one and nothing draws it;
  the sketch above has no status row because the bundle adjustment sets none,
  and the row is absent rather than blank when an operation says nothing.
- **Elapsed** counts up from the instant the operation started, which is the
  number the person is actually watching. A frame is asked for every 100 ms so
  that it keeps counting through a stage that reports nothing.
- **Cancel** is present always and enabled only when the operation can be
  cancelled, with a tooltip saying so when it cannot. The alternative, hiding
  the button, leaves the reader wondering whether they missed it.
- **The phase table** is the live form of what the Action Log entry will hold.
  Completed phases show their cost, the open one is marked and shows the time it
  has been open so far, and phases that have not started are not shown, because
  the panel does not know they are coming. It shows whichever level of timing is
  running, so ticking **Detailed timing** before starting a long operation is how
  somebody watches a kernel's internals
  ([../gui/operation-progress.md](../gui/operation-progress.md) § "Two levels").
  **Nothing folds here, and no two things are ever combined into one row.**
  Each run of a stage is its own row with the cost that run took and the note
  that run gave, each message is its own row, and two runs that said different
  things say both, separately. This is the opposite of what an Action Log entry
  does with the same events, and deliberately so: an entry is read afterwards
  and answers where the time went, for which `round x2` with the ends of its
  note joined is the right summary. A reader watching is asking what the viewer
  is doing now and what it has done so far, and a summary of something they can
  watch unfold tells them less than the thing itself. The entry stays the
  summary of exactly what the panel showed, with every run counted in the row it
  folds into and the costs adding up.

  The table is therefore a log rather than a table, and a long detailed
  operation writes a lot of it: a three-round, sixty-iteration adjustment with
  **Detailed timing** on opens `linearise` and its two siblings five hundred and
  forty times, and every one of those is a row. It is virtualized on a uniform
  row height, as the Action Log's list is, so only the rows in view are drawn.
  The table **follows its tail**: the stage that is running is the
  newest row, and this panel is narrow enough that the stages which finished
  first fill it. Without that, a 102 second solve showed `gather arrays`, which
  cost 2 ms, for the whole of it, and the reader had to scroll to find out what
  the viewer was doing. It holds still the moment the reader scrolls up, which
  is the Action Log's rule for its own list.
- **A row too wide for the column is truncated and says the whole of itself on
  hover**, as an Action Log detail row does. The panel is a fraction of the
  window's width, so that is the common case here rather than the rare one, and
  it applies to the node's label on the idle row as much as to a stage's note.
  The cost is reserved before the names are drawn: a long label would otherwise
  push the number off the panel, and the number is what a reader came here
  for.

### What the rest of the viewer does meanwhile

Everything that reads the scene keeps working. The node shows the version at its
cursor, which is the value the worker was handed and which nothing is mutating,
so orbiting, selecting, opening the Point Track panel, taking a screenshot and
opening a second file all behave exactly as they do when nothing is running.

What is refused is anything that would change the node the operation is running
on. An edit, an undo, a redo, a history jump, a save or a close of that node is
refused with *"{label} is busy: {operation} is still running."* Other nodes are
untouched and remain fully editable. The busy node's Edit menu items are greyed
with the same sentence as their tooltip, so the refusal is visible before it is
provoked.

**One operation at a time, viewer-wide.** Starting a second is refused with
*"{operation} is still running on {label}."* These operations saturate the
machine, so running two would make both slower and neither would finish sooner;
a queue is state with no demand behind it, and is a non-goal below.

Quitting while an operation runs abandons it. The result would have nowhere to
land, and a viewer that refuses to close is worse than a solve that has to be
run again. Refusing a *node* close is a different matter and does happen, which
means `close_node` and `close_all` return a `Result` where they used to return
nothing: a close that cannot be refused cannot be one of the operations
`busy_refusal` covers.

**A job that panics reports a failure.** It cannot be detected by the channel
disconnecting, because the collector the worker and the panel share holds a
sender of its own and the channel therefore stays open; so the worker catches
the unwind and sends `Failed` itself. Without that a panicking kernel would
leave an operation that never finishes, a panel that never clears and a node
that stays busy for the session.

### When it finishes

The finished value arrives on the channel with its row map and the sentence the
edit would have recorded. On the next frame the GUI thread pushes the version,
moves the selection forward through the map, and writes exactly the Action Log
entry the synchronous edit writes today, with the worker's phases attached to
it. The panel drops back to its idle form showing that same operation.

The entry's actor is whoever asked for the operation, not the viewer: an agent
that started a solve is the actor of the version it produced, however many
minutes later it lands.

Its **cost covers the whole operation**, not the frame that pushed it, and its
detail is what the worker reported. Both arrive through one call,
`ActionLog::record_done(kind, started, text, detail)`
([../gui/operation-progress.md](../gui/operation-progress.md), "The Action Log's side"): the
entry is written now, its cost is measured from an instant already past, and it
settles on the frame that draws the result as every other entry does. Without
the start instant a two-minute solve would report the six milliseconds of the
frame that installed it, which is the sort of number that discredits a whole
column.

A **cancelled** operation writes a failed entry, *"{operation} of {label}
cancelled"*, and pushes no version. A **failed** one writes the refusal the
kernel returned, as the synchronous edit does.

Both are timed from `started` like a successful one, and both keep what the
operation reported before it stopped. A solve cancelled seventeen seconds in
spent those seventeen seconds, and a row costing it at the milliseconds of the
frame that collected the answer is the number the paragraph above warns about,
whichever way the operation ended. The elapsed is not repeated in the sentence
for the same reason: the cost column already carries it, and two spellings of
one number can only disagree. Keeping the breakdown is what makes a cancelled
solve informative rather than merely abandoned, since it shows the round it
reached and the median it had got to by then.

Nothing is logged when an operation *starts*. The log records outcomes, not
intentions ([gui/action-log.md](../gui/action-log.md)), and what is running is
what the panel is for.

## Which operations go to the background

The rule is cost, not kind. A point edit is microseconds of work; sending it to
a worker would add a thread hop and a frame of latency to an operation that has
neither today, in exchange for nothing. So:

- **Point edits stay synchronous.** Delete point, add observation, remove
  observation, create point, move camera.
- **Bulk edits go to the background.** Bundle adjust first, since it is the one
  that freezes the window for minutes and the one the wire already warns about.
  Resect in place and delete image follow, once the first has settled.

**Opening a file is not a candidate**, though an earlier version of this draft
assumed it was the other thing that froze a fresh session. Measured, an open of
the 45 MB dino set is 1.45 s of which the read and the derived-index build are
**141 ms**: the rest is 434 ms of GPU upload, which cannot move
(§ "Non-goals"), and 868 ms of the renderer starting, which happens once.
Backgrounding it would move a seventh of the wait off the thread and complicate
the load path for it.

The **materialisation** an edit performs before calling a kernel is the one
plausible further adopter: it is a pure function over a value nothing else
holds, so it fits the mechanism without extending it. Whether it is worth
anything is unmeasured, and now measurable: it has a `materialise` phase, and a
reader can settle the question from the Action Log before anybody writes the
code.

## Reporting progress

The channel from a worker to this panel is
[../gui/operation-progress.md](../gui/operation-progress.md)'s `Progress`: one parameter the
kernel takes, carrying phases, messages, progress counts and the cancel flag.
The worker builds it over a collector the GUI thread shares, so the panel reads
what has been reported by locking that collector each frame rather than by
receiving it. Nothing about a phase or a message crosses the channel below; the
channel carries only the fact that something changed, and at the end the value.

**How much a given kernel reports is its own business**, and a kernel that
reports nothing still works: it is one phase, named by the caller, with a
spinner under it. The panel is finished when it can draw phases, messages, a
count and a spinner; each kernel then decides how much of that it fills in.

An earlier version of this draft staged the first adopter that way, with
`bundle_adjust` spinning and a disabled Cancel until somebody threaded a
`Progress` into its iteration loop. That has happened
([../gui/operation-progress.md](../gui/operation-progress.md)), so the first
version of this panel starts further along than it planned to:
`bundle_adjust` already counts its rounds against the schedule and its LM
iterations against the budget, so the bar is live rather than a spinner, and it
already polls the cancel flag between rounds and between iterations, so Cancel
is enabled rather than explained away. The first `Operation` to declare
`cancellable: true` is the first one built.

## Rust API

`crates/sfm-explorer/src/background/`: `mod.rs` owns the process and the
channel, `panel.rs` the egui view, `tests.rs` the tests. The process is a field
of `AppState`, so the busy check is where every method that would need it
already is.

```rust
/// A long operation running off the GUI thread.
pub(crate) struct BackgroundProcess {
    /// What it is, for the panel and the refusals, and what it claims about
    /// itself.
    pub operation: Operation,
    /// The node it will install its answer into, and the node it locks.
    pub node: ReconId,
    pub label: String,
    pub started: std::time::Instant,
    /// Who asked, so the entry this writes belongs to them and not to the
    /// viewer, however many minutes later it lands.
    pub actor: Actor,
    /// Which run this is, so a handle names an operation rather than merely
    /// the fact that one was running.
    pub id: u64,
    /// Where the worker reports, and where the panel reads. Shared, taken by
    /// `&`, never borrowed mutably
    /// ([../gui/operation-progress.md](../gui/operation-progress.md) § "In the viewer").
    pub collector: Arc<progress::Collector>,
    /// Set to ask the operation to stop. The kernel polls it through the
    /// `Progress` built over `collector`; one that never polls is not
    /// cancellable, and the panel says so.
    pub cancel: Arc<AtomicBool>,
}

/// One kind of background operation: what to run, and what it claims about
/// itself.
///
/// `cancellable` is a **declaration**, not something discovered: a kernel that
/// never polls the flag is indistinguishable from one that has not reached a
/// poll yet, so nothing can infer this. The wrapper knows which kernel it
/// calls, so the wrapper states it, and a test holds it to the claim by
/// cancelling each operation that says `true` and asserting it stops.
pub(crate) struct Operation {
    pub name: &'static str,
    pub cancellable: bool,
}

/// What crosses the channel, worker to GUI thread. Phases, messages and counts
/// do not: they are in the collector both sides hold.
pub(crate) enum Report {
    /// Something was reported. Carries nothing; it exists to wake the loop.
    Progressed,
    /// The operation is over, one way or another.
    Done(Box<Finished>),
}

/// How an operation ended.
///
/// Three ways rather than a `Result`, because only the job knows it was
/// cancelled: the kernel is what met the flag and said so, and deciding at poll
/// time from the flag alone races a solve that finished on its own between the
/// last poll and the cancel.
pub(crate) enum Finished {
    Produced {
        /// The next value, and the map from the input's rows to its own.
        value: SfmrReconstruction,
        map: PointMap,
        /// The version's label, and the Action Log sentence up to the serials,
        /// which only the GUI thread can know.
        version_label: String,
        text: String,
    },
    Cancelled,
    Failed(String),
}
```

and on `AppState`:

```rust
impl AppState {
    pub fn background(&self) -> Option<&BackgroundProcess>;

    /// Why an edit of `id` is refused right now, or `None`.
    ///
    /// Every method that gives a node a new version calls this first, so one
    /// sentence covers the menu's greying, the method's refusal and the wire's.
    pub fn busy_refusal(&self, id: ReconId) -> Option<String>;

    /// Start `operation` on `id`. Refuses when anything is already running.
    pub fn start_background(&mut self, operation: Operation, id: ReconId)
        -> Result<(), String>;

    /// Apply every report the worker has sent.
    ///
    /// The frame asks two questions of this, not one: whether to repaint, which
    /// is true of every report, and whether a version landed, which is true
    /// only at the end and is what tells the panels to drop the caches they
    /// keep about a table that has just been renumbered. Answering with one
    /// bool would flush every texture on every report, which during a long
    /// solve is a thousand flushes for one renumbering.
    pub fn poll_background(&mut self) -> Polled;

    /// Ask the operation to stop.
    pub fn cancel_background(&mut self);

    /// Why a cancel is refused right now, or `None`.
    ///
    /// The sentence has to exist somewhere readable, because the button carries
    /// it as a tooltip and the wire carries it as a refusal.
    pub fn cancel_refusal(&self) -> Option<String>;
}
```

`poll_background` runs **in phase 0, before the MCP drain**. A completed
operation's version is then on screen in the frame it landed, and an agent's
call in that same frame reads the new value rather than the old one. The worker
wakes an idle event loop the way the MCP server does, with
`EventLoopProxy::send_event`; the `UserEvent` variant is unconditional rather
than behind the `mcp` feature, because the background panel is not an MCP
feature.

**Reports carry nothing, so they cannot pile up.** A worker that reports per
iteration sends a `Progressed` with no payload; the GUI thread drains whatever
is waiting, learns only that it should repaint, and reads the current state out
of the collector once. A thousand reports in a frame cost one lock rather than a
thousand repaints, and no event is ever in two places.

## On the wire

A tool that starts a background operation answers **one of two ways, decided by
how long the operation takes**, not by which tool it is.

An operation that finishes within `REPLY_DIRECTLY_WITHIN` replies exactly as it
does today, with its normal result. An agent adjusting a small reconstruction
sees no change at all, and the wire break is paid only by the calls that were
already broken. One still running at the threshold replies with a handle:

```jsonc
{
  "running": true,                          // present and true only in this case
  "operation": "Bundle adjust",
  "reconstruction_label": "dino_dog_toy-embedded",
  "operation_id": 2                         // names this run, not merely "one is running"
}
```

`running` is the discriminator, so a reader tests one field rather than sniffing
the shape. `operation_id` names *which* run, so a poll still answers about an
operation that has since finished and been replaced.

The alternative, replying with a handle always, would change the shape for every
caller including one whose solve was over before the reply was written. The
other alternative, the call as it exists, times out on any reconstruction worth
adjusting (§ "The problem"), so an agent gets an error for a solve that is going
fine and cannot tell it from one that failed.

**The threshold is a perception one.** Below roughly 100 ms a reply reads as
instantaneous; up to about a second a caller stays in flow and simply sees the
system working; past that, attention wanders and a wait wants explaining.
`REPLY_DIRECTLY_WITHIN` is 200 ms: just past instantaneous, well short of
anything anyone would call slow, so a handle comes back only for operations that
genuinely are. It is deliberately not justified by what any one reconstruction
costs, since a constant argued from a fixture ages the moment the fixture does.

**Nothing blocks to reach it.** Waiting out the threshold on the GUI thread
would trade one long freeze for a short freeze on every call. The tool starts
the operation and defers its reply through the path screenshots already use
([../gui/mcp-server.md](../gui/mcp-server.md) § "screenshot"), and each frame
the resolver sends the result if the operation has finished or the handle if the
threshold has passed, whichever is true first. A deferred reply also has to
request a repaint, or an idle viewer never reaches the clock that would answer
it.

**The timeout's own message follows what is running.** It used to read "It may
be showing a modal dialog, or be mid-drag", which names two things that are not
what happened and omits the one that did. A call that times out while an
operation is running now names the operation and the node it is on and points at
`get_background_process`; a call that times out with nothing running keeps the
message it had, which is then true. The thread composing that sentence is the
one thread that cannot ask `AppState`, since it is composing it precisely
because the GUI thread did not answer, so it reads a small shared notice,
`background::BusyNotice`, written where `AppState::background` is written and
nowhere else.

A read tool, **`get_background_process`**, reports what is running: the
operation, the label, the seconds elapsed, `fraction` of the whole where
anything reported one, whether it can be cancelled, the progress as `done`,
`total` and `unit` where there is one, the status, the open phase, and the
stages so far in the shape `get_action_log { "detail": true }` returns them in
([../gui/mcp-server.md](../gui/mcp-server.md) § "get_action_log"), so an agent
that reads a finished operation and a running one parses one shape. The rows are
the panel's transcript rather than the entry's folded summary, for the reason
the panel does not fold (§ "Running"): this tool answers about an operation
being watched. `get_action_log { "detail": true }` is where the summary is. With nothing
running it reports the last operation of the session, marked `finished`, so one
call answers both "is it done" and "what did it cost". `elapsed_s` carries both
halves of that: seconds so far while it runs, seconds in total once it is over.

**The phase list is capped at the size an entry's is**,
`ActionLog::DETAIL_EVENTS`, with a `"{n} earlier events dropped"` row. Not a
size limit chosen for the wire: it is the number the Action Log already applies
to these rows. It keeps the **last** of them rather than the first, which is the
opposite of an entry's cap, because this list is the panel's transcript and
nothing has collapsed the repetition in it: the first 128 rows of a long solve
are its first few seconds. An entry's first 128 rows are its shape, because the
fold got there first.

`get_scene` gains a `background` block beside `status_message`, so an agent that
already polls `get_scene` learns that the viewer is busy without a second call,
and knows not to send an edit that would be refused. It is the same shape with
everything unbounded left out: `running`, `operation`, `reconstruction_label`,
`operation_id`, `elapsed_s` and `fraction`, and `null` with nothing running.
The phase table belongs to the tool an agent asks when it wants it, because
`get_scene` is the most-polled call on the surface and a block that grew with
the solve would be paid for on every poll; the open phase and the status line go
with it, being narrative rather than something a caller acts on. `null` rather
than the last operation for a second reason as well: a block that outlived its
operation would make `background != null` stop meaning "the viewer is busy",
which is the one thing it is read for.

A **`cancel_background`** tool cancels what is running, and refuses when the
operation cannot be cancelled, with the same sentence the button's tooltip
carries.

## Testing

`crates/sfm-explorer/src/background/tests.rs`, headless, driving a fake worker
over the real channel so the tests are about the seam and not about a solver:

- A report moves the panel's progress, appends its phases, and asks for a
  redraw.
- `Done` with a value pushes exactly one version, moves the selection through
  the map, and writes one Action Log entry whose cost spans the whole operation
  rather than the frame that pushed it.
- `Done` with an error writes one failed entry and pushes nothing.
- A cancel writes one failed entry naming the elapsed time, and pushes nothing.
- Every edit, undo, redo, jump, save and close of the busy node is refused, with
  the same sentence in each case; the same operations on another node are
  allowed.
- Starting a second operation is refused while one runs.
- **The value was not copied to run in the background**: the base `Arc` the
  worker was handed is pointer-equal to the one the node is still drawing.

Panel, through `test_support::run_frame_headless`:

- The three states paint what they should: idle with nothing run, idle with a
  finished operation, and running with a bar and with a spinner.
- Cancel is disabled, with its tooltip, for an operation whose kernels do not
  poll the flag, and live for one that does.
- A status replaces the previous one rather than adding a line, and the panel
  shows nothing there for an operation that has never set one.
- **Every `Operation` that declares `cancellable` really is**: cancelling each
  one stops it and writes the cancelled entry. A declaration nothing checks is a
  declaration that rots.

`crates/sfm-explorer/src/mcp/tests.rs`, over a fake operation held open on the
editing fixture's node, so the wire is read at an instant the test chose:

- `get_background_process` answers one shape running and finished: the same
  `operation`, `reconstruction_label` and `operation_id` either way, a cost no
  shorter than the elapsed it was read at, and `running` telling the two apart.
  A session that has run nothing answers `running: false, finished: false` and
  no more.
- **Its `phases` is the transcript and the entry is the summary of it**: every
  run the wire reported is counted in the row the entry folds it into, and the
  entry is then held to the panel's own rows through the agreement assertion
  every other breakdown test ends on.
- A breakdown longer than `DETAIL_EVENTS` is the dropped line and the last 128
  rows, and reads the same after the operation ends as it did during it.
- **`get_scene`'s `background` block does not grow with the solve**: its whole
  key set is asserted, so a field added to the most-polled reply on the surface
  is a deliberate act. It is `null` before an operation and again after it.
- The apply timeout's message names the operation only while one is running, and
  the notice it reads is empty before the operation, set during it, and empty
  again afterwards.

`crates/sfm-explorer/tests/ui_basic.rs`: the Background panel is in the
accessibility tree, and **Panels ▸ Background** ticks it.

`crates/sfm-explorer/src/layout/tests.rs` and `dock/tests.rs`: the default
layout's left column is a top-bottom split of Scene over Background at 0.28, and
Background's home position is the left edge with Scene as its group-mate.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| left column split (`Layout::default`) | `0.72` | Scene's share of the left column; Background takes the rest |
| Background home edge / share | left / `0.18` | Same edge and share as Scene, whose group-mate it is |
| `REPLY_DIRECTLY_WITHIN` | `200 ms` | How long a tool waits before answering with a handle instead of a result (§ "On the wire") |
| repaint tick while running | `100 ms` | The elapsed counts up between reports, and a worker deep in a silent stage sends none for a frame to ride on |
| seconds shown to | one decimal | The cost column is read here while it moves, and at ten frames a second a hundredths digit only spins ([../gui/action-log.md](../gui/action-log.md)) |

## Non-goals

- **A queue.** One operation at a time, and a second is refused rather than
  queued. These saturate the machine; two at once finish no sooner.
- **Backgrounding point edits.** They cost microseconds, and a frame of latency
  is a worse deal than the freeze it avoids.
- **Moving GPU uploads off the frame.** They run against the device and the
  queue on the GUI thread, and shortening them is a separate piece of work with
  a different mechanism. What remains of the freeze after this proposal is
  whatever the upload costs, which is 6 ms for a bundle adjustment and the
  larger part of an undo across one.
- **Editing a node while an operation runs on it.** The operation is a function
  of the value it was handed, and an edit underneath it would produce a version
  whose parent is not the version it was computed from.
- **Persisting a running operation.** Quitting abandons it.
- **Progress from a kernel that does not report it.** The panel shows a spinner
  and the open phase's name, and says nothing it cannot measure.

## Open questions

- Whether an operation should be allowed per node rather than one viewer-wide.
  Per node is the natural generalisation and costs little in the state; it is
  held back because nothing has asked, and because two solves at once is a
  slower way to do one thing.
- Whether the idle panel should keep more than the last operation. A short
  history of what was run and what it cost is the Action Log's job, and the
  panel would be duplicating it; but the panel is where a reader looks for cost,
  and three rows there may save a scroll.
- Whether a background operation should be allowed to start while a modal dialog
  is open. Today the dialog is what starts it, so the question is only about the
  file dialogs, which stop the GUI thread anyway.
