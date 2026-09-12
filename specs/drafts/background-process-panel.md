# The Background panel: an operation that outlives a frame

**Status:** Draft

Amends [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md) § "Non-goals",
[gui/document-model.md](../gui/document-model.md) § "Two kinds of edit",
[gui/panel-layout.md](../gui/panel-layout.md) § "Home positions",
[gui/action-log.md](../gui/action-log.md),
[gui/mcp-server.md](../gui/mcp-server.md) and
[gui/operation-progress.md](../gui/operation-progress.md), whose collector keeps
a status, a count and a fraction that nothing draws until this is built.

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
under a toggle that works as the Action Log's does. A session that has run
nothing says `Nothing running` and no more.

### Running

```
┌ Background ──────────────────┐
│ Bundle adjust                │
│ dino_dog_toy-embedded        │
│ ██████░░░░░░░░░░░  round 2/3 │
│ 42.7 s elapsed       [Cancel]│
│                              │
│ gather arrays          2 ms  │
│ residuals before      10 ms  │
│ solve              ▸ 42.6 s  │
│   round x2         ▸ 42.6 s  │
│     linearise x71     2.9 s  │
│     normal equations  1.9 s  │
│     damping ladder   37.8 s  │
└──────────────────────────────┘
```

- **The name and the node** are the operation and the label it runs on, so a
  glance says what the window is busy with.
- **Progress** is a bar with the kernel's own count and unit when anything
  underneath reports one, and a spinner with the open phase's name when nothing
  does. The bar is a statement about how much of the work is behind you, never a
  prediction of when it will end: it steps at every phase boundary, because
  something did finish, and moves smoothly only across the stages that actually
  report counts ([../gui/operation-progress.md](../gui/operation-progress.md) § "Nesting").
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
  number the person is actually watching.
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
  Repeated stages fold as they do in an entry, which is what makes the table a
  table rather than a log: the sketch's `round x2` and `linearise x71` are two
  rows rather than seventy-three, and their counts climb while the reader
  watches.

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
run again.

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
cancelled after {elapsed}"*, and pushes no version. A **failed** one writes the
refusal the kernel returned, as the synchronous edit does.

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

pub(crate) struct Finished {
    /// The next value, and the map from the input's rows to its own.
    pub outcome: Result<(SfmrReconstruction, PointMap), String>,
    /// The version's label, and the Action Log sentence up to the serials,
    /// which only the GUI thread can know.
    pub version_label: String,
    pub text: String,
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

    /// Apply every report the worker has sent. Returns whether anything
    /// changed, so the frame knows to redraw.
    pub fn poll_background(&mut self) -> bool;

    /// Ask the operation to stop. Silently does nothing when it cannot.
    pub fn cancel_background(&mut self);
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

A tool that starts a background operation **returns when the operation starts**,
not when it finishes:

```json
{"started": "Bundle adjust", "label": "guard"}
```

The agent then polls, or reads the outcome out of the log. This is a break with
what `bundle_adjust` does today, and it is the right break: the alternative is
the call that exists, which times out on any reconstruction worth adjusting
(§ "The problem"), so the agent gets an error for a solve that is going fine and
has no way to tell that from one that failed. Returning immediately makes the
two distinguishable, and the wire already has the vocabulary to follow up.

**The timeout's own message needs the same correction.** It reads "It may be
showing a modal dialog, or be mid-drag", which names two things that are not
what happened and omits the one that did. Once an operation can be in the
background, a call that times out while one is running should say so and point
at `get_background_process`, and a call that times out with nothing running
keeps the message it has, which is then true.

A new read tool, **`get_background_process`**, reports what is running:
the operation, the label, the seconds elapsed, the progress as `done`, `total`
and `unit` when there is one, the open phase, and the completed phases in the
shape `get_action_log { "detail": true }` already returns them in
([../gui/mcp-server.md](../gui/mcp-server.md) § "get_action_log"), so an agent
that reads a finished operation and a running one parses one shape. With
nothing running it reports the last operation of the session, marked `finished`,
so one call answers both "is it done" and "what did it cost".

`get_scene` gains a `background` block of the same shape, beside
`status_message`, so an agent that already polls `get_scene` learns that the
viewer is busy without a second call, and knows not to send an edit that would
be refused.

A **`cancel_background`** tool cancels what is running, and refuses when the
operation cannot be cancelled, with the same sentence the button's tooltip
carries.

The MCP spec's warning that a long `bundle_adjust` will time out while the solve
goes on is deleted, because the behaviour it describes is gone.

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
