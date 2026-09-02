# Action Log

## Purpose

A viewer that a human and an agent both drive needs a record of who did what,
and when. SfM Explorer already reports each action in a one-line status
message painted on the 3D viewport, and the MCP control surface prefixes its
own actions there with `MCP:` so the person watching can tell them from their
own. But that line holds exactly one message. The moment the next action runs,
the previous one is gone; there is no way to scroll back, no way to copy it
into a bug report or an agent transcript, and no time on it to correlate with
anything else.

The Action Log is the longer-term record that status line was standing in for.
It is a dock panel, docked by default as a second tab beside the Image Browser
strip, that shows a scrolling, timestamped, terminal-style list of every action
taken in the viewer: opening and closing files, changing the scene graph,
moving the selection, framing the view, playing the image animation, and every
call the MCP endpoint applies. Each entry says which of three actors did it —
the user at the window, an agent over MCP, or the viewer itself — and the
viewport status line becomes a view of the log's most recent entry rather than
a separate piece of state.

The log records outcomes, not intentions. An action that changed nothing
(selecting the image already selected) writes no entry. A request that failed
writes an entry marked as failed with the reason. Continuous manipulation —
orbiting the camera, dragging a slider, scrubbing through images, an agent
polling for a screenshot — is held to a reasonable granularity by folding a run
of like entries into one, so the log stays readable during exactly the periods
when it would otherwise scroll fastest.

## What the user sees

### Placement

The panel is a seventh dock tab, `Tab::ActionLog`, titled **Action Log**. In
the default layout it joins the bottom node beside the Image Browser as the
second, non-active tab, so the viewer still opens on the image strip:

```
┌────────┬──────────────────┬───────────────┐
│        │    3D Viewer     │ Image Detail  │
│ Scene  ├──────────────────┴───────────────┤
│        │  Image Browser │ Action Log      │  ← tab group, Image Browser active
└────────┴──────────────────────────────────┘
```

The user can drag it anywhere `egui_dock` allows, as with every other tab. It
is not closeable, like every other tab, and there is no menu item for it: the
layout is not persisted and no View menu exists (see
[viewport-hud.md](../gui/viewport-hud.md) for why), so a closeable tab would be
lost until the next launch.

### The list

The body is a vertical scroll area of one-line, fixed-height rows in the
monospace font, oldest at the top, newest at the bottom:

```
14:03:07  Viewer  SfM Explorer 0.2.0 started
14:03:07  Viewer  MCP endpoint listening on http://127.0.0.1:8787/mcp
14:03:08  User    Opened seoul_bull from C:\data\seoul_bull.sfmr
14:03:41  User    Selected image IMG_0007.jpg in seoul_bull
14:04:02  MCP     get_scene
14:04:02  MCP     Opened global from C:\data\global.sfmr
14:04:03  MCP     Aligned global → seoul_bull: 15/17 cameras, RMS 0.031
14:04:05  MCP     screenshot 1280×720
14:04:19  MCP     select_camera_image failed: No loaded reconstruction is labelled `globl` — loaded: `seoul_bull`, `global`.
14:05:10  User    Closed all (2)
```

Three columns: the local time of day to the second, the actor, and the text.
Hovering a row shows the full timestamp with date and UTC offset. A text longer
than the row is truncated with an ellipsis and shown whole in the same tooltip;
rows never wrap, because the list is virtualized on a uniform row height.

Colour carries the rest of the entry's shape, so the columns stay clean:

| Entry | Time | Actor | Text |
|-------|------|-------|------|
| Action by the user | weak | default | default |
| Action over MCP | weak | hyperlink colour | default |
| Action by the viewer | weak | weak | default |
| Query over MCP (a read-only tool) | weak | weak | weak |
| Failed, any actor | weak | as above | `error_fg_color` |

The actor column is what makes an MCP row visually distinct — there is no
`MCP:` prefix in the text, because the text of an action never depends on who
took it (§ "One text per action, whoever took it").

### Following the tail

The list sticks to the bottom: while the view is scrolled to the newest entry
it stays there as entries arrive, and the moment the user scrolls up it stops
following and holds still, so an entry can be read while an agent keeps
working. This is `egui::ScrollArea::stick_to_bottom` — the panel keeps no
follow state of its own. A **Latest** button in the toolbar scrolls back to the
end.

### Toolbar

A single row above the list:

- The entry count, `n entries`, and `(m dropped)` once the capacity has been
  hit.
- **Latest** — scroll to the newest entry.
- **Copy** — put the whole log on the clipboard as plain text, one entry per
  line, with the full date in the timestamp so a pasted excerpt stands alone:
  `2026-09-01 14:04:03  MCP     Aligned global → seoul_bull: …`. Failed entries
  carry a `!` between the actor and the text, since colour does not survive
  the clipboard.
- **Clear** — empty the log. This also clears the viewport status line, which
  is derived from the log. Not confirmed; the log is a convenience, not data.

### The viewport status line

The status message painted under the scene stats in the 3D viewport, and in
red in the empty-state panel when nothing is loaded, is **the text of the
most recent entry that is not a successful query**, prefixed `MCP: ` when that entry's actor is the
agent. That is the whole contract: the status line is a one-row window onto
the log and has no state of its own. Consequences that differ from before:

- A selection change now shows in the status line, whoever made it. This
  closes the gap where the MCP selection tools were meant to announce
  themselves and did not.
- A refused MCP action now shows in the status line (as a failed entry), where
  before only its success did.
- A successful action no longer leaves a stale error on screen, because the
  new entry replaces it; the four sites that cleared the field to achieve that
  go away.
- MCP queries — `get_scene`, `screenshot`, and the rest — are logged but, when
  they succeed, never reach the status line, so an agent polling `get_scene`
  does not read its own polling back as the viewer's status. A refused query
  does reach it, like any other failure.

## What gets logged

The catalogue, by kind. **Kind** decides whether a run of entries coalesces
(§ "Coalescing"); **actor** is whoever applied the action. Texts are the exact
strings, with `{…}` for the values that vary.

| Kind | Coalesces | Actor | Text |
|------|-----------|-------|------|
| Session | no | Viewer | `SfM Explorer {version} started` |
| Session | no | Viewer | `MCP endpoint listening on {url}` |
| File | no | User / MCP | `Opened {label} from {path}` |
| File | no | User / MCP | `Reloaded {label}` |
| File | no | User / MCP | `Failed to load {path}: {error}` — **failed** |
| File | no | User / MCP | `Failed to reload {path}: {error}` — **failed** |
| File | no | User / MCP | `Closed {label}` |
| File | no | User / MCP | `Closed all ({n})` — one entry, not one per node |
| File | no | User | `Loaded demo data` |
| Scene | no | User / MCP | `Soloed {label}` / `Ended the solo` |
| Scene | no | User / MCP | `{label} hidden` / `{label} shown` |
| Scene | no | User / MCP | `{Points|Camera images|Patches|Points at infinity} of {label} hidden/shown` |
| Scene | no | User / MCP | `{label} made non-interactive` / `interactive` |
| Scene | no | User / MCP | `Tint of {label}: {Red|Green|…|None}` |
| Scene | no | User | `Reset transform of {label}` |
| Scene | no | User | `Aligned {src} → {tgt}: {i}/{n} {cameras|points}, RMS {rms:.3}` — existing text |
| Scene | no | User | `Align {src} → {tgt} failed: {reason}` — existing text, **failed** |
| Scene | no | User | `Resected {image} in {label}: …` — existing text |
| Scene | no | User | `Resect {image} in {label} refused: {reason}` — existing text, **failed** |
| Selection | yes | User / MCP | `Selected reconstruction {label}` |
| Selection | yes | User / MCP | `Selected image {name} in {label}` |
| Selection | yes | User / MCP | `Selected camera intrinsics #{k} in {label}` |
| Selection | yes | User / MCP | `Selected point {pt3d_id}` |
| Selection | yes | User / MCP | `Cleared selection` / `Deselected image` / `Deselected camera intrinsics` / `Deselected point` |
| View | yes | User / MCP | `Framed the scene` / `Framed {label}` / `Framed camera #{k} of {label}` |
| View | yes | User / MCP | `Looking through {name}` / `Left camera view` |
| View | yes | User | `Levelled the horizon` / `Reset the view` |
| View | yes | MCP | `Camera placed` / `Camera restored` / `Field of view {fov:.1}°` |
| Display | yes | User | `{Control} {on|off}` for HUD checkboxes, e.g. `Grid off` |
| Display | yes | User | `{Control} {value}` for HUD sliders, e.g. `Point size 3.0`, `Scene scale 0.031` |
| Animation | no | User | `Animation playing at {fps} fps` / `Animation paused at {name}` |
| Animation | no | Viewer | `Animation reached the end at {name}` |
| Animation | no | User | `Animation rate {fps} fps` |
| Query | yes | MCP | `get_scene` / `list_camera_images {label} {offset}..{end}` / `get_camera_image {label} {name}` / `get_camera_intrinsics {label} #{k}` / `get_point {pt3d_id}` / `screenshot {w}×{h}` |
| any | never | MCP | `{tool} failed: {reason}` — **failed**, for any MCP tool the viewer refuses |

Rules that the table implies:

- **Only a change is logged.** Selecting the already-selected image, hiding a
  hidden node, or `set_solo` on the current solo writes nothing. The `AppState`
  method that owns the change is the one that knows, so it is the one that
  logs.
- **Composite actions log once.** `Closed all (3)` is one entry, not three;
  `Cleared selection` is one entry even though it deselects the image, the
  point and the intrinsics; a resection logs its result, not the internal node
  append it ends with. Inner calls are muted while the outer method runs
  (§ "Rust API"). The one exception is deliberate: a
  `set_reconstruction_display` naming several fields records **one entry per
  field it changed**, because each of those fields has its own row above and is
  its own thing to the person watching the window — the call is a batch, not an
  action.
- **`Looking through {name}` marks a deliberate entry into camera view** — a
  double-click, `Z`, a Scene-tree or track-panel double-click, MCP
  `set_view {look_through}`. The *instant* switch that `,` / `.` and animation
  playback make while already in camera view records nothing: it follows a
  selection step whose own `Selected image …` entry already names the image, and
  a `Looking through …` between every two of those would break the coalescing
  that keeps a scrub to one line.
- **The go-to-point dialog, the viewport click, the browser strip, the scene
  tree and the `,` / `.` keys all log the same `Selected image …` text**,
  because all of them end in `AppState::select_image`. Which control was used
  is not recorded.
- **A failed entry is never coalesced away**, whatever its kind.

Not logged, by design:

- Continuous navigation: drag-orbit, pan, scroll-zoom, pinch, WASD fly, and
  Alt-click to set the orbit target. These are how the user looks, not what
  they did; the discrete framing commands are logged instead.
- Hover, anywhere.
- Per-frame selection advances during animation playback are logged through
  the ordinary selection path, so they *are* entries — but they coalesce into
  a single `Selected image …` line that keeps updating under the
  `Animation playing` entry, which is the granularity wanted.
- Dock layout changes (dragging a tab, resizing a split). `egui_dock` owns
  these and the app never observes them.
- MCP requests the viewer never sees: a call whose arguments fail the schema is
  refused on the HTTP thread as a protocol error, and a request that times out
  or is dropped is reported to the client only. The log records what the
  viewer did; a request that never reached it is in the agent's own transcript.

## Coalescing

A run of like entries folds into one. When a new entry arrives, it **replaces**
the newest existing entry — text and timestamp both — instead of being appended
if all of these hold:

1. both have a kind that coalesces (Selection, View, Display, Query);
2. they have the same kind — and for queries, the same tool name;
3. they have the same actor;
4. neither is a failed entry;
5. the new entry's time is within `COALESCE_WINDOW` (1 s) of the existing
   entry's time.

The window is measured from the *replaced* entry's timestamp, which is itself
the time of the last replacement, so an unbroken run coalesces indefinitely: a
17-image scrub with the arrow keys is one line, a slider drag is one line, an
animation playing at 4 fps is one line, and an agent taking a screenshot every
200 ms is one line. The run breaks on any pause longer than the window, on a
different kind, on a change of actor (the user clicking during an agent's
selection run), or on a failure. The user's own scrub and the agent's
selection therefore never merge into each other.

Coalescing is decided at record time and is not reversible; the entries it
replaces are gone. This is deliberate — the log is a readable record, not an
audit trail — and is the reason failures are exempt.

## Timestamps

Each entry carries a `jiff::Timestamp`, the instant it was recorded, and the
panel formats it in the system's local time zone, resolved once when the log is
constructed. The list shows `HH:MM:SS`; the tooltip and the clipboard copy show
`YYYY-MM-DD HH:MM:SS` with the UTC offset in the tooltip. Sub-second precision
is kept in the timestamp but not shown: it is what the coalescing window is
measured against, not something the reader needs.

Wall-clock time rather than time-since-launch, because the log's main job is to
be read *next to* something else — an agent's transcript, a CI log, a screen
recording — and only wall-clock time lines those up. Local rather than UTC
because a human at the window reads it.

This is the crate's first date/time dependency (`jiff`, default features).
Chosen over `chrono` for a smaller dependency graph and over `time` because
`time`'s local-offset lookup is unsound on Unix without an opt-in flag. The
workspace MSRV is 1.95; `jiff` needs 1.70. On Windows and macOS the zone comes
from the OS; on Linux from `/etc/localtime` or `TZ`, falling back to UTC with a
one-time `log::warn!` if neither resolves, which is `jiff`'s own behaviour.

## Rust API

`crates/sfm-explorer/src/action_log/`: `mod.rs` owns the buffer and the
recording rules, `panel.rs` the egui view, `tests.rs` the tests. The buffer is
a field of `AppState`, `pub action_log: ActionLog`, so every place that already
has the state to mutate has the log to write.

```rust
/// Who took an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Actor {
    /// Someone at the window: a click, a key, a menu item, the command line.
    User,
    /// An agent, through the `--mcp` endpoint.
    Mcp,
    /// The viewer on its own: startup, an animation reaching its end.
    Viewer,
}

/// What an entry is about; decides whether a run of them coalesces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Session,
    File,
    Scene,
    Selection,
    View,
    Display,
    Animation,
    /// A read-only MCP tool, by name. Coalesces per tool.
    Query(&'static str),
}

impl Kind {
    pub(crate) fn coalesces(self) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    pub at: jiff::Timestamp,
    pub actor: Actor,
    pub kind: Kind,
    pub failed: bool,
    pub text: String,
}

pub(crate) struct ActionLog { /* VecDeque<Entry>, the current actor, a mute depth, the zone, a dropped count */ }

impl ActionLog {
    pub(crate) const CAPACITY: usize = 10_000;
    pub(crate) const COALESCE_WINDOW: jiff::SignedDuration = jiff::SignedDuration::from_secs(1);

    /// A log formatting in the system's local time zone.
    pub(crate) fn new() -> Self;
    /// A log formatting in `zone`. The tests use a fixed one.
    pub(crate) fn with_zone(zone: jiff::tz::TimeZone) -> Self;

    /// Record a successful action as the current actor, now.
    pub(crate) fn record(&mut self, kind: Kind, text: impl Into<String>);
    /// Record a failed action as the current actor, now. Never coalesces.
    pub(crate) fn fail(&mut self, kind: Kind, text: impl Into<String>);
    /// Record one entry as `actor`, restoring the standing one afterwards.
    pub(crate) fn record_as(&mut self, actor: Actor, kind: Kind, text: impl Into<String>);
    /// Record a read-only MCP tool call: `Kind::Query(tool)`, coalescing per tool.
    pub(crate) fn query(&mut self, tool: &'static str, text: impl Into<String>);
    /// Record `response.changed()` as one entry, building the text only then.
    pub(crate) fn changed(&mut self, response: &egui::Response, kind: Kind,
                          text: impl FnOnce() -> String);
    /// The `record` / `fail` primitives with an explicit instant, for tests.
    pub(crate) fn record_at(&mut self, at: jiff::Timestamp, kind: Kind, failed: bool, text: impl Into<String>);

    /// Who subsequent entries are attributed to. `User` by default.
    pub(crate) fn set_actor(&mut self, actor: Actor);
    pub(crate) fn actor(&self) -> Actor;

    /// Suppress recording until the matching `unmute`. Nests.
    pub(crate) fn mute(&mut self);
    pub(crate) fn unmute(&mut self);

    /// Every entry, oldest first. Reversible, for `status_line`.
    pub(crate) fn entries(&self) -> impl ExactSizeIterator<Item = &Entry> + DoubleEndedIterator;
    /// One entry by position — what the virtualized list asks for, since it
    /// lays out a slice of the rows rather than all of them.
    pub(crate) fn get(&self, index: usize) -> Option<&Entry>;
    pub(crate) fn len(&self) -> usize;
    pub(crate) fn dropped(&self) -> usize;
    pub(crate) fn clear(&mut self);

    /// The most recent entry that is not a successful query, as the status line shows it:
    /// prefixed `MCP: ` when its actor is `Mcp`.
    pub(crate) fn status_line(&self) -> Option<String>;

    /// The whole log as clipboard text, one line per entry.
    pub(crate) fn to_clipboard_text(&self) -> String;
    /// One entry in that same form — also the `log::info!` mirror's line.
    pub(crate) fn line(&self, entry: &Entry) -> String;
    /// `at` in this log's zone, through `jiff`'s `strftime`.
    pub(crate) fn format(&self, at: jiff::Timestamp, fmt: &str) -> String;
}

/// The panel body. Draws the toolbar and the virtualized list into `ui`.
pub(crate) fn show(ui: &mut egui::Ui, log: &mut ActionLog);
```

The module also owns the **texts a scene-graph change produces**, because the
Scene panel and the MCP `set_reconstruction_display` tool both produce them and
two rows that read the same must have done the same thing:

```rust
/// Which of a node's layers a visibility toggle governs.
pub(crate) enum Layer { Node, Points, CameraImages, Patches, PointsAtInfinity }

pub(crate) fn visibility_text(label: &str, layer: Layer, shown: bool) -> String;
pub(crate) fn interactive_text(label: &str, interactive: bool) -> String;
pub(crate) fn tint_text(label: &str, tint: crate::scene::NodeTint) -> String;
```

And on `AppState`, replacing the `status_message: Option<String>` field:

```rust
impl AppState {
    /// What the viewport status line shows. See `ActionLog::status_line`.
    pub fn status_message(&self) -> Option<String>;

    /// The scene and the log at once — a split borrow, for the two callers
    /// that write a node's display state and record what they wrote in the
    /// same breath (the Scene panel's toggles, and the MCP display tool).
    pub fn scene_and_log(&mut self) -> (&mut [SceneNode], &mut ActionLog);
}
```

Four `AppState` methods are new or changed shape, all so that an action has one
owner that knows it happened:

| Method | Why |
|--------|-----|
| `load_file(&Path) -> Result<ReconId, String>` | The failure is returned, not written, so the caller words it (§ "Threading and the MCP seam") |
| `reload_node(ReconId) -> Result<ReconId, String>` | The same, in place of the old `Option` |
| `set_solo(Option<ReconId>)` | The set form the MCP tool needs, with `toggle_solo` resolving a click into it — so one method owns the entry |
| `clear_selection()` / `deselect_point()` | So that "drop everything" is one entry rather than the three deselects it is made of |

### Why it is shaped this way

**An ambient actor, not an actor argument.** The alternative — every mutating
`AppState` method taking an `Actor` — would touch a dozen signatures and every
GUI call site for the benefit of exactly one caller that is not the user. The
MCP drain is the only place an agent's action enters the viewer, and it is one
function called once per frame, so it sets the actor to `Mcp` before applying
the frame's commands and restores `User` after (§ "Threading and the MCP
seam"). Nothing else ever changes it except the viewer's own startup and
animation-end entries, which use `record_as(Actor::Viewer, …)` — one entry with
the standing actor restored, rather than a set the caller has to remember to
undo. A `debug_assert!` that the actor is `User` at the top of the batch catches
an unbalanced set.

**Mute as a depth counter, not a flag.** Composite actions nest — `close_all`
over `close_node`, `resect_image` over `append_node`, `clear_selection` over
three deselects — and each layer only knows about itself.

**One text per action, whoever took it.** The MCP `write` and `view` modules
no longer compose their own messages and no `announce` helper exists. They call
the same `AppState` and `Viewer3D` methods the GUI calls, and those methods
log. That is what makes the actor column trustworthy: two rows with the same
text did the same thing. It also fixes the drift where the MCP selection tools
were specced to announce and did not — they now cannot fail to, because the
logging is in `select_image`, not in the tool.

**`status_line` returns a fresh `String`.** It is called once per frame and
prefixes at most eight bytes; storing the prefixed form would mean two copies
of every text. `get_scene` renders it into its reply as `status_message`,
unchanged in shape.

**`Kind::Query(&'static str)`** rather than a `Query` variant plus a text
match: coalescing per tool needs the tool's identity, and the tool table
already holds every name as a `&'static str`.

**`record_at` exists for tests only.** `record` is `record_at(Timestamp::now(),
…)`; the coalescing window and the local-time formatting are both tested with
fixed instants and a fixed zone, never with the wall clock.

### Example

```rust
// In AppState::toggle_solo, after the state change is known to be a change:
let text = match self.solo {
    Some(id) => format!("Soloed {}", self.scene.label(id)),
    None => "Ended the solo".to_owned(),
};
self.action_log.record(Kind::Scene, text);

// In `mcp::apply_as_agent`, once per frame:
state.action_log.set_actor(Actor::Mcp);
for command in commands {
    let outcome = mcp::apply(state, viewer_3d, command);
    // … reads and refusals are logged here, see below …
}
state.action_log.set_actor(Actor::User);
```

## Threading and the MCP seam

The log lives on the GUI thread and nothing else touches it. MCP commands are
already applied on the GUI thread, drained once per frame as Phase 0 of
`run_ui_and_paint` ([mcp-server.md](../gui/mcp-server.md) § "Threading"), so
an entry for an agent's action is written in the same call that applies it,
with no channel and no lock. The HTTP thread never logs; the events it alone
sees (schema refusals, timeouts) are deliberately outside the log's scope.

The drain splits into two: `mcp::apply_as_agent`, a plain function over
`(&mut AppState, &mut Viewer3D, Vec<Command>)` that moves the actor, applies
each command and writes the entries; and `App::drain_mcp`, which adds the
channel — collecting the requests, calling it, and handing each answer back
down its `oneshot`. The split is what puts the *attribution* under the same
headless test as the command vocabulary, since `App` needs a GPU and a window
and `apply_as_agent` needs neither.

Within that batch, each applied command yields exactly one entry before
coalescing, with the one exception named above (a multi-field
`set_reconstruction_display` records one per field it changed):

- A mutating tool's entry is written by the `AppState` / `Viewer3D` method it
  calls, with the text from the catalogue and actor `Mcp` because the drain
  set it. The tool itself writes nothing on success.
- A read-only tool's entry is written by the drain from the command, as a
  `Query` — `get_scene`, `screenshot 1280×720`, and so on — because there is
  no state method for a read to log through. A deferred screenshot logs when it
  is *applied* (the frame the request was drained), not when the pixels come
  back, so its line appears in order with the commands around it.
- A refusal — `apply` returning a tool error — is written by the drain as a
  failed entry, `{tool} failed: {message}`, with the same message the agent
  receives. So that a failure is not logged twice, once by the method in its
  own words and once by the drain, **a method returns its failure rather than
  logging it, and the caller logs**: every `AppState` method the MCP layer
  calls that can fail returns a `Result`. `load_file` therefore returns
  `Result<ReconId, String>` and writes no entry on `Err`; the File menu logs
  `Failed to load …` from that `Err`, and the drain logs
  `open_reconstruction failed: …` from the same `Err`. One failure, one entry,
  in the vocabulary of whoever asked. This also removes the MCP writer's
  scrape of the status field to recover a load error. The catalogue's
  `Failed to load` and `Failed to reload` rows are the GUI caller's texts;
  the `Align … failed` and `Resect … refused` rows are logged by the methods
  themselves because no MCP tool calls them.

`get_scene` continues to report `status_message`, now from
`AppState::status_message()`. Its value is the newest entry that is not a successful query, so an
agent reading it back after its own mutating call sees that call, as the
spec already promises.

## Implementation notes

- **Where the `record` calls go** is the whole of the implementation, and the
  seams are these. `AppState` mutating methods: `load_file`, `reload_node`,
  `close_node`, `close_all`, `select_recon`, `select_camera`, `select_image`,
  `select_point`, `clear_selection`, `deselect_point`, `set_solo`, `align_node`,
  `resect_image`, `reset_node_transform`, plus the demo loader. `Viewer3D`'s
  two camera-view *entries*, `enter_camera_view` and
  `animated_switch_camera_view`, which take the log as an argument — the node
  and the scene are borrowed out of the same `AppState` for the whole call, so
  a `&mut AppState` alongside them would not typecheck; framing is recorded by
  the caller instead (`dock.rs::zoom_to_fit`, the `Z` key, MCP `set_view`),
  because only the caller knows whether it framed a node, a camera or the
  scene. The scene tree's eyes, cursor and tint radios and the HUD's checkboxes
  and sliders, which write straight into the node or state and record on
  `clicked()` / `changed()` — `ActionLog::changed` and the panel's own `logged`
  helper keep those to one line each. The image browser's play/pause/rate
  handling. Startup in `lib.rs` before the CLI paths load — which is why the
  CLI loads moved below `start_mcp`, so the log reads in the order the session
  happened — and the MCP server's listen report.
- **`response.changed()` alone does not mean somebody did something.** An
  `egui::Slider` clamps and rounds the value it was handed and reports *that*
  as a change, so the Scene slider — first drawn over a `length_scale` the
  upload phase had only just derived from the point cloud — announced
  `Scene scale 0.655` before the window had been touched. `ActionLog::changed`
  therefore also requires the pointer to be down on the widget, a click, a
  finished drag, or keyboard focus: an action needs an actor.
- **The viewport's background click** clears the image and the point together.
  Both inner calls are muted and one entry is recorded for the click:
  `Cleared selection` when it dropped both, `Deselected image` or
  `Deselected point` when it dropped one.
- **Ordering with `append_node`.** `resect_image` writes its status last
  because `append_node` used to clear the field. With the log, `append_node`
  writes nothing and the clear is gone; `resect_image` records once at the end
  with `append_node` muted. `load_file` records after `append_node` returns
  the label, since the text needs the deduplicated label (`global (2)`), not
  the file stem.
- **The status field is removed, not shadowed.** Every reader goes through
  `status_message()`; the overlay and the empty-state panel take the `String`
  by reference as they take the `&str` today. The four clearing sites are
  deleted, not converted.
- **`select_image(None)` during `close_node`** and similar cascades are inside
  the outer method's mute, so a close logs `Closed x` and not also
  `Deselected image`.
- **Virtualized rows** use `ScrollArea::show_rows` with the monospace row
  height, as the scene tree's image list does. The uniform height is what
  forbids wrapping and what makes 10 000 entries free to scroll. Truncation is
  `Label::truncate()`, and the tooltip carries the full text. The time and
  actor columns are **painted** into fixed-width allocated boxes rather than
  laid out as labels, so the three columns line up down the list however wide
  their contents are. **Latest** is a one-frame `vertical_scroll_offset` past
  the end, which the scroll area clamps — the panel keeps no follow state.
  A `Label` keeps its whole job text however few glyphs it drew, so a test
  asking whether a row was truncated has to read the galley's *width*, not its
  string.
- **`stick_to_bottom` and `show_rows` compose** as long as the total row count
  passed is the current count; the log grows monotonically between clears, so
  the scroll offset stays valid.
- **Entries are mirrored to `log::info!`** under the target
  `sfm_explorer::action_log`, one line each in the clipboard format, so a
  `RUST_LOG` capture of a session carries the same stream. The existing
  `log::debug!("MCP: applying {command:?}")` in the drain stays; it is the
  wire-level view, this is the outcome view.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ActionLog::CAPACITY` | `10_000` | Entries kept; the oldest is dropped past this and the toolbar reports the count dropped |
| `ActionLog::COALESCE_WINDOW` | `1 s` | Maximum gap between like entries for the newer to replace the older |
| bottom split fraction (`default_dock_state`) | `0.8` | Unchanged; the log shares the Image Browser node rather than taking its own |
| time format, list | `%H:%M:%S` | Local time, `jiff::tz::TimeZone::system()` resolved once |
| time format, tooltip and clipboard | `%Y-%m-%d %H:%M:%S` | Tooltip adds the offset, `%:z` |

## Testing

`crates/sfm-explorer/src/action_log/tests.rs`, run with
`pixi run cargo test -p sfm-explorer --lib` (the crate is outside
`pixi run test-rust`). Headless throughout; the panel runs through
`test_support::run_frame_headless` and its output is read with
`test_support::painted_texts`.

Buffer rules, with `record_at` and fixed instants:

- Past `CAPACITY`, the oldest entry goes and `dropped()` counts it.
- Two `Selection` entries 0.5 s apart become one, carrying the second text and
  the second time; 1.5 s apart they stay two. The window is measured from the
  replaced entry's time, so three entries at 0, 0.8 and 1.6 s are one.
- A `File` entry never coalesces. A failed `Selection` entry never coalesces,
  and a `Selection` after it does not replace it.
- Different actors do not coalesce; different query tools do not coalesce; the
  same query tool does.
- `status_line()` is `None` when empty, skips `Query` entries, and prefixes
  `MCP: ` exactly when the actor is `Mcp`.
- `mute` nests: two `mute`s and one `unmute` still record nothing.
- `to_clipboard_text` renders a fixed zone (`TimeZone::fixed(-7 h)`) as
  `2026-09-01 14:04:03  MCP     text`, and a failed entry as `MCP   ! text`.

Panel, headless:

- The rows paint in order, oldest first, each as time, actor and text.
- A text wider than the panel is laid out no wider than the panel — read from
  the galley's width, since the string survives truncation — and the frame does
  not panic.
- **Clear** empties the buffer and the next frame paints no rows.

Attribution, in `mcp/tests.rs` beside the existing status-line test. Every
helper there routes through `apply_as_agent` rather than bare `apply`, because
attribution is part of what an MCP call *is* and a test that skipped it would
exercise a path the viewer never takes:

- Applying each mutating `Command` records exactly one entry with actor `Mcp`,
  and the actor is `User` again afterwards.
- Applying each read-only `Command` records one `Query` entry that
  `status_line()` does not report.
- A deferred `screenshot` is logged when it is *applied*, at the size the
  picture will come back at.
- A refused command records one failed entry whose text is the refusal, and it
  reaches the status line.
- `open_reconstruction` on an unreadable path records exactly one failed entry.
- The existing `a_mutating_call_announces_itself_in_the_status_line` passes
  unchanged against `status_message()`.

The fixture those start from selects an image first and then clears the log:
only a change is logged, so a command that asked for the state the scene was
already in would record nothing and the test would be asserting the fixture.

Layout, in `dock/tests.rs`:

- `default_dock_state()` has a bottom node holding `[ImageBrowser, ActionLog]`
  with `ImageBrowser` active.

Scene-graph and resection tests that read `state.status_message` moved to
`status_message()` with no change in the strings they assert.

## Non-goals

- **Persistence.** The log is per session and lives in memory; **Copy** is the
  export. A file sink is the `log` crate's job, and the `log::info!` mirror
  gives it the same stream.
- **Filtering and search.** At 10 000 entries and one second of coalescing the
  list is scrollable; an actor filter is the obvious first addition if it stops
  being.
- **Raising the tab on a failure.** The status line already puts the failure
  on the viewport; raising a tab the user may have docked elsewhere would move
  their layout under them.
- **Recording which control was used.** `Selected image …` does not say
  whether it was the strip, the tree, the viewport or the keyboard.
- **An undo stack.** The entries are text; they are not replayable.
- **Logging the HTTP thread.** Schema refusals and timeouts stay with the
  client that caused them.

## Open questions

- Whether CLI-argument loads should be marked, e.g. `Opened x from … (command
  line)`. They are logged as `User`, which is true, and the session start entry
  immediately above them makes the origin obvious; left unmarked until someone
  misses it.
- Whether a `Display` entry for a slider should record the value at the end of
  the drag only (`drag_stopped`) rather than coalescing every intermediate
  value. Coalescing gives the same final line and needs no extra widget
  plumbing, so it is what the implementation does.
- Whether a multi-field `set_reconstruction_display` should collapse to one
  entry after all. Per-field is right while the fields read as separate things
  in the tree; an agent that habitually sets four at once would make the case
  for a single `Display of {label} changed` line.
