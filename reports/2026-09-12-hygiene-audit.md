# Hygiene audit — 2026-09-12

Read-only structural survey of the whole codebase (Python `src/sfmtool/` + `tests/`,
Rust `crates/`, top-level layout) for oversized multi-concern files, duplication,
misleading names, directory smells, and dead code. Produced by the `audit-hygiene`
skill against HEAD `6111797c`. **Supersedes `reports/2026-08-29-hygiene-audit.md`,
retired in the same commit** — not because that report was worked out, but because it
went stale against 121 commits and 582 files in fourteen days. Every still-open finding
from it is carried forward below, **re-measured here rather than copied**.

Every line count, line range and diff figure was re-derived at `6111797c`. Where this
report and the retired one disagree, this one is the measurement.

**Scale:** 166,421 non-test Rust lines + 99,593 Rust test lines; 36,553 Python
(`src/sfmtool/`, 152 modules) + 45,232 Python test lines (154 modules); 164 specs.
2,962 Rust `#[test]`s, 1,544 Python test functions, 584 Rust files.

## Headline: the growth is not evenly distributed, and neither is the staleness

Since `94d3739` — 14 days, 121 commits, 582 files, +133,387/−27,544:

| Area | Δ lines | |
|---|---|---|
| `crates/sfm-explorer` | **+41,179 / −3,382** | the whole story |
| `crates/sfmtool-core` | +31,385 / −4,243 | |
| `specs/` | +22,764 / −5,216 | |
| `tests/` | +7,711 / −1,156 | |
| `crates/sfmtool-kdf-format` | **+5,504 / −0** | an entire new crate |
| `crates/sfmtool-py` | +5,233 / −538 | |
| `src/sfmtool/` | **+1,166 / −230** | 3% of the Rust churn |

Non-test Rust is up **40%** in a fortnight (118,972 → 166,421). Python is up **2.6%**
(35,617 → 36,553). That asymmetry is the most useful fact in this report and it
explains its two halves: nearly every Python finding below is carried forward with its
numbers *unchanged*, because nobody touched those files; nearly every new finding is in
Rust, and most of those are in `sfm-explorer`.

### The last snapshot's headline was too pessimistic, and the correction is more useful

2026-08-29 concluded that "an acquittal in a report is a measurement with a three-week
half-life". Re-measuring its **entire** "Explicitly not flagged" list at HEAD:

| Cleared 2026-08-29 | Then | Now | Δ |
|---|---|---|---|
| `patch/view_selection.rs` | 1051 | 1051 | **0%** |
| `patch/cluster_refine/kernels.rs` | 799 | 799 | **0%** |
| `patch/keypoint_subpixel.rs` | 861 | 861 | **0%** |
| `keypoint_subpixel/kernels/render.rs` | 826 | 826 | **0%** |
| `spherical/tile_rig.rs` | 1091 | 1091 | **0%** |
| `spherical/per_tile_source_stack.rs` | 866 | 866 | **0%** |
| `features/sift/scale_space.rs` | 881 | 881 | **0%** |
| `patch/cluster_refine/mod.rs` | 899 | 901 | 0% |
| `spherical/photometric_ransac.rs` | 839 | 841 | 0% |
| `geometry/rotation_init.rs` | 891 | 866 | −2% |
| the six cleared Python files | — | — | **0%** |
| `geometry/resect_images.rs` | 1227 | 1314 | +7% |
| `patch/cloud.rs` | 1023 | 1156 | +13% |
| `sfmtool-py/…/sfmr_reconstruction.rs` | 1051 | 1188 | +13% |
| **`geometry/focal_vote/column_scan.rs`** | 1170 | **1964** | **+67%** |
| **`sfm-explorer/src/app.rs`** | 879 | **1631** | **+85%** |

Fourteen of sixteen held, most of them to the line. **Acquittals do not decay with
time; they decay where feature work is landing** — and a snapshot can say where that
is. Better, the last report *predicted both* movers: it flagged `app.rs` with "a
`menu.rs` extraction is again defensible… do `dock.rs::ui` first; re-measure this
after". `dock.rs::ui` was duly done. This is that re-measure, and the answer is yes.

`column_scan.rs` is the counter-example that keeps the rule honest: it grew **67%** and
**the acquittal still holds**. It was cleared because its longest function was 75 lines;
today the file is 1,964 lines and its longest function is **78** (`scan_rotation`,
1662). It grew by adding small functions, which is what healthy growth looks like.
*A file's line count is not the measurement; its longest function and its concern count
are.* Every acquittal in this report therefore carries a number a future audit can
re-run, not a verdict it would have to re-derive.

### The second theme: this repo now knows how to fix these, and does not reuse the fix

Three times in this window a contract was solved properly in one place and then ignored
or re-broken next door:

- `matches-format/verify.rs` became a 108-line orchestrator over 14 per-section helpers
  (#379). **Zero** of the four sibling entry points adopted it — two of them *grew* —
  and the newest format crate reproduced the un-split shape from scratch.
- The sibling-`tests.rs` convention was completed workspace-wide last cycle (seven
  inline blocks moved out). **Four new inline blocks** have landed since, all in the two
  newest subsystems — and one is named `mod profiling`, exactly the spelling the last
  report warned the next scan would have to catch. It did.
- `sfmtool-archive-io` exists to own the container primitives the five format crates
  share, and ships both ends of the content-hash rule. The middle of that rule — the
  digest fold that *defines* every file's identity — is hand-written **nine times, in
  two spellings**.

Against that, the best-engineered thing in the codebase is also brand new: the MCP
surface's `the_wire_vocabulary_holds_across_the_catalog` asserts its own public naming
rule over all 40 tools, and pins the single permitted initialism *by name* so a second
cannot arrive quietly. That is the shape every finding below aims at, and it is cited
as the model throughout.

---

## Rust — `sfm-explorer`

The crate grew 41k lines in a fortnight and now holds six subsystems the last snapshot
never saw: `mcp/` (12,490 lines), `action_log/` (3,796), `background/` (2,519),
`layout.rs` (1,330 + 1,299 tests), `document.rs` (593 + 225), `state/edits.rs` (1,334 +
1,454). **Most of that code is the best-structured in the repository** — see the
acquittals — and the findings cluster in two places: `app.rs`, which nobody split while
everything around it was being split, and the MCP surface, where one public API is
maintained by hand in six places.

**`run_egui_pass` is a 600-line method holding five concerns — the growth the last
snapshot predicted**
> _Carried forward as a **prediction**, not a finding. 2026-08-29 acquitted `app.rs`
> with: "`run_egui_pass` is back to **245** lines from the 180 the last snapshot
> measured, and a `menu.rs` extraction is again defensible — but `dock.rs::ui` is the
> same shape with three times the payoff. Do that one first; re-measure this after."
> `dock.rs::ui` landed as `aa00ae0`. This is the re-measure._
- Location: `crates/sfm-explorer/src/app.rs` (879 → **1631**, +85%, the largest
  single-file growth in the repository); `run_egui_pass` **732–1331** (**600**, was
  245, was 180); `prepare_uploads` **361–680** (320); `run_ui_and_paint` **83–347** (265)
- Problem: The module doc (4–13) calls `run_ui_and_paint` "a thin orchestrator that
  wires four per-frame phases" and names them. One of those four is now 600 lines, and
  it is not one phase — its own banner comments separate at least five concerns:
  DirectManipulation gesture gathering (~748–800), the menu bar with its File / Edit /
  Panels / Camera menus (~800–1070, the bulk), keyboard-shortcut arbitration between
  egui text fields and the accelerators (~1080–1100), three modal dialogs answered
  inline (Go to Point, Bundle Adjust, the close prompt, ~1150–1290), and the dock area
  plus tessellation (~1290–1331). The arms share `self` and nothing else. The file has
  also grown a tail concern the module doc does not claim: `save_as_with_dialog`
  (1480), `save_dirty_before_closing` (1505), `save_outcome` (1539), `edit_outcome`
  (1579) and `forget_selected` (1594) are ~150 lines of File-menu command handling,
  which is neither "the core rendering loop" nor "platform-specific helpers".
- Proposed fix: `app/menu.rs` for the menu bar and the shortcut arbitration it is
  interleaved with, `app/modals.rs` for the three dialogs, `app/save.rs` for the five
  tail functions. `run_egui_pass` returns to ~120 lines of frame plumbing. The crate
  demonstrates this extraction twice already (`dock.rs::ui` → `TabContext` methods,
  `state.rs` → `state/ops.rs`), with the same shape and risk profile. There is a
  further precedent worth following: `layout.rs:1245–1330 panels_menu` is *already* an
  extracted menu body, with a doc explaining that it left `app.rs` specifically so a
  headless frame could draw it and read it back. If `app/menu.rs` exists, the Panels
  menu should move into it from `layout.rs`, which also resolves the one arguably-odd
  thing in `layout.rs`.
- Effort: low (movement along boundaries the banners already draw)
- Risk: low — no signature changes; `app.rs`'s only caller is `lib.rs`'s frame loop.

**Adding an MCP tool means editing six places, and only one edge of that is
machine-checked**
- Location: `crates/sfm-explorer/src/mcp/tools.rs` **56–904** (`catalog`, 849 — the
  longest function in the workspace) and **1194–1569** (`parse`, 376); `mcp/mod.rs`
  **74–273** (the 40-variant `Command`), **550–783** (`apply_with_window`),
  **1201–1244** (`tool_name`), **1359–1416** (`kind`)
- Problem: The 40 tool names are written out three times and each argument name four —
  the schema in `catalog()`, the `reject_unknown(&[…])` key list in `parse`, the
  `args.foo("key")` read, and the `Command` field. There are ~**88** argument-key
  literals in `reject_unknown` restating what `object(&[…])` declared 800 lines above.
  The two seams are unequally guarded. `every_advertised_argument_is_one_the_parser_knows`
  (tests.rs:3433) walks the catalog and covers one direction — advertised ⊆
  parser-known — so a key the parser accepts but nobody advertises makes the closed
  schema a lie, silently. And **nothing at all ties `catalog()`'s names to
  `Command::tool_name()`**: the two tests that touch `tool_name` iterate hand-written
  `Vec<Command>` lists of 7 and 20, not the 40, so a tool advertised as `foo_bar` that
  logs itself as `foobar` would ship. Worse, **`ToolKind::Read` and `Command::kind() ->
  Kind::Query` are two hand-maintained classifications of the same 40 tools**
  (`tools.rs:33–43` vs `mod.rs:1359–1416`). They select exactly the same 12 today —
  checked member by member — and nothing says so. One drives the `readOnlyHint`
  annotation an agent trusts (`server.rs:323`); the other decides whether a refusal
  folds in the Action Log and whether the status line is suppressed. A tool added as
  `ToolKind::Read` but classified `Kind::Edit` would advertise itself read-only to
  agents while filing as an edit, and both existing tests would pass, because each
  asserts its own side against a separate literal list (tests.rs:3580 and :1622).
- Proposed fix: three steps, independently useful and all small. (1) Have
  `Args::reject_unknown` look its tool up in a `OnceLock<Vec<ToolSpec>>` and take the
  key list from the schema — the 88 literals delete themselves. (2) Add one test that
  builds a `Vec<Command>` covering all 40 variants (the compiler's exhaustiveness check
  keeps it maintained) and asserts the set of `command.tool_name()` equals the set of
  `catalog()` names. (3) In the same test, assert
  `(spec.kind == ToolKind::Read) == matches!(command.kind(), Kind::Query(_))`, or
  simply derive one from the other.
- Effort: medium for (1), low for (2) and (3)
- Risk: low — the schema and `reject_unknown` key sets were compared mechanically
  across all 40 tools and agree exactly, and the two `Read`/`Query` sets agree member
  by member, so every step starts from a clean baseline.

**The reply vocabulary forks three ways on one handle, because the rule is tested on
requests only**
- Location: `crates/sfm-explorer/src/mcp/render.rs:208` vs `mcp/read.rs:364–369` vs
  `read.rs:435`; the rule is `tools.rs:6–13`, the enforcement gap is `tests.rs:3492–3527`
- Problem: `tools.rs`'s module doc states the surface's one rule as absolute — "one
  entity, one spelled-out word, in tool names, arguments **and reply fields** alike".
  The request key is always `camera_intrinsics_index`. The replies give it three ways:
  a `list_camera_images` row emits a flat `"camera_intrinsics_index": <int>`
  (render.rs:208), `get_camera_image` buries the same handle at `"camera_intrinsics":
  { "index": …, "model": … }` (read.rs:366), and `get_camera_intrinsics` reports a bare
  top-level `"index"` (read.rs:435). An agent that found the handle through
  `list_camera_images` can paste it straight back; one that found it through
  `get_camera_image` has to know it is nested. `the_wire_vocabulary_holds_across_the_catalog`
  *does* enforce the rule and is the best enforcement in this repo — but it iterates
  `spec.schema["properties"]`, i.e. request keys, so the half of its own stated rule
  about reply fields has never been checked. (Lower confidence, same shape:
  `render.rs:68` reports a reconstruction's handle as `"label"` where 15 other reply
  sites say `"reconstruction_label"`. That one is defensible — it is nested inside the
  reconstruction's own block — but it is the key `get_scene`'s description tells agents
  to carry to every other tool.)
- Proposed fix: Emit `camera_intrinsics_index` alongside the nested block in
  `get_camera_image` (additive, breaks no client), then extend the vocabulary test to
  the reply side: walk a fixture scene through every read tool and assert no entity has
  two spellings. Close the gap in the test rather than in review.
- Effort: medium — the reply builders are imperative `json!` calls, so the reply-side
  test needs a fixture rather than a declarative walk.
- Risk: low — additive on the wire.

**`get_history` is the one MCP tool that does not spell its entity**
- Location: `crates/sfm-explorer/src/mcp/tools.rs:250`; the entity's own spelling at
  `layout.rs:216` and `dock.rs:64`
- Problem: Every other tool naming a viewer entity spells it out — `get_action_log`,
  `get_window_layout`, `get_image_detail_display`, `get_background_task`,
  `get_timing_detail`. `get_history` reads the **Edit History**, whose panel title is
  "Edit History" and whose wire name *on the same MCP surface* is `edit_history` —
  reachable only as `show_panel(panel_name="edit_history")`. So one surface calls the
  entity `edit_history` and another calls it `history`. This is `tools.rs`'s own stated
  rule ("no word that names two things") applied to a word that, in this viewer, also
  names the Action Log to a reader. Its sibling `jump_to_version` does spell its entity.
- Proposed fix: Rename to `get_edit_history`, accepting `get_history` in `parse` for a
  release while advertising only the new name.
- Effort: low
- Risk: medium — the name is public API to any agent config or saved prompt, and it
  gets more expensive with every client written.

**`mcp/mod.rs` carries six concerns and two of them contain no MCP**
- Location: `crates/sfm-explorer/src/mcp/mod.rs` (1643). Seams, several already drawn
  as banner comments: **74–535** the `Command` vocabulary and reply types; **537–917**
  `apply_with_window` dispatch; **918–1094** wire-handle resolution; **1095–1186**
  `apply_as_agent`; **1187–1557** "The Action Log's view of a command"; **1558–1637**
  "Where a panel is, in the picture of the window"
- Problem: Two of the six have no dependence on the MCP protocol at all. **1558–1637**
  is 80 lines of four pure functions over `(&DockState<Tab>, f32)` —
  `panel_body_points`, `panel_body_size`, `panel_body_pixels`, `panel_crop` — and their
  own banner says so ("Both functions here are pure over `(&DockState<Tab>, f32)`").
  That is dock geometry, and `dock.rs` owns the vocabulary. **1187–1557** is 370 lines
  classifying a command for the Action Log (`tool_name`, `edits`, `renumbers`, `run`,
  `kind`, `query_text`), whose subject is `action_log`, not the protocol.
  `selection_reply` (1637–1643) is then orphaned *underneath* the panel-geometry
  banner, four functions from the `render::selection` it wraps. The longest function,
  `apply_with_window` at 234 lines, is **not** the complaint — it is a 40-arm dispatch
  at 5.85 lines per arm with no logic in any arm.
- Proposed fix: `mcp/panel_rect.rs` for 1558–1637 (or into `dock.rs`), `mcp/logged.rs`
  for 1187–1557, and `selection_reply` back beside `render::selection`. `mod.rs` lands
  at ~1100 of vocabulary, dispatch and resolution.
- Effort: low — pure moves, no logic change; all four geometry functions are private
  or `pub(super)`.
- Risk: low

**The version-transition sentence is hand-composed at 11 sites, next to the module
built to prevent exactly that**
- Location: `crates/sfm-explorer/src/state/edits.rs` 115, 245, 317, 421, 553, 701, 846,
  1032, 1072, 1148 and `background/mod.rs:525`; the abstraction that should own it is
  `action_log/mod.rs` **1197–1242**
- Problem: `action_log/mod.rs:1197` opens a banner section titled "The texts shared by
  the GUI and the MCP surface", whose stated purpose is "One text per action, whoever
  took it… So the wording of a scene-graph change lives here rather than being composed
  twice." Three helpers live there: `visibility_text`, `interactive_text`, `tint_text`.
  The single most-repeated Action Log fragment in the crate — the
  `({parent} → {serial})` version suffix — is not one of them. It is written out
  longhand **11 times** across two modules in four shapes
  (`"{text} ({parent} → {serial})"`, `"{text}: {outcome} (…)"`, `"Undo: {label} (…)"`,
  `"Go to: {label} (…)"`). This is not private formatting: `mcp/edit.rs:442` documents
  the shape *to agents* ("that is the form a human reads off a row and off
  `Go to: … (v11 → v12)`"), `mcp/edit.rs:429–436` reads the Action Log text back
  verbatim onto the wire, and `resolve_serial` accepts `"v12"` back because of it. A
  stray space or a changed arrow at any one of the 11 sites is an MCP surface change.
- Proposed fix: `pub(crate) fn version_step_text(body, parent, serial) -> String`
  beside `visibility_text`, called from all 11; the three splicing sites pass the
  already-spliced body.
- Effort: low
- Risk: low — every one of these texts is asserted verbatim in `state/edits/tests.rs`
  and `mcp/tests.rs`, so a wording slip fails loudly rather than silently.

**`undo`, `redo` and `jump_to_version` are three copies of one shape, held together by
prose**
- Location: `crates/sfm-explorer/src/state/edits.rs` **1001–1035** (`undo`, 35),
  **1041–1074** (`redo`, 34), **1091–1148** (`jump_to_version`, 58)
- Problem: `undo` and `redo` share **26 of 34 lines verbatim (76%)**. The diff is four
  things: `history.undo()` vs `history.redo()`, `follow_selection_backward(id, undone)`
  vs `follow_selection_forward(id)`, whether the version label is read before or after
  the step, and the verb. Everything else — the `started`/`Collector`/`busy_refusal`/
  scene-position preamble, the three phase names `"history step"`, `"selection follow"`,
  `"forget images"` in that order, and the `record_done(Kind::Edit, …)` tail — is
  identical. `jump_to_version` is a third copy of the same preamble, phases and tail.
  **The invariant is held entirely by prose**: `redo`'s doc says "The same three stages
  [`AppState::undo`] reports, under `redo`", `jump_to_version`'s says "It reports the
  stages [`AppState::undo`] does", and the module doc calls jump "the two of them
  repeated". Nothing checks it, and a phase renamed in one is silently a different
  breakdown in the Background panel *and on the wire* via `get_action_log(detail=true)`.
- Proposed fix: one private `fn step_cursor(&mut self, id, verb, step: impl FnOnce(&mut
  History) -> Option<(VersionSerial, VersionSerial)>, follow: Follow)` owning the
  preamble, the three phases and the record; `undo`/`redo` become ~8 lines each and
  `jump_to_version` keeps only its walk.
- Effort: medium
- Risk: medium — the borrow dance around `self.scene[index]` and `follow_selection_*`
  is why they were written flat, so the closure must take `&mut History`, not
  `&mut AppState`. The phase names and their nesting are asserted by breakdown tests in
  `mcp/tests.rs:1988–2493` and `action_log/tests.rs`; a refactor that changes nesting
  depth changes the wire breakdown.

**The node-lookup preamble is inlined 11 times, and its refusal has three wordings
across 19 sites**
- Location: preamble at `crates/sfm-explorer/src/state/edits.rs` 95/101, 161/167,
  269/275, 345/351, 463/469, 596/624, 742/769, 901/906, 1004/1007, 1044/1047,
  1094/1097; plus `state/save.rs` 126/152 and `state.rs:808`. Sentence forks at
  `mcp/mod.rs:958`, `:996`, `mcp/read.rs:454`, `camera_lock.rs:98`
- Problem: `busy_refusal(id)?` followed by `scene.iter().position(|n| n.id ==
  …).ok_or("That reconstruction is no longer loaded.")` appears 11 times in `edits.rs`
  alone; `position(|n| n.id == …)` appears 17 times crate-wide with no
  `AppState::node_index(id)` helper. The literal has **19 non-test occurrences in three
  spellings** (verified by grep): "That reconstruction is no longer loaded." ×14
  (edits.rs ×12, save.rs ×2, background/mod.rs ×1, mcp/edit.rs ×1), "**The**
  reconstruction is no longer loaded." ×3 (mcp/mod.rs ×2, mcp/read.rs ×1), and one with
  no full stop (`camera_lock.rs:98`). The `mcp/` split is the one that matters: the same
  failure, reported to the same agent, says "That" from `edit.rs` and "The" from
  `mod.rs` and `read.rs`.
- Proposed fix: `fn node_index(&self, id) -> Result<usize, String>` on `AppState`
  carrying the sentence as one `pub(crate) const RECON_GONE`, plus a `busy_node_index`
  pairing it with `busy_refusal`; `mcp/` builds its `ToolError` from the same const.
- Effort: low
- Risk: low — the only behavioural question is whether all 11 sites want the busy check
  before the lookup, which they currently all do.

**Three `directmanipulation` examples, 1,088 lines, that nothing compiles**
- Location: `crates/sfm-explorer/examples/{win32,winit,winit_wgpu}_directmanipulation.rs`
  (279 + 416 + 393 = **1,088**), declared at `crates/sfm-explorer/Cargo.toml:20–33`
- Problem: All three carry `required-features = ["directmanipulation"]`, and
  `directmanipulation` is a non-default feature (`Cargo.toml:36`). Grepping
  `.github/workflows/*.yml` and `pixi.toml` for `--features`, `--all-features` and
  `directmanipulation` returns **nothing**, so no CI job and no pixi task ever enables
  it: `cargo check --workspace`, `cargo clippy --workspace` and the entire CI matrix
  skip all three files. They were last touched **2026-07-18**, nearly two months ago,
  and a clone scan finds them sharing 60 / 35 / 35 long lines with one another — three
  variations of one spike. They are not dead (`specs/gui/viewport-navigation.md`
  references them as the DirectManipulation reference programs) but they are *unbuilt*,
  and the crate around them is taking winit and wgpu major-version bumps. The first
  person to need them will discover they no longer compile, at exactly the moment they
  can least afford it.
- Proposed fix: Add one `cargo check -p sfm-explorer --features directmanipulation
  --examples` step to the Windows CI job — they are Win32 examples, so that is the
  natural host. Alternatively fold the three into one and say in the spec that it is a
  historical spike pinned to the versions of its date. The cost of the check is one
  compile; the cost of neither is silent rot.
- Effort: low
- Risk: low — a check can only reveal breakage, and if they are already broken that is
  the finding.

**Two of the six new subsystems chose bare `pub` where the other four chose
`pub(crate)`**
- Location: `crates/sfm-explorer/src/document.rs` (33 `pub`, 0 `pub(crate)`) and
  `state/edits.rs` (17 `pub`, 5 `pub(crate)` — `pub fn undo` at 1001 beside
  `pub(crate) fn bundle_adjust_job` at 896)
- Problem: Every module in this crate is declared privately in `lib.rs` (`mod
  document;` — no `pub mod` anywhere), so a bare `pub` item is `pub(crate)` with extra
  letters and no way to tell a deliberately crate-wide item from a module-internal one.
  The four other new subsystems are uniformly disciplined — `mcp/` 0 `pub` / 55
  `pub(crate)`, `action_log/` 0/75, `background/` 0/24, `layout.rs` 0/42 — and these
  two, landing in the same fortnight, went the other way. Crate-wide the drift is older
  and wider (331 `pub` / 539 `pub(crate)`, 21 files mixing), so the six did not cause it
  — but they had the chance not to join it.
- Proposed fix: `pub` → `pub(crate)` in `document.rs` and `state/edits.rs` (50 items).
  Leave the older 19 mixed files for a separate sweep.
- Effort: low — mechanical, and `cargo check` catches every miss.
- Risk: low

**`Tab::wire_name`'s doc states a derivation rule that one arm breaks, on a public wire
surface**
- Location: `crates/sfm-explorer/src/layout.rs` **204–218** against `dock.rs:60–72`
- Problem: The doc says "The panel's name on the wire: its title, lower-cased and joined
  with underscores." Eight of nine arms obey exactly. `Tab::Viewer3D` has title
  `"3D Viewer"` (dock.rs:65) and wire name `"viewer_3d"` (layout.rs:210) — the rule
  gives `3d_viewer`. These names are the `panel_name` enum for `show_panel` /
  `hide_panel` / `screenshot` and the keys of the layout file, so a client that derives
  a name from the documented rule guesses wrong for the one panel most likely to be
  screenshotted — and `screenshot` refuses `hud: false` for anything but
  `panel_name: "viewer_3d"` (tools.rs:873–877), so the refusal is unexplainable.
- Proposed fix: Amend the doc to state the rule and name the exception. Do **not**
  change the string — it is in `LAYOUT_VERSION = 2` files.
- Effort: low
- Risk: low

**The MCP tool count is written in prose twice, and both numbers are wrong, differently**
- Location: `crates/sfm-explorer/src/mcp/server.rs:258` ("Twenty-three tools are cheap
  to re-fetch") and `specs/gui/mcp-server.md:2381` ("Thirty-six tools are cheap")
- Problem: `catalog()` advertises **40**. The same sentence was copied from code into
  spec and both copies then drifted, in opposite directions, to two different wrong
  numbers. This is the identical defect class as the `AGENTS.md` module counts, which
  #444 closed **today** — after three audits running corrected them — and it closed
  them by *deletion*. The argument that settled it there settles it here.
- Proposed fix: Delete the number from both. The sentence works without it.
- Effort: low
- Risk: low

**`mcp/tests.rs` is one 5,427-line file over a module already split fifteen ways**
- Location: `crates/sfm-explorer/src/mcp/tests.rs`
- Problem: 5,427 lines, ~178 tests, **20 banner sections**, and the only `mod tests`
  declaration in `mcp/` — so one file covers all fifteen source modules at once, from
  the HTTP transport over a real socket (3651–3951) to wgpu screenshot plumbing
  (2493–2690) to the schema/parser pair (3426–3651) to the editing surface (3951–4373)
  to the background surface (4373–5103). It is 1.95× the next-largest file in the
  crate. The crate's own pattern is a sibling `tests.rs` per module —
  `image_detail/tests.rs` and `image_detail/intrinsics/tests.rs` coexist, as do
  `scene_renderer/upload/tests.rs` and `scene_renderer/auto_point_size/tests.rs`.
- Proposed fix: Give `server.rs`, `frame.rs`, `layout.rs` and `view.rs` their own
  `mcp/<name>/tests.rs` along the existing banners, leaving `mcp/tests.rs` at ~3,800
  for dispatch and vocabulary. The shared `Fixtures` block (39–255) moves to a
  `mod fixtures` the siblings import — the same shape as `tests/xform/conftest.py` on
  the Python side.
- Effort: medium — the fixtures split needs care.
- Risk: low

---

## Rust — `sfmtool-core`

**`solve_lm` still re-derives its const generic as a runtime bool, and grew 20%**
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (1375 → **2254**,
  +64%); `solve_lm` **1352–1968** (**617**, was 514); the re-derivation at **1376**,
  branch sites **1382, 1511, 1601, 1840**; `bundle_adjust_staged` **1974–2251** (278)
- Problem: The pattern is verbatim intact: `let opt_bspline = CAM_COLS ==
  BSPLINE_CAM_COLS;` at 1376, a `debug_assert!` on it, then four runtime branches — so
  the two monomorphizations each carry the other's dead branches. Only two widths exist
  (`BASE_CAM_COLS = 8` at 453, `BSPLINE_CAM_COLS` at 459); **no third camera
  parameterization arrived**, and `bundle_adjust_staged` already makes the instantiation
  choice explicitly at 2175–2205 (`if opt_bspline { solve_lm::<BSPLINE_CAM_COLS> } else
  { solve_lm::<BASE_CAM_COLS> }`), so the runtime bool inside is pure redundancy.
  **The +879 lines are not the solver.** They are #400 (`57912fa4`, point constraints,
  +833/−156): `PointConstraint` 78–100, `DistanceReference` 101–131, `PointConstraints`
  132–255, `PointConstraintsError` 256–320, `FreePointPolicy` 322–346, `Constraints`
  764–863, `DistanceOrigin` 864–877, `reestimate_points` 972–1056, `robust_cost`
  1064–1103, `residual_norms_depths` 916–945; then `bcd80c4b` (+142, bulk-edit entries)
  and `9ea86c85` (+58, stage reporting). That constraint machinery — roughly 480 lines
  across 78–346 and 764–877 — is a self-contained concern that has nothing to do with
  Levenberg–Marquardt.
- Proposed fix: `trait CameraColumns { const COLS: usize; const BSPLINE: bool; }` with
  two impls, so each branch reads `if C::BSPLINE` and monomorphizes away. Separately,
  move the constraint types to `bundle_adjust/constraints.rs`.
- Effort: medium
- Risk: medium — this is the LM inner loop and every reconstruction number is
  downstream of it; a mis-threaded const changes the normal-equation column layout
  silently. `bundle_adjust/tests.rs` is 4,464 lines and covers both rungs
  (`opt_bspline_*` at 2773–3448), which is real protection.

**`focal_vote`'s entry point is still one function over three policy families, behind a
four-deep wrapper chain**
> _Partially resolved. The `log_median` half of the old finding is **done**: `log_median`
> (413–420) now delegates to `crate::numeric::median_in_place` and its doc says so
> explicitly. `log_iqr` (389–402) is 8 lines and correctly local. Both stay._
- Location: `crates/sfmtool-core/src/geometry/focal_vote.rs` (1073 → **1476**);
  `focal_vote` 750–767 (18) → `focal_vote_with_min_disp` 773–794 (22) →
  `focal_vote_with_options` 804–829 (26) → **`focal_vote_impl` 833–1224 (392)**
- Problem: The 396-line entry point was renamed, not split — the body moved into
  `focal_vote_impl` behind three wrappers. Its own banners still mark the three
  families: `── Pair tables ──` (882), `── Epipolar votes ──` (930), `── Rotation votes
  ──` (1027). The top-level constant count is 16, down from 24, but that is
  bookkeeping: 25 more now sit in `column_scan.rs:52–113`, so the tuning surface for
  **one decision is 41 constants split across two files**.
- Proposed fix: `focal_vote/epipolar_votes.rs` (882–1026) and
  `focal_vote/rotation_votes.rs` (1027–1123), each taking the constants only it uses;
  `focal_vote_impl` becomes a validate-gather-fold driver.
- Effort: medium
- Risk: low — the phases communicate through named vote vectors, and
  `focal_vote/tests.rs` is 1,105 lines.

**`column_scan.rs` grew 68%; the length acquittal holds, but 29% of it is now `unsafe`
SIMD sitting inside the policy that calls it**
> _The mechanical acquittal **survives** and is the report's headline counter-example:
> longest function `scan_rotation` 1662–1739 = **78** (was 75), then `epipolar_vote` 75,
> `fit_epipolar` 66, `freeze_rotation_support` 61, `fit_rotation` 58. Nothing is
> oversized. What changed is the concern count._
- Location: `crates/sfmtool-core/src/geometry/focal_vote/column_scan.rs` (1170 →
  **1964**). Nine banner-marked sections; the SIMD arm is **520–988** (469), plus
  `epipolar_residuals_avx2` 1118–1162, `rotation_cosines_avx2` 1394–1432 and
  `acos_slice_avx2` 1434–1447
- Problem: ~570 lines, 29% of the file, and **all five of its `unsafe fn`**, are
  hand-written `std::arch::x86_64` intrinsics with scalar fallbacks — a different kind
  of code interleaved with the policy that dispatches to it. The house already has the
  right pattern in two places: `geometry/simd.rs` and `features/sift/simd.rs` are
  dedicated SIMD modules.
- Proposed fix: `column_scan/simd.rs` for 520–988 plus the three stray f64 AVX2 kernels
  and their scalar twins, taking the parent to ~1390 and separating `unsafe` from
  policy. A second cut along `epipolar cell` / `rotation cell` is available later.
- Effort: low — the sections are contiguous and already banner-delimited.
- Risk: low — moving `#[target_feature(enable = "avx2")]` functions across a module
  boundary is mechanical, and `column_scan/tests.rs` (878) plus `focal_vote/tests.rs`
  (1105) cover scalar/AVX2 equivalence.

**`keypoint_localize.rs`: the subdirectory split landed, the tail block did not**
> _Partially resolved, and the file moved `camera/` → `patch/`. It gained five
> submodules (`basis.rs` 111, `kernels.rs` 460, `params.rs` 224, `prof.rs` 265,
> `search.rs` 707) — and still grew 1483 → **1633**._
- Location: `crates/sfmtool-core/src/patch/keypoint_localize.rs`; driver
  `localize_patch_keypoints_with_basis` **598–1086** (**489**, was 424); the tail block
  **1090–1436** (347)
- Problem: Everything the old finding named is still in the parent and still
  contiguous: `TailGeometry` (1090–1106), `basis_template` (1107–1157, 51),
  `within_max_shift` (1163–1178), `register_tail` (1207–1375, **169**, the file's
  second-longest function), `seed_offset` (1388–1406), `finalize` (1411–1436). One
  347-line block with one entry point, in a module that has already demonstrated five
  times over that it knows how to make a submodule.
- Proposed fix: `keypoint_localize/tail.rs` for 1090–1436, as originally proposed. The
  parent lands at ~1290 with the driver as its clear subject.
- Effort: low
- Risk: low — mechanical move; `tests.rs` (2,525) exercises `register_tail` through the
  driver.

**`grow_reconstruction` is a 612-line function with a 96-line nested function inside it**
> _Carried forward. Re-measured: 606 → **612**, so nothing moved._
- Location: `crates/sfmtool-core/src/geometry/reconstruction_growth.rs:488–1099`
  (612 of a 1102-line file, 56%); nested `run_grow_ba` **592–687** (96)
- Problem: The next-longest top-level item in the file is 76 lines. Its own banners mark
  the seams: seed state (534), registration order (546), per-image rows and
  covisibility (549), initial structure (565), `── Growth loop ──` (578),
  `── Finishing: release the focal ──` (1010), re-triangulation (1075). `run_grow_ba` is
  a *nested* `fn`, not a sibling — already a self-contained unit, just not hoisted.
- Proposed fix: Hoist `run_grow_ba` to a file-level function over an explicit state
  struct, then split the finishing pass (1010–1099) into `finish_with_released_focal`.
  That takes the driver under 450 without touching the loop.
- Effort: medium
- Risk: medium — the loop mutates shared pose/point/covisibility state, so hoisting
  means naming that state, and getting the borrow split wrong changes acceptance order.
  `tests.rs` is 860 lines.

**`camera/distortion.rs`'s second impl block is projection, not distortion — and the
blocker is a two-word fix**
> _Partially resolved. The `kernels.rs` half **landed** as `585b258`:
> `kernels/{brown,equidistant,thin_prism,rad_tan,sfmtool_fisheye,sfmtool_pinhole,blend}.rs`,
> 1,672 lines, well sized. The projection half was left because two kernels it calls
> are private to `camera::distortion`. That blocker is **verified still true**, and it
> is smaller than it looked._
- Location: `crates/sfmtool-core/src/camera/distortion.rs` (1233); `impl CameraModel`
  **113–805** (693, genuinely distortion); `impl CameraIntrinsics` **811–1224** (**414**,
  12 public methods) plus the free `pixel_jacobian` 1041–1067
- Problem: The second impl block is the pixel↔ray projection API in a file named for
  distortion: `project`, `unproject`, `project_batch`, `unproject_batch`,
  `pixel_to_ray`, `ray_to_pixel`, `ray_to_pixel_with_jacobian`, `min_pixel_scale`,
  `pixel_radius_to_world`, `pixel_radius_to_angle`, `ray_to_pixel_batch`,
  `pixel_to_ray_batch`. So the canonical path for `CameraIntrinsics::project` is
  `camera::distortion`. `CameraIntrinsics` is **already impl'd in four other files**
  (`intrinsics.rs:560`, `intrinsics/registry.rs:314`, `distortion/pinhole_fit.rs:15`,
  `distortion/ray_grid.rs:74`), so a fifth home is the house pattern, not a novelty.
  The blocker: `radial_fisheye_ray_jacobian` is `pub(in crate::camera::distortion)` at
  `kernels/equidistant.rs:466` and `sfmtool_fisheye_ray_jacobian` at
  `kernels/sfmtool_fisheye.rs:215`, both called from `distortion.rs:946/951/956`.
- Proposed fix: Move 811–1224 to `camera/projection.rs` and widen those two kernels
  from `pub(in crate::camera::distortion)` to `pub(in crate::camera)`. That is a
  two-word change to each and does not leak them out of `camera`.
- Effort: low
- Risk: low — a pure move plus a compiler-checked visibility widening;
  `distortion/tests.rs` is 4,144 lines.

**The two Newton undistort solvers are still unmerged, one file apart**
- Location: `crates/sfmtool-core/src/camera/distortion/kernels/thin_prism.rs:144–226`
  (`newton_thin_prism`, 83) and `kernels/rad_tan.rs:182–297`
  (`newton_rad_tan_thin_prism`, 116)
- Problem: Both are the same driver — same `UNDISTORT_MAX_ITER` loop, same
  `res_u.abs() + res_v.abs() < UNDISTORT_EPS` exit, same `det.abs() < 1e-30` guard,
  same 2×2 inverse (`delta_uu = (j11*res_u - j01*res_v) * inv_det`, **byte-identical**),
  same ten-step `alpha *= 0.5` backtracking that recomputes the residual, same update,
  same return. Roughly **32 lines are byte-identical**; only the model `F` and its
  Jacobian differ. The `kernels/` split put them one file apart, which is exactly the
  merge the last report predicted a per-family split would "put in front of whoever
  does it next".
- Proposed fix: One `newton2(x_d, y_d, uu0, vv0, model: impl Fn(f64, f64) -> ((f64,
  f64), [f64; 4]))` in `kernels/mod.rs` (45 lines today), each kernel supplying only its
  residual-and-Jacobian closure.
- Effort: low
- Risk: medium — these run per-pixel in undistort maps, so a closure that fails to
  inline costs measurably. Check `benches/` before landing.

**`camera/remap.rs` still hosts the crate's most-used image containers**
> _Carried forward. Re-measured: **1167**, flat since the last snapshot._
- Location: `crates/sfmtool-core/src/camera/remap.rs` — `ImageU8` **52–170**,
  `ImageU8Pyramid` **173–215**, `ImageF32WithGrad` **810–880**
- Problem: **233 lines are container definitions**; the other ~934 are remap and sample
  kernels. **20 non-test files across three crates** name these types — `sfmtool-core`
  ×10, `sfm-explorer` ×6, `sfmtool-py` ×3, plus `benches/patch_render.rs` — and every
  one writes `use crate::camera::remap::ImageU8`, a path that reads as neither a camera
  nor a remap concept.
- Proposed fix: `sfmtool-core/src/image.rs` (sibling to the existing `numeric.rs` /
  `progress.rs`) holding the three containers and `sample_bilinear_u8` /
  `sample_bilinear_with_grad_u8` (229–275), re-exported from `camera::remap` for one
  release so the 20 call sites move in a follow-up.
- Effort: low
- Risk: low — move plus re-export; the compiler finds everything.

**`run_gpu_levels_prebuilt` is a 484-line method, two thirds of its file**
> _Carried forward. Re-measured **254–737** (484) of a 741-line file — bit-for-bit the
> same as the last two snapshots. Untouched in the window._
- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs:254`
- Problem: Second-longest function in the file is 73. Four phases are already
  banner-marked inside it: pool sizing plus per-level uniform buffers and bind groups
  (269–503), final-upsample buffers (504–601), single-command-buffer encode (602–703),
  submit and readback (704–737).
- Proposed fix: `gpu/encode.rs` for the bind-group construction, returning a per-level
  plan; the encode and submit stay in `mod.rs`.
- Effort: medium
- Risk: medium — wgpu resource lifetimes are load-bearing (buffers must outlive the
  single submit), so a bad split turns a working keep-alive into a use-after-free the
  validation layer catches only at runtime. `gpu/tests.rs` (625) runs on the noop
  backend.

**Eight `prof.rs` modules re-declare the same profiling scaffolding**
- Location: `crates/sfmtool-core/src/patch/{keypoint_localize,keypoint_subpixel,
  normal_refine,cluster_refine,view_selection}/prof.rs`, `geometry/focal_vote/prof.rs`,
  `features/cluster_match/covisibility/prof.rs`, `camera/remap/prof.rs` — **1,690 lines
  total**
- Problem: Five of the eight share a **byte-identical `enabled()` and a byte-identical
  `struct Phase` + `impl Phase { new, reset, time }`**; a sixth (`camera/remap`) shares
  `enabled()` exactly. Diffing `keypoint_localize/prof.rs:15–64` against
  `normal_refine/prof.rs:14–63` and `cluster_refine/prof.rs:15–64` gives **2 differing
  lines out of 50**, both banner comments. The doc scan confirms it independently:
  `/// Whether \`SFMTOOL_PROFILE\` is set (cached on first query).` appears in **8**
  files, `/// Run \`f\`, attributing its wall time to this phase when profiling is on.`
  in **7**, `/// One accumulating phase counter…` in **6**. Beyond that block,
  `count(&AtomicU64, u64)` is identical in five, and every `report()` opens with the
  same ~18-line `for p in PHASES { eprintln!("[sfmtool-profile]   {:<16} …") }` table
  loop. Roughly **350 of the 1,690 lines are the same machinery eight times**; only the
  `static … Phase::new("name")` declarations and the trailing custom summaries are
  per-module.
- Proposed fix: `sfmtool-core/src/prof.rs` (sibling to `numeric.rs` and `progress.rs`)
  holding `enabled()`, `Phase`, `count()` and `report_phases(header, phases, total)`.
  Each module keeps only its statics, its `PHASES` array and its bespoke trailing lines;
  the five `patch/*` files drop to about a third of their size.
- Effort: low
- Risk: low — profiling is `SFMTOOL_PROFILE`-gated and off in tests and production; the
  worst outcome is a wrong number in a diagnostic dump.

**`polar.rs` and `sweep.rs` are the same matcher with a different sort key**
- Location: `crates/sfmtool-core/src/features/feature_match/polar.rs` (712) and
  `sweep.rs` (518) — **44 shared long lines, the highest pair in `sfmtool-core`**
- Problem: Item for item: `struct GeometricInputs<'a>` (polar 258–294 / sweep 41–53 —
  identical but for two extra `positions{1,2}` fields, 6 of polar's 8 non-comment lines
  shared); `struct SortedGeometry<'a>` + `impl { forward, backward }` (polar 417–452 /
  sweep 169–205 — **25 of polar's 31 non-comment lines identical**, again only the
  `positions` fields differ); the one-way sliding-window matcher (`polar_match_one_way`
  118 lines / `match_one_way_sweep_inner` 111); the mutual cross-check driver (109 /
  79); and the two public geometric wrappers each. The doc scan finds 12 shared doc
  lines, including `/// * \`threshold\` — Optional L2 distance ceiling.` five times
  across the two files. The last cycle merged these files' plain/`_geometric` halves;
  it did not merge the two files.
- Proposed fix: `feature_match/window_match.rs` owning `GeometricInputs` (with the
  `positions` fields always present and `None`-able), `SortedGeometry`, the one-way
  window matcher and the mutual driver, parameterized on an ordering plus a per-row
  window predicate. `polar.rs` keeps `cartesian_to_polar` / `compute_angle_offset` /
  `Wraparound` / `angular_order`; `sweep.rs` keeps `argsort_by_y`.
- Effort: medium
- Risk: medium — the polar wraparound window is genuinely different from sweep's linear
  window, and folding both into one predicate is where a subtle off-by-one would land.
  `sweep/tests.rs` is 639 lines; polar's coverage is thinner.

**`cluster_census` is a 310-line function whose author already numbered its six phases**
- Location: `crates/sfmtool-core/src/analysis/cluster_census.rs:425–734` (310 of 737,
  42%; next-longest item is 86)
- Problem: The seams are banner comments that literally number themselves:
  `── Candidate poses ──` (465), `── 1. Viewpoint groups… ──` (490),
  `── 2. Cluster placement… ──` (516), `── 3. Evidence eligibility… ──` (626),
  `── 4. Per-pair census ──` (641), `── 5. Companion: global satisfaction ──` (679),
  `── 6. Companion: group consistency (opt-in) ──` (696). **Phase 6 already lives in a
  sibling** (`cluster_census/group_consistency.rs`, 604 lines), so the split direction
  is established — it just was not applied to phases 1–5.
- Proposed fix: `cluster_census/{groups,placement,pair_census}.rs` for phases 1, 2 and
  4, matching the existing `group_consistency.rs`. The driver drops to ~120 lines.
- Effort: medium
- Risk: low — the phases hand each other named vectors; `cluster_census/tests.rs` is
  1,336 lines.

**`edited.rs` holds five concerns in 1,604 lines, and `RowMap` is independent of all of
them**
- Location: `crates/sfmtool-core/src/reconstruction/edited.rs`. Longest function
  `build_point_set` **964–1069** (106), so this is **not** a length finding
- Problem: Five separable concerns: (1) `EditError` + `Display` + `Error` **33–118**;
  (2) `PointView`, the read-side borrow over the overlay, **173–292**; (3)
  `EditedReconstruction` identity, queries, capability predicates and mutation
  **294–871** (578); (4) materialization — `materialize` 89, `build_point_set` 106,
  `merge_observation_source` 59, `merge_halfvec`, `merge_bitmaps` — **873–1208** (336);
  (5) `RowMap` **1225–1503** (**279**), edited-index ↔ materialized-index translation
  (`by_scan` 89, `forward`, `inverse`, `forward_dense`, `inverse_dense`,
  `new_to_base_slot`, `holes_below`). **`RowMap` names `EditedReconstruction` zero times
  across its 279 lines**, depending only on `SfmrReconstruction` through `by_scan(before,
  after)` — the clean cut. What makes this worth flagging rather than tolerating is the
  module doc at 8–10: "`specs/core/reconstruction/edited-reconstruction.md` is the
  design; **this module is the whole of it**", which reads as a standing argument
  against ever splitting, and which a directory with a `mod.rs` carrying the same
  paragraph would satisfy just as well.
- Proposed fix: `edited/row_map.rs` (`RowMap` + `ScanRows` + `is_subsequence`) and
  `edited/materialize.rs`. `edited.rs` becomes `edited/mod.rs` at ~900, keeping the
  design paragraph reworded to describe the directory.
- Effort: medium
- Risk: low — pure code motion inside one crate; `edited/tests.rs` (682) covers both
  halves and the `pub use` keeps external paths unchanged.

**Four edit error enums repeat the same variants, docs and `Display` strings — with a
`u32`/`usize` fork**
- Location: `crates/sfmtool-core/src/reconstruction/add_observation.rs:25–115`,
  `create_point.rs:24–100`, `remove_observation.rs:20–65`, `move_camera.rs:23–57`,
  against `edited.rs:33–118`
- Problem: `ImageOutOfRange { image, image_count }` is declared in all four with
  byte-identical field docs and one byte-identical `Display` arm — `"image {image} is
  past the {image_count} images of the reconstruction"` at add_observation.rs:86,
  create_point.rs:72, move_camera.rs:43, remove_observation.rs:45. **`move_camera` types
  `image` as `usize`, the other three as `u32`**, and `EditError::ImageOutOfRange`
  (edited.rs:37) is a third shape entirely (`observation`, `image_index`,
  `image_count`) — five spellings of one condition. `NotEmbeddedPatches`,
  `PixelOutsideImage { pixel, size }` (identical docs *and* identical `Display`) and
  `ViewsMissing { got, expected }` are each duplicated between `add_observation` and
  `create_point`. The guards are duplicated too: `add_observation.rs:255–287` and
  `create_point.rs:210–239` are the same four checks in the same order, ~30 lines
  differing only in the enum prefix — including an identical five-line
  `!(pixel[0] >= 0.0 && … < h as f64)` expression down to its line breaks.
- Proposed fix: A `PreconditionError` holding the shared variants, embedded as
  `Precondition(PreconditionError)` in each of the four (they already embed
  `Edit(EditError)`, so the shape is established), plus one
  `check_pixel_edit_preconditions(edited, views, image, pixel)`. Settle `image: u32`
  in the same change.
- Effort: medium
- Risk: medium — the variants are public API through `sfmtool-py` and the explorer's
  MCP surface, so every exhaustive `match` needs a new arm and the `usize`→`u32` change
  is visible at the Python boundary. The compiler finds all of it; the `Display` strings
  are asserted in four `tests.rs` totalling 2,131 lines.

**Twelve forwarding accessors copy their callee's doc comments instead of linking them**
- Location: `crates/sfmtool-core/src/reconstruction/data.rs:303–360` and **405–460**,
  against `data/point_set.rs:167–384`
- Problem: `SfmrReconstruction` forwards 12 accessors to the identically-named
  `PointSet` method and each carries a **verbatim copy** of the callee's doc rather
  than a link — including the long ones: `validate_observation_columns` (9 lines,
  duplicated at data.rs:340–348 / point_set.rs:309–317), `observation_row` (6),
  `keypoints_xy` (4). That is ~45 lines of prose in two places, describing invariants
  ("`from_sfmr_data` builds these in lockstep, but the in-memory editors can leave them
  out of step") that will be right in one copy and stale in the other the first time
  the invariant moves. This is the 33 shared long lines the clone scan reports between
  the two files; the accessor *bodies* are legitimate delegation and are fine.
- Proposed fix: One-line summary plus ``See [`PointSet::<name>`].`` The rustdoc gate
  (warnings-as-errors, private items included) then verifies the reference survives a
  rename, which a copied paragraph never does.
- Effort: low
- Risk: low — doc-only.

**The module-file convention forks inside `patch/`**
- Location: `crates/sfmtool-core/src/patch/`
- Problem: Workspace-wide the adjacent-file form (`foo.rs` beside `foo/`) leads 132 to
  44 over `foo/mod.rs`; `sfmtool-core` is 100 / 19. There is a legible rule underneath
  — crate-root subsystems use `mod.rs`, leaf modules use the adjacent file — and it
  holds at the top level. It breaks one level down, most visibly in `patch/`, where
  nine sibling modules of the same kind split **6 adjacent** (`cloud`,
  `keypoint_localize`, `keypoint_subpixel`, `member_coherence`, `spawn`,
  `view_selection`) against **3 `mod.rs`** (`cluster_refine`, `localizability`,
  `normal_refine`). A reader cannot predict where a module's own code lives. Separately,
  `sfmtool-py` is the inverse of the workspace: **10 `mod.rs` to 1 adjacent** — each
  crate self-consistent, disagreeing across the boundary, which is the classic shape.
- Proposed fix: Write the rule in `AGENTS.md` ("a directory that is a crate-root
  subsystem carries `mod.rs`; anything below it is `foo.rs` beside `foo/`") and converge
  `patch/`'s three. An undocumented convention is how the next module forks one.
- Effort: low
- Risk: low

**`*Options` vs `*Params` forks cleanly at the `geometry`/`camera` boundary**
- Location: `crates/sfmtool-core/src/{geometry,camera}` against
  `{patch,features,spherical,analysis}`
- Problem: Against the workspace baseline (`*Params` 35, `*Options` 20, `*Config` 2,
  `*Settings` 2), the fork is not scattered — it is one module boundary. `geometry/` is
  **12 Options / 1 Params**, `camera/` **1 / 0**, against `patch/` 0/7, `features/`
  0/18, `spherical/` 0/4, `analysis/` 1/5. So `geometry` + `camera` supply **13 of the
  workspace's 20 `*Options`** and every other module is unanimously `*Params`. The
  `geometry` names: `AbsolutePoseOptions`, `AlignOptions`, `EpipolarCurveOptions`,
  `FocalVoteOptions`, `FundamentalOptions`, `GrowOptions`, `HomographyOptions`,
  `IntrinsicsOptions`, `RayEssentialOptions`, `RayRotationOptions`, `RepairOptions`,
  `ResectImageOptions`, `ResectOptions`, `VerifyOptions`. Two `*Config` strays:
  `features/feature_match/geometric_filter.rs:26` and `spherical/sphere_points.rs:30`.
  **This is arguably a legible split rather than noise** — `geometry`'s bags really are
  "options to an estimator" while `patch`'s are "algorithm parameters" — which is why it
  is filed as a convention finding rather than a rename demand.
- Proposed fix: Decide, then write it down. Either state the boundary in `AGENTS.md` as
  the rule, or rename the 14 to `*Params`. Do one; leaving it undocumented means the
  next `geometry` module coin-flips. The two `*Config` strays should become `*Params`
  either way.
- Effort: low (document) / medium (rename — 14 public types, cross-crate)
- Risk: low / medium (a rename touches `sfmtool-py` re-exports)

---

## Rust — format crates

**The per-section split proven in `matches-format/verify.rs` was carried nowhere, and
the four monoliths grew 8%**
> _Carried forward. Re-measured brace-to-brace; two of the four **grew**, and the two
> that look renamed were not split._
- Location: `crates/sfmr-format/src/verify.rs:18–651` (`verify_sfmr`, **634**);
  `sfmr-format/src/read.rs:66–655` (`read_sfmr`, **590**);
  `sfmr-format/src/write.rs:269–804` (`write_sfmr_into`, **536**);
  `matches-format/src/write.rs:54–479` (`write_matches_into`, **426**). Secondary:
  `sfmr-format/src/write.rs:1025–1310` (`validate_dimensions_with`, **286**, still
  carrying `#[allow(clippy::too_many_arguments)]` over 8 parameters) and
  `matches-format/src/write.rs:563–860` (`validate_dimensions`, **298**, was 277)
- Problem: `verify_sfmr` 514 → **634 (+23%)**, `read_sfmr` 517 → **590 (+14%)**;
  the four total 2,186 lines, up from 2,022. The two that appear renamed are not split:
  `write_sfmr_with_options` (write.rs:212–256) and `write_matches` (write.rs:32–52) are
  now 45- and 21-line wrappers, with the body moved intact into `write_sfmr_into` and
  `write_matches_into` by the `EntrySink` / `write_atomically` layer in #407.
  `verify.rs` is the extreme case: the **file is 651 lines and one function is 634 of
  them**, carrying 14 `// === Section ===` banners that name the helpers nobody
  extracted. `write_sfmr_into` carries 9, `write_matches_into` 7. **`read_sfmr` has no
  banners at all** — its eight concerns (metadata and version gate 74–100, cameras
  102–144, images 145–221, thumbnails 222–233, depth statistics 234–247, points3d
  248–430, tracks, rigs/frames) are separated only by blank lines.
  Meanwhile the pattern demonstrably works: `matches-format/src/verify.rs`'s
  `verify_matches` (174–281) is **108 lines** over 14 helpers
  (`verify_two_view_geometries_section` 103, `verify_cluster_patches_section` 91,
  `verify_clusters_section` 88, `verify_image_pairs_section` 82, `verify_images_section`
  64, `check_section_hash`, `check_pair_ordering`, …), landed as `7ce63cff` (#379). One
  file out of five, and **the same crate's `write.rs` still writes all five sections in
  one 426-line body**.
- Proposed fix: Copy the `matches-format/verify.rs` shape literally. One
  `fn <section>_section(…) -> Result<u128, E>` per banner returning that section's
  digest; the entry point becomes a 60–90-line orchestrator. Do `verify_sfmr` first —
  cleanest banners, no partial-write hazard, worst ratio. `read_sfmr` needs its concerns
  *named* before it can be split. Splitting `validate_dimensions*` per section also
  retires the `too_many_arguments` allow.
- Effort: medium (per file; five files)
- Risk: low for `verify.rs` (pure reads); medium for the writers — **the section digest
  order *is* the content hash**, so any reordering during extraction silently
  invalidates every existing `.sfmr` and `.matches`. `sfmr-format/src/tests.rs` (3,090)
  and `matches-format/src/tests.rs` (2,966) pin round-trips and hashes, which is the
  guard, but the extraction must preserve write order exactly. Do the `SectionDigests`
  finding below **first** — it turns digest order into a value the orchestrator holds
  rather than a property of statement order.

**A new instance in the newest crate: `write_into` is 46% of `kdf-format/write.rs`**
- Location: `crates/sfmtool-kdf-format/src/write.rs:395–724` (**330** of 724)
- Problem: The same shape as the four above — metadata (407–449), images and origin
  blocks (450–525), descriptor corpus (555–600), geometry corpus (600–640), tree chunks
  (640–698), content hash (699–723) — six sections in one body with a `section_digests`
  vec threaded through, and **no banner comments at all** this time. That the newest
  format crate, written *after* #379, reproduced the shape from scratch is the evidence
  that the refactor never became a convention. **Credit where it is due, and it belongs
  in the finding:** `kdf-format/src/read.rs` is the counter-example — 1,756 lines but
  ~60 items, longest function `open` at **209** (12% of the file). The reader was
  written decomposed; only the writer was not.
- Proposed fix: `write_images_section`, `write_descriptor_corpus`,
  `write_geometry_corpus`, `write_tree_chunks`, each returning its digests;
  `write_into` becomes a ~60-line driver. Cheaper here than in sfmr/matches because the
  format is new and `validation_tests.rs` (786) already pins the layout.
- Effort: low
- Risk: low — one crate, one format version, digest order asserted by
  `validation_tests.rs`.

**The content-hash fold is hand-written nine times, in two incompatible spellings, in a
workspace that has a crate for exactly this**
- Location: `crates/sfmr-format/src/write.rs:783–788` and `verify.rs:637–641`;
  `matches-format/src/write.rs:460–464` and `verify.rs:267–271`;
  `sift-format/src/write.rs:55–97` and `verify.rs:62–93`;
  `camrig-format/src/write.rs:60–117` and `verify.rs:58–60`;
  `sfmtool-kdf-format/src/write.rs:699–709`
- Problem: Every format's content hash is defined as "XXH128 over the big-endian
  concatenation of the per-section XXH128 digests, in write order". That rule exists in
  **nine copies, in two spellings**: `sfmr`/`matches`/`kdf` accumulate `Vec<u128>` and
  fold with `.flat_map(|d| d.to_be_bytes())`; `sift`/`camrig` `extend_from_slice(
  &xxh3_128(x).to_be_bytes())` straight into a `Vec<u8>`. Both compute the same thing
  today, which is precisely what makes the fork durable — nothing fails if one drifts,
  and the write-side and verify-side folds *within one crate* must agree byte-for-byte
  or every file that crate writes fails its own verifier. `sfmtool-archive-io/src/lib.rs:4–14`
  claims ownership of this in its own module doc ("per-section XXH128 content hashes
  computed over the *uncompressed* bytes… owns only the container primitives they
  share") and ships **both ends** — `write_binary_entry_hashed` (392) and `format_hash`
  (498) — but no accumulator for the middle. So the part that actually defines a file's
  identity is the part that was left out. All five format crates already depend on
  `sfmtool-archive-io`, so the home exists.
- Proposed fix: Add `SectionDigests` to `sfmtool-archive-io` — `push(u128)`,
  `push_bytes(&[u8])` (hashing for you), `finish() -> u128` — with the big-endian fold
  as its one implementation and a doc comment stating the rule once. Nine call sites
  collapse to it, 3–6 lines each.
- Effort: low
- Risk: low — a strict refactor: the bytes hashed are unchanged, and every format crate
  has round-trip and hash-stability fixtures that fail loudly on any divergence.

**A three-place copy of the matches backbone rule, already textually drifted**
- Location: `crates/matches-format/src/read.rs:167–180`, `238–243`, `299–306`;
  `write.rs:539–558` (inside `validate_structure`); `verify.rs:104–144` (inside
  `structure_errors`)
- Problem: The rule "a pairwise file carries `image_pair_count`/`match_count` and never
  `cluster_count`/`cluster_member_count`, and a cluster file the reverse" is implemented
  three times with four error strings that are byte-identical between `read.rs` and
  `verify.rs` and **already differ from `write.rs`**: write.rs:552 has `"cluster-bearing
  file requires metadata.cluster_count and metadata.cluster_member_count"` on one line
  while read.rs:299 and verify.rs:104 wrap it across two. That is proof the copies are
  maintained independently rather than by search-and-replace. The three differ
  structurally too: `read.rs` returns `Err` on first violation, `write.rs` via a local
  `invalid!` macro, `verify.rs` accumulates into a `Vec<String>`.
- Proposed fix: One `fn check_backbone_counts(metadata: &MatchesMetadata) ->
  Vec<String>` in `types.rs` beside `MatchesMetadata`; `verify` extends its vector,
  `read` and `write` take the first entry as an `InvalidFormat`.
- Effort: low
- Risk: low — the strings are asserted in `matches-format/src/tests.rs`.

**`WorkspaceContents` and `WorkspaceMetadata` are declared byte-identically in two
format crates, and both docs say so**
- Location: `crates/matches-format/src/types.rs:58–79` and
  `crates/sfmr-format/src/types.rs:109–125`
- Problem: Both crates declare `pub struct WorkspaceContents { feature_tool,
  feature_type, feature_options, feature_prefix_dir }` and `pub struct
  WorkspaceMetadata { absolute_path, relative_path, contents }` with byte-identical
  fields **and byte-identical doc comments** — including the same parenthetical,
  "(mirrors `.sfm-workspace.json`)". That parenthetical is the code admitting a
  duplication with nothing behind it: two crates independently model one on-disk JSON
  file, and `matches-format`'s own second doc adds "Same structure as in `.sfmr` files"
  in prose. Both crates already depend on `sfmtool-archive-io`.
- Proposed fix: Move both structs to `sfmtool-archive-io` and re-export from each format
  crate. The workspace stanza is a container-level concept, which is what that crate is
  for.
- Effort: low
- Risk: low — the serde shape is unchanged, so no on-disk change; both crates' round-trip
  tests cover it.

**`LazyKdForestOptions` names a type from the crate above it**
- Location: defined at `crates/sfmtool-kdf-format/src/types.rs:117`; the type it is
  named for, `LazyKdForest`, is `crates/sfmtool-core/src/features/kdforest/persistent.rs:304`
- Problem: `sfmtool-core` depends on `sfmtool-kdf-format`, not the reverse. But the
  format crate's own reader, `KdfFile::open` (read.rs:200), takes an options struct
  named after a type in the downstream crate it cannot see, and half its eight fields
  (`max_address_map_bytes`, `max_metadata_bytes`, `max_chunk_bytes`,
  `max_compressed_bytes`) are pure container limits with nothing to do with a forest. A
  reader of `kdf-format` alone meets a name with no referent. The stats types fork the
  same way — `KdfIoStats` (format crate) beside `LazyQueryStats` (core) are two counter
  structs for one query path under two naming schemes. Lower-value and related: four of
  the five on-disk format crates are `<format>-format`; the fifth took the `sfmtool-`
  prefix the workspace otherwise reserves for the *non*-format crates (`sfmtool-core`,
  `sfmtool-py`, `sfmtool-archive-io`), while `AGENTS.md` lists all five together as one
  group.
- Proposed fix: Rename to `KdfOpenOptions`; `sfmtool-core` re-exports it under whatever
  name suits `LazyKdForest`. Optionally rename the crate to `kdf-format` — nothing
  outside the workspace consumes it by name, so it is a `Cargo.toml` plus `use` sweep,
  and it only gets more expensive.
- Effort: low
- Risk: low — compiler-checked; `sfmtool-py/src/spatial/kdf.rs` and
  `matching/cluster.rs` are the only out-of-crate users.

**`sfmtool-kdf-format` is 63% undocumented at its public surface**
- Location: `crates/sfmtool-kdf-format/src/types.rs` (**100 of 129** public items
  undocumented, 78%); crate total **121/193 (63%)**; also `cache.rs:18,25,33,99,281`
  and ten `KdfFile` accessors at `read.rs:410–431,528,822`
- Problem: This resolves the measured 7.4% doc-comment density — the lowest in the
  workspace by a wide margin. Undocumented public items per crate:
  **kdf-format 63%**, sfmr-format 26%, matches-format 13%, sift-format 9%,
  camrig-format 4%, `sfmtool-archive-io` **0%**. It is concentrated in one file, and
  `lib.rs:19` re-exports that file wholesale (`pub use types::*`), so `types.rs` **is**
  the crate's public vocabulary. The shape of the gap is specific: almost every *type*
  has a one-line doc and almost no public *field* has one. That falls hardest on the two
  structs a caller is obliged to fill in — `KdfWriteOptions` (97–101) with **4 of 4**
  fields undocumented, and `LazyKdForestOptions` (117–126) with **7 of 8**, where only
  `max_compressed_bytes` is documented, sitting among seven silent siblings, which is
  the tell that this was time pressure rather than policy. Units are unrecoverable from
  the names: `origin_block_rows` is rows, `target_chunk_bytes` is bytes,
  `max_leaf_features` a count, `query_workers` a thread count. Also `KdfIoStats` 12/12
  (including the non-obvious `duplicate_load_waits` and the
  `resident_bytes`/`in_flight_bytes`/`peak_*` distinction), `Verification` 6/6,
  `KdfSiftSources` 6/6.
  **Counter-evidence, which belongs in the finding:** the on-disk layout is *not*
  undocumented. `specs/formats/kdf-file-format.md` is 610 lines, current for v2
  including the co-blocked geometry corpus and a "Hash composition" section, with repo
  links back into the crate. The verdict is "an undocumented Rust API for a
  well-documented format" — not an undocumented format, and not terse mechanical code
  either, since options and stats structs are the opposite of mechanical.
- Proposed fix: Document the fields of the seven `pub` structs in `types.rs`,
  prioritising `KdfWriteOptions` and `LazyKdForestOptions` — one line each stating unit
  and effect, linking the spec section that sizes it rather than restating it. Then
  `cache.rs`'s five items. About 60 lines of prose, taking the crate from 63% to under
  20%. The ten bare `KdfFile` accessors are one-liners; `cache.rs`'s items are `pub`
  inside a private module and could honestly be `pub(crate)` instead.
- Effort: medium
- Risk: low — doc-only, and the rustdoc gate already documents private items so any
  intra-doc link added here is checked.

**Four inline `#[cfg(test)]` modules — a convention closed last cycle, re-broken in the
two newest subsystems**
- Location: `crates/sfmtool-core/src/features/kdforest/persistent.rs:910–1253` (`mod
  tests`, 343 lines); `crates/sfmtool-kdf-format/src/cache.rs:484–694` (`mod tests`,
  210); `crates/sfmtool-kdf-format/src/read.rs:1551–1756` (**`mod profiling`**, 205);
  `crates/sfmtool-kdf-format/src/summary.rs:311–344` (`mod tests`, 33)
- Problem: **791 lines of inline test code against 155 sibling `tests.rs` files** — and
  `sfm-explorer` is 37/0 clean, `sfmtool-core` outside `kdforest/` has exactly one
  violation, so these four are the whole set. The last snapshot closed seven of these
  and warned that "the check the next snapshot should run is not `git grep "mod tests
  {"` but a scan for a `#[cfg(test)]` line followed by any `mod ... {`, which is what
  found the last two". That scan is exactly what found `read.rs`'s `mod profiling`. It
  inflates the three files' apparent size by 27%, 30% and 12% — `persistent.rs` is 1,253
  lines of which **910 are production**, and `read.rs` is 1,756 of which **1,550 are**.
  `persistent.rs` is the sharpest case: its directory already holds `kdforest/tests.rs`
  (406) and `kdforest/distance/tests.rs` (59), so the convention is established one
  level up from the violation.
  **There is a concrete mechanical cost beyond tidiness.** `numeric/tests.rs::the_workspace_has_one_median`
  — the grep-based gate the last report's Top 3 asked for — exempts test code by
  **skipping files named `tests.rs`** (`rs_sources`, numeric/tests.rs:254–270). An
  inline `#[cfg(test)] mod tests` is not skipped, so a test helper named
  `expected_median` inside any of these four would fail a workspace-wide production-code
  gate, with a message pointing at a production file. Any future source-scanning gate
  written on the same exemption inherits the hole.
- Proposed fix: `kdforest/persistent/tests.rs`, `kdf-format/src/cache/tests.rs`,
  `kdf-format/src/summary/tests.rs`, and `kdf-format/src/read/profiling.rs` for the
  `#[ignore]`d `KDF_PROFILE_PATH` diagnostic — genuinely a different kind of test, which
  argues for its own honestly-named file, not for inlining. Note `mod profiling` reaches
  `file.corpus`, a private field, so it must be a child of `read`.
- Effort: low
- Risk: low — moving `#[cfg(test)]` code; each `use super::*` gains a level.

---

## Rust — `sfmtool-py`

**The shared prologue spread from three bindings to five instead of being lifted**
> _Partially resolved. The `parse_patch_window` / `parse_sampler` half of the old
> finding **landed** as `f51a109`; `args.rs` is now 93 lines and all ten inline matches
> route through it. The prologue half did not land, **and it got worse.**_
- Location: `crates/sfmtool-py/src/patches/` — `localize_keypoints.rs:190–216`,
  `refine_keypoints.rs:160–186`, `refine_normals.rs:157–186`,
  `member_coherence.rs:190–214`, `select_views.rs:110–136`
- Problem: `localize_keypoints.rs:190–218` and `refine_keypoints.rs:160–188` are **29
  consecutive byte-identical lines** (first divergence at the 30th). Against the other
  three, **25 of 27 non-comment lines are byte-identical**, the two differing only in an
  identifier and its message string. The whole-file duplicate scan has got worse, not
  better: `localize_keypoints ‖ refine_keypoints` = **58** shared long lines (was 50),
  `member_coherence ‖ select_views` = **42** (was 29), plus seven more pairs over 28 —
  **nine pairs above 28 across five files**. The old finding named three bindings; there
  are now five.
  Related and the same edit: these five take **27 / 26 / 19 / 18 / 16** positional args
  and each carries `#[allow(clippy::too_many_arguments)]` (10 such allows across 9 files
  in `patches/`). Unlike the SIMD dispatch triples elsewhere in the workspace — where
  the signatures *must* match — **this cluster is drift**: they are flattened kwarg bags
  with nothing constraining their shape.
- Proposed fix: `views.rs` (408 lines) already owns `resolve_scene` (161),
  `resolve_pyramids` (181) and `PosedViews` (117). Add
  `pub(super) fn resolve_patch_scene(recon, cloud, view_arg, view_arg_name) ->
  PyResult<(PosedViews, Option<&Recon>, u32)>` doing the resolve, the `point_indexes`
  length check, the point-range check and the parameterized `CameraViews` guard.
  Removes ~135 duplicated lines across the five.
- Effort: low
- Risk: low — keeping the option name a parameter preserves the guard messages
  verbatim, and `tests/patch/` asserts them.

**Four sibling bindings spell one argument four ways, and one is the crate's lone
`_indices`**
- Location: `crates/sfmtool-py/src/patches/` — `localize_keypoints.rs` and
  `refine_keypoints.rs` take `view_sets`, `refine_normals.rs` takes `view_indices`,
  `member_coherence.rs` takes `member_views`, `select_views.rs` takes `candidate_views`
- Problem: These five share the 27-line prologue above and take the *same* conceptual
  argument — a per-patch list of image indexes — under four names. And `view_indices` is
  one of only **4** `*_indices` identifiers in all of `sfmtool-py` against **~110**
  `*_indexes` (25 `image_indexes`, 13 `cluster_indexes`, 12 `point_indexes`, 12
  `camera_indexes`, 11 `track_image_indexes`, …). At the Python API surface the
  spelling is unambiguously `indexes`, and `point_indexes` is consistent across all five
  of these very bindings. The other stray is `camera_indices` at `matching/image.rs:130/143/182`
  against 12 `camera_indexes` elsewhere.
- Proposed fix: Settle on `view_indexes`, or fold all four into one `views` argument
  when the prologue is lifted — the two fixes are the same edit. Rename
  `camera_indices` → `camera_indexes`.
- Effort: low
- Risk: medium — these are Python-visible kwargs, so a rename breaks callers silently at
  the call site; check `src/sfmtool/` and add a deprecation cycle.

**Sixteen binding names leak their language of origin into the Python API, at the top
level, under two conflicting suffixes**
- Location: `sfmtool._sfmtool.analysis` — `apply_se3_to_camera_poses_py`,
  `build_covisibility_pairs_py`, `build_frustum_intersection_pairs_py`,
  `filter_tracks_by_point_mask_py`, `find_point_correspondences_py`,
  `merge_points_and_tracks_py`, **`estimate_alignment_rs`**, **`ransac_alignment_rs`**;
  `sfmtool._sfmtool.matching` — `match_image_pair_py`, `match_image_pairs_batch_py`,
  `match_one_way_sweep_py`, `match_one_way_sweep_geometric_py`,
  `mutual_best_match_sweep_py`, `mutual_best_match_sweep_geometric_py`,
  `polar_mutual_best_match_py`, `polar_mutual_best_match_geometric_py`
- Problem: `_py` and `_rs` are Rust-side disambiguators between the binding wrapper and
  the core function. From Python, `match_image_pair_py` means "the Python one" in a
  module where everything is. The two suffixes are **inconsistent within one submodule**
  — `analysis` carries both `estimate_alignment_rs` and `filter_tracks_by_point_mask_py`.
  Because `sfmtool/__init__.py:11–12` does `import *` from both submodules, **all 16
  land on the top-level package** as `sfmtool.match_image_pair_py` (verified against the
  built extension, alongside 178 public top-level names and no `__all__`). Python
  callers already treat the names as wrong, and the workaround has itself forked: four
  import sites alias the suffix away (`feature_match/_core.py:17–18` →
  `_rust_match_image_pair`, `_image_pair_graph.py:10–11`), while three use the suffixed
  name bare (`merge/correspondences.py:13`, `merge/reconstructions.py:77`,
  `_point_correspondence.py:11`).
- Proposed fix: Give each `#[pyfunction]` an explicit unsuffixed `#[pyo3(name = "…")]`,
  leaving the Rust symbol as-is; drop the four aliasing imports and update the three
  bare call sites. `tests/rust_bindings/test_analysis_registration.py:9–18` asserts the
  current names and is updated in the same commit. Separately, an explicit `__all__` in
  `sfmtool/__init__.py` would stop future binding additions silently widening the
  top-level surface.
- Effort: medium — needs `maturin develop --release`.
- Risk: medium — public API; keep the old names as deprecated aliases for one release.

**`clone_with_changes` is a 652-line function that is a 27-arm string-keyed dispatch
table**
> _Carried forward. Re-measured: file 771 → **904**, function 599 → **652 (+9%)**._
- Location: `crates/sfmtool-py/src/reconstruction/clone.rs:76–727` (72% of the file;
  the other two functions are 100 and 59)
- Problem: Lines **105–521** are one `match key_str.as_str()` with **27 arms**, each
  6–50 lines of extract-validate-assign (`"positions"` 108, `"colors"` 159, `"errors"`
  183, `"normals"` 202, `"normal_confidence"` 234, `"point_constraints"` 262, …
  `"rig_frame_data"` 498, `"world_space_unit"` 508). Lines **523–727** are a fixed
  post-processing sequence with its own banners: image rebuild, observation-source
  recombination, histogram resize, deferred patch bitmaps, track rebuild, derived fields.
- Proposed fix: `clone/{point_fields,image_fields,track_fields}.rs`, each exposing
  `fn apply(&mut recon, key, value) -> PyResult<bool>`; the driver becomes a chain of
  three tries plus a `finalize_clone`.
- Effort: medium
- Risk: medium — the arms have order dependencies the comments call out ("deferred so
  this wins over the 'patches' clear"). A silent reorder produces a wrong reconstruction
  rather than an error. `tests/xform/` (23 modules) is the guard.

**Three bindings hand-transcribe the same CSR docstring block, and two duplicate its
resolver**
- Location: `crates/sfmtool-py/src/geometry/focal_vote.rs` (475),
  `geometry/estimate_intrinsics.rs` (228), `analysis/cluster_radii.rs` (261)
- Problem: Two layers. **(a) Docstrings**: `estimate_intrinsics.rs:31–62` and
  `focal_vote.rs:285–315` share **15 byte-identical `///` lines** — the whole
  `cluster_starts` / `member_images` / `member_positions` / `width` / `height` argument
  block, transcribed rather than referenced; `cluster_radii.rs` shares 11 with
  `focal_vote.rs`. They describe a contract the core enforces in one place, so they
  drift silently. **(b) Code**: `estimate_intrinsics.rs` correctly imports `vote_source`
  from `focal_vote.rs`, but `cluster_radii.rs:47–95` (`radii_source`, 49 lines)
  reimplements it — same `source.cast::<PyMatchesFile>()` dispatch, the **verbatim
  identical** error string `"the first argument must be a MatchesFile or a (n_clusters +
  1,) uint32 cluster_starts array"`, the same refusal, the same nondecreasing and
  closes-at-member-count validation, against `vote_source` (focal_vote.rs:82–124). Its
  `member_shapes_f32` likewise mirrors `member_positions_f32` — the
  accept-f32-else-cast-f64 coercer, which appears a **fourth and fifth** time in
  `matching/covisibility.rs:56` and `spatial/kdtree.rs:73/354`.
- Proposed fix: A `sfmtool-py/src/matches_args.rs` with the shared `MatchesOrCsr`
  resolver, one `ndarray_f32_or_f64::<N>()` coercer, and one
  `#[doc = include_str!("../docs/csr_observations.md")]` fragment the three splice in.
- Effort: medium
- Risk: low — the error strings are asserted in the Python tests.

---

## Python — `src/sfmtool/`

Python moved +1,166/−230 in the whole window, so with two exceptions every finding here
is carried forward with numbers **re-verified and essentially unchanged**. That is not a
reason to discount them; several are actively divergent duplications.

**A render-farm SDK is a hard dependency for two path-formatting functions, imported
twelve ways**
> _Carried forward. Re-verified exactly: still 12 sites, still a 6/6 split with no rule._
- Location: `pyproject.toml:17`. **Top-level (6):** `sift/file.py:17`,
  `_commands/sift.py:11`, `_commands/solve.py:11`, `_global_sfm.py:10`,
  `_incremental_sfm.py:11`, `_sfmr_naming.py:12`. **Deferred into a function body (6):**
  `analyze/summary.py:429`, `camrig/cp.py:48`, `feature_match/_run.py:391`,
  `motion/recon_discontinuity.py:584`, `_commands/motion.py:93`, `_compare.py:373`
- Problem: `deadline` — the AWS Deadline Cloud client — is a runtime dependency, and the
  entire use of it is two pure functions: `summarize_path_list` and
  `summarize_paths_by_sequence`. A repo-wide grep for `deadline` returns those 12 lines
  and the pyproject entry, nothing else. There is no local seam: replacing, vendoring or
  stubbing them means editing twelve files, and `_sfmr_naming.py` — which imports both
  and whose whole job is filename construction — is the obvious place that seam should
  have been.
- Proposed fix: One `_path_summary.py` re-exporting both; all 12 sites import from
  there at module top level. "Do we still need `deadline`?" then becomes answerable by
  reading one file.
- Effort: low
- Risk: low — pure re-export. Measure `sfm --help` startup either side, since six sites
  were presumably deferred for import cost.

**`_commands/solve.py` carries solver orchestration below its Click declaration**
> _Carried forward with the **framing corrected**. The file measurement holds; the "28
> of 29" claim does not survive re-measurement and is dropped._
- Location: `src/sfmtool/_commands/solve.py` (508 → **526**); the `solve` Click command
  **122–342** (221 lines, 16 params); `_run_sequential_overlap_sfm` **345–465** (121);
  `_run_sfm` **468–526** (59)
- Problem: The two helpers are unchanged at 180 lines. But counting module-level
  non-Click functions across all 29 command modules: **19 of 29 have zero**, 10 have
  some, and the largest is **not** solve.py — `_commands/estimate_intrinsics.py` carries
  **285** helper lines. The distinction that survives is *kind*, not size:
  estimate_intrinsics' helpers are report and `.camrig` I/O and its docstring says so,
  whereas solve.py's two are solver orchestration — windowing arithmetic,
  `--seq-overlap` parsing, dispatch into `_incremental_sfm`/`_global_sfm` — which is
  exactly what `feature_match/_run.py:5–9` says belongs outside the wrapper. As it
  stands the sequential-overlap solve cannot be called except through Click.
- Proposed fix: Move both to `_solve_driver.py` beside `_incremental_sfm.py` and
  `_global_sfm.py`, and defer-import from the callback.
- Effort: low
- Risk: low — one import edit; `_run_sfm` is called only from within solve.py.

**`_global_sfm.py` and `_incremental_sfm.py` are 88% the same 109-line function, and
have already drifted**
- Location: `src/sfmtool/_global_sfm.py:19–127` (`run_global_sfm`, 109 lines, 15 params)
  vs `src/sfmtool/_incremental_sfm.py:25–132` (`run_incremental_sfm`, 108, 15)
- Problem: A direct diff produces **13 changed hunks over ~109 lines** — **61 shared
  long lines, the highest cross-file count anywhere in the Python package.** The genuine
  differences are five: `GlobalPipelineOptions()` vs `IncrementalPipelineOptions()`,
  `mapper_options.mapper.bundle_adjustment.refine_sensor_from_rig` vs
  `mapper_options.ba_refine_sensor_from_rig`, `pycolmap.global_mapping` vs
  `incremental_mapping`, `tool_name="glomap"` vs `"colmap"`, and
  `tool_options {"algorithm": "global"}` vs `{}`. Everything else — argument list, seed
  handling, output-path resolution, error wrapping — is copied, **and the copies have
  visibly drifted**: `_global_sfm.py:67–68` uses `Path`/`mkdir` where
  `_incremental_sfm.py:66–67` uses `os.path.join`/`os.makedirs` for the same directory,
  and `_incremental_sfm.py` calls `pycolmap.set_random_seed` **twice** (46 and 96) where
  the global copy calls it once (91).
- Proposed fix: One `_run_pycolmap_sfm(...)` taking a small strategy record
  (`options_cls`, `mapper_fn`, `refine_rig_setter`, `tool_name`, `tool_options`); the
  two public functions become ~10-line wrappers. Removes ~95 duplicated lines.
- Effort: medium
- Risk: medium — this is the solve hot path for both `sfm solve -i` and `-g`, and the
  pycolmap options objects are not interchangeable, so the strategy record must be
  tested against both engines.

**`xform/_arg_parser.py`: a 440-line argv loop, a 20× repeated guard, and four functions
that reduce to one**
> _Carried forward. Re-verified, and the duplication is now proven byte-exact._
- Location: `src/sfmtool/xform/_arg_parser.py` (**832**); `parse_transform_args`
  **393–832** (440); the `"--X requires an argument"` guard at 408, 431, 453, 466, 548,
  563, 585, 597, 613, 630, 651, 671, 686, 701, 715, 721, 727, 756, 769, 809 (**20
  times**, and nowhere else in the repo); `parse_refine_normals_params` **125–170**,
  `parse_refine_keypoints_params` **194–239**, `parse_localize_keypoints_params`
  **269–315**, `parse_to_embedded_patches_params` **330–372** (**182 lines**, not the
  ~203 previously reported)
- Problem: Normalizing the four bodies by substituting only three tokens — the option
  name, the `_*_KEYS` constant, the transform class — and diffing shows **all four
  reduce to the same 35 code lines, byte-identical**; the only surviving differences are
  docstring wording and one comment. That is **105 lines of pure copy**. Separately the
  440-line loop dispatches ~20 options through a hand-written argv walk whose 20
  repeated guards are themselves the argument for a table.
- Proposed fix: (a) one `_parse_kv_params(param, option_name, keys, transform_cls)`
  replacing all four (−105 lines); (b) a `_take_arg(argv, i, option)` for the 20 guards;
  (c) drive the loop from a `{option: (arity, builder)}` table.
- Effort: medium
- Risk: medium — the error text is user-visible and asserted; `tests/xform/` (23
  modules) must be green either side, and transform ordering is semantic.

**`sift/file.py` holds four concerns and its name describes one**
> _Carried forward. Ranges re-verified unchanged; the consumer graph now names each
> destination._
- Location: `src/sfmtool/sift/file.py` (**882**). File I/O proper **229–597** (~370);
  extraction pipeline `image_files_to_sift_files` **598–766** + `_opencv` **769–791**
  (192); visualization `draw_sift_features` **799–882** (84); stranded xxh128 helpers
  **61–110**; pure feature geometry `compute_orientation` / `feature_size*` **118–167**
- Problem: The consumer graph makes each piece's home obvious. The xxh128 pair is
  consumed by `_patch_compaction.py:44`, `_undistort_images.py:227,230`,
  `_workspace.py:40` and all three `sift/extract_*.py` — workspace hashing, not `.sift`
  I/O. `image_files_to_sift_files` is consumed by `colmap/db_export.py:14`,
  `colmap/db_setup.py:18`, `feature_match/_run.py:55` and `_commands/sift.py:21`, and
  sits beside three `sift/extract_*.py` backends it drives by deferred import from
  inside its own body. `draw_sift_features` has exactly one consumer
  (`_commands/sift.py:19`) and there is a `visualization/` package.
- Proposed fix: `sift/extract.py` for the pipeline (the deferred import becomes a normal
  one), `visualization/_sift_display.py` for the drawing, and a workspace-hashing home
  for the xxh128 pair. `sfmtool/__init__.py:26–41` already re-exports all 16 names, so
  the public surface is unchanged.
- Effort: medium
- Risk: low — pure moves behind existing re-exports.

**Strip modules form one closed pipeline that should be a `strips/` subpackage**
> _Carried forward. Sizes and the edge count both re-verified exactly._
- Location: `_solve_strips.py` (486), `_compare_strips.py` (479), `_inspect_strips.py`
  (241), `_strip_montage.py` (210), `_patch_ncc.py` (178) — **1,594 lines across 5 flat
  top-level siblings**
- Problem: Internal edges: `_compare_strips → _solve_strips, _strip_montage`;
  `_inspect_strips → _solve_strips, _strip_montage`; `_solve_strips → _patch_ncc`.
  **Inbound edges from outside the set: exactly two** — `_compare.py:239` and
  `_commands/inspect.py:177`. A five-module cluster with two entry points is a package.
  Additionally `_patch_ncc.py` is **misnamed for its position**: its own docstring says
  it serves "the `compare --strips` engine", yet it sorts alphabetically among the
  patch-pipeline modules `_patch_compaction.py` / `_embed_patches.py` /
  `_cluster_patches.py`, which it has nothing to do with.
- Proposed fix: `src/sfmtool/strips/` with `_solve.py`, `_compare.py`, `_inspect.py`,
  `_montage.py`, `_ncc.py` and an `__init__.py` exporting the two entry points.
- Effort: low
- Risk: low — two inbound imports to rewrite; `tests/test_cli_inspect_strips.py` imports
  two by path.

**`draw_epipolar_visualization`: 509 lines, 16 parameters, a 2×3 mode matrix — and its
sibling already shows the fix**
> _Carried forward. Re-verified unchanged._
- Location: `src/sfmtool/visualization/_epipolar_display.py` (**619**); the function
  **111–619** (509, **16 positional parameters**, 82% of the module). Mode branches:
  `rectify` 397 / `undistort` 458 / neither 518, crossed with `side_by_side` 583 vs 594
- Problem: Six mutually exclusive paths in one body; the mode is threaded through rather
  than resolved once. The precedent for the fix is its own sibling:
  `visualization/_flow_display.py` faced the same problem and **has already** split its
  entry point into `_draw_flow_only_mode` (381–453) and `_draw_comparison_mode`
  (456–591) behind a dispatcher.
- Proposed fix: Mirror `_flow_display.py` — `_draw_rectified_mode` /
  `_draw_undistorted_mode` / `_draw_direct_mode`, and fold the 16 params into a
  dataclass (`_commands/epipolar.py:299,325,356` already passes them by keyword at three
  call sites).
- Effort: medium
- Risk: medium — no unit test targets this function directly; verify by rendering the
  same output before and after.

**Duplicated helper pairs: all six re-verified, three actively divergent, two upgraded**
> _Carried forward. Every pair re-checked at HEAD; the count went **up**, and the old
> finding named the wrong rotation-angle pair._
- Location and per-pair verdict:
  - **`_apply_range_filter`** — `_commands/to_colmap_bin.py:89–119` vs
    `_commands/to_nerfstudio.py:136–166`: 31 lines each, **exactly one differing line**
    (`print(` vs `click.echo(`). The shared surface is wider than the function: the two
    modules share **35** long lines, including the `--range` / `--filter-points` Click
    option declarations (26–35 / 51–60) and the validation guard (75–76 / 96–97).
  - **`_load_gray`** — `feature_match/_flow_matching.py:154–159` vs
    `motion/flow_stats.py:12–17`: an AST-level exact-duplicate scan over all 152 modules
    found this as **the only byte-identical cross-file function body in the package**.
    Docstrings differ. `motion/image_sequence.py:15` already imports the `flow_stats`
    copy, so one is redundant.
  - **`_classify_ratio`** — `motion/report.py:245–258` vs
    `visualization/_discontinuity_display.py:209–219`: **still divergent, and worse than
    recorded.** Report uses `_RATIO_UPPER = 1.0/_RATIO_LOWER` = 1.3333…; the display
    hardcodes `1.33`; the empty case is `None` vs `""`. The same threshold pair is in
    fact written **four ways across three modules** — add
    `_discontinuity_display.py:236` (`1.0 / 0.75` inline) and
    `motion/image_sequence.py:209` (a user-facing string `"outside [0.75, 1.33]"`).
  - **Sequence-descriptor naming** — `_sfmr_naming.py:58–76` vs
    `feature_match/_run.py:391–410`: **still divergent.** `_sfmr_naming` falls back to
    `f"{first_name}-total-{total_count}-images"` when the paths are not one sequence;
    `_run.py` initializes `descriptor = ""` with no `else`, so a `.matches` file
    silently gets no descriptor where the `.sfmr` would get one.
  - **`_camera_centers`** — **upgraded from 2 copies to 3**: `analyze/images.py:18–27`,
    `rig/panorama.py:56–65`, `_embed_patches.py:240–256`. All compute `C = −Rᵀt`. Note
    the `analyze/` copy is a private name imported across a subpackage boundary at
    `xform/_select_by_distribution.py:30`.
  - **`_rotation_angle_deg`** — confirmed a **name collision, not a duplicate**
    (`_compare_fragments.py:332` takes an `Se3Transform`;
    `motion/recon_discontinuity.py:23` takes two quaternions). But the scan found the
    **real** duplicate the old finding missed: `analyze/images.py:30–32`
    `_compute_rotation_angle(quat_a, quat_b)` is the same expression as
    `recon_discontinuity.py:23–25` under a different name.
- Proposed fix: (a) `_apply_range_filter` plus its two Click options into a shared
  `_commands/_range_options.py` decorator taking an `echo` callable; (b) delete one
  `_load_gray`; (c) one `_classify_ratio` with one `_RATIO_LOWER`/`_RATIO_UPPER` pair,
  deriving `image_sequence.py:209`'s message string from them; (d) `_run.py` calls
  `_sfmr_naming._generate_image_descriptor`; (e) one `camera_centers()`, three call
  sites; (f) rename one `_rotation_angle_deg` and merge `_compute_rotation_angle` into
  the surviving quaternion one.
- Effort: low each, medium in aggregate
- Risk: medium for (c) — unifying on `1.0/0.75` moves the threshold by 0.0033, which can
  flip a borderline frame's classification and any golden output capturing it. Also
  medium for (d): fixing it *changes* `.matches` names in the multi-sequence case, which
  is the point. Low for the rest.

**Flat modules whose only consumer is one sibling, and seven phase labels that
contradict themselves**
> _Carried forward. Sizes, the single-entry structure and the wrong labels all
> re-verified unchanged._
- Location: `_rectification.py` (212) — sole production importer is
  `visualization/_epipolar_display.py:14`, used at 310 and 426. The compare trio
  `_compare.py` (818) + `_compare_fragments.py` (411) + `_compare_strips.py` (479) =
  **1,708 lines** behind the single entry `_commands/compare.py:11`. Phase labels in
  `_compare.py`: **107 `[1/6]`, 113 `[2/6]`, 117 `[3/6]`, 128 `[4/6]`, 171 `[5/6]`, 203
  `[6/7]`, 220 `[7/7]`**
- Problem: A user watching a run sees it count toward 6, then the denominator change to
  7 mid-run. Seven phases exist, so **five of the seven labels are wrong**. Separately,
  `_compare_fragments` has exactly one production importer and `_compare_strips` one —
  1,708 lines behind a single CLI entry point, flat, while every other multi-module CLI
  topic already has a subpackage.
- Proposed fix: Renumber all seven to `[N/7]` — a one-line-each fix worth doing
  immediately and independently. Then move `_rectification.py` into `visualization/` and
  group the trio as `src/sfmtool/compare/`.
- Effort: low (labels) / low (moves)
- Risk: low — `tests/test_compare.py:18` and `tests/test_epipolar.py:323,347` import
  these by path and need updating.

**Three `xform` transform modules each redeclare the same enum whitelist and its
validation**
- Location: `_WINDOWS` / `_SAMPLERS` defined identically at
  `src/sfmtool/xform/_localize_keypoints.py:33–34`, `_refine_keypoints.py:31–32`,
  `_refine_normals.py:40–41`; the matching validation block at `_refine_keypoints.py:76–84`,
  `_refine_normals.py:112–128`, `_localize_keypoints.py:115–123`
- Problem: `("gaussian_disk", "gaussian", "uniform")` and `("bilinear", "bilinear_mip",
  "anisotropic")` are written three times, with the six `raise ValueError(f"… must be
  one of {…}")` lines beside them. Pairwise duplicate-line counts among the three are
  32 / 30 / 28. Adding a sampler to the Rust binding requires three coordinated Python
  edits with nothing to catch a missed one. **This is the Python end of the same family
  as the `sfmtool-py` `args.rs` finding** — those bindings parse the argument these
  three classes then re-validate.
- Proposed fix: One `xform/_patch_params.py` holding `_WINDOWS`, `_SAMPLERS` and a
  `validate_patch_params(...)` the three constructors call.
- Effort: low
- Risk: low — `tests/xform/test_{refine_normals,refine_keypoints,localize_keypoints}.py`
  cover the messages.

**`AlignToTransform` and `AlignToInputTransform` differ only in where the target comes
from, and one of them uses a class-level mutable stash**
- Location: `src/sfmtool/xform/_align_to.py` (69) vs `_align_to_input.py` (72)
- Problem: The two classes share their entire `apply` body from `source_name_to_idx = …`
  onward — `_align_to.py:28–66` vs `_align_to_input.py:31–69`, **39 lines identical
  except for two references** to the target reconstruction. The only real difference is
  provenance: one loads a `.sfmr` from a path, the other reads
  `AlignToInputTransform._original_input`, a **class-level mutable stash** set by a
  `set_original_input` classmethod (19–21) — a hidden global that lines 24–28 must guard
  against at runtime with a `RuntimeError`.
- Proposed fix: One base class holding the shared `apply` with an abstract
  `_target_recon()` hook; the two subclasses supply the path load and the stash read.
- Effort: low
- Risk: low — both are exercised via `tests/xform/test_align.py`.

**The `_private` module-name convention is forked three ways and carries no information**
- Location: all-private packages — `feature_match/` (10 modules, 0 public), `xform/`
  (24, 0), `visualization/` (7, 0), plus the top level (27 `_*` modules + `cli.py`).
  All-public packages — `align/` (4), `analyze/` (5), `camera/` (3), `camrig/` (4),
  `colmap/` (5), `merge/` (3), `rig/` (6), `sift/` (4). Mixed — `motion/` (5 public +
  `_recon_console.py`)
- Problem: 41 modules follow one rule, 34 the opposite, and one package does both.
  Crucially **the distinction is inert**: no subpackage `__init__.py` re-exports
  anything — each is a copyright header plus a one-line docstring — and consumers reach
  straight through in both styles (`_commands/motion.py:95` does `from
  ..motion.image_sequence import …`; `_commands/match.py` does `from
  ..feature_match._run import …`). So the leading underscore predicts nothing about
  whether a module is reachable from outside its package, and a reader cannot use it to
  navigate. `motion/_recon_console.py` is the one module whose name disagrees with its
  own package.
- Proposed fix: Pick one rule and write it into `AGENTS.md`. Cheapest coherent option:
  keep the underscore only where an `__init__.py` actually defines the boundary, and
  either populate those with `__all__` or rename `motion/_recon_console.py` so at least
  no package is internally inconsistent.
- Effort: medium (a rename touches importers) / low if scoped to `motion/` plus the note
- Risk: low — caught immediately by import errors.

**Five modules still use `typing.Optional` where the other 147 use `X | None`**
- Location: `feature_match/_cluster_matching.py:19`, `feature_match/_flow_matching.py:24`,
  `_sfmr_naming.py:10`, `_to_nerfstudio.py:14`, `_workspace.py:6`
- Problem: **8 `Optional[...]` uses across 5 of 152 modules**; every other module uses
  PEP 604. `requires-python = ">=3.13"` (pyproject.toml:11), so there is no
  compatibility reason. A second small fork rides along: `_to_nerfstudio.py:14` imports
  `Callable` from `typing` while `_patch_ncc.py:16` imports it from `collections.abc`.
  Ruff's selection (`pyproject.toml:41`, `select = ["E4","E7","E9","F"]`) excludes the
  `UP` rules that would catch this.
- Proposed fix: Rewrite the 8 annotations and the one `Callable` import. Adding `UP` to
  the ruff selection would make it mechanical, though `pyproject.toml:36–40` documents a
  deliberate policy of not widening that list casually — so this may stay a one-off.
- Effort: low
- Risk: low

---

## Tests, scripts, and top-level layout

**`tests/rust_bindings/` is 57 flat modules and 17,585 lines, organized by language
boundary rather than by subject**
- Location: `tests/rust_bindings/` — 57 modules, **17,585 lines**, larger than the other
  six test subpackages combined (13,840)
- Problem: Every other test subpackage groups by *topic* — `camrig/` (5), `matching/`
  (9), `patch/` (19), `rig/` (8), `sift/` (4), `xform/` (23). `rust_bindings/` groups by
  the fact that the code under test crosses PyO3, which is an implementation detail:
  `test_focal_vote_rust_bindings.py` is a geometry test that happens to go through a
  binding, and it sits 40 files away from `tests/matching/`. **The grouping keys are
  already in the directory**: eight `test_*_registration.py` modules
  (`analysis`, `flow`, `geometry`, `matching`, `reconstruction_patches`, `sift`,
  `spatial`, `spherical`) each assert the expected class and function names on one
  `_sfmtool.<submodule>` namespace, so the binding surface's own submodules supply the
  subdirectory names with no design decision needed.
- Proposed fix: `tests/rust_bindings/{geometry,analysis,matching,reconstruction,sift,
  spatial,spherical,flow}/`, each keeping its `test_*_registration.py` as the index of
  what belongs there.
- Effort: medium — 49 file moves; no code changes (the package has no `conftest.py`).
- Risk: low

**SPDX headers: universal in Python and in Rust production code, 65/35 in Rust
`tests.rs`**
- Location: 54 files, all `tests.rs`; the fork is *inside* `sfmtool-core`
- Problem: Rust non-test files are **530/530** — perfect. Python is **123/123** in
  `src/sfmtool/` and **125/125** in `tests/` — perfect. Rust `tests.rs` files are
  **101 of 155 (65%)**, and the split is not a rule: `sfmtool-core` has **61 with** and
  **46 without**, `sfm-explorer` 36 with and 1 without. A convention that holds
  everywhere else and forks 61/46 inside one crate is drift, not policy. Also
  `scripts/bench_normal_refine.py` and `scripts/exp_plus_descent_localize_compare.py`
  are the only two of 22 Python scripts without one.
- Proposed fix: Add the two-line header to the 54 + 2 files. **End it in enforcement**:
  this repo already has the mechanism — `numeric/tests.rs::the_workspace_has_one_median`
  scans workspace sources and fails on a violation — so a sibling test that every `.rs`
  and `.py` file begins with the SPDX line is a few lines in an established pattern, and
  is the only thing that stops this recurring.
- Effort: low
- Risk: low

**The `viz_*.py` driver loop is a true three-way copy — and `_compose` is correctly
*not*, so the #444 decision was right**
> _Carried forward from #444's one deliberate non-fix, **independently re-examined and
> upheld**, plus the duplicate it left behind._
- Location: `scripts/viz_keypoint_localization.py:275–306`,
  `viz_keypoint_localization_strips.py:313–365`, `viz_view_selection_strips.py:308–359`
- Problem: **First, the `_compose` verdict, which was asked for and is confirmed:
  `_compose` is three real layouts, not three copies.** The three functions
  (`viz_keypoint_localization.py:235–272` = 38 lines,
  `viz_keypoint_localization_strips.py:257–310` = 54, `viz_view_selection_strips.py:267–305`
  = 39) differ in column structure — A renders a fixed 2-tile row at `header_w=330`; B
  renders 2 reference tiles, a `sep = 14` gutter, then a variable-length context strip
  at `header_w=120`; C renders a variable-length strip only at `header_w=360` — with
  different row heights, tile counts and per-row text blocks (4 / 3 / 2 lines at
  different y-offsets). Parameterizing them would need about as many arguments as the
  bodies have lines. Mechanically: of the 48 / 37 / 28 shared long lines between the
  three files, **only 3 fall inside `_compose` at all**. The decision not to consolidate
  was correct and should not be revisited.
  **But the duplication the scan is detecting is real and was left unaddressed.** The
  17-line block from `args.out_dir.mkdir(parents=True, exist_ok=True)` to the final
  `print(f"{args.label}: wrote {out}  {stats}", flush=True)` is **byte-identical across
  all three**, varying only in the render function called and the output filename stem.
  On top of that, 8 identical `p.add_argument(...)` lines (`sfmr`, `--out-dir`,
  `--rows`, `--prioritize-infinity`, `--sample`, `--resolution`, `--tile`, `--seed`) are
  declared three times **with drifting defaults**: `--rows` 8/9, `--sample`
  300/300/400, `--tile` 120/56.
- Proposed fix: Add `common_parser()` (the 8 shared arguments; each script adds its own)
  and `run_over_recons(args, render_fn, stem)` (the 17-line loop) to the
  `scripts/_viz_common.py` created for exactly this. ~60 lines removed; `_compose`
  stays put, three ways.
- Effort: low
- Risk: low — developer tooling, not package code, and not covered by CI.

**`AGENTS.md` says the workspace has "nine crates" 195 lines after saying it has ten**
- Location: `AGENTS.md:73` ("`crates/` — Cargo workspace, 10 crates") vs `AGENTS.md:268`
  ("inherited by all **nine** crates")
- Problem: There are **10**, all ten inherit `rust-version.workspace = true` (verified),
  and the tenth is `sfmtool-kdf-format`, added in this window by `0396ad95` (#419). The
  wrong half sits in the MSRV paragraph — the one that tells the reader to trust
  `Cargo.toml` over dependency metadata — so it is the sentence least able to afford
  being wrong. Same defect class as the module counts #444 closed today.
- Proposed fix: Drop the number: "inherited by every crate in the workspace".
- Effort: low
- Risk: low

**Three `specs/` paths cited from code and specs do not resolve**
- Location: `crates/sfm-explorer/src/progress.rs:13` → `specs/drafts/operation-progress.md`;
  `specs/drafts/sift-gpu-amendment.md:52` → `specs/core/features/gpu-sift.md`;
  `specs/core/geometry/bundle-adjustment.md:6` → `specs/core/geometry/cluster-pinhole-bootstrap.md`
- Problem: Of 122 distinct `specs/…` paths cited across code, specs and docs, **9 do not
  exist**; six of those are cited only by `reports/2026-09-05-spec-audit.md`, which is
  *proposing* them, so they are legitimate. Three are live rot: a source file pointing
  at a draft that is not in `specs/drafts/`, and two spec-to-spec links. The second one
  matters structurally — `AGENTS.md` requires an amendment draft and its standing spec
  to link **both ways**, and `sift-gpu-amendment.md` links to a standing spec that does
  not exist, so the invariant is unsatisfiable as written.
- Proposed fix: Repoint or remove the three. **End it in enforcement**: a test (or a
  `docs-build` step) that every `specs/…` path cited from `crates/`, `src/` and `specs/`
  resolves — the same shape as the SPDX and one-median scans, and it makes this the last
  time an audit has to run the check by hand.
- Effort: low
- Risk: low
- Note: which linked spec is *correct* is `audit-specs`' remit, not this one; the live
  `reports/2026-09-05-spec-audit.md` is being worked separately and is untouched here.

---

## Carried-forward items now resolved

Verified closed at `6111797c`, and recorded so a future audit does not re-open them.
**The five closed by #444 were re-verified, not taken on trust** — a resolved-finding
list nobody checks is how the last cycle's acquittals went stale. All five hold. Note
#444 was squash-merged as **`3e76cbdc`**, so the per-commit shas in the retired report
(`b0bc9b2`, `85d915e`, `746b08c`, `7ad8c91`, `c09491c`) do not exist on `main`.

- **`AGENTS.md`'s module counts, wrong by 61% and 21%** (`3e76cbdc`) — closed by
  deletion after three audits corrected them. Verified: no `~N modules` figure remains,
  and `skills/` is documented at `AGENTS.md:135–137` including the symlink into
  `.claude/skills/` (symlinks confirmed on disk). *(The same defect recurs twice in this
  report — the MCP tool count and the "nine crates" line — which is the argument for
  deleting counts rather than correcting them.)*
- **`tests/matching/test_densify.py` was misnamed** (`3e76cbdc`) — split three ways with
  **no test lost**: 32 `def test_` before, 11 + 11 + 10 = **32** after, across
  `test_densify.py` (145), `test_epipolar_geometry.py` (132), `test_sweep_matching.py`
  (431).
- **`tests/patch/test_embed_patches_compaction.py` mixed four topics** (`3e76cbdc`) —
  split, **no test lost**: 16 before, 8 + 5 + 3 = **16** after, across
  `test_embed_patches_compaction.py` (450), `tests/test_progress.py` (64),
  `tests/patch/test_embed_patches_rounds.py` (186).
- **Patch visualization helpers in `scripts/`** (`3e76cbdc`) — `scripts/_viz_common.py`
  exists (181 lines, 12 helpers) and all three viz scripts import it. *(The
  deliberately-unfixed `_compose` is upheld above; the driver loop it left behind is a
  new finding.)*
- **The `tests/conftest.py` solve-retry loop was duplicated verbatim** (`3e76cbdc`) —
  **exactly one** retry loop now: `_solve_with_retries` (113–173) holds the sole
  `for attempt in range(1, max_attempts + 1)` at 149, with `_canonicalize_best` (176);
  the two former copies are call sites at 346 and 802.
- **`resect_images.rs` reintroduced three primitives the crate already owns** — stayed
  fixed. Verified at HEAD: the file imports `camera::report::angle_between` (39),
  `geometry::rotation::orthonormalized` (43) and `numeric::median_in_place` (44), and
  defines **zero** local `median` / `angle_between` / `orthonormalized`. Its +87 lines
  (1227 → 1314) are one feature commit (`bcd80c4b`, #415, resect-in-place), not drift.
- **Numeric-helper duplication / two `mod numeric`** (`db4ec30`) — holds.
  `numeric/tests.rs::the_workspace_has_one_median` still scans the workspace and
  `kdforest/build.rs:234::median_value` is in its allowlist with a substantive reason.
  `focal_vote::log_median` now delegates to `median_in_place` and says so.
  **No new hand-rolled statistic anywhere in the two new subsystems**;
  `sfmtool-kdf-format` contains no median, percentile or mean at all.
- **`verify_matches` was a 710-line function** — holds at **108** lines over 14
  per-section helpers. *(That it remains the only one of five is the format-crate finding
  above.)*
- **`camera/distortion/kernels.rs` was 43 functions in one flat file** (`585b258`) —
  holds: seven family modules, 1,672 lines, well sized. *(The `camera/projection.rs`
  half is still open above, with the blocker now measured as a two-word fix.)*
- **`dock.rs::ui` was a 377-line `TabViewer` method** (`aa00ae0`) — holds; longest
  method in `dock.rs` is now `show_image_detail` at 180.
- **`AppState` mixed selection state with reconstruction operations** (`e5478e1`) —
  holds; `state/ops.rs` (327) is separate, and the crate went on to add `state/edits.rs`
  and `state/save.rs` in the same shape.
- **`scene_graph/mod.rs` grew 40% and was the only module in its directory**
  (`81d0dcd`) — holds; `cameras.rs`, `menus.rs`, `widgets.rs` split out.
- **The G-buffer contract was declared six times** (2026-08-30) — holds, **and the
  enforcement held too**: `frustum.rs` reads `GBUFFER_DEPTH_STATE` (105),
  `gbuffer_targets(...)` (112) and `QUAD_VERTEX_LAYOUT` (72) at HEAD.
- **The displacement-field summary was computed twice** (`872ca75`), **`draw_overlays`'
  five copy-pasted arms** (`35b70d1`), **`point_track_detail::metrics`** (`5ff41ed`) —
  all hold.
- **Seven inline `#[cfg(test)] mod tests` blocks** (`ed2a8b5`) — those seven are gone
  and `sfm-explorer` is 37/0 clean. *(Four new ones landed elsewhere; see the format-crate
  finding. The convention was closed and re-broken, which is why that finding leads with
  the history.)*

---

## Explicitly not flagged

**Read the headline before trusting this list.** Every entry carries a mechanical number
the next snapshot can re-run in one command — a longest function, a duplicate-line
count, a ratio — deliberately, because the last cycle's prose acquittals were the part
that aged worst. Fourteen of the last list's sixteen entries held exactly; the two that
did not are findings above.

**Long-but-coherent files** (longest function, brace-to-brace, non-test):

- `crates/sfm-explorer/src/mcp/tools.rs:56–904 catalog` — **849 lines, the longest body
  in the repository, and it should stay.** One `vec![]` of 40 `ToolSpec` struct
  literals: **zero branches, zero loops, one concern.** Every repeated schema fragment
  is already factored into nine helpers (905–1190), and `panel_name_schema` even derives
  its enum from `Tab::ALL.map(wire_name)` rather than restating it. Splitting it per
  family costs the property the module doc names as the reason for the shape — catalog
  and parse in one file so a schema and its parser cannot drift — and buys nothing a
  `grep -n 'name: "'` does not. The real value here is the seam that *is* flagged
  (deriving `reject_unknown` from it).
- `crates/sfmtool-core/src/progress.rs` (746) — longest function **31**
  (`split`, 404–434). The flattest file in the workspace; 40+ small methods.
  Its explorer counterpart `sfm-explorer/src/progress.rs` (494) is a clean
  producer/consumer split, not a duplicate — core owns the sink interface, the explorer
  owns the collector, and both module docs say so.
- `crates/sfm-explorer/src/action_log/mod.rs` (1242) — longest **49** (`write`, 540),
  **62 functions**. The healthiest large file in the crate.
- `crates/sfm-explorer/src/layout.rs` (1330) — longest **66** (`from_value`, 607), 50
  functions, 8 banners; the module doc declares exactly four concerns and justifies each.
- `crates/sfm-explorer/src/background/mod.rs` (674) — longest **85** (`finish`, 482).
- `crates/sfm-explorer/src/document.rs` (593) — longest **37**, 34 functions.
- `crates/sfm-explorer/src/mcp/mod.rs:550–783 apply_with_window` — **234**, a 40-arm
  dispatch at 5.85 lines per arm with no logic in any arm. (The *file* is flagged; this
  function is not.)
- `crates/sfmtool-core/src/geometry/translation_averaging.rs` (988) — longest **127**,
  then 102 / 93 / 92 / 58.
- `crates/sfmtool-core/src/reconstruction/point_estimation.rs` (965) — longest **166**
  (`decide`, 432).
- `crates/sfmtool-core/src/reconstruction/edit.rs` (760) — longest **284**
  (`subset_by_image_indices`, 242): filters ~15 parallel columns in lockstep, and
  splitting it is what would break the lockstep its own comments guard.
- `crates/sfmtool-core/src/reconstruction/bundle_adjust.rs` — longest **308**
  (`bundle_adjust`, 243). One solver loop with a spec.
- `crates/matches-format/src/select.rs` (517) — longest **298** (`select_clusters`,
  160). One algorithm with a spec; not a section entry point.
- `crates/sfmtool-core/src/reconstruction/data/conversion.rs` (595) —
  `from_sfmr_data` **214**, `to_sfmr_data` **185**. Column-for-column transcription;
  re-measure if either passes 250.
- `crates/sfmtool-kdf-format/src/read.rs` (1756) — longest **209** (`open`, 200), 12% of
  the file, ~60 items. The decomposed counter-example to its own crate's writer.
- `crates/sfmtool-kdf-format/src/cache.rs` (694) — longest **140** (`get_or_load`, 281):
  one concurrency-critical admission path where the interleaving is the point.
- `crates/sfmtool-core/src/reconstruction/edited.rs` — longest **106**
  (`build_point_set`, 964). The finding there is concern count, not length.
- `crates/sfmtool-py/src/reconstruction/edited.rs` (856) — **never audited before,
  audited now.** Longest **89** (`record_from_dict`, 93), then 69/56/47/45/44/44. One
  small method per edit op. Watch only the hand-written dict codec pair
  (`record_from_dict` / `record_to_dict`, 185–253), whose two halves enumerate the same
  12 keys independently.
- `crates/sfmtool-py/src/reconstruction/sfmr_reconstruction.rs` (1051 → 1188, +13%) —
  longest **69**; 40+ small `#[pymethods]`. Still the flattest large file in the crate.
- `crates/sfmtool-py/src/spatial/kdf.rs` (801) — longest **102**, then 51/43/35/34.
- `crates/sfmtool-core/src/patch/cloud.rs` (1023 → 1156, +13%) — longest **199**
  (`build_patch_cloud`, 706). Re-check if it passes 1300.
- `crates/sfmtool-core/src/spherical/{tile_rig,per_tile_source_stack,photometric_ransac}.rs`
  — longest **167 / 170 / 164**; all three flat since the last snapshot.
- `crates/sfmtool-core/src/geometry/rotation_init.rs` (866, **shrank** from 891) —
  longest **247**.
- `crates/sfmtool-core/src/analysis/infinity/discover.rs` (631) — longest **263**
  (`find_points_at_infinity`, 365). **Borderline**: two functions (263 + 205) are 74% of
  the file. Not flagged because the file is small and those are its only two public
  entry points, but **this is the one to re-measure first next round.**
- `src/sfmtool/_embed_patches.py` (851) — longest function **478**
  (`embed_patches`, 374), **the largest function in the Python subtree**, ahead of
  `parse_transform_args` (440) and `draw_epipolar_visualization` (509 — which *is*
  flagged). Not flagged because it is the documented orchestration entry for the patch
  pipeline whose write tail has **already** been extracted to
  `_patch_compaction.compact_to_embedded_patches` (named at `_embed_patches.py:11`),
  and ~155 of those lines are the docstring over a linear staged pipeline whose stages
  are extracted. **This is the closest call in the list**: re-flag if it passes ~550
  lines or gains a third mode branch.
- `src/sfmtool/_commands/embed_patches.py` (440) — **426 of 440 lines are one Click
  declaration** with 23 parameters and **zero** helper code. The largest command module
  after solve.py and a pure wrapper, which is the convention. The 23-option surface is a
  product question, not drift.
- `src/sfmtool/_commands/xform.py` (418) — the `xform` command is **199** lines and the
  module has **zero** helper functions. Length is the ~24-transform option surface, not
  logic. Re-run: flag if the helper count becomes non-zero.
- **The thin-Click-wrapper convention holds two-thirds of the tree by count**: of the 29
  `_commands/` modules, **19 have exactly zero** module-level helper functions, and the
  number with more than 100 helper lines is **2** (`estimate_intrinsics.py` 285,
  `solve.py` 180). Re-run that pair of counts; flag if the second reaches 4.
- `src/sfmtool/_commands/estimate_intrinsics.py` (484) — **285 helper lines, more than
  `solve.py`'s 180**, longest `_report_lines` **93**. Not flagged because its docstring
  (11–19) states that the bulk is report and `.camrig` I/O with the algorithm in the
  kernel, and the helpers are formatters. That is the convention honoured, which is
  exactly why the `solve.py` finding above had to be reframed. Re-flag if a helper
  starts calling the vote kernel more than once, i.e. becomes control flow.
- `src/sfmtool/visualization/_flow_display.py` (710) — larger than `_epipolar_display.py`
  but has **already** performed the mode decomposition that its sibling has not, so it
  is the model rather than a finding.
- `src/sfmtool/{colmap/io.py, analyze/summary.py, motion/recon_discontinuity.py,
  _densify.py, _undistort_images.py}` — **0% growth**, all five.
- `tests/conftest.py` (571 → **841**, +47%) — longest function **132**
  (`build_cluster_reconstruction`, 244), then 76 and 61; **exactly one** retry loop
  after `3e76cbdc`. The growth is shared-fixture surface for 154 test modules and two
  dataset families, and the obvious fix does not work: the three session-scoped dataset
  fixtures are consumed by `tests/camrig/`, `tests/patch/`, `tests/rig/` **and**
  `tests/xform/`, so moving them into any one subpackage's `conftest.py` makes them
  unreachable from the others, and a `tests/_fixtures.py` sibling would need
  re-registering in every subpackage conftest. It also just *shed* its only duplicated
  block. Re-flag above ~1,000 lines, or when a fifth dataset fixture lands.
- `tests/rust_bindings/test_bundle_adjust_rust_bindings.py` (1269) — the largest test
  module; append-mostly and 1:1 with one binding.

**Duplication scans that came back clean:**

- **Render pipelines are not a duplication finding, despite the numbers.**
  `scene_renderer/pipelines/*.rs` share **42–52 long lines pairwise** (`frustum.rs` ‖
  `track_ray.rs` 47, ‖ `points.rs` 45, ‖ `image_quad.rs` 43), which looks alarming. I
  read `frustum.rs` (132 lines) against two siblings line by line: the shared lines are
  wgpu descriptor keywords — struct-literal field names, `..Default::default()`, closing
  braces — and the one genuine shared invariant, the G-buffer triple, **is already one
  constant** (`GBUFFER_DEPTH_STATE`, `gbuffer_targets(...)`, `QUAD_VERTEX_LAYOUT`, in
  use at frustum.rs:105/112/72 from last cycle's fix). A builder would trade a
  declarative descriptor for an argument list of the same length. *What the next
  snapshot should re-check is not the count but whether a new non-keyword invariant has
  entered the shared set.*
- **`action_log/panel.rs` ‖ `background/panel.rs`: 6 shared distinct code lines >25
  chars.** Deliberate reuse, not duplication — `background/panel.rs:31` imports
  `action_log::{detail_row, detail_text, Breakdown}` and its module doc says why ("a
  second spelling of a row would be a second thing that can be wrong"); it also routes
  through the shared `crate::elide::middle`. Only the two egui painters differ, and they
  differ because the column layouts do. One five-line rationale comment about
  `Label::truncate` is copied verbatim at `action_log/panel.rs:416–420` and
  `background/panel.rs:346–351` — too small to be worth a helper.
- **`document.rs` ‖ `state/edits.rs`: 1 shared distinct code line >25 chars.** No
  undo/redo bookkeeping duplication between them; `document.rs` is the model and
  `edits.rs` is the `AppState` layer over it, and `background/mod.rs:523` reuses
  `state::edits::version_before` rather than re-deriving it. The duplication in
  `edits.rs` is *within* the file.
- **`mcp/read.rs` ‖ `mcp/render.rs`: 3 shared distinct code lines.** `read.rs` calls
  five `render::` builders.
- **Schema-vs-parser key divergence across the MCP surface: 0**, all 40 tools and ~88
  keys, both directions, verified mechanically. (The finding above is that nothing
  *keeps* it at 0.)
- **Test-side duplication is healthy.** A ≥25-char duplicate-line scan across all 154
  Python test modules tops out at **13** shared long lines
  (`test_camrig.py` ‖ `test_camrig_cp.py`), against 61 for the worst pair in `src/`.
- **The `entries::*` overlap between each format crate's read/write/verify is not
  duplication.** 47/42/42 call sites in sfmr, 36/35/41 in matches — but `entries.rs` is
  the right abstraction and its module doc states the case ("a name written three times
  is a name that can disagree three ways"). The residual — which entries belong to which
  section, in what order — is exactly what the section-split finding fixes.
- **`s_conjugate_relative_pose` (matches-format) / `s_conjugate_sensor_pose`
  (camrig-format) are a *justified* copy, and both say so.** Four-line bodies duplicating
  `sfmtool_core::geometry::convention::relative_pose_conjugate_s`, each with a doc
  comment giving the reason (both crates sit below `sfmtool-core` and the operation is an
  exact component permutation, so the copy is loss-free). Noted rather than flagged —
  but if the `WorkspaceContents` finding moves shared types into `sfmtool-archive-io`,
  this is the obvious second passenger, since archive-io sits below both.

**Convention tallies measured, and deliberately *not* flagged:**

- **Error-message capitalization is not a fork.** The raw numbers look like one — 350
  lowercase vs 127 uppercase workspace-wide, with `sfm-explorer` split 110/122 inside one
  crate. Reading the strings settles it: every library crate is uniformly lowercase
  (matches-format 41/0, sfmr-format 49/0, sfmtool-core 36/0, sift-format 21/0,
  kdf-format 21/1 — the 1 is the acronym "SIFT"), and the explorer's uppercase strings
  are **UI labels and Action Log entries** ("Adjusted {name}", "Camera Intrinsics
  ({count})") while its lowercase ones are identifiers, table cells and genuine errors
  ("could not read {}: {e}"). Two spellings carrying two meanings. The only real
  outliers are `camrig-format/src/verify.rs:74,78` ("Structural validation failed:",
  "Could not parse archive contents:") against 168:2 lowercase across the library crates
  — a two-line fix, noted here rather than promoted to a finding.
- **`indices` vs `indexes` is not one fork but three separate stories.** Workspace-wide
  340/209 looks like drift; read per-subtree it is not. `sfm-explorer` is 25 `indexes` /
  7 `indices`, and **all seven** `indices` quote an inherited `sfmtool-core` API name
  (`subset_by_image_indices`, `track_image_indices`) or an `egui_dock` term; on the wire
  there is exactly one plural index key (`camera_image_indices`) and no competing
  spelling, so **nothing agent-visible forks**. `sfmtool-py`'s Python surface is
  unambiguously `indexes` (~110 to 4), and those 4 *are* flagged above. The genuine
  inconsistency is inside `sfmtool-core`, which exports both `subset_by_image_indices`
  and `feature_indexes()` — worth settling when one of those functions is next touched,
  not worth a sweep.
- **`#[allow(clippy::too_many_arguments)]`: 193 workspace-wide** (sfmtool-core 125,
  sfmtool-py 51, sfm-explorer 16, sfmr-format 1). Recorded as a **baseline number, not a
  finding**, because sampling contradicts the headline: the worst file,
  `patch/keypoint_localize/kernels.rs` (7), is SIMD dispatch triples
  (`compute_channel_grids` / `_scalar` / `_avx2` at 13 args; `score_cell_one_channel` ×3
  at 11) whose signatures are *required* to match. The one cluster that **is** drift —
  `sfmtool-py/src/patches/`, 10 allows on five bindings taking 27/26/19/18/16 flattened
  kwargs that nothing constrains — is flagged above as the same edit as the prologue.
- **Doc economy: nothing over the 0.5 threshold.** Per-crate doc density
  (non-test): sfmtool-archive-io 27.0%, camrig-format 24.1%, sfmtool-core 22.4%,
  sfmtool-py 21.6%, sfm-explorer 18.7%, sfmr-format 17.6%, matches-format 14.6%,
  sift-format 13.8%, sfmr-colmap 8.7%, **sfmtool-kdf-format 7.4%** (flagged above, and
  the finding is a *missing*-doc one). Per-item on the `sfmtool-py` wrapper layer the
  maximum ratio is **0.45** (`geometry/estimate_intrinsics.rs`), with most files
  0.29–0.38 — none over 0.5. The seven longest contiguous doc blocks in the workspace
  (93–125 lines) are all `sfmtool-py` bindings; sampling
  `patches/localize_keypoints.rs:24` (122 lines) shows disciplined `Args:` / `Returns:`
  reference prose a REPL user genuinely needs, carrying an `#[allow]` with an
  explanation for why the docstring form conflicts with rustdoc. **Explicitly the thing
  the skill says not to flag.** `mcp/edit.rs` at 30.3% is the thinnest wrapper and the
  most documented, and its docs are rationale, not restatement.
- **`*Change` in `sfm-explorer` is not an option-bag fork.** The crate has zero
  `*Params` and zero `*Options`; its idiom is `*Change` (`DisplayChange`,
  `ImageDetailDisplayChange`, `IntrinsicsChange`, `WindowChange` — 4/4), which is a
  partial-update delta, a different thing from a parameter bag, applied without
  exception.
- **UI ellipsis spelling: 9/9 ASCII `...`** in menu labels; `…` appears only in prose.
- **Test-only helper naming**: 3 `_for_test` vs 1 `_for_tests` vs ~6 unsuffixed. No
  majority worth enforcing; measured so it is not re-derived.

**Structural checks that came back clean:**

- **No dead Python modules.** The import graph was rebuilt twice independently (an AST
  walk resolving relative imports and function-body deferred imports, plus a
  regex cross-check): **every one of the 152 non-`__init__` modules under
  `src/sfmtool/` has at least one production importer.** No `_old` / `_v2` / `.bak` /
  `.orig` anywhere in `src/`, `tests/` or `scripts/`.
- **Only 5 `#[allow(dead_code)]` in the whole Rust workspace**, all on small
  enum variants and test helpers.
- **`specs/` index integrity is perfect**: 164 specs; of the 142 in the five indexed
  areas (`cli/` 34, `core/` 67, `formats/` 8, `gui/` 30, `workspace/` 3), **every one is
  listed in its area's `README.md`** — 0 unindexed. (Three broken outbound links are
  flagged above.)
- **CLI help text carries no implementation leaks.** All 30 command modules plus
  `cli.py` grepped for `pycolmap|glomap|_sfmtool|PyO3|numpy|cv2|sqlite|\.so` in `help=`
  strings: **1 hit** (`_commands/from_colmap_bin.py:36`), naming `colmap`/`glomap` as
  legitimate user-facing tool identifiers.
- **`scripts/` unreferenced files** — not re-reported, per the 2026-08-20 decision.
  **`skills/` → `.claude/skills/` symlinks** — not re-reported; documented at
  `AGENTS.md:135–137` and confirmed on disk.
- **`src/sfmtool/patch*` is a near-miss, not a finding.** `_cluster_patches.py` (240) +
  `_embed_patches.py` (851) + `_patch_compaction.py` (278) = 1,369 lines flat, while
  `tests/patch/` (19 modules) and `specs/core/patch/` (17 specs) both group this work.
  Unlike the strip cluster it is **not closed** — `_patch_compaction.py` has an inbound
  edge from `xform/_localize_keypoints.py:158` — so a subpackage would not have a clean
  boundary. **Recheck if that one edge disappears.**
- **`kdf-format` having both `src/tests.rs` (408) and `src/validation_tests.rs` (786)**
  — a second naming fork, but defensible: validation tests pin the on-disk layout, unit
  tests do not. Worth a one-line module doc naming the boundary, not a finding.
- **`kdforest/build.rs`** is at a non-root path, so Cargo does not treat it as a build
  script. Safe; recorded so it is not re-flagged.

---

## Top 3

1. **`app.rs::run_egui_pass` → `app/{menu,modals,save}.rs`.**
   The best effort-to-value ratio in the report, and the one the last snapshot already
   did the analysis for. `app.rs` is the largest single-file growth in the repository
   (879 → **1631**, +85%) and `run_egui_pass` went **180 → 245 → 600** across three
   snapshots while holding five unrelated concerns its own banner comments already
   separate. It is a pure extraction along drawn lines, with no signature changes and
   one caller — and the crate has demonstrated exactly this refactor twice in the last
   month (`dock.rs::ui` → `TabContext` methods, `state.rs` → `state/ops.rs`), both
   landing cleanly. The last report deferred it explicitly ("do `dock.rs::ui` first;
   re-measure this after"), that work landed, and this is the re-measure. There is even
   a testability payoff with precedent: `layout.rs:1245 panels_menu` was extracted from
   `app.rs` specifically so a headless frame could draw a menu and read it back, so
   `app/menu.rs` inherits a proven pattern and should absorb `panels_menu` on the way.

2. **`SectionDigests` in `sfmtool-archive-io`, then `verify_sfmr`.**
   Two items, paired because the first de-risks the second and is worth doing alone.
   The content-hash rule that defines every `.sfmr`, `.matches`, `.sift`, `.camrig` and
   `.kdf` file's identity is hand-written **nine times in two incompatible spellings**,
   in a workspace whose `sfmtool-archive-io` crate exists for this, claims the rule in
   its own module doc, and ships both ends of it while omitting the middle. The fix is
   one 20-line type and nine call-site edits of 3–6 lines, it changes no bytes, and
   every format crate has hash-stability fixtures that would fail loudly. Then
   `verify_sfmr`: **634 lines of a 651-line file** — 97% — carrying 14 `// === Section
   === ` banners that name the helpers nobody extracted, up 23% since the last snapshot.
   The remedy is not a design question, because **this repo already built it**:
   `matches-format/verify.rs` is a 108-line orchestrator over 14 per-section helpers
   (#379). Doing `verify_sfmr` after `SectionDigests` turns the one real hazard — digest
   order is the content hash, and statement order currently encodes it — into a value
   the orchestrator holds explicitly.

3. **The `sfmtool-py` patch prologue, and the four inline test modules.**
   Two cheap ones that share a lesson: both are contracts that were solved last cycle
   and then spread anyway. The prologue was flagged at three bindings; `args.rs` was
   created for it, the parser half landed as `f51a109`, and the prologue half is now at
   **five** bindings with **29 consecutive byte-identical lines** between two of them and
   nine file pairs over 28 shared lines. `views.rs` already owns `resolve_scene` and
   `PosedViews`, so the fix is one function there and five call sites, removing ~135
   lines — and it is the same edit as settling the four spellings of that argument
   (`view_sets` / `view_indices` / `member_views` / `candidate_views`) and retiring the
   `too_many_arguments` allows on the one cluster where they are genuinely drift.
   Meanwhile the sibling-`tests.rs` convention was closed workspace-wide last cycle and
   **four new inline blocks** (791 lines) have landed in the newest two subsystems — one
   of them named `mod profiling`, exactly the spelling the last report told the next scan
   to look for. It has a mechanical cost, not just a tidiness one:
   `numeric/tests.rs::the_workspace_has_one_median` exempts test code by skipping files
   *named* `tests.rs`, so these four sit inside a workspace-wide production-code gate,
   and any future source-scanning gate inherits the hole.

**Runner-up, because it is the cheapest correctness-shaped fix here:** the two MCP
assertions. `ToolKind::Read` and `Command::kind() -> Kind::Query` are two hand-maintained
classifications of the same 40 tools that agree exactly today and are tied by nothing;
one drives the `readOnlyHint` an agent trusts, the other decides how a refusal is logged.
Separately, nothing ties `catalog()`'s 40 names to `Command::tool_name()`. Both are a few
lines **inside a test that already walks the catalog** — and that test,
`the_wire_vocabulary_holds_across_the_catalog`, is the best piece of enforcement in this
codebase and the model the rest of this report keeps pointing at. Extending it is the
cheapest way to keep it that way.

---

## Appendix — design topics carried forward

Not hygiene findings — unspecced feature proposals inherited from
`reports/2026-07-07-next-steps.md` via the 2026-08-08 and 2026-08-29 snapshots, kept
because `AGENTS.md` says to fold unfinished items into the next regenerated report
rather than let them evaporate. **Each premise re-verified at `6111797c`:**

- **A — Camera bookmarks (save/restore named viewpoints) in SfM Explorer.**
  `specs/gui/viewport-navigation.md` still carries the unticked
  `- [ ] Save/restore camera positions`, and a case-insensitive grep for `bookmark`
  across `crates/sfm-explorer/src/` returns **nothing**. Still open. The warning attached
  to this item last time is now much stronger: the explorer gained 41k lines in the last
  fortnight alone, including `layout.rs` (which already persists window and panel state
  to a versioned layout file) and `camera_lock.rs`. **Re-read `layout.rs`, `state.rs` and
  `viewer_3d/` before acting on any sketch of this** — the layout file is very likely
  where bookmarks now belong, which was not true when the topic was written.
- **B — `sfm xform --crop` (3D bounding-volume crop).** No crop transform in
  `src/sfmtool/xform/`, zero `crop` matches in `_commands/xform.py`, and none in
  `specs/cli/reconstruction/xform/xform-command.md`. Still open, entirely unchanged.
- **C — Pose-aware per-tile source stacks (parallax-correct panoramas).**
  `PerSphericalTileSourceStack` still exposes only `build_rotation_only`
  (`spherical/per_tile_source_stack.rs:271`). Its dependency does exist —
  `WarpMap::build_with_pose_impl` is at `camera/warp_map.rs:276`, reached from three
  call sites — so only the per-tile consumer is missing. Still open.
