# Spec audit — 2026-08-30

**Sample:** 13 of 121 specs read against their code. Seed `20260830`; pool was
the 41 specs never covered by a prior spec-audit report (`reports/2026-07-07-spec-audit.md`
covered 82). Ten drawn at random from that pool; three included regardless of the
draw — `member-coherence-validation.md` (top duplicate-prose hit of the corpus,
5 shared lines), `rotation-init.md` (3 shared lines), and `keypoint-reach.md`
(specs code that landed in 600b52e, after the last audit). Corpus-wide mechanical
checks below cover all 121.

Drawn at random: `observation-coverage`, `sfmtool-fisheye-kernels`,
`cluster-covisibility`, `covisibility-selection`, `absolute-pose`,
`epipolar-estimation`, `focal-vote`, `pose-verification`, `translation-averaging`,
`cluster-patch-refinement`.

Nothing here is a claim about the 108 specs that were not read. The mechanical
section acquits a spec on the checks it runs; its silence acquits nothing else.

---

## Mechanical findings (all 121 specs)

### 1. Documented defaults vs actual defaults — clean, with one blind spot

106 documented values were parsed out of spec parameter tables and diffed against
Click `default=`, `#[pyo3(signature = ...)]` and `impl Default` across `src/` and
`crates/`. Six apparent mismatches, **all verified false positives** from table
parsing (measurement tables, "Alternatives considered" tables, per-dataset result
tables). Spot-checked and confirmed correct in code: `edl_strength = 0.7` and
`edl_radius = 2.4` (`sfm-explorer/src/scene_renderer/uniforms.rs:252`,
`state.rs:374`); `BRACKET_MAX_STEPS = 24` and `BRACKET_LOG_TOL = 1e-3`
(`camera/epipolar.rs:119,121`).

**The check has a blind spot worth recording, because the deep pass found drifts
it missed.** The matcher keys on parameter *name*, so a name used by several
commands is acquitted as soon as any one of them matches. `resolution` is
documented as `24` in three `xform` specs and `25` in `cluster-patches.md` — and
`25` is what the code ships — so the check passed, while
`specs/cli/image-feature/cluster-patches-command.md:31` documents **15**. A
future run should key on (spec → owning command → parameter), not on the bare
name.

Real drifts, found by reading:

| where | documented | actual | code |
|---|---|---|---|
| `cli/image-feature/cluster-patches-command.md:31` `--resolution` | 15 | **25** | `_commands/cluster_patches.py:49` |
| `core/patch/cluster-patch-refinement.md:537` `--patch-size` | 8.0 | **12.0** | `_commands/cluster_patches.py:36` |
| `core/patch/cluster-patch-refinement.md:55` `MATCHES_FORMAT_VERSION` | 3, readers accept 1–3 | **5**, gates at ≥4 and ≥5 | `matches-format/src/types.rs:120`, `read.rs:98,111` |
| `core/geometry/rotation-init.md:116` `rotation_init(...)` | `seed`/`min_images`/`max_images` positional | **keyword-only** (`*` in signature) | `sfmtool-py/src/geometry/rotation_init.rs:51` |
| `core/features/cluster-covisibility.md:183` `from_arrays(...)` | 4 params | **6** (`member_accepted`, `positions_xy`, `seed`) | `sfmtool-py/src/matching/covisibility.rs:128` |
| `core/features/covisibility-selection.md:30` `from_arrays(...)` | omits `member_accepted` | present, and third positional | same |

The last three are worse than a stale table: a reader who copies the documented
call gets a `TypeError`, or worse, passes `positions_xy` into the
`member_accepted` slot positionally.

> _Status (2026-09-03): Done — all six rows corrected in commit 7ddbe97; see
> the Top priority 2 annotation, which also records where this finding's own
> wording was off (the `member_accepted` slot number, and the count of the
> `MATCHES_FORMAT_VERSION` gates). The methodology note for the next run — key
> the check on spec → owning command → parameter, not the bare name — stands._
>
> _A seventh drift, found later and outside the check's reach:
> `core/patch/patch-cloud.md:221-222` put `PatchExtent::default()` and the
> binding's `extent_value` at `5.0`; `patch/cloud.rs:933` and
> `sfmtool-py/src/patches/cloud.rs:65,127` are both **2.5**. Corrected in commit
> 7d4d3a4, with the missing distinction the 5.0 came from: `factor` is a
> half-extent multiplier, so the full patch edge is 5× the projected feature size
> and that is what the CLI's `--extent-value` states before halving it. **A
> second methodology note**, then: this value is stated in a `//` comment inside
> a ` ```rust ` fence rather than in a parameter table, so the scan never saw it.
> A future run should also read declared defaults out of fenced code, not only
> out of tables._

### 2. Duplicate prose between a spec and its implementing code

Long prose lines (≥55 chars, normalized, code fences stripped) bucketed across
every spec and every source file:

| shared lines | spec | code |
|---|---|---|
| 5 | `core/patch/member-coherence-validation.md` | `patch/member_coherence/decide.rs` |
| 3 | `core/analysis/observation-coverage.md` | `analysis/observation_coverage.rs` |
| 3 | `core/geometry/pose-verification.md` | `geometry/pose_verification.rs` |
| 3 | `core/geometry/rotation-init.md` | `geometry/rotation_init.rs` |
| 3 | `core/geometry/translation-averaging.md` | `geometry/translation_averaging.rs` |
| 2 | `core/analysis/adjacency-surfel-normals.md` | `analysis/adjacency_surfel_normals.rs` |

**Exact-match counts understate this badly, and that is the finding.** Every one
of these that was read deeply turned out to carry a 35–125-line module or
function doc re-deriving the spec's whole argument in *near*-verbatim paraphrase,
of which the exact matches are only the residue:

| doc block | lines | re-derives |
|---|---|---|
| `patch/member_coherence/decide.rs:182-304` | 123 | spec §§201-352, same calibration numbers, same four-step list |
| `camera/distortion/bspline.rs` (module) | 95 of 247 | the gauge, the `N ≥ 2` argument, two-stage monotonicity |
| `geometry/pose_verification.rs:4-48` | 45 | Screens A and B, both load-bearing properties, Repair |
| `geometry/focal_vote.rs:4-43` | 40 | the two-family degeneracy argument, log-median, bimodality |
| `geometry/rotation_init.rs:4-44` | 40 | Purpose plus all four Mechanism stages |
| `geometry/translation_averaging.rs:4-38` | 35 | the objective, the projector, the null-space argument |
| `geometry/absolute_pose.rs:4-32` | 29 | Lambda Twist rationale, determinism, the ArrayVec note verbatim |
| `analysis/observation_coverage.rs:6-18` | 13 | spec lines 12-22 word for word |

In nearly every case the module doc already carries a `See specs/...` pointer
*below* the re-derivation. **The fix is uniform and mechanical: the doc keeps what
the code cannot say — frame conventions, panic contracts, the `S = diag(1,−1,−1)`
conjugation, the no-FMA note — and shrinks to the pointer for the rationale.**
Two exceptions found: `decide.rs:4-10` is the one doc in its module that does
*not* link the spec, which is likely why it grew into a standalone essay; and
`keypoint_reach.rs:4-27` is already correctly sized.

The third copy is usually a PyO3 binding docstring (`focal_vote.rs:85-160`, 72
lines; `pose_verification.rs:100-180`, 58 lines). Those mostly earn their length —
a REPL user cannot click into `specs/` — but they should be reference, not
rationale.

> _Status (2026-09-03): Done for the eight-row table — 420 lines of
> re-derivation come down to 177, commit c11c885. Per-block before → after:
> `decide.rs`'s `decide_member_coherence` 123 → 46, `bspline.rs`'s module header
> 30 → 24 (plus `bspline_is_monotone`'s two-stage derivation),
> `pose_verification.rs` 44 → 21, `focal_vote.rs` 40 → 23, `rotation_init.rs`
> 41 → 20, `translation_averaging.rs` 35 → 18, `absolute_pose.rs` 29 → 17,
> `observation_coverage.rs` 17 → 8. Each keeps what the code alone cannot say —
> frame conventions, the `S = diag(1,−1,−1)` conjugation, the log-focal median
> convention, the center-anchored gauge, identity and panic contracts, the
> mechanism in the order the code runs it — and points at the spec for the
> rationale. `decide.rs:4-10`, the exception this section names, now links its
> spec. The finding's premise that every block already carried a pointer is
> wrong in one case: `bspline.rs` carried none, and nothing else in
> `camera/distortion/` does either._
>
> _The PyO3 docstrings are deliberately left: nothing in them was found false,
> and the "reference, not rationale" trim is a separate pass. The two `covisibility`
> third copies this section's per-spec siblings name (`covisibility.rs:10-13`,
> `selection.rs:64-69`) are also untouched — they were not in the eight-row
> table._

### 3. Spec shape

Over the 59 `specs/core/` specs:

- **34 carry no ` ```rust ` interface block.** Three convey an interface in
  Python or shell instead. The deep pass confirmed the hit is real, not a scan
  artifact, in every case it checked. `focal-vote.md` is the extreme: 487 lines
  with **zero** lines inside any code fence, never naming `focal_vote`,
  `FocalVoteOptions` or `focal_vote_with_options`. `translation-averaging.md`,
  `pose-verification.md`, `covisibility-selection.md`,
  `sfmtool-fisheye-kernels.md` and `keypoint-reach.md` likewise name no Rust
  signature — several document only the Python binding, in a spec for a Rust
  core module.
- **4 bury the block past 60%** — `member-coherence-validation.md` (71%),
  `observation-adjacency-graph.md` (70%), `adjacency-surfel-normals.md` (69%),
  `sift.md` (66%). For member-coherence the teaching before it is genuinely
  load-bearing; only the 49-line calibration narrative at 431-479 is derivation
  sitting where orientation belongs.
- **Not one sampled spec carried a usage example.** Thirteen of thirteen. This is
  the most consistent shortfall in the corpus — more consistent than the missing
  signatures, since several specs do convey the interface in prose but none show
  a call.

### 4. Openings

**74 of 121 specs — 61% — open with a `**Status:** ...` line rather than a
sentence saying what the thing is for**, and in all 74 that line is the first
prose in the document. Not one puts purpose first. `specs/TEMPLATE.md` never
mentions a Status section, and its one absolute rule is the opposite: "The
plain-English opening ... No exceptions to this one."

The line is rarely a bare stamp. The typical opening is status + date + two or
three repo paths + the symbols inside them, so the first nouns a cold reader
meets are file paths:

- `cluster-covisibility.md:3` — "Promotes the image-grouping-by-shared-clusters
  machinery from the pinhole bootstrap experiments ... into `sfmtool-core`":
  a status, then a *change*, then three paths, before any statement of purpose.
- `covisibility-selection.md:3` — "**Status:** Implemented — extends
  `ClusterCovisibility`": the opening rests on a bare symbol defined in a
  different spec, and the Purpose paragraph then describes the world *before* the
  change ("today exist only as per-caller array code").
- `sfmtool-fisheye-kernels.md:3` — "**Status:** Implemented." then six lines of
  file inventory.
- `focal-vote.md`, `pose-verification.md`, `absolute-pose.md`,
  `epipolar-estimation.md`, `rotation-init.md`, `cluster-patch-refinement.md` —
  the same shape; in each the real purpose sentence is stranded 8–25 lines down
  and is usually fine once reached.

Two specs passed cleanly: `observation-coverage.md` ("Several per-image decisions
need to know which parts of an image are already accounted for...") and
`keypoint-reach.md` ("One question, asked per image of a track set: which other
keypoints lie inside this keypoint's own disk?").

This is a house convention that has quietly overridden the template's one
non-negotiable, and it is the cheapest fix in the report — the purpose sentence
usually already exists a few lines below. Replacement first sentences drafted by
the deep pass are in the per-spec sections.

> _Status (2026-09-03): Done — every lifecycle stamp outside `specs/drafts/` is
> gone, from 75 specs. The convention chosen is not the one Top priority 4
> proposed: rather than sanctioning Status as a metadata line below the opening,
> **a standing spec carries no Status line at all** — location encodes lifecycle,
> and `specs/TEMPLATE.md` now says so. See the Top priority 4 annotation for the
> rule and the wording._
>
> _Three corrections to this section's own numbers, all in the direction of "more
> than it says". First, the count. §4 says 74 of 121; the pattern it grepped —
> `**Status:**` at column 0 — matches **52** files today, and 121 specs is now
> 138. The gap is spelling: the same stamp is also written `**Status**:` (colon
> outside the bold), `**Status: implemented** in …` (label inside the bold),
> `*Status: Implemented*`, `_Status: **implemented** — …_`, and as a `> Status:`
> blockquote. Counting all of them: **75 specs**, and three of those carry a
> second or third stamp mid-file (`bundle-adjustment.md` ×2, `focal-vote.md`,
> `patch-keypoint-localization.md`, `patch-view-selection.md`,
> `sift-to-patch-reconstruction.md` ×3). A future grep should be
> `-iE '^[_*>| ]*[_*]*status[:*]'`, not `^\*\*Status:\*\*`._
>
> _Second, "in all 74 that line is the first prose in the document" is right, and
> understates the damage in a subset: in eleven specs the stamp was the **only**
> statement of what the thing is, so deleting it left nothing and a purpose
> paragraph had to be written from scratch. They are listed in the Top priority 4
> annotation._
>
> _Third, the section reads the Status line as pure overhead. About two-thirds of
> them carried something worth keeping — the pointer at the implementing code,
> and in a handful of cases a real behavioural fact (`epipolar-curves.md`'s
> per-feature anchor depth, `bundle-adjustment.md`'s "one staged loop, not two",
> `viewport-hud.md`'s **File** being the only menu). Those are kept, in the
> section where they belong. Four stamps also pointed at code that had moved or
> never existed; see the Top priority 4 annotation for the list._

### 5. Work-order residue

Two patterns, both explicitly prohibited by `TEMPLATE.md` ("Present tense,
describing what the code is. Not a work order").

**a. Dated errata blocks — 29 across 12 specs.** A `> Status (YYYY-MM-DD):
implemented, with notes against the text below` block sits between the title and
the body, correcting prose that was never updated.

| blocks | spec |
|---|---|
| 7 | `core/patch/cluster-patch-refinement.md` |
| 5 | `core/patch/patch-normal-refinement.md` |
| 3 | `gui/camera-intrinsics.md`, `core/patch/patch-cloud.md` |
| 2 | `core/patch/fronto-parallel-patch-cache.md`, `core/geometry/rotation-init.md`, `core/features/randomized-kdtree-forest.md` |
| 1 | `core/patch/cluster-patches.md`, `core/geometry/reconstruction-growth.md`, `core/geometry/pose-verification.md`, `core/features/covisibility-selection.md`, `cli/reconstruction/xform/refine-keypoints-command.md` |

**Every spec read this run that had one showed the note and the body in active
disagreement**, with nothing telling a reader which wins:

- `focal-vote.md:92-99` describes one uniformly sampled member pair per cluster;
  the code enumerates every covisible pair exhaustively
  (`focal_vote.rs:645,672-681`) — as the note at `:11-22` says.
- `rotation-init.md:65-66` says displacement tables come from covisibility
  selection; note (3) at `:24-29` says they are built in-kernel, and
  `build_pair_tables` (`rotation_init.rs:171-212`) agrees with the note.
- `cluster-patch-refinement.md:182` specifies `GaussianDisk { sigma: 15.0/4.0 }`;
  the note at `:117-118` and `params.rs:104` both say **0.5** — two values 65
  lines apart in one document.
- `covisibility-selection.md` note (5) *admits* the `nearest`/`farthest`
  signatures in the body are missing their `min_shared` parameter, rather than
  fixing them.

> _Status (2026-09-03): Partially done — 32 blocks across 14 specs folded into
> the prose they corrected and deleted, the code read first in every case. Every
> disagreement this section names is resolved in the code's favour:
> `focal-vote.md`'s Pair tables section describes the exhaustive enumeration,
> `rotation-init.md` §1 builds the tables in-kernel and §4 carries the 8 px trim
> gate and 10-survivor floor, and `pose-verification.md`'s note (5) — which is
> where the `nearest`/`farthest` admission actually lives, not
> `covisibility-selection.md`; see the Top priority 2 annotation — is gone, its
> `min_shared` default and the other eleven undocumented tunables now in a
> Parameters table. Not done: `cluster-patch-refinement.md`'s seven, held back
> for the wholesale rewrite Top priority 3 asks for._
>
> _The table above undercounts, because the survey grepped for the word
> "Status". The same construct is also spelled `Deviation`, `Note` and
> `Correction`: 39 blocks across 16 specs, not 29 across 12. The extra ten are
> `absolute-pose.md` (2), `epipolar-estimation.md` (2), `focal-vote.md` (1 — the
> one this section itself cites), `refine-normals-command.md` (1), and four more
> in `camera-intrinsics.md`, which has seven rather than three. All ten are
> folded here._
>
> _Widen the grep once more — to any blockquote carrying a date — and a further
> family appears under labels the audit did not look for at all. What remains
> open, after this pass and excluding `cluster-patch-refinement.md`'s own two
> (`Addition (2026-07-10)`, `Revision (2026-07-11)`): the coordinated
> `**Precondition — shipped (2026-06-25):**` set in `compare-command.md:126`,
> `xform-command.md:330` and `render-patches-command.md:14` with its sibling
> `**Re-layered (2026-06-25):**` in `embed-patches-command.md:10` (the matching
> one in `refine-normals-command.md` is folded here, since that file was already
> being edited); `sift.md:216` (`Revision`); `track-cluster-matching.md:309`
> (`Performance pass`); `keypoint-localization-search-cache.md:155,170`
> (`Incremental LOO consensus`, `Convergence fix`);
> `keypoint-localization-consensus-basis.md:9`;
> `patch-normal-refine-view-subset.md:11,13`;
> `sift-to-patch-reconstruction.md:46`; and `multi-panel-image-browser.md:248,
> 253, 300` (`Added`, `Added`, `Changed`), which annotate the GUI panel work of
> #344-#349 and are the freshest of the set. Sixteen blocks in eleven specs — a
> second pass of the same shape, and the next audit's grep should be written to
> catch them._
>
> _Status (2026-09-04): Done — `grep -rnE '^>.*\(20[0-9]{2}-[0-9]{2}-[0-9]{2}\)'
> over `specs/` is empty. The inventory was written against line numbers that had
> already moved and overcounts by six: at the start of this pass the wide grep
> found **ten** blocks in seven specs, not sixteen in eleven.
> `embed-patches-command.md`'s `Re-layered`, `keypoint-localization-consensus-basis.md:9`,
> `patch-normal-refine-view-subset.md:11,13` and `sift-to-patch-reconstruction.md:46`
> no longer existed — they went with the `**Status:**` sweep of commits 1883d2d /
> 89c381d, which rewrote the tops of those files. `cluster-patch-refinement.md`'s
> two were already gone with its rewrite, as this section's own last annotation
> records._
>
> _The ten, folded into the prose each corrected with the code read first:
> `compare-command.md:126` — `--strips` is deliberately ungated, and the
> "`scripts/exp_*`/`cmp_*` probes" it cited do not exist (only
> `scripts/exp_plus_descent_localize_compare.py` matches the glob, and no `cmp_*`
> does), so the sentence now names `PatchCloud.from_reconstruction` staying
> dual-mode and `_solve_strips.py`, which is what actually holds;
> `xform-command.md:330` and `render-patches-command.md:14` — the
> `embedded_patches` precondition is body prose in each, in
> `render-patches-command.md` next to the conversion recipe that was already
> there; `sift.md:216` — cap-aware coarse-to-fine detection is §7's own
> paragraph, keeping the 94%-of-detection-pixels and 300 s → 141 s numbers;
> `track-cluster-matching.md:309` — a `### Where the time actually goes`
> subsection under `## Cost Analysis`, keeping the DinoLedge measurement and the
> determinism statement, and stating each optimization as a constraint on how the
> matcher may be changed rather than as a diff;
> `keypoint-localization-search-cache.md:139,154` — the Gram-space LOO consensus
> and the round-over-round convergence metric are two prose paragraphs after the
> round-loop pseudocode; `multi-panel-image-browser.md:254,259,306` — the two
> `Added` blocks are fixed by correcting the `Tab` enum itself against `dock.rs`
> (it now lists all seven variants, in the code's order, with a table naming the
> spec that owns each of the four later ones), and `Changed` becomes the sentence
> saying the grid is the layout the panels start in._
>
> _Two code comments now point at notes that no longer exist and should be
> redirected rather than deleted, since what they say is still true:
> `focal_vote.rs:647` ("see the spec's deviation note" → the Pair tables
> section) and `epipolar_estimation/tests.rs:433` (→ Testing requirements,
> Contamination sweep). `absolute_pose.rs:28-32` carries a "Deviation from the
> spec (2026-07-14)" module-doc note whose premise the spec no longer holds; it
> should shrink to the plain rationale. In `sfm-explorer`,
> `image_detail/intrinsics/axes.rs:18-19,435-443`,
> `intrinsics_detail/derived.rs:88-97` and `image_detail/intrinsics/field.rs:36`
> quote or argue with spec wording that this pass replaced._
>
> _Status (2026-09-03): Done — every comment this paragraph lists is fixed,
> commit 7d4d3a4. `focal_vote.rs:647` names the spec's "Pair tables" section;
> `epipolar_estimation/tests.rs:433` names Testing requirements → Contamination
> sweep; `absolute_pose.rs:28-32`'s deviation note is now the plain rationale for
> the plain `Vec` (in the module-doc shrink, c11c885); and the four `sfm-explorer`
> comments state what the spec now states rather than disputing it — radial
> models keep the axes straight and bunch the ticks, the tick ladder takes the
> finest step that clears the spacing, and the distortion maximum bounds its own
> domain._
>
> _Four more of the same family, found while doing it and fixed in the same
> commit, since the paragraph's inventory was not complete. `focal_vote.rs:425`'s
> `PairAccum` doc still described "the sampled pass" and "how many clusters
> sampled this pair", against a field that has been a true shared-cluster
> covisibility count since the tables went exhaustive. `absolute_pose.rs:47-50`
> documented `KABSCH_RANK_EPS` as a test on the **smallest** singular value; the
> code tests the second against the first, and the in-body comment at `:284-288`
> already said why. `patch/normal_refine/mod.rs:71-72` and `params.rs:229-231`
> both said the input `u_axis` is reprojected onto the candidate plane with
> `v = n × u` — `repose_patch` passes `base.v_axis` as `from_center_normal`'s
> `up_hint`, so it is `v` that is reprojected and `u = v × n` that follows. And
> `params.rs:196-198` promised `max_refine_views` a "conditioning fallback" that
> `view_subset.rs:104-112` and `patch-normal-refine-view-subset.md:95-113` both
> say does not exist and explain why not; the code is right and the doc now
> matches it, naming the two cases where the subset genuinely does widen back to
> every view (a point at infinity, and no front-facing view to anchor on)._
>
> _Status (2026-09-03): Done — `cluster-patch-refinement.md`'s seven blocks, the
> last of the set, are gone with the wholesale rewrite of Top priority 3, along
> with the two dated blocks under other labels this section counted separately
> (`Addition (2026-07-10)`, `Revision (2026-07-11)`). The `sigma` contradiction
> this section cites is resolved in the code's favour: the spec's only statement
> of the window default is now `GaussianDisk { sigma: 0.5 }`, in the Parameters
> table and in the Theory paragraph that explains the unit. Sixteen blocks in
> eleven specs remain, listed above._

**b. Imperative construction language and phase headings.** `## Prior state
(before this change)` and `## Consumers & migration`
(`batch-triangulation-api.md:54,287`); `## Implementation plan (production)` with
Phase 1/2 (`fronto-parallel-patch-cache.md:191-241`); Steps 1–9 each suffixed
"— DONE" (`gui/camera-views.md:1474-1572`); `## Implementation Plan` with Steps
1–4 (`gui/image-animation.md:102-122`); and six `New module ...` lines
(`track-cluster-matching.md:508`, `cluster-patch-refinement.md:137`,
`batch-triangulation-api.md:83`, `patch-normal-refine-view-subset.md:153`,
`refine-normals-command.md:361`, plus `sift.md:688`'s "not yet implemented").

22 specs carry an `Open questions` section. Sampled ones contained settled items
presented as live: `cluster-patch-refinement.md:567-571` says `sfm match
--cluster` "does not yet write cluster-bearing files", but
`tests/patch/test_cluster_patches.py:42-43` shells out to exactly that.

> _Status (2026-09-03): Partially done — the two items this section pins on
> `cluster-patch-refinement.md` are gone with its rewrite: the `New module …`
> line at `:137` (which was also wrong about the module's file list), and the
> settled "does not yet write cluster-bearing files" claim, whose whole §5 test
> sketch went with it. The rewritten `Open questions` carries two genuinely
> undecided items — the localizability threshold's resolution dependence and the
> reference-selection policy. The other five `New module …` lines and the phase
> headings named above are untouched._
>
> _Status (2026-09-04): Done — every remaining item of this finding is gone, and
> the greps it was written from come back empty across `specs/` outside
> `specs/drafts/`: no `## Implementation Status` / `Plan` / `phases` / `Order` /
> `Staging` / `Phasing` / `Plumbing` heading, no `— DONE`, `[x]` or `NOT
> STARTED` marker, no `New module …` / `New file …` line._
>
> _The nineteen sections: `sift.md`'s `## Phasing`;
> `track-cluster-matching.md`'s `## Implementation Status` and
> `### Implementation order (suggested)`;
> `keypoint-localization-consensus-basis.md`'s and
> `patch-normal-refine-view-subset.md`'s `## Plumbing` **and** their
> `## Task-completion checks (from AGENTS.md)` sections (the latter restated
> `AGENTS.md` and carried nothing about the design);
> `batch-triangulation-api.md`'s `## Prior state (before this change)` and
> `## Consumers & migration`; `image-warping.md`'s `## Implementation Order`,
> which the audit's grep missed because the word is "Order"; and in `specs/gui/`,
> `camera-intrinsics.md`'s `## Implementation phases`, `camera-views.md`'s Steps
> 1-9, `image-animation.md`'s Steps 1-4, `multi-panel-image-browser.md`'s whole
> `## Implementation Plan` including Phase B, `patch-rendering.md`'s
> `## Implementation sketch (v1)` and `## Implementation Status`,
> `point-cloud-rendering.md`'s, `viewport-hud.md`'s `## Staging` and
> `## Implementation Status`, `viewport-navigation.md`'s, and
> `scene-graph.md`'s `## Implementation Phases`. `goto-point.md`'s lone `[x]`
> was never a checklist item — it is the close box in an ASCII mock of the
> dialog's title bar, now `[×]`._
>
> _Every ticked item was verified against the code before its line was deleted,
> and **five turned out to be wrong**, all in `specs/gui/`. (1)
> `viewport-hud.md`'s "Remove the View menu outright, leaving File as the only
> menu": the View menu is gone, but `app.rs:661,718,735` shows **File, Go,
> Panels** — the spec's opening paragraph said "the only menu" too, and now says
> the menu bar has no View menu. (2) `viewport-hud.md`'s `## State ownership`
> said "the HUD opens collapsed each launch" while `Viewer3D::hud_open` starts
> `true` and `hud/tests.rs:344` asserts it opens expanded. (3)
> `multi-panel-image-browser.md`'s `### Phase B — NOT STARTED` listed five
> "potential additions", of which point hover, hover-based track highlighting
> and the feature overlay modes all **exist** (`state.rs hovered_point`,
> `image_browser.rs hover_track_images`, `OverlayMode`), specified since in
> `cross-panel-hover.md`; only co-track highlighting is genuinely absent, and it
> is now a Non-goal. (4) Its Step 7 named `observations_for_point` /
> `track_image_indices` helpers that do not exist — call sites index
> `observation_offsets` directly, which `batch-triangulation-api.md`'s Reuse map
> also mis-stated. (5) `track-cluster-matching.md`'s status claimed the bindings
> are "exposed as `sfmtool.background_floor_clusters` /
> `sfmtool.clusters_to_pair_matches`"; nothing re-exports them at the package top
> level — they live in `sfmtool._sfmtool.matching`, which is where
> `_cluster_matching.py` imports them from — and its `#### Python package
> surface` section had asked for a re-export that was never done. Two smaller
> ones went the same way: `camera-views.md`'s Step 4 named a
> `bg_image_loaded_index: Option<usize>` field that is now
> `bg_image_loaded: Option<ImageRef>`, and `image-animation.md`'s Design section
> put the minibar transport in `dock.rs` when it is in `image_browser.rs`._
>
> _Nothing that only a checklist stated was lost. What moved into the body, by
> spec: `track-cluster-matching.md` — the three-layer code pointers and the
> four-dataset end-to-end reproduction now lead `## Production Implementation`,
> and the `d = 28 → 10` sweep is `### Choosing d` under `## Parameters`, where the
> defaults table already pointed;
> `keypoint-localization-consensus-basis.md` — a new `## How the ranking inputs
> reach the kernel` carries the `view_scores` / `track_view_counts` optional
> inputs, `select_views`'s `track_view_count` output, and the `basis_pick` /
> `tail_register` profiling phases with the `N_BASIS` / `N_TAIL` /
> `N_TAIL_NO_BASIS` counters; `patch-normal-refine-view-subset.md` — a
> `## Where it lives`, and its `## Validation harness` now describes
> `scripts/validate_refine_subset.py`, which exists (the section asked for a
> `.sh` that does not); `batch-triangulation-api.md` — the max-track-angle
> comparison is its own section rather than a "prior state" table;
> `camera-views.md` — Step 9's design (which navigation keeps camera view, the
> FOV rules, the camera-to-camera transition, `CameraViewMode`) is promoted to
> `## Persistent camera view and free-look navigation`, since four cross-links
> pointed into it; `camera-intrinsics.md` — the module paths for
> `camera/report.rs`, `intrinsics_detail/` and `image_detail/intrinsics/`, the
> `sfmr_format::{RigFrameData, …}` re-export, and the `_CAMERA_PARAM_NAMES`
> desync consequence for `sfm inspect`; `multi-panel-image-browser.md` — the
> `Tab` enum, corrected against `dock.rs` and given a table of which spec owns
> each later tab. Unticked items became Non-goals sections in
> `viewport-navigation.md` (five), `point-cloud-rendering.md` (four, plus one
> Open question), `camera-views.md` (five), `multi-panel-image-browser.md`
> (three, two of them carried over from `plan.md`) and
> `track-cluster-matching.md` (`sfm solve --cluster`, which does not exist)._
>
> _Four unticked items were genuine intentions with design detail behind them,
> and became amendment drafts under the convention Top priority 4 settled:
> `specs/drafts/sift-gpu-amendment.md`,
> `specs/drafts/sift-incremental-extraction-amendment.md`,
> `specs/drafts/patch-normal-refine-zncc-weighted-selection-amendment.md` and
> `specs/drafts/patch-rendering-flat-shaded-amendment.md`. Each is linked from
> one present-tense sentence in the spec it amends and names that spec back.
> `specs/drafts/README.md` is new and indexes them; `specs/README.md`'s drafts
> row now links it._
>
> _`specs/gui/plan.md` is retired. It was a roadmap, dated 2026-06-10, whose
> `## Current Implementation Status` is now duplicated in more detail by the
> per-panel specs; its spec table listed 6 of the 21 files in `specs/gui/`, its
> status list predates the Scene Graph, MCP server, Action Log, Camera
> Intrinsics panel, panel layout and Go-to-Point work, it called
> `viewport-hud.md` "(proposed)" though it shipped, and its Performance Targets
> table is a verbatim copy of `architecture.md`'s. Two ideas in it were found
> nowhere else and were carried into `multi-panel-image-browser.md`'s Non-goals
> — a grid mode for the browser strip, and epipolar lines to the selected image
> as an overlay mode. `specs/gui/README.md`'s "Planning and Reference" section
> is now "Reference"._

**c. Dead references.** Three cited artifacts do not exist and, per `git log`,
never did: `scripts/exp_pinhole_bootstrap.py` (`cluster-covisibility.md:8,216`),
`scripts/exp_cluster_patch_clusters.py` (`cluster-patch-refinement.md:14`, called
"the behavioral reference"), and
`reports/2026-07-09-exp-pairwise-sift-warp.md` (`cluster-patch-refinement.md:10,600`).
> _Status (2026-09-03): Partially done — the phantom citations in two of the
> specs the Top priority 3 pass was already editing are gone.
> `cluster-patches.md:3-5` cited `reports/2026-07-09-exp-pairwise-sift-warp.md`
> plus harnesses `scripts/exp_pairwise_sift_warp.py` and
> `scripts/exp_cluster_patch_clusters.py`, and
> `fronto-parallel-patch-cache.md:14` cited
> `reports/2026-06-15-patch-cache-status.md`. `git log --all` finds no commit
> that ever added any of the four, so the count of never-existing artifacts is
> five, not three — `exp_pairwise_sift_warp.py` and the two reports are not in
> the list above. `randomized-kdtree-forest.md` also pointed at
> `crates/sfmtool-py/src/py_kdforest.rs`, which is `spatial/kdforest.rs`, and
> named a Python test `test_exhaustive_budget_matches_brute_force` that does not
> exist; both are corrected. Still open: the citations inside
> `cluster-patch-refinement.md`, which the wholesale rewrite will take, and
> `sift-to-patch-reconstruction.md:61`'s
> `reports/exp/2026-06-21-mvs-normal-refinement.md` — `reports/exp/` is not a
> directory in this repo._
>
> _Status (2026-09-03): Done for this spec — the rewrite drops both citations
> inside `cluster-patch-refinement.md` (`scripts/exp_cluster_patch_clusters.py`
> "the behavioral reference" at `:14`, and
> `reports/2026-07-09-exp-pairwise-sift-warp.md` at `:10` and `:600`) together
> with the "cross-check with the prototype" test item they made unrunnable. The
> spec no longer refers to a prototype at all; the measured calibration those
> citations stood in for is stated where it is used, and the design-level numbers
> stay in `cluster-patches.md`. One more sentence of the same family went from
> `cluster-patches.md:220` ("The experiment scripts stay as the behavioral
> reference until the Rust kernel lands"), which named no file but pointed at the
> same phantoms. Still open: `cluster-covisibility.md`'s
> `scripts/exp_pinhole_bootstrap.py` and `sift-to-patch-reconstruction.md:61`._

Also dangling: `specs/.../cluster-pinhole-bootstrap.md` and the symbol
`MAX_CLUSTERS`, neither of which exists anywhere in the repo.

### 6. Coverage, both ways

**CLI: complete.** All 27 subcommands have a spec except `explorer` (the GUI
launcher, covered by `specs/gui/`) and the `ws` group (covered by
`ws-init-command.md`, its only subcommand). Every area `README.md` indexes every
spec beside it — zero missing rows.

**Specs never linked from the code they describe: 50 of 121**, including 13 under
`specs/core/`: `image-pair-graph`, `reconstruction-alignment`, `image-warping`,
`projection-jacobian`, `sfmtool-fisheye-kernels`, `sfmtool-pinhole-kernels`,
`flow-based-matching`, `gpu-optical-flow`, `optical-flow`,
`randomized-kdtree-forest`, `relative-pose`, `reprojection-residuals`,
`point-correspondence`. The house style is contract-then-link; these are the
modules where a reader sitting in the code has no route to the design.

**Highest-churn module since the last audit** is `patch/cluster_refine` (32
commits), whose spec is in this sample and is the most drifted document found.
Next: `patch/normal_refine` (28), `patch/keypoint_localize` (20),
`reconstruction/data` (19), `camera/distortion` (18).

---

## Per-spec findings

### specs/core/analysis/observation-coverage.md
**Summary:** Per-image coverage grid over observation disks, with four queries. Accurate on grid geometry, the cell-center rule, clipping, saturation, the sector convention and `cell_px = 4`. Its gaps are all at the edges.
**Implementing code:** `analysis/observation_coverage.rs` (`ObservationCoverage::{build, counts_at, covered_fraction, uncovered_sectors, image_covered_fraction, grid, image_count, cell_px}`, `DEFAULT_CELL_PX`, `MAX_SECTORS`); `sfmtool-py/src/analysis/observation_coverage.rs`.
**Inconsistencies:**
  - `grid` signature wrong: spec:82 says `-> (&[u8], width_cells, height_cells)`; actual is `Option<(&[u8], u32, u32)>` (`:135`).
  - `MAX_SECTORS = 32` (`:31`) undocumented, and the layers *disagree* on violating it: Rust returns all-zero masks (`:238-240`), Python raises `ValueError` (`py:186-190`). Same split for out-of-range image indexes (Rust 0/0.0 at `:141,263`; Python rejects at `py:77-81,202-207,239-243`). Neither documented, nor that `cell_px == 0` and slice-length mismatches panic (`:80-91,146,177,228`).
  - Non-finite *keypoints* are dropped (`:304,347`); the spec's drop list (50-52) covers only non-finite radius and far-outside keypoints.
  - `image_count()`, `cell_px()` and the Python getters/`__repr__` appear in no section. The `// default 4` comment at spec:38-46 misleads — core `build` has no default; only the PyO3 signature supplies one (`py:41`).
  - No usage example. F1 and F2 pass — the Overview reads cold.
**Third copies:** `observation_coverage.rs:6-18` reproduces spec:12-22 essentially verbatim (13 lines), `:35-41` restates 26-34, `py:17-22` is a third rendering. **The Rust module doc should shrink** to one sentence plus its existing pointer at `:20`; keep the struct doc (a `grid()` caller needs the cell geometry) and the Python docstring.
**Recommendation:** update spec — add an errors/limits subsection, fix the `grid` signature, list the accessors, add one worked call; then trim `:6-18`.
**Unclear / incorrect / suspicious:** The Rust-degrades / Python-raises split looks deliberate but is stated nowhere as a decision. `DEFAULT_CELL_PX`'s doc ("The spec's default cell size") points authority at the spec, which is backwards for a standing constant. Nothing outside tests uses `ObservationCoverage`, so its three motivating consumers are still hypothetical.
> _Status (2026-09-03): Done for the "Third copies" item — `observation_coverage.rs:6-18` is one sentence plus its existing pointer, the module doc going 17 lines to 8, commit c11c885. The struct doc at `:35-41` is deliberately kept: a `grid()` caller needs the cell geometry. `DEFAULT_CELL_PX`'s doc still points authority at the spec, and this section's other findings are untouched._

### specs/core/camera/sfmtool-fisheye-kernels.md
**Summary:** The fisheye model whose radial map is a monotone cubic B-spline correction on the equidistant base. Technically accurate — basis gauge, fold gate, safeguarded Newton, two-stage monotonicity and the bit-identity short-circuit table all match. It also doubles as the home for the B-spline basis that `SFMTOOL_PINHOLE` defers to.
**Implementing code:** `camera/distortion/bspline.rs` (`BSPLINE_SUPPORT`=4 `:37`, `MIN_BSPLINE_COEFFS`=2 `:43`, `basis_at :120`, `delta_and_deriv :187`, `bspline_is_monotone :220`); `camera/distortion/kernels.rs:1088-1330` (`recover_radial_bspline :1161`, `sfmtool_fisheye_ray_jacobian :1281`); dispatch `distortion.rs:160,417,594,725,952`.
**Inconsistencies:**
  - **F1 fails.** Opens `**Status:** Implemented.` plus six lines of paths. Proposed: "A fisheye camera whose radial map is a monotone cubic B-spline correction on top of the equidistant base, so a real lens can be calibrated past 90° without a polynomial that folds — these are the kernels that project, unproject and differentiate it."
  - **F2 confirmed and it bites.** No signature anywhere. `bspline_is_monotone` is described as checking "over `[0, θ_max]`" (spec:114) while the code takes a third `d_span` argument and checks `[0, min(d_span, d_max)]` (`bspline.rs:220,236`). `recover_radial_bspline`'s `(d, converged)` return and `d_max` argument are inferable only from prose. Unstated: every kernel is `pub(super)`, so the reader's actual entry point is `CameraModel::SfmtoolFisheye` via `CameraIntrinsics::{project, ray_to_pixel_with_jacobian}`.
  - Spec:36-37 attributes the identity fallback to `basis_at`, which does the opposite — it `debug_assert!`s the preconditions (`bspline.rs:130-131`). The fallback lives in `delta_and_deriv` (`:188-190`) and `bspline_is_monotone` (`:222-228`).
  - Spec:106 says the Jacobian is `None` at `rz < 0`; code is `rz <= 0.0` (`kernels.rs:1297`), plus `None` on the zero ray (`:1293`), unmentioned. Spec:104 gives the on-axis test as absolute `ρ < 1e-15`; the Jacobian uses a *relative* `rho <= EQUIDISTANT_AXIS_EPS * n2.sqrt()` with `1e-12` (`kernels.rs:938,1049`).
  - Newton's budget is never named: `UNDISTORT_MAX_ITER = 100`, `UNDISTORT_EPS = 1e-10` (`distortion.rs:68,71`). A spec that says "converges" should say within what.
  - "Testing requirements" (`:174`) is future-tense for tests that exist.
**Third copies:** The largest overlap found. `bspline.rs` carries 95 doc lines in a 247-line file re-deriving the center-anchored gauge, the `N ≥ 2` argument, the held-constant tail and two-stage monotonicity in the spec's own words; `kernels.rs:1088-1330` adds 63 more; `intrinsics.rs:170-215` adds 39 more. `specs/formats/sfmtool-camera-models.md:186-201` is a fourth statement of the projection equation. **Shrink `bspline.rs`'s module header** to the shape of the data plus a spec pointer.
**Recommendation:** update spec — purpose opening, a real Rust interface block (including `d_span` and the `(d, converged)` return) plus one call through `CameraIntrinsics`, fix the `basis_at` attribution, present-tense the testing section.
**Unclear / incorrect / suspicious:** The B-spline basis is shared machinery for two models but specced under one, with `sfmtool-pinhole-kernels.md:24-30` deferring here — the code has the cleaner split (`bspline.rs` is its own module), so that section wants its own spec. `distortion.rs:952` wraps an already-`Option` return in `Some(...)`, giving `Option<Option<_>>` unwrapped at `:966`; correct, but not what "None from both, together" prepares a reader for.
> _Status (2026-09-03): Partially done — `bspline.rs`'s module header is 30 lines to 24, and `bspline_is_monotone`'s doc no longer re-derives the two-stage argument, commit c11c885. The re-derivation was not the only problem there: the file — and nothing else in `camera/distortion/` — carried **no** spec pointer at all, which is the likeliest reason its header grew into a standalone derivation. It now names this spec for the basis and the monotonicity invariant, and `formats/sfmtool-camera-models.md` for why the gauge is anchored at the centre. `kernels.rs:1088-1330` and `intrinsics.rs:170-215` are untouched, as is every finding against the spec itself._
>
> _Status (2026-09-03): Done for F1 — the `## Summary` opening is this section's proposed sentence, verbatim, and the six-line file inventory that was the Status line is now the lead of `## Basis evaluation` as relative links, commit 1883d2d. The sentence that followed leaned on a link to parse ("The computation behind the `SFMTOOL_FISHEYE` camera model defined in …"); it now names the model spec as where the parameterization lives and says this one is the computation. F2 and the six inconsistencies remain open._

### specs/core/features/cluster-covisibility.md
**Summary:** The shared-cluster count matrix `W[i,j]`, its acceptance mask, dense storage bound, seed-group iterator and ranking. The algorithm, complexity and determinism content is accurate and earns its place; the scaffolding around it is residue from the promotion-from-experiments change it was written for.
**Implementing code:** `features/cluster_match/covisibility.rs` (`MAX_DENSE_IMAGES :27`, `SeedGroupParams :105`, `ClusterCovisibility::{from_clusters :177, from_clusters_with_positions :205, rank_by_covisibility :351, seed_groups :394, next_seed_group :412}`); `sfmtool-py/src/matching/covisibility.rs`.
**Inconsistencies:**
  - F1 fails (quoted in §4). Proposed: "Cluster covisibility measures how many match clusters each pair of images shares, so a caller can pick mutually-overlapping image groups and rank candidate views before any reconstruction exists."
  - Three dead references (§5c). The real consumer today is `geometry/reconstruction_growth.rs:620,1027` (`cv.thin_to(...)`), which the spec never names.
  - Bindings block (`:183-191`) documents a 4-parameter `from_arrays`; actual is 6 (`py:126-127`). A reader of *this* spec alone gets a wrong signature.
    > _Status (2026-09-03): Done — the Bindings block now shows the full signature `from_arrays(cluster_starts, member_images, num_images, member_accepted=None, positions_xy=None, seed=0)`, commit 7ddbe97. The `cov.counts` error claim, the dead references and the missing `next_seed_group` are untouched._
  - `cov.counts  # ... errors above dense bound` (`:188`) — the getter never errors (`py:206-216`); the bound is enforced only at construction (`:213`).
  - `matcher_options.d` (`:80,102`) is not a symbol; it is `BackgroundFloorParams::d`, default 10 (`cluster_match/mod.rs:52,64` — value correct).
  - `next_seed_group` (`:412`) is public API — the step function both iterators drive — and appears in neither covisibility spec.
  - No example call. Interface *rationale* is present and good ("Core stays I/O-free: raw CSR slices in", `:110-112`).
**Third copies:** The cluster-vs-post-reconstruction naming rationale is stated three times (spec:34-38, `covisibility.rs:10-13`, `py:72-74`); the seed-group algorithm twice at full length (spec:158-175, `:388-393` + `:424-472`). The core module doc already has a spec pointer at `:15-19` — **the re-derivation above it should go.**
**Recommendation:** update spec — purpose opening, delete the three dead references, correct `from_arrays` and the `counts` error claim, add `next_seed_group` and one example call.
**Unclear / incorrect / suspicious:** The Complexity section's empirical numbers ("mean span ≈ 3.3 across three campaign datasets") came from the deleted `exp_pinhole_bootstrap.py` and are unreproducible. Keep the bound, drop the constants.

> _Status (2026-09-03): Done for F1 — `## Purpose` opens with this section's proposed sentence, verbatim, commit 1883d2d. The Status line went with it, and with it the "Promotes the image-grouping-by-shared-clusters machinery from the pinhole bootstrap experiments" framing and its two dead citations (`scripts/exp_pinhole_bootstrap.py`, `cluster-pinhole-bootstrap.md`). The two cross-references it carried are worth keeping and are now a **See also** paragraph at the end of the opening section rather than the first thing a reader meets. The code pointer is the lead of `## Rust API`, as links. Still open: the third dead citation in `## Validation`, the `cov.counts` error claim, the missing `next_seed_group`, and the absent example call._

### specs/core/features/covisibility-selection.md
**Summary:** Three queries layered on the same type — sampled pair displacement, banded thinning, and reach. Correct against the code, but it cannot be read cold and its only signature block is wrong.
**Implementing code:** `features/cluster_match/covisibility/selection.rs` (`sweep_order :19`, `thin_in_order :42`, `thin :70`, `thin_to :78`, `reach :127`); sampling pass `covisibility.rs:248-305`; bindings `py:222,239,341,358,372`.
**Inconsistencies:**
  - F1 fails twice: the opening rests on a symbol from another spec, and the Purpose paragraph (`:19-24`) describes the world *before* the change. Proposed: "Three queries over a set of images' shared-cluster counts: how far apart two covisible images are in appearance, which subset survives redundancy-thinning, and how much of the capture a chosen subset connects to."
  - **F2 confirmed.** No `rust` block. The only fenced block (`:30-33`) is a pseudo-signature that **omits `member_accepted`** — a reader following it positionally passes `positions_xy` into the mask slot. The Rust constructor these queries require, `from_clusters_with_positions` (`:205`), is named only inside the status note.
    > _Status (2026-09-03): Partially done — the Construction block now carries the real signature and names `from_clusters_with_positions` as the core-side constructor; the 2026-07-18 errata block, whose note (1) admitted the signature was wrong, is deleted and its content folded into the section it corrected, commit 7ddbe97. The missing Rust interface block itself remains open._
  - `:44` "squared-root pixel distances accumulate" — the code accumulates plain Euclidean distance via `f64::hypot` (`covisibility.rs:283`).
  - `thin` unconditionally keeps the first swept image regardless of the band (`selection.rs:46-49`); the spec's band description implies every image is band-tested. The code doc says it; the spec does not.
  - `thin_to`'s search range `[1, median per-image row peak]` and its fixed 25 iterations (`selection.rs:95-104`) are load-bearing and absent.
  - The dated errata block (`:8-15`) and future-tense "Testing requirements" (`:80-91`) are residue; the tests exist (`covisibility/tests.rs:294-527`).
**Third copies:** The `[tau/8, tau)` band rationale appears three times (spec:52-56, `selection.rs:64-69`, `py:333-337`); the sampled pass four times (spec:42-44, `:199-204`, `:270-272`, `py:118-123`). Both docstrings should shrink to signature-level facts plus the pointer.
**Recommendation:** discuss — this spec and `cluster-covisibility.md` describe one module's queries across two documents, and this one cannot stand alone. Either fold it in as a "Selection queries" section, or give it a self-contained opening and a real Rust interface block.
**Unclear / incorrect / suspicious:** Both spec (`:63`) and code (`selection.rs:74-77`) assert the kept count grows monotonically with `tau`, and `thin_to`'s binary search depends on it — but both band edges move with `tau` (`selection.rs:56`), so it is not obvious and may not hold. The code hedges by tracking the closest-so-far candidate across all 25 iterations rather than trusting the bisection, which suggests the author did not fully believe it either. Worth verifying or restating as a heuristic.

> _Status (2026-09-03): Done for F1 — both halves. The opening no longer rests on `ClusterCovisibility`, a symbol from another spec: the Status line is gone and `## Purpose` opens with this section's proposed sentence, verbatim. The paragraph that described the world before the change ("three selection primitives that today exist only as per-caller array code") is gone with it, keeping the true half — all three are order-free and deterministic given a seed. The code pointer is the lead of `## Construction`, as links. Commit 1883d2d. F2 — no `rust` interface block — and the four factual findings remain open._

### specs/core/geometry/absolute-pose.md
**Summary:** P3P (Lambda Twist) plus a LO-RANSAC estimator and a trimmed pose-only refiner. Accurate on the algorithm, option semantics and every binding default; three concrete claims have drifted and the interface sections are transcriptions.
**Implementing code:** `geometry/absolute_pose.rs` (`p3p_solve`, `kabsch`, `estimate_absolute_pose`, `local_optimize :499`, `AbsolutePoseOptions::default :343`); `geometry/pose_refine.rs`; consumer `geometry/reconstruction_growth.rs:255`.
**Inconsistencies:**
  - Spec:199-201 says only the polynomial fisheye family and equirectangular lack an analytic Jacobian, with `EQUIDISTANT_FISHEYE` the fisheye exception. `intrinsics.rs:493-499` returns true for `EquidistantFisheye`, `SimpleRadialFisheye` **and** `SfmtoolFisheye` (plus `SfmtoolPinhole` via `needs_ray_path`). Stale by two models.
  - Spec:102-103 "no allocation beyond the fixed-capacity result" — `plane_quadric_dirs` allocates a `Vec` per call (`:192`), `kabsch`/SVD allocate, and the result is a plain `Vec`.
  - Spec:155-156 "keeps the returned pose the best *refit* pose, not a raw 3-point solution" — `local_optimize` discards the refit whenever `new_count <= count` (`:499-509`).
  - Spec:43-44 claims `w = 0.05` is tractable; the in-crate sweep needs `max_iterations: 200_000` to reach it (`tests.rs:291`) while the documented default is `50_000` (`:349`) — below the ~55k trials the spec's own formula requires. The spec never says the default cap truncates its advertised floor.
  - Spec:241-245 says `(N, 3)` input means caller-supplied bearings and an angular threshold; the binding also accepts a `camera` with `(N, 3)` and derives the angle from `max_error_px` (`py:176-187`).
  - `AbsolutePoseOptions::default()` (`max_angular_error: 0.01`) is undocumented; only binding defaults are given. `p3p_solve`'s binding is named in the Status block but absent from the Bindings section.
**Third copies:** `absolute_pose.rs:4-32` re-derives purpose, Lambda Twist rationale and determinism, and reproduces the spec's ArrayVec deviation note verbatim; **the module doc is the one to shrink.** `pose_refine.rs:6-19` restates spec:191-207 nearly word for word. Spec:107-140 and 172-189 transcribe the struct/fn declarations *including their doc comments*.
**Recommendation:** update spec — fix the Jacobian model list, the allocation and "best refit" claims, reconcile `w=0.05` with `max_iterations`, and replace the copied declarations with rationale plus one real call.
    > _Status (2026-09-03): Partially done — the two 2026-07-14 `Deviation` blocks, which §5a's grep missed, are folded and deleted. `p3p_solve`'s published signature is the real `-> Vec<(UnitQuaternion<f64>, Vector3<f64>)>`, with the reason kept in the "pure function" paragraph: a plain `Vec` reserved once for four poses, because the workspace carries no `arrayvec` dependency. The Kabsch degeneracy paragraph now says the collinearity test reads the **second** singular value against the first (`σ₁ < KABSCH_RANK_EPS · σ₀`, `1e-9`), not the third — three points are always coplanar, so the third vanishes for every triple and its direction is fixed by the determinant correction. The section's other findings are untouched. Two follow-ons: `absolute_pose.rs:28-32`'s module-doc deviation note now contradicts the spec it cites, and `absolute_pose.rs:47-50` documents `KABSCH_RANK_EPS` as the "smallest" singular value where the code (correctly) uses the second — the in-body comment at `:284-288` has it right._
**Unclear / incorrect / suspicious:** `absolute_pose.rs:505-506`'s comment reads "Keep the refined pose only if it did not shrink the consensus", but the branch it sits in is taken on `new_count <= count` and *discards* the refit — the comment describes behaviour the code does not have (the sibling `local_optimize_f` comment is correct). F1: purpose is stranded at line 8; proposed opening — "Registers one camera against known 3D structure: given image bearings paired with world points, most of which may be wrong matches, recover the camera's rigid pose."
> _Status (2026-09-03): Done for the "Third copies" item — `absolute_pose.rs:4-32` is 29 lines to 17, keeping the −Z bearing convention, the world-to-camera pose convention and the bit-stability contract, commit c11c885. The "Deviation from the spec (2026-07-14)" note inside it, whose premise the spec no longer holds, is now the plain rationale it always was: a plain `Vec` reserved once for four poses, because the workspace carries no `arrayvec` dependency. The follow-on this section's 2026-09-03 annotation named is fixed too — `KABSCH_RANK_EPS`'s doc said "smallest" singular value where the code (correctly) tests the second against the first, commit 7d4d3a4. `pose_refine.rs:6-19`, the spec's transcribed declarations, and the `local_optimize` comment at `:505-506` this paragraph flags are untouched._
>
> _Status (2026-09-03): Done for F1 — `## Purpose` opens with this section's proposed sentence, verbatim, commit 1883d2d. Both Status paragraphs are gone, and the pointer they carried (`absolute_pose.rs` and `pose_refine.rs`, with the three binding names) is the lead of `## The minimal solver` — chosen over `## Bindings`, which sits 180 lines further down past the theory, on the rule that the pointer belongs where a caller first meets a signature. The five stale factual claims this section lists remain open._

### specs/core/geometry/epipolar-estimation.md
**Summary:** 7- and 8-point fundamental solvers, Sampson gating, LO-RANSAC and the Bougnoux focal. Solver descriptions, the Sampson definition, the Bougnoux formula and every default match exactly. The failures are a false cross-reference, a contract the code does not implement, and an overtaken Non-goals list.
**Implementing code:** `geometry/epipolar_estimation.rs` (`fundamental_7pt :165`, `fundamental_8pt :226`, `estimate_fundamental :427`, `local_optimize_f :354`, `focal_from_fundamental :467`, `NULLSPACE_RANK_EPS :38`); consumer `geometry/focal_vote.rs:753,774`.
**Inconsistencies:**
  - **Spec:115 says `fundamental_8pt` returns `None` for "a rank-deficient design matrix". It does not.** The guard tests the **largest** eigenvalue against zero — `eig.eigenvalues.iter().fold(0.0, f64::max) <= 0.0` (`:226`) — which only rejects an all-zero design. A design with fewer than 8 independent constraints passes and returns an arbitrary null direction. `fundamental_7pt:165` does it correctly: `eig.eigenvalues[idx[2]] <= NULLSPACE_RANK_EPS * lmax`. A code defect against a documented contract, not a doc nit.
    > _Status (2026-08-30): Done — guard fixed to require a 1-D null space, with regression tests for the coplanar / zero-baseline / repeated-point / collinear configurations, and the spec's refit section expanded to name them. Details and two follow-on findings in the Top priorities annotation._
  - Spec:33 "`compute_epipole`'s null-space extraction is shared." It is not — `focal_from_fundamental` runs its own SVD (`:467-469`), and `compute_epipole` (`camera/epipolar.rs:71`) has no caller in this module (only `camera/rectification.rs` and tests).
  - Non-goals (`:279-283`) list essential-matrix/relative-pose decomposition and homography + F-vs-H model selection as out of scope. Both exist: `geometry/relative_pose.rs`, `geometry/homography_estimation.rs`, and `focal_vote.rs:52,758` performs exactly the H-domination selection the spec defers.
  - Spec:163-164's "best refit matrix, not a raw 7-point solution" has the same defect as its sibling: `local_optimize_f` keeps the 7-point `F` when the refit does not grow the count (`:354-362`).
  - Undocumented: after the loop, `estimate_fundamental` re-enforces rank 2, renormalizes and **rescores the mask**, then applies `min_inliers` to that post-enforcement count (`:427-433`). The spec implies `min_inliers` gates the consensus found during sampling.
  - `hartley_normalize` and `vec_to_mat3` are `pub(crate)` and consumed by `homography_estimation.rs:22`; the code doc records this (`:58-60`), the spec presents them as module-internal.
**Third copies:** `epipolar_estimation.rs:4-26` restates purpose, the 7-point cubic derivation and determinism from spec:1-12 and 89-101. Spec:13-20's deviation note re-derives what the `ZERO_F_EPS` doc comment says better (`:40-44`). Spec:120-151 is a verbatim copy of `FundamentalOptions`/`FundamentalEstimate` including doc comments.
**Recommendation:** update spec **and code** — fix `fundamental_8pt`'s rank guard to match the documented `None` contract, delete the `compute_epipole` sentence, retire the two overtaken non-goals.
    > _Status (2026-09-03): Partially done — the two 2026-07-16 `Deviation` blocks, which §5a's grep missed, are folded and deleted. "Focal length" now lists the three rejection checks in the order the code applies them: a non-finite entry or a Frobenius norm below `ZERO_F_EPS = 1e-12` **before** the rescaling, because that is what rotation-only zero-baseline motion produces and the normalized direction of such a matrix is pure round-off the `f₁² ≤ 0` sign test does not catch; then `|den| < BOUGNOUX_DEN_EPS = 1e-12`, whose value is set by the denominator's inherent near-cancellation (order `1e-8` at unit Frobenius norm); then the sign test. The Contamination sweep testing bullet states both floors and why they differ — `w⁷` sampling needs ~5×10⁵ trials for 0.999 confidence at `w = 0.2`, so the in-crate sweep floors at 0.35 and the 0.2 end runs against the release-built extension in `test_epipolar_estimation_rust_bindings.py`. The `compute_epipole` sentence, the overtaken non-goals, the `local_optimize_f` refit claim and the post-loop rescoring remain open. Follow-on: `epipolar_estimation/tests.rs:433`'s "See the spec's deviation note" now dangles and should point at Testing requirements → Contamination sweep._
**Unclear / incorrect / suspicious:** `epipolar_estimation.rs:224-225`'s comment ("A rank-deficient design ... leaves the null direction ambiguous — reject when the largest eigenvalue is ~0") is a non sequitur papering over the missing check. F1: proposed opening — "Recovers the epipolar geometry between two images from matched pixels alone — the fundamental matrix, robust to a match set that may be mostly wrong, plus the focal length it implies when the principal points are known."
> _Status (2026-09-03): Done for the dangling pointer — `epipolar_estimation/tests.rs:433`'s "See the spec's deviation note" now reads "See the spec's Testing requirements → Contamination sweep", which is where the two floors and the reason they differ ended up, commit 7d4d3a4. `epipolar_estimation.rs:4-26`'s re-derivation was not in Top priority 5's eight-block list and is untouched, as are the spec findings still open above._
>
> _Status (2026-09-03): Done for F1 — `## Purpose` opens with this section's proposed sentence, verbatim, followed by the old opening's precise statement as a "Formally: …" clause, commit 1883d2d. Deleting the Status block would have promoted the `compute_fundamental_matrix` paragraph to the top of the file, which the new convention forbids, so it moved to the end of Purpose with its path as a link. The code pointer leads `## The minimal solver (7-point)`. The `compute_epipole` sharing claim and the overtaken Non-goals list remain open._

### specs/core/geometry/focal-vote.md
**Summary:** Estimates focal length from feature tracks before any reconstruction exists. Technically accurate — **every one of 24 documented thresholds matches the code exactly**, the cleanest constants table audited this run — but it is 487 lines of unbroken prose with zero code fences, and it never names the Rust entry points.
**Implementing code:** `geometry/focal_vote.rs` (`focal_vote :533`, `focal_vote_with_min_disp :559`, `focal_vote_with_options :588`, `FocalVoteOptions :289`, `family_consensus`, `rotation_self_calib_focal`); `geometry/focal_vote/column_scan.rs`; `geometry/homography_estimation.rs`.
**Inconsistencies:**
  - **Live contradiction (F5).** "Pair tables" (`:92-99`) still says each cluster contributes one uniformly sampled member pair; the code accumulates *every* covisible member pair exhaustively (`:645,672-681`), as the spec's own deviation note (`:11-22`) says.
    > _Status (2026-09-03): Done — "Pair tables" now describes the exhaustive per-cluster enumeration and the last-observation-wins per-image dedupe, and says why the counts are the true covisibility (a sampled single pair starves the `30`-cluster epipolar and `25`-cluster rotation gates on exactly the parallax-poor captures the estimator exists for). The deviation note is deleted. "Determinism" said "All sampling (pair tables, RANSAC) derives from the input seed" — the same falsehood one section further on — and now says the tables draw no randomness at all. The section's other findings (no `rust` fence, `FocalVoteOptions` unnamed, the mid-spec `**Status:** Implemented (2026-08-08)` stamp, `PairAccum`'s stale doc at `focal_vote.rs:425`) are untouched._
  - **F2/F3 total.** No signature, struct or example call in 487 lines. `FocalVoteOptions{seed, epipolar_min_disp_frac, columns}` (`:289-305`) is never named. Only the Python binding is given, in prose (`:415-426`), and even that omits the keyword-only `*` boundary (`py:163`).
  - `estimate_homography` documented as `(points1, points2, max_error_px=3.0, seed=0)`; the binding also takes `confidence=0.999, max_iterations=10_000, min_inliers=4, local_optimization=true`, all keyword-only (`py/homography_estimation.rs:71-81`).
  - `camera_model` and `columns` are result fields but appear only in prose, not the Output table (`:71-90`). `parallax_poverty` is `0.0`, not absent, when no pair reaches 16 F inliers (`:884-888`) — undocumented.
  - The "Camera-Model Columns" section carries its *own* second `**Status:** Implemented (2026-08-08)` stamp with validation numbers (`:203-218`) — a change-log paragraph embedded mid-spec.
**Third copies:** Three. `focal_vote.rs:4-43` (40 lines) re-derives the Overview, the degeneracy argument, log-median rationale and bimodality rule sentence for sentence; `column_scan.rs:44-112` restates each constant's spec rationale verbatim; `py:85-160` (72 lines) re-describes the whole result dict. The theory should stay in the spec; **the Output table and Binding section should point at the pyfunction docstring** rather than duplicating it.
**Recommendation:** update spec — rewrite "Pair tables" to the exhaustive pass and delete the deviation note; purpose opening; add one `rust` fence with `FocalVoteOptions` + a `focal_vote_with_options` call, and one Python one-liner.
**Unclear / incorrect / suspicious:** `PairAccum`'s doc (`:425`) still says "from the sampled pass: how many clusters sampled this pair" — stale against its own field, which is a true covisibility count. `focal_vote.rs:765` hardcodes `16.0_f64.max(0.8 * n_f)` rather than naming `RATIO_MIN_F_INLIERS` / `ROTATION_DOMINATION_FRAC` — a code nit.
> _Status (2026-09-03): Done — both code items this section names are fixed, commit 7d4d3a4. `PairAccum`'s doc (`:425`) no longer says "sampled pass": it describes the exhaustive per-cluster pass, and a count that is the pair's true shared-cluster covisibility. The in-body comment at `:647` pointed at the deviation note #351 deleted and now names the spec's "Pair tables" section, which carries the same argument. Separately the module doc `:4-43` is 40 lines to 23 (c11c885), keeping the log-focal median convention and the column and determinism contracts. `column_scan.rs:44-112`, the 72-line binding docstring, and the `16.0_f64.max(0.8 * n_f)` code nit are untouched._
>
> _Status (2026-09-03): Done for the openings — this section had no drafted replacement, but `## Overview` opened on the bare symbol `focal_vote`, which the template's opening rule disallows, and now reads "The focal vote estimates a shared focal length from cluster-track observations before any reconstruction exists." The top Status block is gone and the pointer (`focal_vote.rs`, `focal_vote/column_scan.rs`, `homography_estimation.rs`) leads `## Binding`. **The mid-spec `**Status:** Implemented (2026-08-08)` stamp this section flags at `:203-218` is gone too**: its validation numbers are kept as a present-tense paragraph, moved to sit after the paragraph that defines what a column is rather than before it, and "the prototype's data-derived values" becomes "the data-derived values from those captures". Commit 1883d2d. F2/F3 — still no `rust` fence, `FocalVoteOptions` still unnamed, still no example call — remain open._

### specs/core/geometry/pose-verification.md
**Summary:** Finds cameras in a finished reconstruction whose poses are wrong and repairs them, using only 2D tracks. Substantively correct and well argued, but its two query signatures are stale, seven of nine tunables are undocumented, and its 135 lines are shadowed by a 45-line module doc.
**Implementing code:** `geometry/pose_verification.rs` (`verify_poses :387`, `repair_poses :520`, `VerifyOptions :76`, `RepairOptions :159`, `measured_relative_rotation :342`); substrate `features/cluster_match/covisibility/displacement.rs` (`nearest :255`, `farthest :262`).
**Inconsistencies:**
  - F1/F5: opening is a status line built from a symbol and three links, followed by a 19-line numbered errata block (`:15-33`). Proposed: "Find the cameras in a finished reconstruction whose poses are wrong, and put them back — using only the 2D tracks, with no reference solve, image ordering, or motion model to check against."
  - Signatures wrong: spec writes `nearest(i, k)` / `farthest(i, k)` (`:56-59`); actual is `nearest(&self, i: u32, k: usize, min_shared: u32)` (`displacement.rs:255,262`). Errata note (5) *admits* the missing parameter rather than the body being fixed.
    > _Status (2026-09-03): Partially done — the body now reads `nearest(i, k, min_shared)` / `farthest(i, k, min_shared)`, commit 7ddbe97. Errata note (5) is left in place for the Top priority 3 folding pass, along with the `displacement.rs` path drift and the undocumented defaults._
    > _Status (2026-09-03): Done — the errata block is folded and deleted. Its five notes are body prose: the Substrate section explains why the shared count is per-cluster deduplicated while the mean displacement is exhaustive over cross-image member pairs (and that the two coincide when clusters hold one member per image), and that `to_arrays`/`from_arrays` round-trip the neighborhood alone; Screen A states that it tests support rather than agreement with the stored pose, which is what makes it complementary to Screen B; Screen B names the whole-sign polar factor and the `S = diag(1, −1, −1)` conjugation the note and `pose_verification.rs:365-368` both carry; Repair states the ascending-image-order walk with accepted repairs feeding later inits. The `displacement.rs` path is corrected, and a Parameters table publishes all sixteen defaults — the twelve `VerifyOptions`/`RepairOptions` fields plus `INLIER_PX`, `REFINE_TRIM_ROUNDS`, `REFINE_KEEP_FRACTION` and `REPAIR_INIT_NEIGHBORS` — with the note's calibration guidance as the paragraph under it. The `verify_poses` signature, the example call and the "Testing requirements" phrasing remain open._
  - `verify_poses`'s 10-argument signature and the two option structs are never shown; no example call in either language.
  - Undocumented defaults: `resect_min_obs = 8`, `resect_accept_gate = 0.30`, `max_neighbors = 4`, `min_pair_correspondences = 30`, `min_h_inliers = 20`, `min_rotation_measurements = 2` (`:106-117`), `RepairOptions::min_obs = 12` (`:194`), `INLIER_PX = 3.0`, `REFINE_TRIM_ROUNDS = 5`, `REFINE_KEEP_FRACTION = 0.6` (`:66-72`). Two are load-bearing: `resect_accept_gate` is what Screen A's "no acceptable consensus" means, and `min_rotation_measurements` is why Screen B abstains with `NaN`.
  - Path drift: spec:5 places `DisplacementNeighborhood` in `covisibility.rs`; it is in `covisibility/displacement.rs:43`. Spec:81 writes `R = K⁻¹HK` "conjugated to the canonical frame" without naming the conjugator; the code uses `S = diag(1, −1, −1)` on both sides (`:365-368`).
**Third copies:** The dominant finding. `pose_verification.rs:4-48` is a 45-line module doc covering Purpose, both screens, both load-bearing properties and Repair in the spec's own words. Exact matches: *"estimate the homography over the pair's shared-cluster correspondences,"*; *"rotation. The per-image score is the **median** angular discrepancy over"*; *"registered neighbours — chordal mean of their rotations, mean of their"*. A dozen more match at 0.76–0.99. A fourth copy is the 58-line binding docstring (`py:100-180`). **The module doc should shrink** to a two-line pointer plus what the code alone can say.
**Recommendation:** update spec — fold errata (1)–(5) into the body and delete the block, purpose opening, correct the `nearest`/`farthest` signatures and the `displacement.rs` path, table all twelve defaults, add one `verify_poses` call; then cut `pose_verification.rs:4-48`.
**Unclear / incorrect / suspicious:** "Testing requirements" (`:114-127`) is phrased as requirements ("construction cost linear in observations under the span cap") and **no test asserts that linearity claim**. The focal-vote spec's parallel section is present tense and is the better model.
> _Status (2026-09-03): Done for the "Third copies" item, this section's dominant finding — `pose_verification.rs:4-48` is 45 lines to 21, commit c11c885. What it keeps is what the spec does not: the canonical −Z frame and the world-to-camera pose convention, and — because those are what make it necessary — the `S = diag(1, −1, −1)` conjugation of Screen B's optical-frame `K⁻¹HK`. Both screens at length, both load-bearing properties and the repair acceptance rule are now the spec's alone, which states all of them after the errata folding. The 58-line binding docstring is deliberately left; nothing in it was found false._
>
> _Status (2026-09-03): Done for F1 — `## Purpose` opens with this section's proposed sentence, verbatim, commit 1883d2d. The 12-line Status block is gone: its dependency list is a short "builds on" clause with relative links, and its code pointer leads `## Inputs and outputs`. This section's path-drift finding needed no fix — by the time of this pass the Status line already named `covisibility/displacement.rs` correctly. The `verify_poses` signature, the missing example call and the "Testing requirements" phrasing remain open._

### specs/core/geometry/translation-averaging.md
**Summary:** Solves camera centres from pairwise translation directions, with the constellation as the form's own null space. Spec and code landed together in d99b19c, so the physics, gauges, null-space rule, rank tolerance and reweighting all match exactly. The gaps are interface-shaped.
**Implementing code:** `geometry/translation_averaging.rs` (`average_translations :209`, `direction_reading :345`, `spectrum :449`, `relative_lengths :752`, `orientation_reading :911`; `TranslationGraph :130`, `DepthRows :554`, `OrientationRays :860`, `AveragingCensus :79`); `sfmtool-py/src/geometry/translation_averaging.rs`.
**Inconsistencies:**
  - Spec:202 says "**Three** functions in the geometry module of the bindings" and then lists **four**; four are registered (`py:438-441`).
  - Defaults written as literal ellipses — `rounds=...` (`:207`), `rounds=..., min_tied=...` (`:217`) — instead of `IRLS_ROUNDS = 5` (`:52`) and `LENGTH_IRLS_ROUNDS = 8` (`:58`). `min_tied`'s 3 *is* given in prose (`:149-151`), so the spec is inconsistent with itself.
  - CG bounds undocumented: `CG_STEPS = 200` (`:64`), `CG_TOL = 1e-12` (`:69`). The five module constants the binding exports to Python (`py:442-446`) are never mentioned.
  - **No Rust interface at all**, in a spec for a Rust core module. `TranslationGraph`, `DepthRows`, `OrientationRays`, `AveragingCensus`, `TranslationAveraging`, `RelativeLengths`, `OrientationReading` appear nowhere.
  - `orientation_reading`'s dict under-described (`:187-190`): the census key is `pts`, not `points` (`py:429`), and `angw_per_obs` (`py:426`) is unmentioned. `margin_frac` is `|2·obs_frac − 1|` (`:697`) — the spec omits the doubling.
  - `rounds` must be ≥ 1 in the binding (`py:172`), while core `average_translations` accepts 0 (`:239`) and `relative_lengths` silently clamps with `rounds.max(1)` (`:826`). None of this three-way split is stated.
  - Spec:92 omits the fallback: `median_floor` returns 1.0 when the median is not positive (`:545-553`) — the case a graph of exact directions actually hits.
**Third copies:** `translation_averaging.rs:4-38` (~35 lines) re-derives the objective, the `P_ij` projector, the null-space-is-the-constellation argument, the colinear-path example and the loose-frame rule near-verbatim against spec:27-84 — and already carries `See specs/...` at `:38`. `py:110-124` is a third telling. **Shrink the core module doc** to one paragraph plus the link.
**Recommendation:** update spec — fix "Three functions", substitute real defaults for the `...`, add a Rust interface and one example call, name the orientation census keys.
**Unclear / incorrect / suspicious:** Spec:172-174 asserts batch triangulation returns "the minimum-norm point in the observable subspace where those rays are exactly parallel, so no point drops out of the vote" — a load-bearing claim this spec states but does not own; if `triangulate_batch` changes, the parallel-ray points silently vote. `spectrum` (`:449`) does a dense `SymmetricEigen` of a `3n × 3n` matrix inside the reweighting loop, so the solve is O(rounds · n³); the Determinism section discusses eigenvector sign but never the cost, and no frame-count ceiling is stated.
> _Status (2026-09-03): Done for the "Third copies" item — `translation_averaging.rs:4-38` is 35 lines to 18, commit c11c885. The objective, the `P_ij` projector, the two gauges and the null-space-is-the-constellation argument all go to the spec, which states them at length; what stays is what the module offers and why the two side operations sit beside the solve. `py:110-124` is untouched, and every finding against the spec itself remains open._

### specs/core/geometry/rotation-init.md
**Summary:** Far-field rotation initialization for captures whose parallax is too weak to seed any other way. All thirteen tuning constants still match. The problem is structural: two dated status notes occupy 30 of 142 lines between the title and "## Purpose", one still correcting the body below it.
**Implementing code:** `geometry/rotation_init.rs` (`rotation_init :637`, `build_pair_tables :171`, `build_edges :249`, `average_rotations :409`, `seed_baseline :474`, constants `:64-101`); `sfmtool-py/src/geometry/rotation_init.rs`.
**Inconsistencies:**
  - **Signature wrong.** Spec:116-118 shows `seed`, `min_images`, `max_images` as positional; the binding makes them keyword-only via `*` (`py:51`). A call written from the spec raises `TypeError`.
    > _Status (2026-09-03): Done — the published binding signature now carries the keyword-only `*` before `seed`, commit 7ddbe97. The two dated status notes and the remaining documentation gaps are left for Top priority 3._
    > _Status (2026-09-03): Done — both notes folded and deleted, and the two contradictions this section names are resolved in the code's favour. §1 opens with the in-kernel pair-table pass (`build_pair_tables`) and says why the kernel builds its own rather than reading the sampled `ClusterCovisibility` tables; §4 carries the 8 px trim gate and 10-survivor floor alongside the 12-point candidacy floor. Notes (1), (2) and the 2026-08-11 note are one paragraph in §4 plus a sentence in Output: the finishing adjustment holds the far field at infinity over a mask it builds itself as the deduplicated union of the component's validated edges' H-inliers, because a finite far cloud rewards baseline collapse into a panorama; the gauge is renormalized to a unit seed baseline afterwards; the mask is internal, and a caller reads the far clusters off the unit rows of `points`. The `**Status:**` header also listed covisibility selection as a dependency — `rotation_init.rs` imports nothing from it — and no longer does. `H_MAX_ERROR_PX`, the cluster-run input contract, the pose convention and the BA budget remain open._
  - **Body contradicts its own note.** Spec:65-66 says displacement tables come "from covisibility selection"; note (3) at `:24-29` says in-kernel over all covisible member pairs — and `build_pair_tables` (`:171-212`) agrees with the note. Same for note (4): the 8 px trim gate (`RESECT_MAX_ERROR_PX :88`) and 10-survivor floor (`RESECT_MIN_INLIERS :90`) exist only in the note, while §4 (`:99-104`) still reads as though 12 points is the whole gate.
  - `H_MAX_ERROR_PX = 3.0` (`:71`) is undocumented, yet it *defines* the far/near partition the whole method rests on.
  - Input contract missing: the binding requires `cluster_indexes` nondecreasing with each cluster a contiguous run (`py:28-29`), because `build_pair_tables` scans runs (`:180-186`). Unsorted input silently produces wrong tables.
  - Pose convention absent: the binding states world-to-camera in the canonical frame, camera along −Z (`py:24-25`), and the core doc adds the `S = diag(1,−1,−1)` conjugation at the pixel-frame boundary (`:39-41`). The spec's Output section says only "rotations (WXYZ) and translations".
  - `DEFAULT_MIN_IMAGES` / `DEFAULT_MAX_IMAGES` are `pub const` (`:99,101`) but the binding hardcodes 8/14 (`py:51`) and the spec states them a third time.
  - BA budget undocumented: `BA_MAX_ITERS = 60`, `BA_MIN_TRACK = 2`, `BA_MIN_OBS = 12` (`:92-96`); the spec says "full default schedule".
**Third copies:** `rotation_init.rs:4-44` (~40 lines) restates Purpose and all four Mechanism stages, the opening near-verbatim against spec:45-54. The binding doc (`py:14-48`) is a third copy. **Shrink the core module doc** — but it uniquely carries the −Z frame and the `S` conjugation, which must move *into* the spec rather than be deleted.
**Recommendation:** update spec — fold both notes into the present-tense body (fixing §1's table claim and §4's gates), correct the keyword-only signature, add the pose convention and the cluster-run input contract.
**Unclear / incorrect / suspicious:** Note (2) chains a *second* note ("see the 2026-08-11 status below"), so a reader must reconcile the body against two errata. Notes (1)–(2) are now largely reflected in the body, making the blocks more stale than the text they annotate. F1: proposed opening — "Far-field rotation initialization poses a first handful of cameras on captures whose parallax is too weak to seed any other way, by letting the distant, parallax-free correspondences fix the rotations first and the near ones supply the metric frame afterwards."
> _Status (2026-09-03): Partially done — `rotation_init.rs:4-44` is 41 lines to 20, commit c11c885. The four Mechanism stages go to the spec, which carries them in full after the folding pass, and the doc names the four private helpers in their running order instead. **The −Z frame convention and the `S = diag(1,−1,−1)` boundary conjugation stay in the doc in full**: this section asks that they move *into* the spec, and after #351 `rotation-init.md` contains neither string, so deleting them from the code would have lost them outright. That half of the finding is still open, with `H_MAX_ERROR_PX`, the cluster-run input contract, the pose convention and the BA budget._
>
> _Status (2026-09-03): Done for F1 — `## Purpose` opens with this section's proposed sentence, verbatim, and the paragraph that follows keeps its two-populations argument while dropping the trailing "it succeeds precisely on captures whose windowed parallax is too weak…", which the new opening now says. The Status line and its three-item dependency list are gone; the pointer plus a short "builds on" clause leads `## Binding`. Commit 1883d2d. The −Z frame and `S = diag(1,−1,−1)` conjugation still are not in this spec, so the code doc still carries them; that half of Top priority 5 stays open._
>
> _Status (2026-09-04): Done — the spec has a `## Frame convention` section
> between `## Inputs` and `## Mechanism`, where a caller reading the input
> contract meets it before any rotation is named. It states that every rotation in
> and out is canonical-frame (camera along `−Z`, `+Y` up, world-to-camera, the
> frame `.sfmr` stores), that `H = K R K⁻¹` is a pixel-frame relation and so
> `K⁻¹ H K` comes out optical (`+Y` down, looking along `+Z`), and that the two
> differ by `S = diag(1, −1, −1)` with `R_canonical = S · R_optical · S`, `S² = I`,
> applied once at the edge-building boundary — plus why the error is silent, since
> `S R S` is still a rotation of the same angle and shows up only as a mirrored
> reconstruction. §1 now says the stored `R_ij` is the polar-orthogonalized
> `K⁻¹ H K` conjugated by `S`, and links the section. Per the task, the module doc
> in `rotation_init.rs` is left at full length in this pass; it could now point at
> the spec instead. `H_MAX_ERROR_PX`, the cluster-run input contract and the BA
> budget are still undocumented._

### specs/core/patch/cluster-patch-refinement.md
**Summary:** Turns a `.matches` file's SIFT clusters into patch clusters. All five numbered steps are genuinely shipped and the kernel, binding, CLI and every `ClusterRefineParams` default match. But the document is still shaped as a work order — the most drifted spec found this run, and the highest-churn module in the repo since the last audit.
**Implementing code:** `patch/cluster_refine/{mod.rs:806, params.rs:96, kernels.rs, consistency.rs:174, prof.rs}`; `sfmtool-py/src/matching/cluster.rs:317`; `src/sfmtool/_commands/cluster_patches.py`, `src/sfmtool/_cluster_patches.py:42`.
**Inconsistencies:**
  - **`--patch-size` documented 8.0 (`:537`); shipped 12.0** (`_commands/cluster_patches.py:36`). The CLI spec has it right; conversely that spec's `--resolution` row says 15 against a shipped 25 — this spec has *that* one right. Each spec is wrong about the parameter the other gets right.
    > _Status (2026-09-03): Done — the usage line now reads `--patch-size 12.0`, and `cluster-patches-command.md`'s `--resolution` row now reads 25, commit 7ddbe97._
  - **`MATCHES_FORMAT_VERSION` documented 3, readers 1–3 (`:55`); actual 5**, with cluster_patches gated at ≥4 and ≥5 (`matches-format/src/types.rs:120`, `read.rs:98,111`). `cluster_refine/mod.rs:26` already says version 5.
    > _Status (2026-09-03): Done — the bullet now says `MATCHES_FORMAT_VERSION = 5`, readers accept 1–5 and reject anything newer, and it spells out all three cluster-section gates: clusters at ≥ 3, cluster-patch `p = A·x_ref + t` at ≥ 4, cluster-patch `S = W·S_ref` at ≥ 5 (`read.rs:74,98,111`). Commit 7ddbe97; the rest of this spec's rewrite stays with Top priority 3._
  - **Contradicted default 65 lines apart:** `:182` writes `PatchWindow::GaussianDisk { sigma: 15.0/4.0 }`; the status block at `:117-118` and `params.rs:104` both say 0.5.
  - Two dead artifacts cited as authoritative (§5c), making the §5 "cross-check with the prototype" item (`:598-601`) unrunnable. Stale AVX2 prescription at `:401` (`_mm256_i32gather_ps`), explicitly replaced by pair loads in the 2026-07-11 block at `:369-385`. §3 return keys (`:513-515`) omit `member_consistency_residual`, which the binding does set (`cluster.rs:457`).
  - **Reuse-map line refs stale, two badly:** `affine_core_map` / `sample_support_affine` cited at `view_selection.rs:314/385` are at **391/496**; `score_raw_against_reference` cited `:538` (twice) is at **669**; `ImageU8Pyramid::build` cited `remap.rs:282` is at **182**; `bilinear_geometry` cited `:385` is at **276** (cited 3×); `sample_bilinear_u8` cited `:338` is at **229**.
  - Migration-site list (`:102-106`) wrong at both ends: `_run.py`, `_db_populate.py`, `_densify.py` do **not** use `pairs_from_matches`; `feature_match/_merge.py:52` does and is unlisted.
  - **Work-order residue throughout:** `:90` "**Still open — derived pairs.**" and `:106` "These MUST be migrated in the same change that..." — both dead, contradicted by the shipped block at `:71-88` and the opening's own "nothing in this spec remains open"; `:137-138` "New module ... with `mod.rs`, `params.rs`, `kernels.rs`, `tests.rs`" (it also has `consistency.rs`, `consistency/`, `prof.rs`); `:341,344` "promote ... and share" / "or extract into `znorm.rs`" — neither happened; `:349-353` "Genuinely new code:"; `:404-412` imperative AVX2 plan; `:504-516` "factor that helper", "Validate lengths", "Register in `matching/mod.rs`" — all shipped per `:467-478`; `:533` "spec to live at `cluster-patches-command.md`" (it exists); `:576-597` test sketches in future tense.
  - **Settled question presented as open:** `:567-571` says `sfm match --cluster` "does not yet write cluster-bearing files"; `tests/patch/test_cluster_patches.py:42-43` shells out to exactly that. Still genuinely open: `:258-263` and `:270`.
  - No example call. F4 is otherwise fine — the Algorithm and Performance sections carry real invariants (determinism under thread schedule, the all-in-frame `+1.0` rule, f32 variance cancellation via centering).
**Third copies:** `cluster_refine/mod.rs:4-37` (34 doc lines) re-derives the algorithm, mip-level rule and the `S = W·A_ref` convention — that one earns its place as the module entry point. `params.rs:76-95,112-119` re-derives the Performance tuning rationale duplicating spec:436-451 and **should shrink** to the pointer it already names. The CLI help text (`_commands/cluster_patches.py:38-43,71-79`) is a third statement, alongside `cluster-patches-command.md:36-52`.
**Recommendation:** update spec — rewrite as present-tense description of what shipped: fold the seven dated blocks into the prose they correct, delete the reuse map / AVX2 plan / §3–§5 imperative bullets, refresh line refs and the format version, and fix `--patch-size` here and `--resolution` in the CLI spec. F1 proposed opening: "Cluster-patch refinement turns a `.matches` file's SIFT feature clusters into patch clusters: per cluster it picks a reference member and fits a photometrically vetted affine warp from the reference's patch onto every other member, so downstream stages read each member's absolute shape and refined position without a `.sift` lookup."
**Unclear / incorrect / suspicious:** `:14` calls a file that is not in the repo and has no git history "the behavioral reference" — either it was never committed (delete both citations and the cross-check item) or it lived outside version control and the spec should say so. Worth a decision rather than a silent carry: `:258-263` documents that the localizability gate's `0.35` threshold is **not** resolution-invariant (1,913 → 372 rejections as resolution goes 15 → 31) while the CLI ships `--resolution` as a freely tunable knob — the gate's strength moves with an unrelated option, recorded as a known defect with no owner.

> _Status (2026-09-03): Done — rewritten wholesale against the code, 616 lines to
> 525, present tense throughout. Every inconsistency above is closed. The seven
> dated blocks and the two under other labels are gone; so are the reuse map with
> its eight stale line references, the imperative AVX2 plan (the shipped
> fused-channel pair-load kernel is described as what it is), the §3–§5
> construction bullets, the `New module …` line, the "Still open — derived pairs"
> block and its wrong migration-site list, and the settled `sfm match --cluster`
> open question. The `sigma` contradiction resolves to `0.5`, the
> `MATCHES_FORMAT_VERSION` bullet is replaced by a pointer to
> `matches-file-format.md`, which owns the sections and the three version gates
> normatively, and the §3 return keys now list `member_consistency_residual`. The
> `--patch-size 12.0` and `--resolution 25` corrections of commit 7ddbe97 are
> preserved in a Parameters table checked against `params.rs` and the Click
> defaults, which also publishes three constants no spec carried before
> (`MIN_ABS_DET`, `SIGMA_CLAMP`, `LOCALIZABILITY_SIGMA_NOISE`) and the four
> cascade-tuning knobs the binding does not expose. The finding's two asks that
> were not mechanical are answered: F1's proposed opening is the new first
> paragraph in substance, and the spec now carries a worked Rust call and a worked
> Python call. The `0.35` resolution-dependence, which the finding wanted decided
> rather than silently carried, is stated as an open question with its
> dino_dog_toy numbers and the fix it implies (re-express the threshold in
> keypoint-frame or source px), not as a defect note buried in an algorithm step.
> `params.rs:76-95,112-119` still re-derives the tuning rationale and should still
> shrink to the pointer it names; that is a code edit and stays open._

### specs/core/patch/member-coherence-validation.md
**Summary:** Decides which cluster members genuinely image the same surface patch. Substantively accurate — every default, constant, verdict branch, tie-break and Python dict key checks out, and it is the **only sampled spec with no work-order residue at all**. Its problems are structural: a 683-line derivation with the API at 71%, no example call, and ~120 lines of its argument re-derived in `decide.rs`.
**Implementing code:** `patch/member_coherence.rs`; `member_coherence/decide.rs` (`decide_member_coherence`, `max_support_block`, `core_coherence`, `core_deficit`); `member_coherence/matrix.rs`; `sfmtool-py/src/patches/member_coherence.rs:168`.
**Inconsistencies:**
  - **Doc contradicts spec and code.** `member_coherence.rs:380-382` documents `retained_deficit` as the deficit "on the **coarsest available grid scale**". The code reads the *first* coarse table — `matrix.zncc_coarse.first()` (`decide.rs:360`, used at `:499`) — and spec:318 and `decide.rs:265` both say first / one-halving explicitly. At the default `resolution = 24` the two tables differ (12×12 vs 6×6), so this is substantive. The sibling `sharpness_deficit` (`:389`) is the one that reads `.last()` (`decide.rs:361`).
    > _Status (2026-09-03): Done — `retained_deficit`'s doc now says the *first* coarse grid scale (one halving), not the coarsest, and points at `MemberMatrix::zncc_coarse`, commit 7ddbe97. No behaviour change. Shrinking `decide.rs:182-304` and adding the spec link to `decide.rs:4-10` stay open._
  - No example call in either language — only signatures (`:549-568`, `:578-586`).
  - F2 judged: most teaching before line 483 *is* load-bearing (the frozen common support and keypoint anchoring both change what `bar` means), but the calibration narrative at `:431-479` — 49 lines of prototype-vs-native Spearman fits, per-dataset split counts, two out-of-reach exemplars — is derivation sitting between the Parameters table and the API. Moving it below would put the callable surface at ~63% with no loss.
  - The API sketch lists `MemberMatrix` fields in a different order from the struct — cosmetic, but a reader diffing them stumbles.
**Verified correct:** all 14 documented defaults and constants (`member_coherence.rs:116-160`, `normal_refine/params.rs:257-260`), the Python signature character for character, and all 19 dict keys.
**Third copies:** The corpus's worst. The five exactly-matching lines are all spec↔`decide.rs` (spec:203-204 = `:195-196`; spec:227 = `:213`; spec:241 = `:108`; spec:300 = `:250`; spec:308 = `:257`), and they are residue of a much larger overlap: `decide_member_coherence`'s doc (`decide.rs:182-304`, **123 lines**) is a near-complete restatement of spec §§201-352, with the same calibration numbers and the same four-step tightening list. **The doc comment should shrink** to the mechanism (~35 lines) plus the spec link. `decide.rs:4-10` is the only module doc here that does *not* point at the spec — `member_coherence.rs:6` and `matrix.rs` both do — likely why it grew into a standalone essay.
**Recommendation:** update code — fix `member_coherence.rs:381` to "first (finest) coarse grid scale", shrink `decide.rs:182-304`, add the spec link to `decide.rs:4-10`. Optionally move spec:431-479 below the API and add a three-line worked call.
**Unclear / incorrect / suspicious:** none.
> _Status (2026-09-03): Done for both code items — `decide_member_coherence`'s doc is 123 lines to 46, and `decide.rs:4-10` gains the spec link it was the only module doc in this directory to lack. Commit c11c885. What survives is the mechanism as six numbered steps in the order the code runs them, with the `effective_bar` / `effective_margin_gate` formulas, the once-only tightening, the inert conditions for each term, and the scored-members rule; the calibration numbers, the circularity argument, the exoneration asymmetry and the one-halving justification are the spec's, which states all of them. The optional spec edits this section suggests — moving `:431-479` below the API, adding a worked call — are untouched._

### specs/core/analysis/keypoint-reach.md
**Summary:** Per-image enumeration of which keypoints fall inside another keypoint's own reach disk. Faithful on semantics — directedness, the self pair, non-finite reach, the sorted-column run, batch invariance. But it documents a Python binding for a crate whose Rust surface it never names, and two of its sections describe callers that do not exist.
**Implementing code:** `crates/sfmtool-core/src/spatial/keypoint_reach.rs` (`pairs_within_reach :128`, `pairs_within_reach_batch :140`, `KeypointRows :43`, `ReachPairs :95`, `KeypointReachError :54`, `BATCH_ROWS = 256 :39`, `cmp_nan_last :292`); `sfmtool-py/src/analysis/keypoint_reach.rs:55`.
**Inconsistencies:**
  - **F2/F3, the headline.** No Rust interface at all. `pairs_within_reach(rows: KeypointRows) -> Result<ReachPairs, KeypointReachError>`, the `KeypointRows` input shape (note `xy_px` is a **flat `2n` slice**, not `(n,2)`), `ReachPairs`'s three parallel vectors, the two error variants, `pairs_within_reach_batch` and `BATCH_ROWS` are all public and none appear. A Rust caller cannot find out what to call; only a Python caller can. No example call in either language.
  - **Present-tense claims about code that does not exist.** §"What consumes it" (`:68-80`) describes two consumers — coarse-observation retirement and same-measurement reconciliation — in the present tense, and §Testing claims "**Caller parity.** The two consuming rules reproduce their reference masks byte for byte." **Verified: `pairs_within_reach` / `keypoint_pairs_within_reach` has zero callers outside its own tests**, and no such parity test exists. 600b52e's own message confirms the callers "were each expanding for themselves in NumPy" and were not switched over. §Mechanism `:63-64` leans on the same absent callers to justify intra-image parallelism.
  - Contract gap: §Mechanism `:60-62` gives the order as "rows in their given order" and never says the stream is **grouped by image in ascending image index** — part of the tested contract (`:120`, `tests.rs:178`).
  - Uncovered but present: the comment at `:276-278` explaining why the distance is written out rather than fused ("an FMA would round the sum once instead of twice") — a genuine numerical hazard tied to the byte-identical parity the spec promises, and exactly what F4 wants in the spec.
  - F1 passes. Nit only: "track set" is repo vocabulary used one sentence before it is defined.
**Verified correct:** directedness and self-exclusion, non-finite-reach skipping, per-row rejection of negative reach, NaN-sorts-last, and inclusive `d <= reach` (`keypoint_reach.rs:153-155,263-292`).
**Third copies:** Mild and healthy. `keypoint_reach.rs:4-27` (24 lines) restates the opening, directedness and mechanism but ends with the spec pointer and stops short of the rationale. The PyO3 doc (`py:20-52`) restates the contract for `help()`, which is its job. **No copy needs to shrink; the spec needs to grow.**
**Recommendation:** update spec — add the Rust interface with `KeypointRows` / `ReachPairs` / `pairs_within_reach` and a two-line call, state the image-ascending output order, fold in the no-FMA note, and demote §"What consumes it" and the "Caller parity" bullet to what they are (the rules this was extracted from, still on their NumPy expansions) until the callers are migrated.
**Unclear / incorrect / suspicious:** The spec sits in `specs/core/analysis/` while the code sits in `crates/sfmtool-core/src/spatial/` — deliberate (the source points back, and the README row is filed there), but it is the only `core/` spec whose directory does not mirror its module.
> _Status (2026-09-03): Done for the demotion — re-verified first: `pairs_within_reach` and `keypoint_pairs_within_reach` still have no caller outside `spatial/keypoint_reach/tests.rs`, the binding's own registration, and `tests/rust_bindings/test_keypoint_reach_rust_bindings.py`. "What consumes it" now says exactly that and names those as the only callers; the two rules move to a new `Open questions` item, written as a migration that could happen and carrying the byte-parity such a migration would have to establish rather than claiming it as tested. The "Caller parity" testing bullet goes with them, and Mechanism no longer justifies intra-image parallelism by what "the callers" do. Commit 93eb19d. The rest of this section — the missing Rust interface, the unstated image-ascending output order, the no-FMA note — is open. `keypoint_reach.rs:4-27` was already correctly sized and is untouched._

---

## Code without specs

| surface | user-facing? | spec |
|---|---|---|
| 27 CLI subcommands | yes | all covered (`explorer` → `specs/gui/`, `ws` → `ws-init-command.md`) |
| `sift-format`, `matches-format`, `sfmr-format`, `camrig-format` | yes | `specs/formats/*-file-format.md` |
| **`sfmtool-archive-io`** | internal, load-bearing | **none** — one passing mention in `specs/formats/README.md:4` |
| `sfmr-colmap` | yes | covered by the three `colmap-interop` CLI specs |
| `sfm-explorer` | yes | 18 specs under `specs/gui/` |
| `sfmtool-py` | yes | bindings documented inside each `core/` spec |
| `sfmtool-core` — 7 module groups | internal | 59 specs under `specs/core/` |
| `geometry/homography_estimation` | internal | inside `focal-vote.md`; module doc links it |
| `geometry/convention` | internal | inside `sfmr-file-format.md`; module doc links it |
| **`geometry/pose_refine`** (233 L) | internal, load-bearing | **none, and no spec link in the module doc** |
| `analysis/point_inspect` | yes (`sfm inspect pt3d_*`) | `cli/reconstruction/inspect-command.md` |
| `reconstruction/data` (1,212 non-test L) | internal | none; `SfmrReconstruction` referenced from 10+ specs |
| `camera/distortion` (2,260 non-test L) | internal | `bspline`/`kernels`/`ray_grid`/`pinhole_fit` all covered |

Only three entries are worth arguing about.

### crates/sfmtool-archive-io
**What it does:** Owns the ZIP + zstd container primitives that all four on-disk
formats share — entry read/write, XXH128 section hashing over uncompressed bytes,
raw-slice reinterpretation. 263 lines, 12 public items.
**Why it matters:** Internal but load-bearing, and it is the *only* thing standing
between four independently-specced formats and four divergent containers. The
duplication shows: `sift-file-format.md`, `matches-file-format.md` and
`sfmr-file-format.md` each carry ~80 lines mentioning zstd / XXH128 / ZIP STORE,
`camrig-file-format.md` 31 — four descriptions of one contract, cross-linked
nowhere. The crate's own module doc (`lib.rs:1-9`) states the shared contract
better than any of them.
**Recommendation:** write a spec at `specs/formats/archive-container.md` carrying
the container contract once, and cut the four format specs back to their own
schema, validation and error type — which is exactly the split the code already
implements.

### crates/sfmtool-core/src/geometry/pose_refine.rs
**What it does:** Trimmed pose-only resection refinement — the robust companion to
the minimal `absolute_pose` estimator, refitting L2 on the best-fitting fraction
each round under Levenberg–Marquardt with an analytic Jacobian.
**Why it matters:** Internal but load-bearing, and it is reached through
`absolute-pose.md`, whose own text about it (`:191-207`) is re-derived in
`pose_refine.rs:6-19`. It is the only sampled module with **no** `specs/` link in
its doc comment.
**Recommendation:** add a note to `specs/core/geometry/absolute-pose.md` — it
already carries the rationale; the module doc should link there and shrink.

### crates/sfmtool-core/src/reconstruction/data
**What it does:** The in-memory `.sfmr` model — `SfmrReconstruction` plus
`conversion` (the format boundary), `recompute` (reprojection errors, depth
statistics) and `demo`. 1,212 non-test lines, 19 commits since the last audit.
**Why it matters:** Referenced by name from more than ten specs, none of which
owns it; its on-disk counterpart is specced but the in-memory boundary is not.
**Recommendation:** acceptable as unspecced — the module doc (`data.rs:4-14`) is
accurate and well-shaped. A one-line pointer from `specs/formats/sfmr-file-format.md`
naming `SfmrReconstruction` as the in-memory boundary would close the gap without
a new document.

---

## Top priorities

1. **Fix `fundamental_8pt`'s rank guard — a code defect against a documented
   contract.** `specs/core/geometry/epipolar-estimation.md:115` promises `None`
   for a rank-deficient design; the guard tests the *largest* eigenvalue against
   zero (`epipolar_estimation.rs:226`), so only an all-zero design is rejected and
   a degenerate 8-point configuration returns an arbitrary null direction that
   propagates into `estimate_fundamental`'s consensus. The sibling
   `fundamental_7pt:165` already has the correct relative test
   (`eigenvalues[idx[2]] <= NULLSPACE_RANK_EPS * lmax`) — mirror it. The
   misleading comment at `:224-225` goes with it.

> _Status (2026-08-30): Done — the guard now requires a 1-D null space
> (`eigenvalues[idx[1]] > NULLSPACE_RANK_EPS * lmax`), mirroring the 7-point
> solver and sharing its constant; the misleading comment is replaced.
> `NULLSPACE_RANK_EPS`'s doc and the spec's refit section now name the
> degenerate configurations and the measured margins. Three tests added:
> `eight_point_rejects_rank_deficient_designs` (coplanar structure, zero
> baseline, <8 distinct points, collinear points),
> `eight_point_minimal_designs_are_almost_always_accepted`, and
> `estimator_survives_a_dominant_plane`._
>
> _Two things the exploration established that the finding did not. First, the
> impact is on the **public function**, not on `estimate_fundamental`: measured
> across dominant-plane sweeps, estimator output is bit-identical either side of
> the fix, because `fundamental_7pt`'s guard already rejects the all-coplanar
> minimal samples before a coplanar consensus can form. The severity is that a
> rank-deficient refit is undetectable downstream — run in isolation on a
> coplanar inlier set it scored 76 of 80 inliers at machine precision while
> sitting 2-4% from the true `F` and voting focals of 215-1274 against a true
> 700. Second, the shared `1e-9` threshold is conservative at exactly `N = 8`,
> rejecting ~3 in 400 general-position minimal designs whose margin lands at
> `1e-10`-`1e-13`; that is the intended trade (local optimization keeps the
> minimal solution) and is now pinned by a test._
>
> _Separate pre-existing issue found while probing this one, **not** caused or
> cured by the fix: from about a 0.85 plane fraction, `estimate_fundamental`'s
> adaptive termination settles on a plane-only consensus and stops early —
> worst measured `7e-4` in `F` and `411px` in focal, identical before and after.
> The stopping rule does not account for the drawn sample's spread. Carried
> forward as an open item; see the note in `estimator_survives_a_dominant_plane`._

2. **Correct the three wrong published defaults and the three wrong published
   signatures**, all of which mislead a user at the point of use:
   `cluster-patches-command.md:31` `--resolution` 15 → **25**;
   `cluster-patch-refinement.md:537` `--patch-size` 8.0 → **12.0** and `:55`
   `MATCHES_FORMAT_VERSION` 3 → **5**; `rotation-init.md:116` (missing the
   keyword-only `*`, so a copied call raises `TypeError`);
   `cluster-covisibility.md:183` and `covisibility-selection.md:30` (`from_arrays`
   missing `member_accepted`, so a positional call silently puts `positions_xy`
   in the mask slot). Also fix `member_coherence.rs:381`'s doc, which says
   "coarsest available grid scale" where the code reads `.first()` — a real
   behavioural miswording, not a typo.

> _Status (2026-09-03): Done — all six corrected, plus one the finding did not
> list. `cluster-patches-command.md`'s `--resolution` row is 25;
> `cluster-patch-refinement.md`'s usage line is `--patch-size 12.0`;
> `rotation-init.md`'s binding signature carries the keyword-only `*` before
> `seed`; `cluster-covisibility.md` and `covisibility-selection.md` both publish
> `from_arrays(cluster_starts, member_images, num_images, member_accepted=None,
> positions_xy=None, seed=0)`, and `covisibility-selection.md`'s 2026-07-18
> errata block — whose note (1) admitted the body's signature was wrong — is
> deleted, its content folded into the Construction section it corrected;
> `member_coherence.rs`'s `retained_deficit` doc now says the *first* coarse
> grid scale (one halving), not the coarsest. The seventh:
> `pose-verification.md`'s body now reads `nearest(i, k, min_shared)` /
> `farthest(i, k, min_shared)`. No behaviour change. Commit 7ddbe97._
>
> _Three corrections to the finding's own wording, established while verifying
> against the code. First, §5a attributes the `nearest`/`farthest` `min_shared`
> admission to `covisibility-selection.md`; that admission is note (5) of
> `pose-verification.md`, which is also where the two stale signatures live —
> `covisibility-selection.md` never mentions either function (the per-spec
> section has this right, only the §5a bullet does not). Second, the §1 table
> calls `member_accepted` "third positional"; it is the fourth parameter, after
> `cluster_starts`, `member_images` and `num_images`. Third,
> `MATCHES_FORMAT_VERSION` has **three** cluster-section gates, not two: a file
> below version 3 may not claim `clusters/` or `cluster_patches/` at all
> (`read.rs:74`), on top of the ≥ 4 and ≥ 5 cluster-patch gates — and readers
> also reject anything above 5 (`read.rs:65`). All three are now in the spec._

3. **Retire the 29 dated errata blocks across 12 specs by folding them into the
   prose they correct.** Every spec read this run that had one showed the note and
   the body in active disagreement, with no signal to a reader which wins —
   `focal-vote.md` (sampled vs exhaustive pair enumeration), `rotation-init.md`
   (table provenance and the resection gates), `cluster-patch-refinement.md`
   (`sigma` 15/4 vs 0.5, 65 lines apart), `covisibility-selection.md` (a note that
   admits the body's signatures are wrong rather than fixing them). This is the
   single highest-yield editing pass available, and `cluster-patch-refinement.md`
   — seven blocks, the repo's highest-churn module, plus dead artifact citations
   and eight stale line references — should be rewritten wholesale rather than
   patched.

> _Status (2026-09-03): Partially done — 32 blocks across 14 specs are folded
> into the prose they corrected and deleted. `cluster-patch-refinement.md`'s
> seven remain, deliberately: they go with the wholesale rewrite this finding
> asks for, as a separate change._
>
> _Folded here: `focal-vote.md`, `rotation-init.md` (2), `pose-verification.md`,
> `reconstruction-growth.md`, `absolute-pose.md` (2), `epipolar-estimation.md`
> (2), `randomized-kdtree-forest.md` (2), `patch-normal-refinement.md` (5),
> `patch-cloud.md` (3), `fronto-parallel-patch-cache.md` (2),
> `cluster-patches.md`, `camera-intrinsics.md` (7),
> `refine-keypoints-command.md`, `refine-normals-command.md` (2).
> `covisibility-selection.md`'s single block went earlier, in commit 7ddbe97.
> In each case the code was read first and the body rewritten to what the code
> does, with the note's rationale kept where it was load-bearing and phrased as
> a reason rather than as a change. Three sections that existed only to host a
> note went with it: `refine-keypoints-command.md`'s `## Future`,
> `randomized-kdtree-forest.md`'s `## Phasing`, and
> `fronto-parallel-patch-cache.md`'s `## Implementation plan (production)` with
> its Phase 1/Phase 2 headings (finding §5b), which is now a present-tense
> `## Implementation`. `cluster-patches.md`'s settled Open question went too._
>
> _The count is larger than the finding's because §5a's grep keyed on the word
> "Status": ten more blocks of the identical shape are spelled `Deviation`,
> `Note` or `Correction`, including the `focal-vote.md` one this finding
> itself cites. The full inventory and what is left is under §5a below._
>
> _Status (2026-09-03): Done — `cluster-patch-refinement.md` is rewritten
> wholesale from the code, 616 lines to 525, and the seven blocks this finding
> held back are gone with it (plus two more of the same shape under `Addition`
> and `Revision` labels). The rewrite follows `TEMPLATE.md`'s shape — purpose,
> Rust API with its rationale and a worked call, theory, implementation notes
> that carry only what the code cannot — and is present tense throughout: no
> status blocks, no `New module`, no phase or migration language, no dead
> citations, and a Parameters table checked against `params.rs` and the Click
> defaults. Content the sibling specs own is now a pointer rather than a copy:
> the `.matches` sections and their version gates to `matches-file-format.md`,
> the motivation and warp-family calibration to `cluster-patches.md`, the
> localizability score to `patch-localizability.md`, the residual to
> `cluster-warp-consistency.md`. Two cross-references that the rewrite would
> otherwise have broken are fixed in place —
> `cluster-patches-command.md`'s "§1" citation and `cluster-patches.md`'s
> description of what this spec contains — and the `specs/core/patch/README.md`
> row now says what the document is. Per-spec detail under
> `specs/core/patch/cluster-patch-refinement.md` above; §5a, §5b and §5c carry
> the items this closes for them._

4. **Decide the `**Status:**`-line convention and write it into `specs/TEMPLATE.md`
   either way.** 61% of the corpus opens with one, in every case before any
   statement of purpose, against the template's one absolute rule. The template is
   currently silent, so neither the convention nor the rule is enforceable. The
   cheap resolution: sanction Status as a metadata line placed *after* the opening
   paragraph, and move the purpose sentence — which usually already exists a few
   lines down — to the top. Replacement first sentences for the ten sampled
   failures are in the per-spec sections above.

> _Status (2026-09-03): Done — decided the other way, and written into
> `specs/TEMPLATE.md`, `specs/README.md` and `AGENTS.md`. **A standing spec —
> anything under `specs/` outside `specs/drafts/` — carries no `**Status:**` line
> at all.** Location encodes lifecycle: a draft in `specs/drafts/` opens with
> `**Status:** Draft` and may use future tense freely; being filed *is* the
> status, so a filed spec describes what exists, in the present tense, purpose
> paragraph first. Filing a draft is three edits — delete the Status line, lift
> the purpose paragraph to the top, convert to present tense — and this gives an
> audit one mechanical check with no judgement in it: no lifecycle stamp outside
> `specs/drafts/`._
>
> _Two rules go with it. **Partial implementation is an amendment draft, never an
> inline marker**: the standing spec says in the present tense that X is not
> implemented and links the proposal in `specs/drafts/`; the draft opens by naming
> and linking the spec it amends; when it ships, its content folds in and the
> draft is deleted. A **non-goal** is distinguished from a draft — one
> present-tense sentence, no draft, no link ("does not support rolling shutter") —
> so the corpus does not sprout amendment drafts for things nobody means to build.
> And **the code pointer moves into the interface section**, as its lead sentence,
> with repo paths written as relative Markdown links so a moved file shows up as a
> broken link. Binding names stay code spans; they are not files. The template's
> one absolute rule also gains the enforcement it lacked: the opening paragraph is
> the first prose in the file, under the title and nothing else._
>
> _The sweep: **75 specs**, in commits 1883d2d (55 under `specs/core/`), 89c381d
> (19 under `specs/cli/`, `specs/gui/`, `specs/formats/`) and bcacb91
> (`patch-cloud.md`). Nine of the ten drafted replacement first sentences were
> used, six verbatim: `sfmtool-fisheye-kernels.md`, `cluster-covisibility.md`,
> `covisibility-selection.md`, `absolute-pose.md`, `epipolar-estimation.md`,
> `pose-verification.md` and `rotation-init.md`. `focal-vote.md` had no drafted
> sentence but its opening led with the bare symbol `focal_vote` and was rewritten;
> `cluster-patch-refinement.md`'s was already applied by the Top priority 3
> rewrite; `observation-coverage.md` and `keypoint-reach.md` needed none (§4
> already acquits them)._
>
> _**Eleven specs needed a purpose paragraph written from scratch**, because the
> Status line was the only description of what the thing was:
> `find-points-at-infinity.md`, `fronto-parallel-patch-cache.md`,
> `cluster-warp-consistency.md`, `keypoint-localization-search-cache.md`,
> `keypoint-subpixel-refinement.md`, `patch-localizability.md`,
> `patch-normal-refine-view-subset.md`, `ray-grid-projection.md`,
> `adaptive-clip-and-grid.md`, `image-animation.md`, `cross-panel-hover.md`.
> Another nine had an opening that described the change rather than the thing
> ("This document specifies **moving** the display controls…", "replacing the
> original single `CentralPanel`", "Callers … need three primitives that **today
> exist only as per-caller array code**") and were rewritten in place:
> `viewport-hud.md`, `multi-panel-image-browser.md`, `point-track-detail.md`,
> `scene-graph.md`, `goto-point.md`, `covisibility-selection.md`,
> `batch-triangulation-api.md`, `tile-batched-consensus-atlas.md`,
> `sfmtool-pinhole-kernels.md`._
>
> _**Four code pointers were wrong**, which is the argument for making them links:
> `ray-grid-projection.md` put `ray_to_pixel_grid` in `camera/distortion.rs` (it is
> in the `distortion/ray_grid.rs` submodule); `keypoint-localization-search-cache.md`
> put the AVX2 kernels in `patch/keypoint_localize.rs` (they are in
> `keypoint_localize/kernels.rs`); `patch-localizability.md` named
> `patch/localizability.rs`, which is a directory; `patch-normal-refinement.md`
> gave a crate-relative rather than repo-relative path. Three cited artifacts that
> have never existed went with the lines that carried them —
> `scripts/exp_pinhole_bootstrap.py` and `cluster-pinhole-bootstrap.md` (§5c's
> first item, now closed), and `scripts/seed_census.py` in `cluster-census.md`.
> `cluster-covisibility.md`'s and `affine-factorization.md`'s **second** citations
> of `exp_pinhole_bootstrap.py`, in their `## Validation` sections, are outside
> this edit and still open._
>
> _Link check: 696 relative `](…)` targets across all 138 specs resolve, the only
> two failures being `TEMPLATE.md`'s deliberately fictional `foo.rs` /
> `foo-gpu-amendment.md` examples. Two pre-existing broken links were fixed in
> passing: `find-points-at-infinity.md`'s `[v2 model]` reference definition and
> `patch-localizability.md`'s `_embed_patches.py`. `zensical.toml` publishes only
> `docs/`, so `docs-build` does not cover `specs/` and was not run._
>
> _**No amendment draft was created, and one was considered.** `patch-cloud.md`
> said `**Status:** Proposed` while everything it specifies ships — the only
> symbol that does not exist is `warp_maps_for_patch`, a four-line batch
> convenience the spec's own Open questions already deferred "until a caller needs
> it". That is a non-goal by the rule above, not an intention someone means to
> build, so the section states in the present tense that there is no batch helper
> and why, and `patch-keypoint-localization.md` stops citing the function
> (bcacb91). **Three genuine amendment-draft candidates surfaced and are left for
> a follow-up**, because each needs a decision this pass was not asked to make:
> `sift.md`'s `## Phasing` (Phase 2 GPU SIFT, and the on-disk incremental-extraction
> extension); `patch-rendering.md`'s `### Planned (v1)` flat-shaded fallback; and
> `sift-file-format.md`'s `**Version 2 (draft)**` chunked-descriptor layout. Under
> the new rule each is a present-tense Non-goals sentence plus a draft, not a
> heading inside the standing spec._
>
> _**The residue this pass deliberately did not touch**, and the next coherent
> piece of work: nine specs still carry an `## Implementation Status` /
> `## Implementation Plan` / `## Implementation phases` / `## Staging` /
> `## Phasing` section — a checkbox list or a numbered plan, usually with every
> item ticked or suffixed `— DONE` — which is the same convention violation the
> Status *line* was, one heading further down.
> `track-cluster-matching.md`, `camera-views.md`, `patch-rendering.md`,
> `plan.md`, `point-cloud-rendering.md`, `viewport-hud.md`,
> `viewport-navigation.md`, `image-animation.md`, `multi-panel-image-browser.md`
> (whose `### Phase B: Evaluation and Refinement — NOT STARTED` is the only one
> that is not merely stale), plus `camera-intrinsics.md`'s six `— *done.*` phases,
> `scene-graph.md`'s five, `keypoint-localization-consensus-basis.md`'s and
> `patch-normal-refine-view-subset.md`'s `## Plumbing` work orders and their
> `## Task-completion checks (from AGENTS.md)` sections, and
> `batch-triangulation-api.md`'s `## Prior state (before this change)` /
> `## Consumers & migration`. §5b names four of these; the real count is
> fifteen-plus. This is now the largest remaining work-order residue in the
> corpus._
>
> _Status (2026-09-04): Done — nineteen sections across sixteen specs, plus the
> ten remaining dated blockquotes and `specs/gui/plan.md`'s retirement. The
> per-spec detail, the five ticked-but-false claims found while verifying, and
> the four amendment drafts created are under Mechanical findings §5b above. The
> two rules this annotation added are what the drafts follow: an unbuilt part is
> one present-tense sentence in the standing spec linking a draft that names the
> spec back, and a non-goal is a fact with no draft and no link — which is what
> the fifteen new Non-goals entries are._
>
> _One correction to this paragraph's inventory: `scene-graph.md`'s five phases
> and `camera-intrinsics.md`'s six were the same construct as the checklists, and
> `image-warping.md` carried a nineteenth section (`## Implementation Order`) the
> word-based grep never had a chance of finding. A future grep wants
> `-iE '^#{2,4} .*(implementation|staging|phasing|plumbing|prior state|task-completion)'`
> rather than an enumeration of the three nouns seen so far._

5. **Shrink the module docs that re-derive their specs, and fix the one spec whose
   consumers are fiction.** Eight doc blocks totalling ~420 lines re-derive their
   spec's argument while already carrying a `See specs/...` pointer below it
   (`decide.rs:182-304` at 123 lines is the worst; then `bspline.rs`,
   `pose_verification.rs:4-48`, `focal_vote.rs:4-43`, `rotation_init.rs:4-44`,
   `translation_averaging.rs:4-38`, `absolute_pose.rs:4-32`,
   `observation_coverage.rs:6-18`). Each keeps what the code alone can say — frame
   conventions, the `S = diag(1,−1,−1)` conjugation, panic contracts, the no-FMA
   note — and drops the rest. Separately, `keypoint-reach.md` describes two
   consumers and a byte-parity test in the present tense that **do not exist**
   (verified: zero non-test callers); demote those sections until the NumPy
   callers are migrated.

> _Status (2026-09-03): Done — all eight blocks shrunk (commit c11c885) and
> `keypoint-reach.md` demoted (93eb19d), plus the Rust doc-comment follow-ons
> the §5a annotation had inventoried (7d4d3a4). Module-doc line counts before →
> after: `observation_coverage.rs` 17 → 8, `bspline.rs` 30 → 24,
> `pose_verification.rs` 44 → 21, `focal_vote.rs` 40 → 23, `rotation_init.rs`
> 41 → 20, `translation_averaging.rs` 35 → 18, `absolute_pose.rs` 29 → 17; and
> `decide_member_coherence`'s function doc 123 → 46, kept as the six-step
> mechanism in the order the code runs it. 420 lines of re-derivation → 177, and
> `decide.rs`'s module doc gains the spec link it was the only one in its
> directory to lack. No behaviour change; the rustdoc gate and the `sfmtool-core`
> doctests pass._
>
> _Three corrections to the finding's premise, found while doing it. First, not
> every block "already carried a `See specs/...` pointer": `bspline.rs` had none
> at all, nor did any other file in `camera/distortion/`, which is the likeliest
> reason its header grew — it now names `sfmtool-fisheye-kernels.md` for the
> basis and `formats/sfmtool-camera-models.md` for the gauge. Second,
> `rotation_init.rs`'s −Z frame and `S = diag(1,−1,−1)` conjugation are still
> **not** in `rotation-init.md` after the #351 folding pass, so the doc keeps
> them in full rather than pointing; the finding's "must move into the spec"
> remains open. Third, `bspline.rs`'s re-derivation is not confined to the module
> header — `bspline_is_monotone`'s doc re-derived the two-stage test, and is
> shrunk with it._
>
> _One item of the finding is deliberately untouched: the PyO3 binding
> docstrings (`sfmtool-py/src/geometry/focal_vote.rs`, `pose_verification.rs`)
> earn their length, since a REPL user cannot click into `specs/`. Nothing in
> them was found false._

---

## Notes on prior reports

`reports/2026-07-07-spec-audit.md` (798 lines, 85 per-spec sections) has all five
of its Top priorities and every honourable mention annotated Done. Its per-spec
sections remain the only reading of the 82 specs this run did not sample, so it
is **not** superseded and should be kept until those specs are re-read. Two of its
findings recur here in new places, which is worth noting for whoever acts on
this one: stale `Non-goals` lists overtaken by shipped modules, and CLI/spec
default drift. 14 specs carry a `Non-goals` section and 28 contain deferral
language ("a natural v2", "not yet implemented"); only `epipolar-estimation.md`'s
was checked this run, and it was stale.

**Methodology note for the next run:** key the defaults check on
(spec → owning command → parameter) rather than the bare parameter name. Two real
drifts hid behind name collisions this time.
