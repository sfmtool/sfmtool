# Intrinsics estimation over the focal vote

## Purpose

`estimate_intrinsics` is the high-level face of the structure-free focal
vote ([focal-vote.md](focal-vote.md)): it takes the same cluster-track
observation arrays the vote takes, runs the camera-model columns, and
returns one typed answer -- the model verdict, whether that verdict is
confirmed, the consensus focal, and the votes that actually belong to the
verdict. Callers that want a camera, not a diagnostic table, call this;
`focal_vote` remains the diagnostic layer underneath, returned intact for
callers that want everything.

It also owns WHEN the camera-model columns are worth running. Those columns
cost two self-consistency scans per candidate pair, which the closed-form
pinhole vote does not run at all, and a capture whose pinhole vote is strong
has nothing for the arbitration to overturn. `ColumnPolicy::Auto` therefore
screens on the pinhole-only vote and re-runs with both columns exactly when
that vote comes back weak, reporting which weak-vote reasons fired.

`estimate_intrinsics` itself does no I/O: it is pure compute over the
cluster-grouped observation arrays, so it serves the CLI command, the seed
pipeline, and tests identically. `estimate_intrinsics_from_matches` is the
same estimate over an already-parsed `.matches` file, which states those
observations in exactly that layout — so a caller holding a file makes one
call instead of taking the file apart, and the reading (the `f32 → f64`
widening, the shared-camera dimensions rule) has one implementation that
both languages reach.

## Rust interface

The estimator lives in
[estimate_intrinsics.rs](../../../crates/sfmtool-core/src/geometry/estimate_intrinsics.rs),
bound as `sfmtool._sfmtool.geometry.estimate_intrinsics`; the
`sfm estimate-intrinsics` command
([estimate-intrinsics-command.md](../../cli/reconstruction/estimate-intrinsics-command.md))
is its caller.

```rust
pub enum ColumnPolicy {
    /// Run `IntrinsicsOptions::vote`'s own `columns`, always.
    Fixed,
    /// Vote pinhole-only first; re-run with both columns only when that
    /// vote comes back weak. Auto picks both column sets itself, so
    /// `vote.columns` is not read under this policy.
    Auto,
}

pub enum EscalationReason {
    NoConsensus,         // "no_consensus"
    RotationRailed,      // "rotation_railed"
    FamilyDisagreement,  // "family_disagreement"
    ThinPool,            // "thin_pool"
}

pub struct IntrinsicsOptions {
    /// Passed through to the vote (seed, epipolar_min_disp_frac, and --
    /// under `Fixed` -- the column set).
    pub vote: FocalVoteOptions,
    /// Whether the column set is the one `vote` names, or one the
    /// estimator escalates its way to.
    pub columns: ColumnPolicy,
    /// Certified rotation-cell votes an equidistant verdict needs in the
    /// equidistant column before it counts as confirmed. The default 1 is
    /// the structural rule: a wrong ray map cannot fake a pure rotation of
    /// rays, so any certified rotation mass at all separates a real fisheye
    /// from an arbitration artifact.
    pub min_rotation_mass: usize,
}

pub struct IntrinsicsEstimate {
    /// The verdict model, `None` when no column produced one.
    pub camera_model: Option<CameraModel>,
    /// `Some(true|false)` for a Fisheye verdict; `None` when the question
    /// does not arise (Pinhole verdict, no verdict, or a single-column run).
    pub confirmed: Option<bool>,
    /// The winning column's consensus focal, in pixels.
    pub focal_px: Option<f64>,
    /// The winning column's certified scan votes -- the per-pair evidence
    /// behind THIS verdict, unlike the raw vote result's flat lists, which
    /// always belong to the pinhole closed-form kernel.
    pub verdict_votes: Vec<ScanVote>,
    /// Under `Auto`: the weak-vote reasons that fired, in check order.
    /// Empty means the pinhole vote stood and no second run happened.
    /// `None` under `Fixed`.
    pub escalation: Option<Vec<EscalationReason>>,
    /// The pinhole-only vote the escalation decision was read off, kept
    /// only when the estimate then re-ran with both columns -- the source
    /// of a caller's PINHOLE numbers, which the two-column `vote` no
    /// longer reports at the top level.
    pub screening_vote: Option<FocalVoteResult>,
    /// The full vote result behind the verdict, untouched.
    pub vote: FocalVoteResult,
    /// The camera the estimate implies, at the call's width and height with
    /// the image centre as its principal point. `None` without a consensus
    /// focal to build it from.
    pub camera: Option<CameraIntrinsics>,
}

pub fn escalation_reasons(
    vote: &FocalVoteResult,
    width: u32,
    height: u32,
) -> Vec<EscalationReason>

/// Cluster-grouped (CSR) observations: `cluster_starts` of `n_clusters + 1`
/// offsets (opening at 0, nondecreasing, closing at the member count), one
/// `member_images` entry and one `member_positions` entry per member, and
/// the shared image size whose centre is the principal point. See
/// focal-vote.md for the contract and its up-front validation.
pub fn estimate_intrinsics(
    cluster_starts: &[u32],
    member_images: &[u32],
    member_positions: &[[f32; 2]],
    width: u32,
    height: u32,
    options: &IntrinsicsOptions,
) -> IntrinsicsEstimate

/// The same estimate over a parsed `.matches` file: the cluster backbone's
/// CSR index and member images borrowed, its member positions widened
/// `f32 -> f64` once, and `(width, height)` off the image table. Errors
/// (`MatchesInputError`) only on a property of the file.
pub fn estimate_intrinsics_from_matches(
    matches: &MatchesData,
    options: &IntrinsicsOptions,
) -> Result<IntrinsicsEstimate, MatchesInputError>
```

Example call: the CLI's `--model auto` is
`estimate_intrinsics_from_matches(&matches, &IntrinsicsOptions { columns:
ColumnPolicy::Auto, ..Default::default() })` -- pinhole first, both columns
on a weak vote, structural confirmation -- and then reads `camera_model`,
`confirmed`, `focal_px` and `escalation` off the result instead of
re-deriving any of them from the vote dict. A caller whose observations are
already in hand as arrays calls `estimate_intrinsics` with the same options
and gets the same bits. A caller that wants both columns
unconditionally leaves `columns` at its `Fixed` default, whose `vote.columns`
is already the pair.

**The shared-camera rule.** The from-matches entry refuses a file whose
images do not all carry the same dimensions. The estimate is of ONE camera
with a centred principal point, so a file mixing resolutions is not one
estimate; checking it at the reading is what keeps every caller from
silently answering it from the first image's size.

## Verdict semantics

The estimate owns the interpretation the callers previously each carried:

- **Winning column.** `camera_model` and `focal_px` are the vote's own
  arbitration (greater certified mass of model-informative scan votes; the
  flat vote fields already follow it).
- **Confirmation.** An equidistant verdict is confirmed when the
  equidistant column carries at least `min_rotation_mass` certified
  rotation-cell votes. The default 1 is structural rather than a tuned
  threshold: measured over the fleet, false fisheye verdicts carry exactly
  zero certified rotation mass while true fisheyes carry 4 to 44, so the
  rule separates them with no band between. The floor is an option because
  a caller running a reduced cell set changes that geometry -- with the
  epipolar cell absent, a single rotation vote has been observed to confirm
  a false verdict -- and such a caller must raise the floor to what its own
  population supports.
- **Verdict votes.** `verdict_votes` is the winning column's certified scan
  votes. The raw result's top-level `epipolar_votes` / `rotation_votes`
  always describe the pinhole closed-form kernel regardless of the verdict;
  callers pairing those with the verdict's scalar fields pair numbers from
  two different columns, which is exactly the mistake this field exists to
  end.

An unconfirmed equidistant verdict is returned as-is (`camera_model` set,
`confirmed == Some(false)`); refusing to act on it is the caller's policy,
not the estimator's.

## The camera

`camera` is the verdict read out as the thing a caller actually solves with:
a `CameraIntrinsics` at the call's `(width, height)`, whose principal point is
the image centre `(width/2, height/2)` -- the same centre the vote itself
assumed, so the camera cannot disagree with the evidence behind it.

Which model it carries is the confirmation question, not the verdict alone:

- **A confirmed equidistant verdict** gives an `EquidistantFisheye` camera at
  `focal_px`, the verdict's own consensus.
- **Everything else** -- a pinhole verdict, an UNCONFIRMED equidistant one, no
  verdict at all -- gives a `SimplePinhole` camera at the raw pinhole vote
  focal. An unconfirmed equidistant verdict rests on nothing but the
  arbitration, so the camera stays on the chart the capture is still assumed to
  obey while `camera_model` and `confirmed` report what the arbitration said.

The raw pinhole vote focal is whichever of these the run holds it in, in order:
`screening_vote`, where an escalated `Auto` run keeps the pinhole-only vote;
otherwise `vote.focal_px`, when `vote` IS the pinhole answer (a pinhole-only
run, or a multi-column run whose verdict is `Pinhole`, the top-level focal
being the winning column's); otherwise the pinhole column's own consensus,
which is the only place that answer survives a multi-column run the pinhole
column lost. The three cases are one reading -- the capture's pinhole focal --
spread over the three shapes a result can take.

`camera` is `None` when the applicable focal is `None`. A confirmed equidistant
verdict with no focal yields no camera rather than a pinhole one: an
equidistant focal parameterizes `θ = r/f` and a pinhole focal `r = f·tan θ`,
and the two agree only near the axis, so substituting one for the other is a
units error rather than a fallback.

## The weak-vote escalation

`ColumnPolicy::Auto` runs the vote pinhole-only, reads `escalation_reasons`
off that result, and re-runs with both columns when any reason fired. The
disjunction is over the pinhole vote's own diagnostics -- four reasons,
checked and reported in this order:

- **`no_consensus`** -- `focal_px` is `None`. Fewer than the vote's
  `MIN_POOL` votes pooled, so there is no pinhole answer to weigh.
- **`rotation_railed`** -- the consensus came from the rotation family and
  that family's median sits within one grid step of the bottom of the
  rotation self-calibration's focal grid, `ORTHO_GRID_LO * max(w, h)`. The
  grid is 48 log-spaced points over `[0.3, 4.0] * max(w, h)`, so one step is
  `(4/0.3)^(1/47) = 1.0566`; an answer there is a scan that ran out of grid
  rather than one that found an interior minimum. That is what a fisheye
  capture looks like through a perspective chart: on the fleet, kerry at
  480 px lands exactly on the floor (ratio 1.000) while the nearest
  rectilinear capture sits 2.2 grid steps above it (1.153).
- **`family_disagreement`** -- the gap between the two families' medians
  exceeds the vote's own `FAMILY_DISAGREEMENT_BAND` (0.25 in log-focal).
  Two independent estimators past the kernel's own bimodality band are
  answering different questions about the same capture, and the reported
  consensus is one family's median with the other discarded.
- **`thin_pool`** -- `n_pool <= 9`. The one cut point without a kernel
  constant behind it: 9 is the tightest bar that still reaches every fisheye
  capture on the fleet (three of them pool exactly 9, and no other reason
  catches them); it is half the vote's `MAX_EPIPOLAR_PAIRS` budget of 18,
  and the rectilinear captures it additionally admits all pool 3 to 8, i.e.
  thinner still.

Over the fleet (42 captures at last measurement, 6 of them fisheye) the
disjunction fires on every fisheye capture and on 9 of the 36 rectilinear
ones, every one of those 9 a genuinely weak pinhole vote (7 tripped the
bimodality band, 5 pooled 8 votes or fewer). What it buys against always
running both columns is arbitration error: run unconditionally, 3 of those
36 rectilinear captures arbitrate to a fisheye verdict.

The screening vote is kept as `screening_vote` when the escalation fires,
because the escalated result's top-level fields report the WINNING column --
the fisheye answer whenever the escalation paid off -- while a caller's
pinhole consensus, pool and spread are the screening vote's. Its pinhole
column would carry the same consensus, but reading those numbers off a
column is exactly the hand-derivation this API exists to end.

The two runs share the closed-form pass, so an escalated estimate costs that
pass twice; a screened-out one costs the scans zero times, which is where the
policy pays for itself.

## Implementation notes

- The function composes `focal_vote_with_options` and reads its output; it
  holds no thresholds beyond `min_rotation_mass` and the escalation's four,
  and re-runs the vote only where `Auto` escalates. Any future change to
  arbitration lives in the vote, not here.
- `ORTHO_GRID_{LO,HI,N}` and `FAMILY_DISAGREEMENT_BAND` are read from
  `focal_vote` as `pub(crate)` constants rather than restated: two of the
  four cut points are questions about the vote's own grid and band, and a
  second copy would be free to drift from the machinery it describes.
- Determinism: same inputs and seed give bit-identical output, exactly as
  the vote guarantees; the estimate adds no randomness and no ordering of
  its own (`verdict_votes` preserves the column's stored vote order).
- The PyO3 binding returns the estimate as a dict with the vote dict nested
  under `"vote"`, so Python callers keep full diagnostic access without a
  second call. Its `columns` argument takes the string `"auto"` for
  `ColumnPolicy::Auto` and a sequence of column names for `Fixed`;
  `escalation` comes back as the reasons' string names, `screening_vote`
  as a nested vote dict, and `camera` as a
  `sfmtool._sfmtool.geometry.CameraIntrinsics` (or `None`).
- The binding takes the same two forms as the Rust surface and only these
  two: `estimate_intrinsics(matches_file, ...)` with a `MatchesFile` handle
  (a selection included), which forwards to `estimate_intrinsics_from_matches`,
  and `estimate_intrinsics(cluster_starts, member_images, member_positions,
  width, height, ...)`. Mixing them -- observation arrays alongside a
  `MatchesFile`, or the array form with an argument missing -- is a
  `ValueError`, and so is every `MatchesInputError`.
