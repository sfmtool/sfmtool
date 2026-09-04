# Intrinsics estimation over the focal vote

**Status:** Implemented in
`crates/sfmtool-core/src/geometry/estimate_intrinsics.rs` (tests in
`estimate_intrinsics/tests.rs`), bound as
`sfmtool._sfmtool.geometry.estimate_intrinsics`
(`crates/sfmtool-py/src/geometry/estimate_intrinsics.rs`, Python tests in
`tests/rust_bindings/test_estimate_intrinsics_rust_bindings.py`). The
`sfm estimate-intrinsics` command is its caller
([estimate-intrinsics-command.md](../../cli/reconstruction/estimate-intrinsics-command.md)).

## Purpose

`estimate_intrinsics` is the high-level face of the structure-free focal
vote ([focal-vote.md](focal-vote.md)): it takes the same cluster-track
observation arrays the vote takes, runs the two camera-model columns, and
returns one typed answer -- the model verdict, whether that verdict is
confirmed, the consensus focal, and the votes that actually belong to the
verdict. Callers that want a camera, not a diagnostic table, call this;
`focal_vote` remains the diagnostic layer underneath, returned intact for
callers that want everything.

The function does no I/O. Reading a `.matches` file, resolving keypoint
positions, and choosing which clusters to admit are the caller's job; the
API is pure compute over arrays, so it serves the CLI command, the seed
pipeline, and tests identically.

## Rust interface

```rust
pub struct IntrinsicsOptions {
    /// Passed through to the vote (seed, epipolar_min_disp_frac, ...).
    pub vote: FocalVoteOptions,
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
    /// The full vote result, untouched, for diagnostics.
    pub vote: FocalVoteResult,
}

pub fn estimate_intrinsics(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    options: &IntrinsicsOptions,
) -> IntrinsicsEstimate
```

Example call: the CLI's `--model auto` is
`estimate_intrinsics(ci, ii, xy, w, h, &IntrinsicsOptions::default())` --
both columns, structural confirmation -- and then reads `camera_model`,
`confirmed` and `focal_px` off the result instead of re-deriving them from
the vote dict.

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

## Implementation notes

- The function composes `focal_vote_with_options` and reads its output; it
  re-runs nothing and holds no thresholds beyond `min_rotation_mass`. Any
  future change to arbitration lives in the vote, not here.
- Determinism: same inputs and seed give bit-identical output, exactly as
  the vote guarantees; the estimate adds no randomness and no ordering of
  its own (`verdict_votes` preserves the column's stored vote order).
- The PyO3 binding returns the estimate as a dict with the vote dict nested
  under `"vote"`, so Python callers keep full diagnostic access without a
  second call.
