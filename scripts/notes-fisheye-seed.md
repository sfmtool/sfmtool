# Fisheye-native seed pipeline — incremental plan

Goal: a confirmed `fisheye_detected` capture gets a real equidistant seed
from this pipeline — not a pinhole solve that knows it is wrong, and not
a hand-off to an external solver. Every phase lands behind the
escalation verdict, so the pinhole fleet stays byte-inert at every step.

## Principles

1. **Escalation-gated.** The fisheye path executes only after the
   confirmed verdict (both-cells rule). Phase gates always include a
   36-dataset pinhole fleet A/B that must be byte-identical.
   > _Updated (2026-08-10, Phase 6): the verdict now ROUTES by default.
   > `SFMTOOL_FISHEYE_SEED` survives as a tri-state override — `"0"`
   > refuses routing (the pre-Phase-6 status quo), unset and `"1"` both
   > route. What no setting can do is route without a confirmed verdict,
   > so an unconfirmed one stays annotation-only exactly as before and
   > the escalation gate itself is unchanged. The pinhole fleet A/B is
   > unchanged as a gate and still passes byte-identically._
2. **Single-parameter equidistant in the seed.** The seed solves
   `θ = r/f` only, represented as `SimpleRadialFisheye` with `k1 = 0` —
   the exact `SimplePinhole` analog (one focal, centered principal
   point), and the model family mirrors the pinhole deferral ladder:
   `SimplePinhole → SimpleRadial` corresponds to
   `SimpleRadialFisheye(k1=0) → SimpleRadialFisheye(k1 free)`.
   Polynomial coefficients are deferred to downstream refinement,
   exactly as the pinhole seed defers radial distortion. The known
   consequence is a systematic ~6% gap against Kannala-Brandt
   calibrations (measured, prototype gate) — accepted at seed stage per
   the solid-starting-point acceptance bar.
3. **Ray-native, never virtual-pinhole.** The captures are >180° FOV;
   a virtual-pinhole remap discards the periphery, which carries the
   model information. Geometry stages consume unit rays; residual gates
   are angular in form with values derived from pixel tolerances through
   `dr/dθ` (the focal-vote spec's convention).
4. **Measured gates, in the established style.** Each phase has an
   acceptance measurement on the four fisheye fleet entries
   (KerryPark480/360, OmniCoast, OmniTemple1). From Phase 3 on, the two
   Kerry entries' rig focals (129.15 / 257.94) become commensurable with
   the solve — the fleet's `err%` column lights up for them and becomes
   the primary metric. Human Explorer inspection remains the GT
   arbiter.

## Phase 1 — Ray layer + core geometric primitives

The smallest stage-1 unit that can be fisheye-correct: a per-capture
camera context `(model, f)` threaded from the confirmed verdict, an
`obs → unit rays` map (equidistant inverse is closed-form), and
ray-native versions of the three primitives everything else composes:

- `triangulate` — ray-midpoint already; feed it fisheye rays instead of
  pinhole rays (audit: current implementation builds rays from `(u,v,f)`
  inline).
- `p3p_resect` — Lambda Twist is bearing-vector-native; check whether
  the native binding accepts rays directly or needs a variant that
  skips the pinhole unprojection. Same for `refine_absolute_pose`
  (trimmed iterations) — its residual must become angular.
- `pose_refine` / `reproj_res_one` — angular residuals under the
  camera context.

**Measured scope note (2026-08-09):** the scripts' geometric core is
already `CameraIntrinsics`-mediated — `make_cam` builds the camera
object and the pose stack (`estimate_absolute_pose`,
`refine_absolute_pose`, `reprojection_residuals`) plus the ray sites
(`pixel_to_ray_batch` / `ray_to_pixel_batch`) all consume it; pixel
thresholds stay valid because residuals are computed through the model.
Phase 1 therefore reduces to: (a) `make_cam` returns
`SimpleRadialFisheye(k1=0)` under the fisheye context, (b) an audit
that the native kernels behind those calls are model-generic
*including* θ ≥ 90° rays (a hidden z=1 normalization is the failure
class to look for), (c) a sweep for the few residual inline-`f` sites
(parts of `triangulate`'s ray build, `perspective_init`). The new
geometry work lives in Phases 2–3, not here.

**Kernel extensions:** possibly none; at most ray-accepting variants of
the absolute-pose bindings.
**Gate:** synthetic equidistant scene (planted poses/points) resects and
triangulates to numerical accuracy; pinhole fleet byte-inert (the ray
layer is a fisheye-branch-only code path).

### Phase 1 outcome (2026-08-09) — DONE

Landed: a `(model, focal)` camera context in both seed scripts, gated on
`SFMTOOL_FISHEYE_SEED=1` **and** the confirmed both-cells verdict, with
`make_cam` building `SIMPLE_RADIAL_FISHEYE { k1 = 0 }` at the verdict's
equidistant focal; `exp_fast_seed` hands the context to
`exp_pinhole_bootstrap` before finalization. The four script primitives
(`triangulate`, `p3p_resect`, `pose_refine`, `reproj_res_one`) recover
planted geometry to 1e-11 or better under that context on a synthetic
equidistant scene with 117 observations past 90° —
`scripts/check_fisheye_seed_primitives.py`. Pinhole byte-parity holds
(b100 / DinoDogToyWS / MurdoSmallAntiqueCat: 25/25 ZIP payload entries
identical, only the metadata timestamp and derived hashes move).

**Kernel audit — the ray layer needed no new kernels.** `pixel_to_ray` /
`ray_to_pixel` are exact `θ = r/f` for `k1 = 0` at every θ up to π
(`recover_theta_equidistant` is a no-op there, `blend_fisheye_ray` is an
identity blend because recovered == undistorted), Lambda Twist is
bearing-native with depth-along-the-ray positivity rather than a `z > 0`
test, and `refine_absolute_pose` / `reprojection_residuals` take their
domain test from `ray_to_pixel`. Rust tests pin all of it.

**One defect found and fixed:** `bundle_adjust`'s inter-round in-front
gate measured `−z_cam` against the `1e-3·f` floor, which discards EVERY
observation at θ ≥ 90° — the whole periphery a >180° capture is
supposed to contribute. The measure is now the range `‖p_cam‖` for
ray-path models (`needs_ray_path()`), `−z_cam` unchanged for the
perspective family.

**Assumptions Phase 2/3 should carry forward:**

- **`opt_f` is a hard error, not a silent degrade.** The binding raises
  `ValueError: opt_f requires a SIMPLE_PINHOLE camera` (the core's own
  `opt_f && matches!(SimplePinhole)` is what degrades silently). Every
  fisheye-context run therefore aborted at the focal release until the
  scripts' BA wrappers clamped `opt_f` to a fixed-focal solve under a
  non-pinhole context. Phase 3 replaces that clamp.
- **Phase 3's `opt_f` extension is smaller than assumed.** The focal
  enters every COLMAP model as a pure multiplier of the distorted
  normalized coordinate, so the kernel's existing `∂u/∂f = (u − cx)/f`
  is already EXACT for the whole equidistant family, `k1 = 0` or not —
  no central difference needed for the focal column. What remains is
  widening the `matches!` and re-deriving the fixed-focal gauge tests.
- **The pixel Jacobian falls back to a central difference.**
  `supports_pixel_jacobian()` is false for every ray-path model, so
  `pose_refine` and `bundle_adjust` central-difference `ray_to_pixel`.
  Verified stable at θ = 95°/105°/130° (all ±h probes stay in-domain),
  but it is ~4× the projections per linearization; if fisheye BA
  wall-clock becomes a Phase-3 gate item, an analytic equidistant
  Jacobian is the lever.
- **`ray_to_pixel_with_jacobian` rejects `rz ≤ 0` before dispatch.** It
  is unreachable for fisheye today (the model returns `None` from
  `distort_jacobian` anyway), but any analytic fisheye Jacobian must
  move that guard inside the perspective branch.
- **Domain edge:** a ray exactly at the antipode (θ = π, `r_xy = 0`)
  projects to the principal point, aliasing θ = 0. Measure-zero, 75°
  outside any real capture, pinned by a test — but a fisheye-native
  stage that synthesizes rays should not assume `ray_to_pixel` is
  injective over the full sphere.
- **Still pinhole below the primitives** (unchanged, and the reason a
  gated run is degraded rather than good): `perspective_init`'s
  `−f₀/scale` weak-perspective conversion, `pair_parallax`'s `K` /
  `E = KᵀFK` / `z = 1` normalization and `z > 0` cheirality,
  `depth_init`'s `rays · (z_pred / rays.z)` scaling, and the scan's
  `tvec · (f_try / f_probe)` rescale.

## Phase 2 — Seed probe and growth on ray-space two-view geometry

Stage 1's pinhole-specific estimators get ray-space equivalents, and
most of the machinery already exists in `column_scan.rs` (#282):

- Pair initialization: ray-space essential estimation (the epipolar
  cell's estimator) → relative pose by E decomposition on rays →
  triangulate → grow by ray-P3P (Phase 1). This replaces the
  covis-window affine factorization for the fisheye branch — affine
  factorization assumes locally-linear projection and is the piece
  least worth porting.
- Rotation-core hypothesis: the rotation cell's ray-rotation fit *is*
  the fisheye rotation core; reuse it.
- Widen ladder: unchanged in structure; its consensus gates move to
  angular form via the camera context.
- Parallax gate / seed-window selection: covisibility grouping is
  model-agnostic; parallax measurement moves to ray angles.

**Kernel extensions:** expose the column-scan pair estimator (E on rays
with consensus) through the binding for direct use, if not already
callable.
**Gate:** the four fisheye entries pose ≥ the pinhole path's 11–12
images with materially better inlier fractions (currently 0.22–0.57);
no regression in seed wall-clock beyond ~2×.

### Phase 2 outcome (2026-08-09) — DONE

Stage 1 has a fisheye-native path behind the same conjunction gate
(`SFMTOOL_FISHEYE_SEED=1` **and** the confirmed both-cells verdict), with
the focal held FIXED at the verdict's equidistant focal from the probe
through to the committed solve. The single test `fisheye_stage1()` — "is
the camera context non-pinhole" — guards every new branch, so no pinhole
run can reach any of it.

**Kernel export.** The column scan's two estimators, evaluated ONCE at a
known camera instead of scanned over candidate focals, now live in
`sfmtool_core::geometry::relative_pose` and are bound as
`estimate_essential_rays` / `fit_ray_rotation`. They share the column
scan's own sampling, consensus, local-optimization and residual code
(`epipolar_rows`, `epipolar_residuals`, `rotation_residuals`,
`null_from_rows`, `kabsch`, `draw_samples` widened to `pub(crate)`; the
row build factored out of `fit_epipolar` unchanged). The epipolar
residual gains a two-sided form — the max of the two one-sided angles —
because a caller estimating geometry wants both constraints, whereas the
scan's one-sided split exists to make the direction-agreement
certificate non-vacuous. Rust tests in `relative_pose/tests.rs`, Python in
`tests/rust_bindings/test_relative_pose_rust_bindings.py`.

**What replaced what, per work item.**

1. *Pair init.* `fisheye_window_seed` replaces `factorize_window` →
   `metric_upgrade` → `perspective_init` outright: rays for every
   covisible pair of the window, `estimate_essential_rays` at an angular
   bound derived from 3 px through `dr/dθ` (= `f` everywhere for the
   equidistant map), four-way decomposition, ray-native chirality (depth
   along the ray positive in both cameras — never `z > 0`), ray-midpoint
   triangulation through the Phase-1 `triangulate_batch`, then ray-P3P +
   pose refine across the rest of the window and a fixed-focal mini-BA.
   There is no reflection twin to grow: chirality picks one of the four
   decompositions, where the metric upgrade's mirror fit a near-affine
   window equally well.
2. *Rotation core.* `rotation_core_rays` replaces the native
   `rotation_init` under a fisheye context. The pinhole core reads its
   skeleton off far-field CONJUGATE HOMOGRAPHIES, which a fisheye map
   induces at no focal; the analog is the rotation cell's ray-rotation
   fit. Pairs a rotation explains are the far-field edges; a
   maximum-consensus spanning tree (with the epipolar gauge edge forced
   in, so skeleton and baseline agree by construction) chains absolute
   rotations, and translations grow by the native rotation-locked
   `resect_translation`, whose ray-space rows are model-agnostic.
3. *Parallax.* `ray_pair_parallax` is the ray form of `pair_parallax`;
   covisibility grouping is untouched. The affine `est_base_depth` upper
   gate is BYPASSED under fisheye (it measures distance from the
   weak-perspective validity band, a property of an estimator this branch
   does not run).
4. *Widen ladder / probe flow audit.* `widen`, `core_parallax`,
   `budget_mask`, `ba_rows`, `_grow_one`, `localize_anchors` and the
   `attempt` flow reach their camera only through `make_cam` and are
   context-clean as they stand — no inline pinhole math survives on the
   fisheye path. The Phase-1 inventory's remaining rows are answered:
   `perspective_init` **replaced** (item 1); `pair_parallax`'s
   `K` / `E = KᵀFK` / `z = 1` **replaced** (item 3);
   `verify_and_repair`'s `k_inv @ H @ k_m` **bypassed** — it is opt-in
   (`SFMTOOL_VERIFY=1`) and called only from
   `exp_pinhole_bootstrap.main()`'s growth stage, never from the seed's
   `finalize_seed_from_dict`; the ray-native replacement already exists
   (`fit_ray_rotation`) should a fisheye growth stage ever need it.
   > _Status (2026-08-12): Moot — `verify_and_repair` is gone. Its neighbour
   > lists came only from the `displacement-knn.npz` sidecar, which the
   > 2026-08-12 ablation measured as unread at defaults on all 41 fleet
   > entries and a net regression when force-enabled; sidecar and function
   > were removed together. `fit_ray_rotation` remains the ray-native
   > primitive should a fisheye growth stage ever want the screen back._
5. `depth_init`'s `rays · (z_pred / rays.z)` is **left with a guard
   comment** for the same reason: opt-in (`SFMTOOL_DEPTH_INIT=1`) and
   inside `grow_loop`, unreachable from the seed.

Two further fisheye-branch corrections fell out of the audit: the probe
focal is now the verdict's EQUIDISTANT focal (the pinhole vote and its
+10% bias correction parameterize a different map), and
`parallax_poverty` / `n_rotation` — which decide whether the rotation
core is tried at all — are read off the WINNING column instead of the
pinhole one.

**Stage-1 acceptance, gate ON, against the Phase-1 gate-ON baseline:**

| capture | posed before → after | inlier<2px before → after | seed wall-clock (pinhole path → fisheye path) |
| --- | --- | --- | --- |
| KerryPark480 | 14/48 → **43/48** | 61.2% → 51.3% | 2.9 s → 5.2 s (1.8×) |
| KerryPark360 | 14/82 → 12/82 | 41.5% → **62.5%** | 3.0 s → 2.5 s (0.8×) |
| OmniCoast | 10/88 → **11/88** | 81.3% → 58.0% | 9.8 s → 6.5 s (0.7×) |
| OmniTemple1 | 10/80 → **14/80** | 35.8% → **51.9%** | 8.2 s → 6.1 s (0.7×) |

Wall-clock is inside the ≤2× bar everywhere. Posed counts improve on
three captures. The inlier column moves both ways, and **it is not a
quality comparison between the arms**: the baseline arm ran a five-point
focal SCAN plus a release and reported the best of them, at focals the
equidistant column says are wrong by 15–120%, while this arm reports one
fixed-focal settle at the measured focal. A mis-modelled focal buys
consensus by shrinking the effective field — the pure-pinhole control
run (gate OFF, same captures) makes the same point from the other side:
OmniCoast scores 22.1% there against the mixed-model 81.3%.

The one capture with an independent fisheye reference settles it.
**KerryPark360 against `recon-infinity-gated.sfmr` (OPENCV_FISHEYE):**

| | baseline (gate ON, Phase 1) | Phase 2 |
| --- | --- | --- |
| camera rotation err | 21.63° mean / 34.81° max | **0.99° mean / 2.82° max** |
| camera centre err | 34.75% mean of subset diameter | **11.74%** |
| focal vs reference | +67.7% | **+6.9%** |
| posed subset span of the reference rig | 14% | **30%** |

The same reading holds on the seed windows themselves, which is the
cleanest apples-to-apples measurement available: on the identical
covisibility group, KerryPark360's probe scores 84.4% where the affine
init at the pinhole probe focal scored 28.3%, and OmniTemple1's 55.4%
against 25.7%. Both Kerry focals land +7.1% / +7.2% against the rig fx
(129.15 / 257.94) — the known Kannala-Brandt gap, and today a property
of the vote rather than of the solve.

**Pinhole byte-parity** holds by the true A/B method (fresh gate-OFF runs
at HEAD before and after the change, each backed up to its own
md5-manifest directory and compared entry by entry): b100 /
DinoDogToyWS / MurdoSmallAntiqueCat, 25 of 27 ZIP entries byte-identical,
the two movers being the metadata blob (creation timestamp) and its
derived content hash. The pre-existing canonicals were stale against
HEAD and were not used as the reference. Synthetic gate:
`scripts/check_fisheye_seed_primitives.py` gains a Phase-2 suite that
drives the whole ray-space seed on a planted equidistant scene with 142
of 984 observations past 90° — epipolar matrix essential to 4e-16,
relative pose to 5e-14°, chirality keeping all 136 inliers (120 of them
`z ≤ 0`), the window solve posing 6/6 to 3e-13°, and
`rotation_core_rays` posing a 6-camera rotating rig and certifying 145
far-field clusters.

**What this changes for Phase 3 and later.**

- **The commit bar lost a term.** The pinhole path arbitrates attempts by
  focal observability first (scan spread), coverage second. At a fixed
  focal there is no spread to measure, so the fisheye bar is posed count
  + coverage reach and the spread slot carries `NaN`; `flat_scan` /
  `edge_scan` / `vote_divergence` are suppressed for the same reason.
  The visible cost is OmniCoast: its rotation core is committable at 11
  posed, so the covisibility windows the baseline explored to (a 95.8%
  probe) are never tried. Phase 3's equidistant scan restores the term
  and should be wired straight back into `committable` / `score`.
- **`f_report` is the equidistant focal** on the fisheye branch, and the
  pinhole vote is no longer a fallback for it — the two are not
  commensurable. Phase 3's release changes what that number means, not
  where it comes from.
- The finalization now receives a seed whose `focal_structure_px` and
  camera model agree (both equidistant); Phase 1 handed it a pinhole
  scan focal wearing a fisheye model. Phase 4/5 audits start from a
  consistent input.
- `rotation_core_rays` certifies far-field clusters (4.9k on
  KerryPark480, 16k on OmniCoast) that the finalization still discards,
  because restricted admission renumbers clusters and the ids cannot be
  bridged. That is a pre-existing gap the fisheye branch now hits
  routinely — worth fixing where Phase 5 touches the handoff.
- Stage 1 still needs a fisheye analog of the vote-vs-structure
  arbitration: with the focal fixed there is nothing to arbitrate, but
  once Phase 3 releases one, the census audit the Phase-3 section already
  names becomes load-bearing.

## Phase 3a — EQUIDISTANT_FISHEYE camera model (precursor)

Verified 2026-08-09: the camera enum has no distortion-free equiangular
model — the current convention is `SimpleRadialFisheye` with `k1 = 0`,
carried by hand at every construction site. Add a native
`EquidistantFisheye { focal_length, principal_point_x,
principal_point_y }` — `SimplePinhole`'s exact parameter list under the
`θ = r/f` map. (Name chosen for consistency with the merged focal-vote
column vocabulary — `CameraModel::EquidistantFisheye`, the
`"equidistant"` binding string, the spec section title — over the
synonym "equiangular".)

- Projection both ways is closed-form and exact (no Newton, no blend);
  ship an analytic `PixelJacobian` — this removes the ~4× central-
  difference cost Phase 1 measured for ray-path models in
  `pose_refine`/`bundle_adjust`, and gives Phase 3's free-focal release
  an analytic focal column.
- Storage: native model name in `.sfmr` (precedent: `EQUIRECTANGULAR`
  is already a native non-COLMAP model); sfmr-format spec table entry.
- COLMAP interop: export as `SIMPLE_RADIAL_FISHEYE` with `k = 0`;
  import `SIMPLE_RADIAL_FISHEYE` as `EQUIDISTANT_FISHEYE` iff its `k`
  is exactly 0, unchanged otherwise. (COLMAP has no distortion-free
  equidistant model, so the k-parameter carrier is the interop shim in
  both directions.)
- Seed scripts and Phase 1's `make_cam` switch from the k1=0 convention
  to the new model; `needs_ray_path()`, `opt_f` (Phase 3b) and the
  in-front gate treat it as the fisheye family member it is.

### Phase 3a outcome (2026-08-09) — DONE

`CameraModel::EquidistantFisheye { focal_length, principal_point_x,
principal_point_y }`, model name `"EQUIDISTANT_FISHEYE"`, is a native
non-COLMAP model alongside `Equirectangular`. Match arms touched:
`model_name`, `has_distortion` (false — the map has no coefficients),
`is_fisheye` (true, so `needs_ray_path` is true), `supports_pixel_jacobian`
(true — the one ray-path model with an analytic one), `focal_lengths`,
`principal_point`, `TryFrom<&SfmrCamera>`, `From<&CameraIntrinsics>`, and in
`camera/distortion.rs` the `distort` / `undistort` / `distort_ray` /
`undistort_to_ray_optical` dispatches plus a new pre-guard branch in
`ray_to_pixel_with_jacobian`.

**Projection, exact, both ways.** Forward is `θ = atan2(ρ, rz)` times the
unit 2D direction; inverse is `θ = r_d` into `[sinθ·û, cosθ]`. No Newton, no
`blend_fisheye_ray` — those exist for the polynomial family and this model
bypasses both. θ ≥ 90° is ordinary, and `distort_ray` never returns `None`.
The antipode (θ = π, `r_xy = 0`) aliases the principal point exactly as under
the k1 = 0 convention, pinned by a test.

**Analytic Jacobian.** With `ρ = r_xy`, `n² = ‖r‖²`, `(ux, uy) = (rx, ry)/ρ`
and `c = rz/n² − θ/ρ`, the optical-frame 2×3 is
`[[θ·uy²/ρ + ux²·rz/n², ux·uy·c, −rx/n²], [ux·uy·c, θ·ux²/ρ + uy²·rz/n²,
−ry/n²]]`; `K` and the `S` flip compose on top. Nothing is guarded on `rz`,
so the Phase-1 finding is honoured: the pre-dispatch `rz ≤ 0` guard moved
inside the perspective branch. On the axis (`ρ → 0`, `rz > 0`) the limit is
`diag(1/rz, 1/rz)` with a zero third column — the pinhole small-angle form,
direction-independent — and that is what the axis branch returns. At the
antipode (`ρ → 0`, `rz < 0`) `θ/ρ` diverges and there is no finite Jacobian;
this returns `None`, the single measure-zero direction where the derivative's
domain is narrower than `ray_to_pixel`'s, now documented in
`specs/core/camera/projection-jacobian.md`.

Validation: 4107 analytic-vs-central-difference samples over a whole 480²
synthetic sensor (image circle at θ = 130°, three ray scales) plus explicit
θ ∈ {60°, 89°, 91°, 105°, 130°} bands at five azimuths — worst relative
error **1.98e-8**, which is the central difference's truncation, against a
1e-6 assertion bar. Plus the axis limit, azimuth-independent continuity,
and the degree-0 homogeneity identities `J·r = 0`, `J(s·r) = J(r)/s`.

**Consumers.** `refine_absolute_pose` and `bundle_adjust` select the
analytic path off `supports_pixel_jacobian()` and now take it. Equivalence
tests solve the same synthetic wide-FOV scene under both representations:
pose refinement agrees to < 1e-10 rad / 1e-10 translation, and the bundle to
< 1e-9 rad / 1e-9 translation / 1e-8 per point — three or more orders inside
each arm's own accuracy against planted truth. Measured cost of the
projection-plus-Jacobian step itself: **8.8×** (200k rays, 0.0292 s
central-difference vs 0.0033 s analytic, release). End-to-end BA wall-clock
moves only ~5% (0.064 → 0.061 s on a 12-image / 400-point / 3394-observation
scene) — the normal-equation solve dominates, so the ~4× projection saving
Phase 1 identified is real but not where this kernel's time goes.

**Storage.** `sfmr-format` validates no model names (`SfmrCamera::model` is a
free string), so `.sfmr` needed no code change; the camera-model table in
`specs/formats/sfmr-file-format.md` gains the row and the carrier rule.

**COLMAP interop, both sides.** Export writes `SIMPLE_RADIAL_FISHEYE` with a
literal `k = 0`; import claims a `SIMPLE_RADIAL_FISHEYE` back as
`EQUIDISTANT_FISHEYE` (dropping `k`) **iff `k == 0.0` exactly** — `1e-300`
stays polynomial. Rust: `claim_native_camera_model` plus carrier-aware
`colmap_model_id` / `camera_params_to_array` in `sfmr-colmap`, applied at the
binary read and at both writers (binary and DB). Python: the same rule in
`colmap_camera_from_intrinsics` (always) and `pycolmap_camera_to_intrinsics`
(behind `claim_native=True`). Claiming is opt-in on the Python import
because that same function also builds the *initial* camera for a COLMAP
solve, where a freshly-initialized `SIMPLE_RADIAL_FISHEYE` has `k = 0` and
must stay one; it is enabled at the two sites that import a *solved*
reconstruction (`colmap/io.py`, `xform/_bundle_adjust.py`).

`_CAMERA_PARAM_NAMES` — the `--camera-model` vocabulary and the
`switch-camera-model` target list — is deliberately unchanged: it feeds
`pycolmap.CameraModelId`, and sfmtool's native models are absent from it for
the same reason `EQUIRECTANGULAR` always was. The native model works as a
switch *source* (the transform reads source parameters generically).

**Measured.** Pinhole byte-parity, gate OFF, `c2f_seoul/b100`: 25 of 27 ZIP
entries byte-identical against the canonical, the two movers being the
metadata blob (creation timestamp only, diffed key by key) and its derived
content hash. Fisheye smoke, gate ON, `KerryPark480`, as a true A/B (HEAD
before, HEAD + change after; the before-run reproduced the canonical 25/27,
so the pipeline is deterministic and the canonical was at HEAD):

| | before (k1 = 0) | after (native) |
| --- | --- | --- |
| stage-1 posed | 43/48 | 43/48 |
| stage-1 inlier < 2 px | 51.3% | **53.7%** |
| stage-1 wall-clock | 8.5 s | 8.6 s |
| finalized points / cams | 61 / 26 | **91 / 31** |
| native BA focal | 144.0 px | 144.0 px |
| post-BA reproj median | 0.33 px | **0.31 px** |

Same posed count and the same settled focal on both sides; the finalization
differences are the changed linearization moving observations across the
trimmed-quantile gates, which the many quantile-driven cull stages then
amplify. Both canonicals were backed up and restored with md5 verification.

**What Phase 3b inherits.**

- **`opt_f` is still `SimplePinhole`-only** (`bundle_adjust.rs`), so the
  scripts' fixed-focal clamp under a non-pinhole context is unchanged. The
  focal column `∂(u, v)/∂f = (u − cx)/f` is exact for this model too (`f` is
  a pure multiplier of `θ`), so the release is a widened `matches!` plus the
  gauge tests — now against `EquidistantFisheye` rather than
  `SimpleRadialFisheye`.
- The scan can rebuild a camera per candidate focal through `make_cam`
  unchanged; the model carries one focal and nothing else.
- `seed-final.sfmr` on the fisheye branch now stores `EQUIDISTANT_FISHEYE`
  where it stored `SIMPLE_RADIAL_FISHEYE { k1 = 0 }`. Anything reading those
  files must be at or past this change — an older build raises
  `unknown camera model: EQUIDISTANT_FISHEYE` (observed while A/B-ing).
- The polynomial fisheye family still central-differences its pixel
  Jacobian; only this model gained an analytic one.

## Phase 3 — Equidistant focal scan and release

- Scan: sweep equidistant `f` over the FOV-derived band (the vote's
  band machinery), rescaling per candidate as the pinhole scan does;
  spread/basin logic carries over unchanged in log-f space.
- Release: free-focal BA under the equidistant model. **Kernel
  extension required:** `bundle_adjust.rs:873` restricts `opt_f` to
  `SimplePinhole`; extend to `SimpleRadialFisheye` (analytic df is
  simple for `k1 = 0`; central-difference otherwise). Fixed-intrinsics fisheye BA
  works today (the kernel's `PixelJacobian` already central-differences
  fisheye models), so the scan can land before the release extension.
- Basin guard and vote-vs-structure arbitration carry over; the census
  needs an audit (its bridge-cluster explanation model may embed
  pinhole projection).

**Gate:** Kerry entries' released equidistant focal vs rig fx — err%
becomes meaningful; target the vote's ±1–2% precedent, accept the ~6%
KB-model gap as the known ceiling. Fleet A/B inert.

### Phase 3b outcome (2026-08-09) — DONE

The focal is scanned and RELEASED on the fisheye branch. `opt_f` admits
`EquidistantFisheye`, stage 1 sweeps a five-point equidistant grid whose
spread feeds `committable` / `score` again, and the release runs free-focal
BA under the equidistant model at both the stage-1 and the finalization BA.

**Kernel.** `opt_f`'s gate widens to `SimplePinhole` **or**
`EquidistantFisheye`, the two single-focal distortion-free models — a plain
`matches!` at the gate, not a model predicate, because releasability is a
property of this kernel's focal column, not of the camera. The condition, stated in the code and the spec, is
that `f` multiplies a distorted coordinate that does not itself read `f`:
`x_d = rx/(−rz)` for the pinhole, `x_d = θ·ûx` with `θ = atan2(ρ, rz)` for
the equidistant map, so `∂u/∂f = x_d = (u − cx)/f` in both. Phase 3a's claim
that the existing analytic column is already exact here is CONFIRMED by
measurement, not assumed: against a central difference of the projection in
`f`, over θ from 5° to 170° at five azimuths and three ray scales (270
samples, 150 of them past 90°), the worst relative error is under `1e-9` —
the difference quotient's own cancellation noise, with no truncation
allowance, which is what an exactly-linear dependence gives. `cam_at` moves
the focal for the same two models. The binding's guard becomes
`opt_f requires a SIMPLE_PINHOLE or EQUIDISTANT_FISHEYE camera`.

The polynomial fisheye family is deliberately NOT released, against the
Phase-3 sketch above: `SimpleRadialFisheye` recovers `θ` from `r/f` before
applying `g(θ²)`, so `(u − cx)/f` is not its focal derivative even at
`k1 = 0`, and a "simple analytic df for k1 = 0" would be a different column
in a different place. It keeps degrading `opt_f` to a fixed-focal solve, now
pinned by a test that asserts its focal comes back bit-identical.

Gauge tests re-derived against `EquidistantFisheye` (Rust + binding): a
released focal recovers a planted 130 px from a −9% start on a scene with
231/1271 observations past 90°; the same solve at `opt_f = false` returns
the input focal bit-identically; the polynomial arm does too under
`opt_f = true`.

**Stage-1 scan.** `f_grid` on the fisheye branch is
`f_verdict · 1.15^{−2..2}` clipped to the focal-vote kernel's own FOV-derived
band `[0.075, 3] × max(w, h)` (`specs/core/geometry/focal-vote.md`). Log-SYMMETRIC,
unlike the pinhole grid: the pinhole grid skews upward because Bougnoux votes
run ~10% low, and the equidistant column carries no such measured bias. Every
fisheye-branch suppression is gone — `committable` measures `scan_spread`,
the level loop's break bar reads it, and `flat_scan` / `edge_scan` /
`vote_divergence` all run. The divergence guard and the flagged fallback now
compare against `f_indep`, the structure-free focal IN THE SOLVE'S OWN
PARAMETERIZATION: the pinhole vote (bias-corrected ×1.1) for a pinhole solve,
the equidistant verdict (raw) for an equidistant one. Same for the seed's
`focal_vote_px`, which the finalization consumes as `arbitrate_vote` — it was
handing a fisheye camera the pinhole vote's focal, a units error.

The scans are well-conditioned on all four captures, with an interior optimum
inside ONE grid step (×1.15) of the verdict every time — two independent
measurements (pairwise vote, multi-view structure) corroborating:

| capture | grid (px) | inlier<2px across the grid | spread | peak |
| --- | --- | --- | --- | --- |
| KerryPark480 | 104.6 … 182.9 | 10.8 / 15.4 / 54.1 / **62.7** / 60.6 | 51.9pp | +1 step |
| KerryPark360 | 209.1 … 365.8 | 23.1 / 38.1 / **63.4** / 59.6 / 55.6 | 40.4pp | at verdict |
| OmniCoast | 441.6 … 772.3 | 47.0 / **60.4** / 57.1 / 42.6 / 36.8 | 23.5pp | −1 step |
| OmniTemple1 | 394.9 … 690.7 | 25.0 / 33.4 / **49.4** / 36.4 / 17.9 | 31.5pp | at verdict |

No capture trips `flat_scan` or `edge_scan`.

**Census audit — NOT inert, and it was wrong.** `seed_census` built a
hardcoded centred `SIMPLE_PINHOLE` at `f` in BOTH paths, and its pure-Python
fallback additionally hand-wrote the pinhole reprojection `−f·x/z + cx` with a
`z > 0` rejection that discards every observation past 90°. It is reachable on
every fisheye seed — it is the finalization's vote-vs-structure arbiter — so
it was a pinhole formula scoring fisheye geometry. The native kernel itself is
model-generic (`pixel_to_ray` in, `ray_to_pixel` out); the fix is to thread the
candidate's own camera through `census_score` / `census_report`, and to
reproject through it in the fallback with a model-aware in-front test. Omitted,
the camera defaults to the old centred pinhole, so the pinhole path is
unchanged.

**Three further defects, all pre-existing and all live on the fisheye branch:**

1. **`save_sfmr` hardcoded `SIMPLE_PINHOLE`.** The dense seed it writes was
   triangulated through `make_cam` — equidistant rays — and then handed to the
   embed, the reprojection culls, the census and the finalization BA wearing a
   pinhole model; the write-back afterwards stamped `EQUIDISTANT_FISHEYE` with
   the pinhole-released focal back onto it. So the entire finalization ran the
   wrong model, and the artifact's declared model described nothing that
   produced it. It now uses the context camera (byte-identical on the pinhole
   default). This single fix is where the finalization numbers below come from.
2. **The `0.3 × max(w, h)` focal plausibility floor rejects a real fisheye
   focal.** KerryPark480's own focal is ~138 px against a floor of 144 — the
   release basin guard and the finalization's guarded-BA band would both have
   thrown away every honest solve. The floor is now FOV-derived
   (`0.075 × max(w, h)`) under an equidistant context, matching the vote's band.
3. **The Phase-1/2 fixed-focal clamp was already dead code.** The stage-1
   release block sat inside the pinhole `else`, so `opt_f = True` never reached
   a fisheye camera from stage 1; and the finalization's `_run_ba` bypasses the
   wrapper the clamp lived in, reaching the binding directly with whatever
   `emb.cameras[0]` carried — which defect 1 guaranteed was a pinhole. Both
   clamps are removed.

**Stage-1 acceptance, gate ON, true A/B (HEAD scripts vs these; the widened
kernel is inert under HEAD scripts, which never pass `opt_f` a fisheye
camera):**

| capture | posed before → after | inlier<2px before → after | wall-clock |
| --- | --- | --- | --- |
| KerryPark480 | 43/48 → 43/48 | 53.7% → **61.9%** | 15 s → 35 s |
| KerryPark360 | 12/82 → 12/82 | 62.5% → 62.5% | 6 s → 9 s |
| OmniCoast | 10/88 → 10/88 | 57.2% → **68.5%** | 17 s → 23 s |
| OmniTemple1 | 14/80 → 14/80 | 51.9% → **55.5%** | 18 s → 27 s |

No posed regression anywhere; consensus improves on three. Wall-clock roughly
doubles on KerryPark480 (five scan candidates plus two refits plus the release
where Phase 2 ran one settle) and stays inside the campaign's ≤2× bar.

**Finalization, same A/B — `seed-final.sfmr` points / cameras:**

| capture | before | after |
| --- | --- | --- |
| KerryPark480 | 91 / 31 | **728 / 43** |
| KerryPark360 | 19 / 7 | **222 / 12** |
| OmniCoast | 29 / 6 | **943 / 8** |
| OmniTemple1 | 6 / 14 | **1460 / 14** |

That is defect 1: a fisheye seed embedded and culled through a pinhole camera
loses almost everything, and the surviving handful is what Phases 1–3a were
measuring. OmniTemple1's six-point artifact was the extreme case.

**Focal accuracy, and a correction to this plan's own bar.** `err%` against
rig `fx` is not the right measure for a released EQUIDISTANT focal, because
the rig calibrations are Kannala-Brandt and the seed solves a pure `θ = r/f`
map — the two differ by a model gap that has to be measured, not assumed at
"~6%". Least-squares fitting `f_eq·θ` to each rig's own KB radial map over its
monotone image circle gives the best pure-equidistant focal a perfect
estimator could report:

| capture | rig `fx` | best-fit equidistant `f` | model gap | fit rms |
| --- | --- | --- | --- | --- |
| KerryPark480 | 129.150 | **135.52** | +4.93% | 2.68 px over 211° |
| KerryPark360 | 257.935 | **271.32** | +5.19% | 5.39 px over 210° |

So the ~6% ceiling is real and now quantified, and the honest accuracy column
is error against that best-fit focal:

| capture | | before (fixed verdict) | after (released) | shipped camera |
| --- | --- | --- | --- | --- |
| KerryPark480 | px | 138.310 | 159.057 | 138.31 |
| | vs rig fx | +7.09% | +23.16% | +7.09% |
| | vs model optimum | **+2.06%** | +17.37% | **+2.06%** |
| KerryPark360 | px | 276.562 | 276.562 | 276.56 |
| | vs rig fx | +7.22% | +7.22% | +7.22% |
| | vs model optimum | **+1.93%** | **+1.93%** | **+1.93%** |

KerryPark360 meets the ±1–2% bar on both sides: its scan peaks exactly on the
verdict and the release does not move it. KerryPark480's release overshoots to
+17.4% and is the one gate miss. Diagnosed, with `SFMTOOL_BA_DEBUG=1`: the
free-focal BA is not the thing that overshoots — it walks steadily DOWNWARD,
159.06 → 150.66 → 146.97 → 139.43 across its three rounds, i.e. the robust
cost's own optimum sits near the verdict. What holds the reported focal at
159.06 is the release loop's keep-best rule, whose metric is inlier-under-2px
over the fixed budget denominator, and that metric prefers the larger focal
(62.7% at 159.1 against 54.1% at 138.3). The scan and the release are
optimizing different functionals and disagree on this capture; the vote
tiebreak already pulled the scan winner down from 182.9 to 159.1, and the
finalization's census arbitration then shipped the verdict focal outright, so
the artifact carries +2.06%. The keep-best rule is shared with the pinhole
branch, so it was left alone here.

Why KerryPark480 and not the others: its seed is a rotation core at 2.94°
parallax, and its scan curve is steep below the verdict and nearly flat above
it (54.1 / 62.7 / 60.6 across the top three grid points). The focal is bounded
well from below and weakly from above — the soft form of the `edge_scan`
pathology, which `edge_scan` cannot see because the coarse peak is interior. A
second candidate mechanism, unmeasured: a larger focal pulls every ray toward
the axis, which conditions the triangulation better, so more clusters
triangulate at all and the inlier NUMERATOR grows against a fixed denominator.

**OmniCoast: the restored term justifies the commit rather than changing it.**
Phase 2 recorded that OmniCoast commits on its 11-image rotation core and
never explores the wider covisibility windows the pinhole baseline reached.
With the focal-observability term restored, the posed count is unchanged
(10/88 before and after) and the same rotation-core window commits — but now
because it MEASURES 23.5pp of scan spread with a clean interior optimum, where
the Phase-2 arm carried `scan spread nanpp` and committed unconditionally. The
concern that the suppressed term was hiding an unexplored better window is
answered in the negative: that window passes the same bar a pinhole window
would. The released focal moves −13.0% off the verdict (507.8 against 584.0),
and the census arbitration keeps the verdict.

OmniTemple1's release moves +11.1% (580.3 against 522.3) but the seed carries
`low_consensus`, so `f_report` falls back to the verdict — the pinhole
branch's own semantics, now available on the fisheye branch because the
fallback target is finally commensurable with the solve. Neither OmniPhotos
capture has a rig GT; both are stable in the sense that the scan's interior
optimum and the vote agree to within one grid step.

**Pinhole byte-parity, gate OFF.** `c2f_seoul/b100`: 25 of 27 ZIP entries
byte-identical against the canonical, the two movers being `metadata.json.zst`
and `content_hash.json.zst`; the metadata blob diffed key by key is 36 of 37
identical with only `timestamp` moving. Stage 1 reproduces 14/100 posed,
62.4% inlier, f = 355.1. Every fisheye change is behind `fisheye_stage1()` or
`_CAM_CONTEXT`, and the two kernel widenings (`opt_f`, `cam_at`) are
different match arms from the pinhole ones.

Synthetic gate: `scripts/check_fisheye_seed_primitives.py` gains a Phase-3b
block — grid in band and log-symmetric, the release floor below the capture's
own focal, `opt_f = False` bit-identical, and free-focal BA recovering
118.3 → 130.00 px against a planted 130 (err 0.00%) at a 9.5e-11 px median
residual with rotations unbent to 8e-3°.

Canonical protection: fresh md5-manifest backup plus a pre-run `sfmr/` listing
per workspace; all five canonicals restored byte-exact and every workspace
diff-clean against its listing afterwards.

**What Phase 4 inherits.**

- **The finalization now genuinely runs the equidistant model**, which is what
  makes Phase 4's audit necessary rather than theoretical: `embed_patches`,
  the congeal/consensus bitmap chain, `localize_anchors`, the ARS normals and
  the reprojection culls are all being handed an `EQUIDISTANT_FISHEYE` camera
  for the first time. They no longer silently fail on a pinhole misread, but
  nothing here establishes that patch geometry near the periphery is right.
  The point counts jumping 8–240× is evidence the culls stopped destroying
  everything, not evidence the survivors are well-formed.
- **The keep-best-inlier rule outranks the release on at least one capture.**
  Whether the release's own soft-L1 optimum is the better focal estimate is
  measurable (KerryPark480 is the test case: the release wants ~140, one step
  from the model optimum 135.5) but changing the rule touches the pinhole
  branch and needs a fleet A/B of its own.
- **`err%` against rig `fx` should be retired as the fleet column** in favour
  of error against the best-fit equidistant focal (135.52 / 271.32), computed
  once from each rig config. Against `fx` a perfect solve reads +5%.
- The scan grid's ×1.15 step is coarser than the ±1–2% bar it is being judged
  against; the release is what is supposed to refine inside it, which is the
  same observation as the keep-best item above.
- `rotation_core_rays`' far clusters are still dropped at the handoff
  (unchanged from Phase 2), and the finalization's `arbitrate_vote` path is
  now the load-bearing focal arbiter on every flagged fisheye seed.

## Phase 4 — Photometric verification and embed under fisheye

The stage-1 anchor localization and the finalization's embed both render
patches through the camera. Audit first: the embed machinery
(`OrientedPatch`, `select_views`, keypoint localizers) consumes
`CameraIntrinsics` — determine how much is already model-correct (the
production embed runs on pycolmap fisheye reconstructions of kerry in
tests) vs. silently pinhole. Then:

- Stage-1 `localize_anchors` under the fisheye context.
- Finalization embed/congeal/consensus-bitmap chain on the fisheye
  seed; patch frames near the periphery are the risk item (strong
  anisotropy under the equidistant map — measure before assuming).

**Gate:** photometric census and vet statistics on the fisheye entries
in family with pinhole captures' (no order-of-magnitude outliers);
Explorer inspection of patch bitmaps at high θ.

### Phase 4 outcome (2026-08-09) — DONE

The photometric layer can see past 90° for the first time. Its render substrate
was already model-correct; what was not was a set of **cheirality tests that
re-implemented a model decision instead of delegating it**, and one **patch
sizing rule** written in optical-axis depth. Together they clipped the fisheye
seed at exactly the 90° horizon — the shipped artifacts' `θ_max` was 90.0 /
89.6 / 81.7 / 90.0° across the four entries, against 17.7–37.2% of each
capture's detected features sitting past 90°.

**Audit verdicts, per component.** The render substrate is genuinely
model-generic and was already right: `WarpMap::from_patch` builds the patch →
camera-frame map affinely (no camera math) and hands it to
`ray_to_pixel_grid`, whose fisheye branch is bounded to 0.02 px by a per-cell
probe; `compute_svd` / `get_jacobian` central-difference that map, so the
per-pixel 2×2 the anisotropic sampler and the sub-pixel GN step consume carries
the equidistant scale variation for free. Downstream of it, ZNCC/ECC/IRLS,
`znorm`, `consensus_phi`, the localizability structure tensor, the ARS surfel
fit and the observation-adjacency graph (which already measures range, not
depth) are model-free by construction. `obliquity.rs` takes foreshortening from
the true surface→camera ray, not the optical axis. `member_coherence.rs` has no
cheirality gate at all — invalid rays arrive as `NaN` from the warp map — which
is the pattern the broken sites should have followed.

| component | verdict | site |
| --- | --- | --- |
| `WarpMap::from_patch` / `ray_to_pixel_grid` | **model-correct** | — |
| warp Jacobian / SVD, anisotropic sampler | **model-correct** | — |
| `normal_refine` search, consensus, obliquity, view-subset, level, support | **model-correct** | — |
| ARS `estimate_adjacency_surfel_normals`, `build_observation_adjacency` | **model-correct** | — |
| `member_coherence`, `localizability` scorer | **model-correct** | — |
| `view_selection::is_in_front` | was-pinhole → **fixed** | `z < 0` half-space gate on every expansion candidate |
| `keypoint_localize::project_unclipped` | was-pinhole → **fixed** | `pc.z >= 0 → None` ahead of `ray_to_pixel`; the only world→pixel entry in localize + sub-pixel + spawn |
| `keypoint_localize::seed_offset` | was-pinhole → **fixed** | no `s > 0` test on the bearing ∩ plane hit |
| `normal_refine::fronto_cache::corner_norm_pts` | was-pinhole → **fixed** | `z >= −1e-9` reject plus a gnomonic `(x/z, y/z)` corner space that is singular at 90° |
| `PatchCloud` `PixelRadius` extent | was-pinhole → **fixed** (`from_reconstruction` only) | `p_cam.z.abs()` as the distance |
| `sfmtool-py` localizability grid→px scale | was-pinhole → **fixed** | `−z` depth, dropping every θ > 90° observation from the median |
| seed writer's surfel-frame solve (`save_sfmr`) | was-pinhole → **fixed** | pinhole projection Jacobian, `max(z, 1e-6)`, extent `r_px·z_ref/f` |
| `_cheirality_keep`, `_member_view_structure`, `_inf_proj_err`, `_relocalize_keypoints`, collapse/contained/inf-gate depths | was-pinhole → **fixed** | `−z` as depth and as the in-front test |
| stage-1 `localize_anchors` | **model-correct by delegation** | builds its camera through `make_cam`; it was broken only through `project_unclipped` beneath it |
| `_normalize_infinity_frames` angular size (`rad_px / f`) | **model-correct** | `dθ = dr/f` is exact for the equidistant map |
| `PatchCloud::FeatureSize` extent | **model-correct** | already `σ·‖p_cam‖/f` |

Every fix is behind `camera.model.needs_ray_path()` (Rust) or
`fisheye_stage1()` (the scripts), so the perspective family takes the identical
branch it always did. The one **semantic** change worth naming: `-z > 0` is not
generalized to a range test but to the camera's own imaged **cone**,
`θ ≤ r_max/f` with `r_max` the inscribed image-circle radius (`_in_field`).
"In front" for a pinhole *is* the θ < 90° cone; for an equidistant map it is the
cone the sensor carries. The pinhole reading is recovered exactly when no
fisheye context is installed.

**Periphery anisotropy — measured, and it is not the risk it was named as.**
For a patch whose plane faces the camera, the equidistant map's own
sphere→image anisotropy is exactly `θ/sin θ` (radial scale `f`, azimuthal scale
`f·θ/sin θ`): 1.11 at 45°, 1.57 at 90°, 1.76 at 100°, 2.42 at 120°. On the real
seeds the measured singular-value ratio of the patch-grid→image Jacobian tracks
that law to within a few percent in every band, **including 90–105°**
(KerryPark480: 1.610 measured vs 1.617 analytic). So peripheral patch frames are
well-formed; the anisotropy they carry is the projection's, not a pathology, and
it stays inside the sampler's clamp. The analytic law and the range extent rule
are pinned on planted geometry by `check_fisheye_seed_primitives.py`'s new
Phase-4 suite (exact to <0.001% over a θ sweep to 120°).

**Periphery photometry — sound at every θ.** Per-member mean pairwise ZNCC from
the same keypoint-anchored `validate_member_coherence` primitive the
finalization vets with (never sampled at reprojections), by θ band, on the
shipped artifacts:

| capture | 0–30° | 30–60° | 60–75° | 75–90° | 90–105° |
| --- | --- | --- | --- | --- | --- |
| KerryPark480 | 0.857 | 0.861 | 0.864 | 0.865 | **0.861** |
| KerryPark360 | 0.898 | 0.872 | 0.878 | 0.904 | **0.903** |
| OmniCoast | 0.847 | 0.900 | 0.933 | 0.950 | **0.878** |
| OmniTemple1 | 0.926 | 0.927 | 0.932 | 0.924 | **0.909** |
| b100 (pinhole control) | 0.863 | 0.894 | — | — | — |

Flat in θ and in family with the pinhole control — the gate's "no
order-of-magnitude outliers" bar, met with room to spare.

**The extent defect, in one number.** `ext = r_px·z_ref/f` is the range rule
times `cos θ_ref`: a detection referenced at 60° got half its true size, at 75°
a quarter, and at 90° a zero-extent frame — which `from_halfvec_arrays` drops as
"no patch" outright. Before the fix the shipped patches' projected half-size
declined with θ (KerryPark480 1.00 / 0.88 / 0.94 / 0.86 across the bands,
OmniCoast 1.00 / 0.83 / 0.43 / 0.28); after it they do not.

**Fisheye A/B, gate ON, true A/B (fresh runs at HEAD before and after):**

| capture | posed | points | cams | θ_max (deg) | post-BA reproj median (px) | embed yield |
| --- | --- | --- | --- | --- | --- | --- |
| KerryPark480 | 43/48 → 43/48 | 728 → **1154** | 43 → 43 | 90.0 → **99.4** | 12.64 → **1.99** | 3169 → **4880** |
| KerryPark360 | 12/82 → **13/82** | 222 → **493** | 12 → **13** | 89.6 → **98.5** | 0.62 → **0.60** | 338 → **795** |
| OmniCoast | 10/88 → **11/88** | 943 → **1410** | 8 → 8 | 81.7 → **94.1** | 5.04 → **0.65** | 2256 → **5555** |
| OmniTemple1 | 14/80 → 14/80 | 1460 → **2160** | 14 → 14 | 90.0 → **104.4** | 1.05 → 1.21 | 2942 → **5688** |

No posed or point regression anywhere. The two reprojection outliers the gate
was written against — KerryPark480 at 12.64 px and OmniCoast at 5.04 px against
the pinhole control's 0.21 px — are gone. The vet statistics move the same way:
late-vet cull rate 9.5 → 4.7% / 8.3 → 3.5% / 10.2 → 4.9% / 3.7 → 3.2%
(pinhole control 2.8%), track-view eviction 5.09 → 3.33% / 2.19 → 1.26% /
2.46 → 1.39% / 1.32 → 1.02%. Stage 1 moves on two captures because
`localize_anchors`' photometric verify now keeps peripheral observations, which
changes which frames it un-poses; both moves are upward in posed count.
Wall-clock 32 → 37 / 7 → 10 / 21 → 30 / 25 → 31 s, well inside the campaign's
≤2× bar.

**Pinhole byte-parity, gate OFF.** `c2f_seoul/b100`, true A/B: 25 of 27 ZIP
entries byte-identical, the two movers `metadata.json.zst` and
`content_hash.json.zst`; the metadata blob diffed key by key is 12 of 13
identical with only `timestamp` moving. Every finalization log line is
character-identical between the arms. Stage 1 reproduces 14/100 posed, 62.4%
inlier, f = 355.1, and the seed 423 points / 0.21 px.

**OmniCoast's Phase-1 embed collapse — resolved, and the diagnosis was
incomplete.** Phase 1 recorded 2962 embedded → 154 after the length-3 filter.
Phase 3b's `save_sfmr` repair fixed the collapse itself (2256 → 1234, a 55%
length-3 survival rate, in family with every other capture). What Phase 3b did
NOT fix is the embed's *reach*: only 2256 of 7498 seed points embedded at all,
because `select_views`/`localize` could not admit a peripheral view. Phase 4
takes that to 5555 of 7498. So the collapse had two causes stacked — a
pinhole-modelled cull chain (Phase 3b) and a pinhole-modelled visibility test
(Phase 4) — and only the first was diagnosed at the time.

**KerryPark480's keep-best overshoot — measured, deferred, and smaller than it
looks.** Re-measured at this HEAD with `SFMTOOL_BA_DEBUG=1`: the free-focal
release walks 159.06 → 150.66 → 146.97 → **139.43**, i.e. the robust cost's own
optimum is 139.43, +2.9% against the model optimum 135.52, while keep-best ships
159.06 (+17.4%) because its inlier-under-2px metric reads 0.632 at 159.06 and
collapses to 0.293 at 139.43. **The release optimum is the better estimate on
this capture — the finding stands.** What the re-measurement adds is that the
overshoot never reaches the artifact: the finalization's census arbitration
compares vote 138.3 (census 0.913) against structure 157.6 (census 0.561),
refuses the structure candidate as unendorsed, and ships 138.31 — **+2.06%**
against the model optimum, inside the ±1–2% bar's neighbourhood. So the defect
is confined to the reported `f_report` / stage-1 focal, not the shipped camera.
Changing the keep-best rule touches the pinhole branch and needs its own
36-dataset fleet A/B, which is out of scope for a photometric-layer phase;
carried to Phase 6 with the measurement in hand.

**The fleet's accuracy column for the Kerry entries is retired.** `err%` against
rig `fx` is replaced by `err%` against the **best-fit equidistant focal**, since
the rigs are calibrated Kannala-Brandt and the seed solves `θ = r/f`: against
`fx` a perfect equidistant estimator reads +4.93% (KerryPark480) / +5.19%
(KerryPark360). `C:/DataSets/workspace-prep/equidistant_gt.py` computes it once
from each `rig_config.json` — least squares of `f_eq·θ` against the rig's own KB
radial map over the monotone prefix inside the inscribed image circle — giving
135.522 px over 211° at 2.68 px rms and 271.321 px over 210° at 5.39 px rms,
reproducing Phase 3b's figures. `seedgc_run.py` prefers it over `resolve_gt`
whenever a KB rig config exists and records which reference it used in a new
`gt_src` column (`equi` / `solve`); `fisheye_ab.py` now indexes the fleet TSV by
header name so the older reference snapshot still loads. Shipped accuracy on
this arm: KerryPark480 138.31 → **+2.06%**, KerryPark360 276.56 → **+1.93%**.

**Synthetic gate.** `scripts/check_fisheye_seed_primitives.py` gains a Phase-4
suite (exit 0, all suites): the camera's own 2×3 Jacobian is finite over a θ
sweep to 120° with `J·x = 0` to 5e-11 and `J(2x) = J(x)/2` exactly; its
minimum-norm right inverse satisfies `J·pinv(J) = I` to 7e-16 at every θ with
its null direction the viewing ray to 1e-12; `_in_field` reproduces the model's
105.8° cone and falls back to the pinhole half-space with no context; and the
range extent rule projects to `r_px` at every θ to <0.001% while the projected
anisotropy matches `θ/sin θ` to <0.001%. Rust tests pin the three kernel fixes
directly (`view_selection`, `keypoint_localize` ×2, `fronto_cache`), each
asserting the perspective arm still refuses the same geometry.

**Two production tests were pinning the defect.**
`test_select_views_admitted_points_are_in_front_of_camera` and
`test_select_views_infinity_admitted_are_in_front` asserted `z_cam < 0` for every
admitted view **on kerry_park, a back-to-back fisheye rig** — i.e. they asserted
the 90° clip on the one fixture that must not have it. They now assert the
model-stated invariant (the camera projects it, and it lands in frame) plus a
new positive assertion that at least one admitted view *does* sit past 90°, so
the production embed's reach into the periphery is guarded rather than
forbidden.

**Canonical protection.** Fresh `fisheye-p4-canon-backup` md5-manifest dir over
all five workspaces (never reused), with a pre-run `sfmr/` + `sfmr/seed-rounds/`
listing; restored byte-exact after each arm (5/5), every `seed-rounds/` snapshot
pruned, all five workspaces verified diff-clean against the listing.

**Post-inspection caveat (2026-08-09, human Explorer review).** The gate's
point-count and reprojection wins are about *projecting correctly*, and they
stand — but point counts on non-static scenes are not a health metric. The
Kerry entries' clouds are dominated by self-consistent false-parallax
content: moving clouds and wind-blown foliage triangulate into small, close,
mutually coherent garbage points (observed bearing span matches the solved
span, median ratio 1.01 — no per-track geometric test can flag them), and the
genuinely distant static scene is largely absent (the finite cloud sits
within ~2.5 max-baselines). Confirmed clean cases at 95–99° (811, 1093) show
the periphery machinery itself is sound; 207 is a SIFT scale-chimera
(members are the same self-similar pattern at different octaves), 1127 a
stale-keypoint depth outlier. OmniTemple1 judged a plausible starting-point
solve by inspection. The separation of static from non-static structure
(multi-hypothesis extraction over image-area coverage) is scoped as its own
experiment, outside this plan's phases.

**What Phase 5 inherits.**

- **`_in_field` is a weaker guard than `-z > 0` was.** The cheirality cull is a
  perspective notion; its fisheye analogue is a cone test, so a point the BA
  pushes to the far side of the scene must now leave the imaged cone to be
  culled. Nothing in the A/B suggests this bit (reprojection medians *fell*),
  but Phase 5's cull audit should decide whether the fisheye branch wants a
  positive-range-plus-in-frame conjunct instead.
- **The projected patch half-size now GROWS with θ** (KerryPark480 1.08 → 1.34
  ×on-axis across the bands; OmniTemple1 1.05 → **2.20**). The range rule is
  right per view, but the extent is set once from the reference view and a
  peripheral point's other views are at a different range. OmniTemple1's 2.2×
  is the one number here that deserves a second look before Phase 5 calls the
  sizing settled.
- **`PatchCloud::from_tracks` still reads all-perspective.** The ray-path flag is
  filled from `from_reconstruction`'s cameras and left empty by `from_tracks`,
  which takes focals rather than cameras. Only `PatchExtent::PixelRadius` reads
  it and no seed path uses that policy, but `sfm xform to-embedded-patches
  --extent pixel_size` on a fisheye recon via the array entry point is still
  wrong. Widening `from_tracks` is a public-API change and was deliberately not
  ridden along here.
- **Two approximations are now model-conditional and undocumented as such.**
  `view_selection::affine_core_map`'s 4th-corner residual is blind to curvature
  *symmetric* about the patch centre, which under the equidistant map is the
  dominant term at any image radius; and `keypoint_subpixel`'s
  `grid_to_source_scale` is a corner secant over the whole core, used only to
  decide tile-vs-direct render. Both are quality knobs, not wrong keypoints, but
  `map.compute_svd()`'s per-pixel `σ_major` is the model-correct measure for the
  second.
- **`search`/`max_offset_px` (patch-grid px) and `max_shift_px`/`offsets_px`
  (image px) are only jointly calibrated when grid px ≈ image px.** Under
  fisheye that ratio varies with θ *within one image*, so one global
  `max_shift_px` is a tighter gate at the periphery than on axis. With the
  visibility gates fixed, this is the next thing that will asymmetrically drop
  peripheral views.
- The fronto-cache's module doc and `specs/core/patch/fronto-parallel-patch-cache.md`
  both claimed the cache is "exact for any camera model". The module doc is
  corrected here; **the spec is not yet** — each half of the composed map is a
  three-corner affine fit, so it is exact only for an affine image map. Phase 5
  should correct the spec text (and `specs/core/patch/patch-normal-refinement.md`'s
  "so distortion / fisheye work unchanged", true of the render and false of the
  cache and the extent sizing).
- Human Explorer inspection is **prepared, not done**:
  `C:/DataSets/workspace-prep/phase4-explorer-inspection/` holds before/after
  `seed-final.sfmr` for KerryPark480 / KerryPark360 / OmniTemple1, high-θ
  consensus-bitmap contact sheets for each, and a README naming exactly what to
  look at.
- `rotation_core_rays`' far clusters are still dropped at the handoff
  (unchanged since Phase 2).

## Phase 5 — Finalization end-to-end + persistence

- Triangulation/reprojection culls, member coherence, late vet, ARS
  normals, infinity classifier: audit each for pinhole assumptions
  (most operate on world geometry + stored keypoints and should be
  model-clean; reprojection-based culls need the camera context).
- Final BA under the fisheye model (Phase 3 extension), then
  `save_sfmr` writing `SIMPLE_RADIAL_FISHEYE` intrinsics — format
  support exists across the model enum.
  > _Correction (2026-08-10): this bullet was already stale when Phase 5
  > began. Phase 3a replaced the `SimpleRadialFisheye { k1 = 0 }`
  > convention with the native `EQUIDISTANT_FISHEYE` model, and Phase
  > 3b's defect 1 fixed `save_sfmr`'s hardcoded `SIMPLE_PINHOLE` to use
  > the context camera — so `seed-final.sfmr` has written
  > `EQUIDISTANT_FISHEYE` since 3b, and the final BA has run free-focal
  > under it since 3b's `opt_f` widening. Nothing in this bullet
  > remained for Phase 5 to do._

**Gate:** seed-final.sfmr for the four fisheye entries loads in the
Explorer with visually sane geometry (the current center-out warp
gone); the full stage-dump waterfall runs; fleet A/B inert.

### Phase 5 outcome (2026-08-10) — DONE

The finalization chain is model-correct end to end. The audit that establishes
it found exactly **one** live defect inside the chain — after Phases 3b and 4 it
is well insulated, and the remaining pinhole assumption sat in a place none of
the culls read, the **stored** per-point reprojection error. Sweeping the core
for the same failure class then turned up a second, larger one just upstream of
it, in the rotation core's translation resection; both ship.

**Per-stage audit.** Every stage of `_finalize_seed`, read and then tested.
"By delegation" means the stage reaches its camera only through
`recon.cameras[0]` / `make_cam` and a model-generic kernel, which is the
pattern the whole chain should follow.

| stage | verdict | how it gets the model right |
| --- | --- | --- |
| `dense_structure` / `triangulate` / `fill_new_points` | **model-correct by delegation** | rays via `make_cam`, native `triangulate_batch` |
| `save_sfmr` surfel-frame solve | **model-correct** (Phase 4) | dual-armed on `fisheye_stage1()`; range + the camera's own 2×3 Jacobian |
| `_cheirality_keep` (pre- and post-BA) | **model-correct** (Phase 4) | `_in_field` — the imaged cone, not the half-space |
| `_reprojection_medians` / `_reprojection_cull_mask` | **model-correct by delegation** | native `reprojection_residuals` → `ray_to_pixel` |
| `_refresh_errors` → `recompute_point_errors` | **was pinhole → FIXED** | `observation_reprojection_error` hard-coded the perspective cheirality test and the gnomonic `(x/−z, y/−z)` projection for every model |
| `_collapse_duplicate_points` / `_reconcile_aliased_tracks` / `_conflicted_pairs` | **model-correct** | `_cam_depth` + `cam.pixel_to_ray_batch` re-triangulation |
| `_cull_contained_inconsistent` / `_contained_pairs` | **model-correct** | `_cam_depth` ranges |
| `_evict_track_views` / `_member_view_structure` | **model-correct** (Phase 4) | `_in_field` + `ray_to_pixel_batch` footprint |
| `_member_coherence_vet` | **model-correct** | native `validate_member_coherence`; no cheirality gate at all — invalid rays arrive as `NaN` from the warp map |
| `_late_vet` (evict replay, member coherence, localizability re-cull) | **model-correct by delegation** | `score_localizability`'s grid→px scale is range-based (Phase 4) |
| `_ars_normals` — `_ars_edges`, `_ars_fit`, expand, promote | **model-correct** | `build_observation_adjacency` measures RANGE from camera centres; congealing through `CameraViews` |
| `classify_points_at_infinity` (native) | **model-correct** | `pixel_to_ray` bearings, world-frame viewing rays, per-ray noise `error/f` |
| `_inf_gate_veto` / `_inf_gate_median_depths` / `_inf_proj_err` | **model-correct** (Phase 4) | `_cam_depth`, `_in_field` |
| `_normalize_infinity_frames` | **model-correct** | `dθ = dr/f` is exact for the equidistant map |
| `_relocalize_keypoints` | **model-correct** (Phase 4) | `_in_field` accept gate, `_cam_depth` move measure |
| `_rerender_mutated_bitmaps` / `embed_patches` | **model-correct** (Phase 4) | `WarpMap::from_patch` → `ray_to_pixel_grid` |
| final BA + census arbitration | **model-correct** (Phases 1/3b) | range-based in-front gate, `opt_f` admits `EquidistantFisheye`, census through the context camera |
| *(upstream, found by the same sweep)* `resect_translation` trim gate | **was pinhole → FIXED** | the rotation core's translation solve; the half-space in-front test scored every peripheral observation invalid |

**The defect, and what it cost.** `observation_reprojection_error` — shared by
`recompute_point_errors` and the points-at-infinity discovery — rejected every
observation with `p_cam.z >= 0` and then projected through
`camera.project(x/−z, y/−z)`. On an equidistant capture that is the entire
past-90° annulus: those observations returned `None` and dropped out of their
point's mean, and a point observed ONLY past 90° came back with the "no valid
observation" error of **0.0** — i.e. reported as the best-fitting point in the
cloud. The fix delegates to `ray_to_pixel` for `needs_ray_path()` models and
leaves the perspective branch bit-identical. Measured on the four fisheye
artifacts (before → after, same run conditions):

| capture | points whose stored error moved | of those, past-90-only | fake `0.0` errors removed | median change | max change |
| --- | --- | --- | --- | --- | --- |
| KerryPark480 | 333 / 1154 (28.9%) | 0 | 0 | +0.007 px | 4.35 px |
| KerryPark360 | 114 / 493 (23.1%) | 3 | 3 | −0.023 px | 1.00 px |
| OmniTemple1 | 147 / 2160 (6.8%) | 54 | 54 | +0.392 px | 5.67 px |
| OmniCoast | — (one point differs) | — | 11 | — | — |

The shipped geometry barely moves — the cull chain judges on
`_reprojection_medians`, which was already model-generic — so this is a fix to
what the artifact *reports*, not to what it contains: the Explorer's quality
shading, and the classifier's per-ray angular noise `max(error, floor)/f`.

**The second defect — `resect_translation`'s trim gate — SHIPPED.** The
module's own doc claims the mechanism is camera-model-agnostic. It is the ROWS
that are model-agnostic; they are also SIGN-BLIND (`[r]ₓ·(R·X + t)` vanishes
for `−r`), so the trim gate is what carries the chirality, and it was written
as the perspective half-space. Under a fisheye camera that scores every
peripheral observation `INVALID_RESIDUAL`, and the rotation core's translations
settle on the on-axis subset. The gate is now positive RANGE along the observed
ray for `needs_ray_path()` models — which still rejects the antipodal
reflection, the one thing the sign-blind rows need it for — with the
perspective branch evaluating the identical `z < 0` expression it always did.
Measured, gate ON, against the plan-HEAD baseline:

| capture | posed | inlier<2px | points | obs | post-BA reproj median | at infinity | shipped focal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KerryPark480 | 43 → **36** | 61.9 → **66.0%** | 1154 → **1299** | 7743 → **8728** | 1.99 → **0.58 px** | 64 → 10 | 138.31 → 153.28 |
| OmniCoast | 11 → **8** | 57.1 → **59.4%** | 1410 → **2566** | 6121 → **11460** | 0.65 → 0.67 px | 23 → 2 | unchanged |
| KerryPark360 | 13 → 13 | unchanged | 493 → 493 | unchanged | 0.60 px | 5 | unchanged |
| OmniTemple1 | 14 → 14 | unchanged | 2160 → 2160 | unchanged | 1.21 px | 4 | unchanged |

Only the two rotation-core captures move; the other two have no rotation core
to grow and change only in stored point error. **The trade is accepted on the
campaign's standing acceptance bar** — a seed is a solid starting point to
refine from, not a frame-count maximiser (see
`feedback_seed_stage_precision.md`). Losing 7 and 3 posed frames buys +145 and
+1156 points, +985 and +5339 observations, and on KerryPark480 a 3.4×
improvement in post-BA reprojection median; the cameras that drop out are the
ones whose translation was only ever resectable because the gate had thrown
away the periphery that disagreed with them. The one cost worth naming is
KerryPark480's shipped focal: its census arbitration now endorses the structure
candidate (153.28) instead of falling back to the vote (138.31), i.e. **+13.1%
against the best-fit equidistant optimum 135.52, up from +2.06%**. That is the
keep-best/arbitration interaction Phase 4 already carried forward, now visible
on a healthier solve rather than a new defect.

**The seven inherited items.**

1. **`_in_field` weaker than `−z > 0` — measured, no change needed.** Measured
   where the cull actually bites (the post-BA state, stage dump 13 on
   KerryPark480, where the BA has pushed points out to θ = 174°):

   | reading | observations culled | points culled |
   | --- | --- | --- |
   | imaged cone `θ ≤ 99.4°` (shipping) | 339 / 11738 | 114 |
   | pinhole half-space `−z > 0` (pre-Phase-4) | 987 / 11738 | 547 |
   | range > 0 **and** projection in frame | 295 / 11738 | 104 |
   | cone **and** in-frame (the proposed conjunct) | 339 / 11738 | 114 |

   The proposed conjunct is EXACTLY the cone test — identical to the
   observation. In-frame alone is strictly weaker: 44 observations sit inside
   the image rectangle but outside the inscribed image circle (the corners),
   so swapping the cone for in-frame would loosen the cull rather than tighten
   it. The Phase-4 worry is answered in the negative, with numbers, and the
   guard is not vacuous — it culls 114 points the pinhole reading would have
   culled alongside 433 legitimate peripheral ones. No change.
2. **Patch extent growing with θ — measured, no change needed.** The projected
   half-size does grow (KerryPark480 1.28×, KerryPark360 1.51×, OmniCoast
   1.09×, **OmniTemple1 2.09×** at 90–105° against on-axis), but most of that
   is the projection's own doing: the equidistant map's azimuthal scale is
   `f·θ/sin θ`, which is **1.72×** in that band. Three of four captures grow
   LESS than the projection alone dictates; only OmniTemple1 exceeds it, by
   1.21×, and that residual is the reference-view rule (`ext = r_px·d_ref/f`,
   set once at the reference range). The decisive datum is the WITHIN-TRACK
   projected-size spread, which is what a per-reference extent actually risks:
   p50 **1.07** (OmniTemple1), 1.17 (KerryPark360), 1.32 (KerryPark480), 1.01
   (OmniCoast) — against **1.53** on the b100 pinhole control. Every fisheye
   entry holds its patch size across a track more tightly than the pinhole
   control does. With Phase 4's flat per-θ ZNCC table, the sizing is settled.
3. **`PatchCloud::from_tracks` — FIXED, and the brief's premise corrected.**
   The core signature gains `cam_ray_path: Option<&[bool]>` beside
   `cam_focals`, read only by `PatchExtent::PixelRadius`; `None` keeps the
   all-perspective reading. **The Python signature does not change**: the
   binding already holds whole `CameraViews` cameras and simply stops throwing
   the model away. *Correction to the Phase-4 note and to this phase's brief:*
   `sfm xform … --to-embedded-patches extent=pixel_size` goes through
   `from_reconstruction`, **not** `from_tracks`, and has always read the
   models — `from_tracks` is reachable from Python only through the direct
   `PatchCloud.from_tracks` binding (tests and `scripts/`), and no seed path
   uses `PixelRadius`. So the fix is correctness for a public API with no
   production consumer today. Rust test
   (`from_tracks_pixel_radius_honours_the_ray_path_flag`: 0.184615 by range vs
   0.032058 by `|z|` at θ = 100°), Python test (`TestFromTracksCameraModel`),
   and a synthetic-gate check.
4. **The two model-conditional approximations — documented; the `compute_svd`
   drop-in REFUTED.** `view_selection::affine_core_map`'s 4th-corner residual
   already documented its blindness to symmetric curvature; what it did not say
   is that how much of the curvature is symmetric is a property of the CAMERA
   MODEL — under `θ = r/f` the radial compression `θ/sin θ` is symmetric about
   the patch centre at every image radius and is the dominant term, so on a
   fisheye view the residual measures the smaller half. What keeps it honest is
   that the asymmetric part scales with the same radius, so a patch large
   enough for the symmetric term to matter fails the 0.5 px bound anyway and
   falls back to the exact warp. `keypoint_subpixel`'s `grid_to_source_scale`
   is now documented as a three-corner SECANT over the whole core, blind to how
   the stretch varies across the grid (θ/sin θ again). **`map.compute_svd()` is
   not a cheap drop-in for it**: there is no `WarpMap` in scope at the gate —
   the first one is built INSIDE `render_refine_tile`, after the gate has
   decided — so using the SVD would cost a full `(R + 2·pad)²` per-pixel
   projection to decide whether to skip building exactly that tile, inverting
   the gate's purpose. `view_selection` faces the same trade and resolves it
   the same way with its closed-form `affine_sigma_major`. Deferred with the
   reason recorded, and the cheap precedent named (a 4th-corner residual, one
   more `ray_to_pixel`, not the SVD).
5. **Mixed px-unit gates — measured, quantified, deferred with the numbers.**
   Two measurements. First, the gate's nominal tightness: `max_shift_px = 3`
   image px expressed in patch-grid px, by θ band, on the shipped artifacts —

   | capture | on-axis | 75–90° | 90–105° |
   | --- | --- | --- | --- |
   | OmniTemple1 | 1.95 grid px | 1.27 (0.65×) | **0.94 (0.48×)** |
   | KerryPark360 | 1.81 | 1.34 (0.74×) | 1.20 (0.66×) |
   | KerryPark480 | 1.97 | 1.63 (0.83×) | 1.54 (0.78×) |
   | OmniCoast | 1.58 | 1.60 (1.01×) | 1.45 (0.92×) |

   At OmniTemple1's periphery a single grid-px step already exceeds the gate,
   so the discrete search can propose offsets the gate must refuse. Second, the
   DROP RATE that decides materiality — the production localizer run at
   `max_shift_px` 3 vs 6, surviving observations by θ band:

   | capture | 0–30° | 75–90° | 90°+ |
   | --- | --- | --- | --- |
   | OmniTemple1 | 1.125× | **1.228×** | **1.252×** |
   | KerryPark480 | 1.106× | 1.096× | 1.106× |
   | KerryPark360 | 1.018× | 1.025× | 1.013× |
   | OmniCoast | 1.005× | 1.066× | 1.408× (n = 49) |

   The asymmetry is real and tracks the grid-px measurement (OmniTemple1's
   0.48× is the worst gate and the worst asymmetry; Kerry's 0.66–0.78× shows
   none), but it maxes at **+11% relative** on one of four entries and is
   absent on two. The larger effect is that the gate is globally tight — at
   3 px it drops 3.4–17.7% of observations — which is a different question and
   a pinhole-branch one. Judged below the bar for changing a production
   localization gate's units; the measurement and the harness
   (`fisheye_p5_shift.py`) are recorded so Phase 6 can decide with them.
6. **Spec corrections — DONE.** `specs/core/patch/fronto-parallel-patch-cache.md` no
   longer claims "the map is exact for *any* camera model": the corner
   parameterization is now stated per family (undistorted-normalized for the
   perspective family, the model's own `ray_to_pixel` for ray-path models,
   because `x/z` blows up at 90°), the composed map is described as
   *parameterization*-independent rather than projection-independent, and a new
   Limitations entry names projection curvature over the patch as a
   camera-model limit — symmetric about the patch centre under the equidistant
   map, and invisible to a three-corner fit that (unlike
   `view_selection::affine_core_map`) never even looks at its fourth corner.
   The "makes the cache exact for distorted and fisheye rigs" design bullet and
   the two "distortion-independence" test claims are restated as bounded
   divergence. `specs/core/patch/patch-normal-refinement.md`'s "so distortion /
   fisheye work unchanged" is scoped to the source-rendering path it is true of,
   with the fronto cache and the model-dependent patch sizing named as the two
   things it does not cover.
7. **Far-cluster handoff — PLUMBING FIXED, admission left conservative.**
   > _Status (2026-08-11): Superseded — the id bridge below is replaced by the
   > cluster-restriction stage (see "Restriction as a pipeline stage")._
   > _Status (2026-08-11): Removed — the whole channel is gone, ablated and
   > deleted (see "Far-field side channel removed"). The opt-in flag, the
   > carried ids and the forcing site no longer exist._

   The finalization re-derived its own restricted selection of the source
   file, so stage-1 cluster ids meant nothing to it. Both loaders recorded each
   cluster's SOURCE id (`selected_cluster_ids`, `src_cl`) and
   `bridge_cluster_ids` composed the two arrays. Admission is
   opt-in (`SFMTOOL_FAR_BRIDGE=1`) because whether a rotation-certified far
   field is worth reconstructing is a content question this seed stage does not
   settle — see the Phase-4 post-inspection caveat. Measured with it ON:

   | capture | stage-1 far clusters | bridged | survivors in them | already native-demoted | forced on H evidence | points | reproj |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | KerryPark480 | 4911 | **4429** (482 lost to restricted admission) | 775 | 15 (2%) | **3** | 1154 → 1141 | 1.99 → 1.89 px |
   | OmniCoast | 16066 | **1027** (15039 lost) | 168 | 1 (1%) | **1** | 1409 → 1410 | 0.65 px |
   | KerryPark360, OmniTemple1 | none certified | — | — | — | — | unchanged | unchanged |

   So bridging alone degrades nothing — but it also barely does anything, and
   the reason is a SECOND bottleneck the id bridge does not touch:
   `_far_point_mask` identifies a seed point in the embedded cloud by its exact
   float64 position, and the collapse merge re-triangulates, so 542 / 357 of
   the far-field points are already unmatched by the time the evidence arrives.
   Fixing the id bridge exposed that; carrying a stable per-point identity
   through the collapse is the actual work, and it is Phase 6's.

**Gate numbers.**

- **Full suites.** `cargo test --workspace`: every target passes except
  `sfm-explorer --test ui_basic`'s
  `a_real_right_click_opens_the_reconstruction_rows_context_menu`, which fails
  identically at plan-HEAD with no changes applied (it needs a real focused
  window; verified by rerunning it on the stashed tree). `cargo clippy
  --workspace --all-targets` and `pixi run doc` clean; `cargo fmt` applied.
  `pixi run -e test test`: **2103 passed, 1 skipped**. `ruff format --check`
  and `ruff check` clean on every touched file.
- **Full pinhole fleet A/B, gate OFF — PASS, and it covers everything except
  the resection gate.** True A/B (fresh runs at plan-HEAD and at the tree
  holding every Phase-5 change *except* the resection fix, each with its own
  `maturin develop --release`), all 40 wslist entries, artifacts captured per
  arm. **36 of 36 pinhole entries byte-parity clean**: 25 of 27 ZIP payload
  entries identical, the two movers `metadata.json.zst` and its derived
  `content_hash.json.zst`; the metadata blob diffed key by key is 12 of 13
  identical with only `timestamp` moving (checked on DnDTabletop,
  MurdoSmallAntiqueCat, 20250907_000240907, vid2). The four fisheye entries are
  the only ones that move.
- **Pinhole parity for the resection gate — structural claim + spot check.**
  The full fleet was NOT re-run for it. The gate is unreachable for the
  perspective family by construction: `needs_ray_path()` is false there and the
  `else` arm is the character-identical `c.z < 0.0` expression that preceded the
  branch. That is asserted directly by
  `perspective_trim_gate_is_bit_identical_to_the_half_space`, which drives a
  pinhole and a `SimpleRadial` camera over geometry straddling the camera plane
  and requires every `z ≥ 0` observation to come back `INVALID_RESIDUAL` rather
  than range-scored. Empirically, a three-dataset spot check isolating this
  change alone (same tree ± the resection fix): **b100, DnDTabletop and
  20250907_000240907 all 25/27 entries identical**, movers `metadata.json.zst`
  (12/13 keys identical, `timestamp` only) and `content_hash.json.zst`. b100
  also reproduces its canonical figures exactly — 14/100 posed, 62.4% inlier,
  f = 355.1, seed 423 points at 0.21 px.
- **Fisheye, gate ON — the resection fix is the mover; nothing else regresses.**
  Final numbers in the resection table above: posed 36/13/8/14, points
  1299/493/2566/2160, post-BA reproj median 0.58/0.60/0.67/1.21 px. Points and
  observations rise on both rotation-core captures and are unchanged on the
  other two; photometric census and vet statistics stay in family.
- **Stage-dump waterfall — runs end to end on KerryPark480 (gate ON).** 26
  checkpoints, stages 00–25, embed 4880 pts → len3 2668 → size-cull 2498 →
  cheirality 2497 → covis-camera-cull 1629 → ARS promote 1783 → BA 1774 →
  late fit 1210 → late vet 1189 → 1156 → 1154 → final 1154, with the per-image
  observation census and delta at every stage.
- **Explorer artifacts prepared** —
  `C:/DataSets/workspace-prep/phase5-explorer-inspection/` holds before/after
  `seed-final.sfmr` for all four entries plus a README naming what changed and
  what to look at. Human inspection is NOT done. **The Kerry entries' content
  is known non-static-dominated** (the Phase-4 post-inspection caveat): for
  those two the inspection question is machinery sanity — does the periphery
  project, shade and render like the rest of the cloud — not scene quality.
  OmniTemple1 is the entry to judge scene quality on.
- **Canonical protection.** Fresh `fisheye-p5-canon-backup` over all 40
  workspaces (never reused), with a pre-run `sfmr/` + `sfmr/seed-rounds/`
  listing; 40/40 restored byte-exact; every `p5*` round snapshot pruned. The
  listing check reported two workspaces "dirty" — a separator artifact, since
  their solve artifacts carry frame-range names containing commas and the
  listing was comma-joined; verified directly that both have zero Phase-5
  leftovers, and the harness now joins on `|`.
- **`scripts/check_fisheye_seed_primitives.py` exits 0**, with a new Phase-5
  suite: rotation-locked resection on a planted wide scene keeps **120 of 120**
  observations including **25 of 25** past 90° (the half-space gate kept 0 of
  them), recovers the planted translation to 1.6e-16 and still refuses the
  antipodal reflection; `from_tracks` sizes a 100° `pixel_radius` patch by
  range; and the far-cluster handoff carries per-cluster evidence into the
  restricted numbering, including the "did not survive the restriction" case.

**What Phase 6 inherits.**

- **`max_error_px = 8.0` in the rotation core's resection**, now that the
  periphery is actually in the solve. The trim bar was chosen against an
  on-axis-only observation set; whether 8 px is still the right gate — and
  whether the posed frames the fix drops (KerryPark480 43 → 36, OmniCoast
  11 → 8) come back under a re-derived one — is unmeasured.
- **KerryPark480's focal arbitration.** With a healthier structure candidate
  the finalization's census now endorses it (153.28) over the vote (138.31),
  taking the shipped focal from +2.06% to +13.1% against the best-fit
  equidistant optimum. This is the keep-best/arbitration interaction Phase 4
  carried forward, and it is now the load-bearing one on this capture.
- **The far-field handoff's second bottleneck**: exact-float64 position
  matching in `_far_point_mask`, which the collapse merge breaks for most
  far-field points. The id bridge is done; a stable per-point identity through
  the collapse is not.
  > _Status (2026-08-11): Moot — the uid carry closed this, and the channel it
  > served has since been removed ("Far-field side channel removed")._
- **`max_shift_px` in image px against a grid-px search**, measured above:
  material on OmniTemple1 (gate 0.48× on-axis in grid px, +11% relative
  peripheral drop), absent on the Kerry entries. Changing it touches the
  pinhole branch.
- **The keep-best-inlier rule** (unchanged since Phase 4, still needs its own
  36-dataset fleet A/B) and the `err%`-against-best-fit-equidistant fleet
  column.
- **Human Explorer inspection of the Phase-5 set** is prepared, not done.
- The promotion gate this phase exists to justify is **met**: the finalization
  chain is model-correct, the full pinhole fleet is byte-inert, and the fisheye
  entries carry no regressions. Flipping the fisheye branch from env-gated to
  default-on-confirmed-verdict is Phase 6's to do.

## Phase 6 — Fleet integration and GT

- Kerry entries: err% against rig fx as a standing fleet column;
  consider promoting a cleaned Kerry reconstruction toward the
  approved-GT pipeline (human inspection).
- OmniPhotos entries: cross-scene focal consistency (same lens) as
  their standing check; optionally enlist `omni_Hilltop` as the third
  scene.
- Retire the `fisheye_detected`-but-degraded status quo: a confirmed
  verdict now routes to the fisheye branch by default; the annotation
  path remains as the fallback for unconfirmed verdicts.

### Phase 6 outcome (2026-08-10) — DONE

The branch is default-on, the fleet columns mean what they say, and the two
items this phase inherited as "measure it" both come back with a number rather
than a change. The headline is that **two of the brief's premises did not
survive measurement** — the FeatureSize fix turns out to be fleet-inert, and
the keep-best rule is no longer where KerryPark480's focal error comes from —
and the one change that does move an artifact is the arbitration fix those
measurements pointed at instead.

**Item 1 — FeatureSize through the uniform Jacobian. DONE, and fleet-INERT.**
`PatchExtent::FeatureSize`'s finite branch sized every model by `σ·‖p_cam‖/f`;
it now routes through `CameraIntrinsics::pixel_radius_to_world` with the
observation's own `σ` as the pixel budget, i.e. the identical rule
`PatchExtent::PixelRadius` uses at a per-observation radius. The infinity branch
already went through `pixel_radius_to_angle` and is unchanged, so both policies
now state one thing and `PatchScene` no longer exposes a bare focal at all.
`specs/core/patch/patch-cloud.md` says it as one rule at two pixel budgets.

The brief predicted this would move every fleet entry. **It moves none**, and
the reason is worth recording because it is not obvious from the call graph: the
seed's `save_sfmr` writes `feature_source="embedded_patches"` with its own
warp-derived extents, so `embed_patches` takes the `embedded = recon` branch and
never calls `to_embedded_patches` — the only `FeatureSize` consumer. The
full-fleet A/B confirms it: **40 of 40 entries identical** on posed / points /
infinity / cameras / observations / reprojection median / focal / err% / flags,
and byte-identically so — 25 of 27 ZIP payload entries match on every one of the
40, the two movers being `metadata.json.zst` (creation timestamp) and its
derived `content_hash.json.zst`.

So this is a correctness fix to a **production path the seed fleet does not
use** — `sfm embed-patches` and `sfm xform --to-embedded-patches` on a
`sift_files` reconstruction, which is the GT-cleanup recipe's own first step.

Measured there instead, on three real `sift_files` reconstructions
(`fisheye_p6_extent.py`, half-extent ratio after/before by off-axis angle):

| reconstruction | camera | θ p50 | ratio p50 | 0–10° | 20–30° | 30–45° | 45–60° | 60–75° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DnDTabletop solve (132 965 pts) | SIMPLE_RADIAL | 21.9° | 0.927 | 0.992 | 0.905 | 0.842 | — | — |
| Murdo solve (141 512 pts) | SIMPLE_RADIAL | 21.5° | 0.925 | 0.991 | 0.901 | 0.838 | — | — |
| KerryPark360 `recon-infinity-gated` (109 594 pts) | OPENCV_FISHEYE | 52.1° | 0.932 | 1.000 | 0.979 | 0.954 | 0.917 | 0.879 |

The two perspective reconstructions track `cos θ` to within a percent
(0.992 vs cos 0-10° = 0.991; 0.842 vs 0.848 at 30–45°, the small extra being the
radial distortion magnification the old rule ignored) — that is exactly the
`sec θ` the range reading was over-sizing by. The polynomial fisheye does NOT
track `cos θ`, correctly: its `σ_min` carries `dr_d/dθ ≠ f`, so its patches
shrink 12% at 60–75° and are unchanged past 75° where the KB principal branch
ends and the rule falls back to the range reading. All three shrink; none by
more than ~16%.

Rust test (`from_tracks_feature_size_reads_the_view_camera_model`) and a
synthetic-gate check pin the two closed forms at 55° off axis.

**And the seed writer now uses that core rule instead of its own copy.** The
reason the fleet does not reach `FeatureSize` is that `save_sfmr` sized its
surfel frames itself, with a hand-written `r_px · scale_ref` where `scale_ref`
was `z_ref/f` on the pinhole arm and `d_ref/f` on the fisheye arm. Two
implementations that agree today is the duplication class this campaign exists
to remove — it is exactly how the writer's hard-coded `SIMPLE_PINHOLE` survived
to Phase 3b, sizing a fisheye seed through a pinhole camera. The writer now
calls the camera itself: a new binding
`CameraIntrinsics.pixel_radius_to_world_batch` (the position-anchored sibling of
the existing `pixel_radius_to_angle_batch`) is evaluated once for every point's
reference view, and the Python-side formula is deleted. `scale_ref` survives
only where it means something else — the in-plane scale of the reference right
inverse in the tilt solve.

**Equivalence proven at the switchover**, which is what licensed the deletion:
on one pinhole entry (`20250907_000240907`) and one fisheye entry
(`KerryPark480`), the produced `seed-final.sfmr` is **byte-identical** to the
Arm-B artifact — 25 of 27 ZIP payload entries match, the movers being only
`metadata.json.zst` (timestamp) and its derived `content_hash.json.zst`. The two
formulas even round identically: `r_px·(z/f)` and `(r_px·|z|)/f` associate
differently, so ULP-level divergence was the expected risk, and it did not
occur on either capture. One behavioural difference is worth naming because it
is latent rather than observed: the old pinhole arm clamped a NEGATIVE reference
depth to `1e-6` (a zero-extent frame, i.e. no patch), while the core reads
`|z|`. No point in either capture exercises it — a reference member is a real
observation — but the core's reading is the correct one, and the old clamp was a
defensive hack rather than a rule.

**Item 2 — the focal arbitration. FIXED, and the brief's diagnosis is
superseded.** Re-measured at this HEAD with `SFMTOOL_BA_DEBUG=1`, KerryPark480's
free-focal release walks **159.06 → 153.24 → 151.83** and stops (two rounds, the
third is inside the 1% stability bar). Phase 4's measurement — 159 → 139, "the
BA's own soft-L1 optimum walks toward the verdict" — was taken *before* Phase
5's resection fix, and on the healthier solve that fix produces **it no longer
holds**. So candidate (a), scoring keep-best on the release's own objective,
would move the reported focal by 0.9% (153.24 → 151.83) against a 13.1% error.
**Keep-best is not the defect any more and was left alone**, which also removes
the pinhole-fleet risk that changing it carried.

What remains is a genuine disagreement between two measurements. The census
arbitration prefers the structure candidate on this capture — census 0.416 for
153.3 against 0.865 for the vote's 138.3 — while the rig says 138.3 is right
(+2.06% against the best-fit equidistant 135.522) and 153.3 is not (+13.1%).
The census is not a neutral referee here: **it is computed from the same
structure it is judging**, triangulating the raw clusters at the candidate's
poses, so on support that is internally self-consistent without being the
static scene it endorses the contaminated focal. The coverage-complement
experiment is the control — on the same capture's clean complement support the
free-focal release lands 0.7% from the vote, and the arbitration never has to
choose.

So the fix is candidate (b), made data-derived: **the census may not endorse a
structure candidate that the structure-free vote contradicts beyond the vote's
own measured precision.** The band is not a constant — it is the vote pool's own
log-focal IQR (`pool_spread`, the focal-vote kernel's existing diagnostic), the
instrument reporting its own dispersion, floored at 2%. The floor is the one
number chosen rather than measured per run, and it is chosen from measurements:
the equidistant column reads +2.06% / +1.93% against the two rig-calibrated
captures, and a release on clean support agrees with the vote to 0.7%, so 2% is
the largest of the measured disagreements and therefore the choice most
permissive to the structure candidate. The rule is shared with the pinhole
branch, parameterized by each column's own dispersion rather than gated on
which branch is running.

Why it is inert nearly everywhere, with the numbers. The veto can only ever
remove a structure candidate, so the only entries at risk are the ones whose
census currently prefers structure — **three of the fleet's forty**:

| entry | vote (centre) | structure | disagreement | vote pool log-IQR | contradicted? |
| --- | --- | --- | --- | --- | --- |
| KerryPark480 | 138.31 (×1.0) | 153.28 | **0.1027** | **0.00076** (equidistant column, 40 votes) | **yes** |
| 20250712_202131684 | 2488.6 (×1.1 → 2737.5) | 2572.4 | 0.0622 | 0.1830 | no |
| 20240906_081206935 | 1402.8 (×1.1 → 1543.1) | 1477.3 | 0.0436 | 0.0548 | no |

The Kerry entries' equidistant votes are effectively unanimous (log-IQR 0.00076
and 0.00039 over 40 and 52 votes) while the pinhole column's runs 0.027–0.399
across the fleet (p50 0.129), which is why one measured band separates the two
regimes without a branch test. The band centre carries the pinhole vote's known
×1.1 bias correction; the fallback candidate is still the raw vote, as before.

Fleet A/B, Item 1 + Item 2 against Item 1 alone: **KerryPark480 is the only
entry whose geometry, focal or point set moves**; 39 of 40 are identical on
every column. One further entry, `20240614_203547691`, gains the
`vote_contradiction` flag alongside the `census_guard` it already carried —
same decision (keep the vote), same artifact, now with both reasons recorded
rather than one. (The first pass of this arm recorded only the contradiction
there, which dropped the census-guard fact; the two are different statements
about the structure candidate and both are kept.)

| KerryPark480 | Item 1 only | Item 1 + 2 |
| --- | --- | --- |
| shipped focal | 153.277 px | **138.310 px** |
| err% vs best-fit equidistant 135.522 | +13.1% | **+2.06%** |
| points / at infinity | 1299 / 10 | 891 / 72 |
| per-point reproj median | 0.671 px | 0.750 px |
| posed | 36 | 36 |

The accuracy win costs 31% of the points and 0.08 px of stored reprojection: at
the smaller (correct) focal more tracks are depth-unresolvable, so the
classifier demotes them and the reprojection cull bites harder. **That trade is
staged for human inspection rather than declared a win** — the campaign's
acceptance bar is a refinable starting point, not a point count
(`feedback_seed_stage_precision.md`), and this capture's cloud is known to be
non-static-dominated, so the columns cannot settle it. The A/B handle
(`SFMTOOL_VOTE_BAND=0`) is kept so the two arms stay reproducible.

**Item 3 — `max_error_px = 8.0`. MEASURED, no change.** Swept through the new
`SFMTOOL_RESECT_MAX_PX` knob with `SFMTOOL_RESECT_TRACE=1` recording each
resection's kept fraction and residual distribution.

| KerryPark480 | bar 4 | 6 | **8** | 10 | 12 | 16 | ∞ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| posed | 37 | 36 | **36** | 39 | 43 | 46 | 48 |
| stage-1 inlier<2px | 66.4% | 67.6% | **66.0%** | 64.4% | 64.8% | 55.9% | 56.4% |
| points | 1032 | 732 | **891** | 1006 | 1089 | 1339 | 1424 |
| reproj median | 0.714 | 0.865 | **0.750** | 0.795 | 0.765 | 0.725 | 0.735 |
| rot agreement vs the bar-8 solve (med / max) | 0.58/1.82° | 1.12/7.07° | — | 0.61/4.20° | 0.49/3.13° | 1.88/3.96° | 1.67/3.90° |
| centre err (% of subset diameter) | 7.7 | 5.1 | — | 7.8 | 6.4 | 8.9 | 8.1 |

Three findings. (1) **The 7 frames Phase 5's resection fix dropped do come
back** — at bar 12 the solve is back to 43 posed — **and they are not garbage**:
the cameras the two solves share agree to 0.49–0.61° median with no systematic
centre drift. (2) **Nothing improves.** Reprojection is flat at 0.71–0.87 px
across 4–12 with no ordering, points rise with the frame count, and past 12 the
stage-1 consensus falls below the 0.60 healthy band and flags the seed. (3) On
OmniCoast the sweep does not even order: posed 12/8/11/11/11 and reproj
6.63/1.10/0.44/0.44/0.42 px at bars 4/8/12/16/∞, with the wide bars landing the
rotation core in a *different* window (bar 16 disagrees with the bar-8 solve by
4.2° median / 26° max on their common cameras). The bar is not a smooth quality
knob on either rotation-core capture.

And 8.0 is already where the data would put it. At the shipping bar KerryPark480
keeps 2517/5168 = 48.7% of the resection's observations with a per-resection
residual p50 of 3.86 px and p90 of 13.32 px (a further 27.8% score out of the
imaged cone entirely and are unscoreable at any bar); interpolating, **8.0 px is
the ≈p73 of the scoreable residual distribution**, which is the trim point a
quantile rule would choose. OmniCoast keeps 60.8% at 8 px against a p50 of
4.79 px. **Closed with the numbers: 8.0 stands**, and the two knobs stay so the
measurement is reproducible.

**Item 4 — far-point matching. FIXED (mechanism), admission unchanged.**

> _Status (2026-08-11): Removed — the mechanism below was measured on the whole
> fleet and deleted with the rest of the channel (see "Far-field side channel
> removed"). The reading it produced stands and is the reason: the
> rotation-certified far field on these captures is largely not at infinity by
> its own observations._

`_far_point_mask` keyed the far field by exact float64 position, resolved at the
demotion site — downstream of a duplicate collapse and an alias reconcile that
both re-triangulate. It is replaced by `_far_point_uids`, which resolves the
same evidence ONCE at the embed (where the position identity is still valid) to
the per-point UID the finalization already carries through every cull for
`det_size` and the infinity trace; the mask at the demotion site is then a set
membership, which no re-triangulation can break. Measured with
`SFMTOOL_FAR_BRIDGE=1`:

| capture | far points reaching the demotion (Phase 5 → now) | unmatched by position (Phase 5 → now) | proposed | vetoed | forced | points | reproj |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KerryPark480 | 775 → **1011** | 542 → **0** | 1004 | 994 | 3 → **10** | 891 → 912 | 0.750 → 0.747 px |
| OmniCoast | 168 → **151** | 357 → **0** | 150 | 150 | 1 → **0** | 2566 → 2566 | 1.104 px |
| KerryPark360, OmniTemple1 | no clusters certified | — | — | — | — | unchanged | unchanged |

The identity defect is gone — nothing is lost to position matching any more, and
the evidence arrives at the gate at full strength (1004 proposals where Phase 5
saw a handful). **The binding constraint is now the bearing veto, not the
identity**, and its numbers are the interesting part: it refuses 994 of 1004 on
KerryPark480 and 150 of 150 on OmniCoast, at a bearing-fit median of 12.29 px
and 3.38 px against a 1.00 px noise floor. That is a measurement, not a
threshold artifact: **the rotation-certified far field on these captures is
largely not at infinity by its own observations** — consistent with the Phase-4
caveat that KerryPark480's far field is moving cloud, which carries real (if
false) parallax. Forcing what survives the veto neither harms nor helps
(+21 points, −0.003 px on KerryPark480; nothing on OmniCoast). Admission stays
opt-in, as scoped.

**Item 5 — fleet and GT integration.**

(a) **The equidistant reference is the standing fleet column.** `seedgc_run.py`
already preferred `equidistant_gt.py` over `resolve_gt` wherever a KB
`rig_config.json` exists and recorded the choice in `gt_src`; what was stale was
the fleet's own documentation. `wslist-notes.txt`'s two Kerry entries said
"PINHOLE-INCOMMENSURABLE, do NOT read the seed's err%", which was true only
while those captures were solved with a pinhole model. They now record the
best-fit equidistant focals (135.522 / 271.321 px) as the reference, why the rig
`fx` is not it (a perfect equidistant estimator reads +4.93% / +5.19% against
`fx`), and that the non-static-dominated caveat is unchanged. The Phase-6 fleet
harness (`fisheye_p6_run.py`) carries the same `gt_src` convention. The old
`seedgc-results.tsv` snapshot predates the `gt_src` column and is moved aside to
`seedgc-results-pre-gtsrc.tsv` so the next fleet run writes the current header
rather than appending to a narrower one.

(b) **OmniPhotos cross-scene focal consistency, three scenes.** `omni_Hilltop`
is promoted to `C:/DataSets/OmniHilltop` and enlisted as fleet entry 41. It was
prepared for the focal-vote prototype, which needs only a clusters `.matches`,
so the seed's `-clusters-patches` file had to be built first
(`sfm cluster-patches`, 230 963 patches). Same body, same lens, same 1920×1920
crop, three unrelated scenes and no calibration anywhere, so the spread of the
shipped equidistant focals is the only reproducibility statement available:

| scene | verdict margin | equidistant cells (epi / rot) | shipped focal | vs the three-scene mean |
| --- | --- | --- | --- | --- |
| OmniCoast | 5.36× | 15 / 44 | 583.99 px | +3.78% |
| OmniHilltop | 4.29× | 15 / 15 | **581.94 px** | +3.41% |
| OmniTemple1 | 1.46× | 15 / **4** | 522.26 px | −7.19% |

**Coast and Hilltop agree to 0.35%** — two unrelated scenes through one lens,
solved independently, landing 2 px apart on a 1920 px sensor. OmniTemple1 sits
10.4% below their mean, and it is exactly the entry the fleet already flags: the
thinnest verdict margin of the four fisheye captures, a rotation cell carrying 4
units of model-informative mass against Coast's 44, and a `low_consensus` seed.
So the three-scene check is informative rather than decorative — **the verdict
margin predicts the focal's reproducibility**, which is a statement no
two-scene comparison could have made (with two scenes a 10% disagreement has no
baseline to be judged against). Hilltop itself solves cleanly: 13/156 posed,
60.3% inlier, 3148 points at a 0.49 px reprojection median, and its census
arbitration keeps the vote.

(c) **GT candidacy: nothing approved, one entry staged.** Per
`project_gt_approval_convention.md` a candidate needs a human pass first. The two
Kerry entries are ruled out for now on content, not on solve quality — their
clouds are false-parallax dominated and the static scene is largely absent, and
the static/non-static separation experiment that would change that is outside
this plan. OmniTemple1 is the best-looking fisheye solve on the fleet and the one
Phase 4's inspection called a plausible starting point, but it has **no
independent reference of any kind**, so approving it would mean approving a
self-consistent solve on its own word. It is **staged** in
`phase6-explorer-inspection/` with that stated; `approved-gts.tsv` and every
`sfmr/cleanup/APPROVED.md` are untouched.

**Item 6 — the default-on flip. DONE.** A confirmed both-cells verdict installs
the equidistant camera context by default. `SFMTOOL_FISHEYE_SEED` becomes a
tri-state override, factored into a testable `fisheye_routing_override(value)`:

| setting | behavior |
| --- | --- |
| unset | route on a confirmed verdict (the new default) |
| `"1"` | identical to unset — the explicit opt-in every Phase-1..5 harness passes, kept a no-op so they all keep working |
| `"0"` | never route: the verdict is annotation only and the seed solves pinhole and degrades gracefully, i.e. the pre-Phase-6 status quo |

The obvious alternative — make `"1"` force routing *regardless* of the verdict —
is **declined, and the reason is measured**: run unconditionally the bare
arbitration returns EquidistantFisheye on three rectilinear fleet captures
(BadlandPanorama, 20240614_224244438, MossyRailing), all three winning on the
epipolar cell with exactly zero rotation-cell mass, which is precisely what the
both-cells confirmation rule exists to reject. A force switch would route a
rectilinear capture to a fisheye seed. And a capture with no verdict at all has
no equidistant focal to build a context from, so there is nothing for a force
switch to force. The override is therefore a veto, never a promotion.

The flip is invisible to the pinhole fleet by construction and by measurement:
no fleet entry earns a confirmed verdict except the four fisheye ones, and the
36 pinhole entries are byte-identical across the Item-1 arm (which runs the flip
with the variable unset) and the base arm. The four fisheye entries were checked
directly (`fisheye_p6_flip.py`) — with the variable **unset** all four route and
reproduce the `"1"` arm's artifact to 25 of 27 ZIP entries (the two movers being
the timestamped metadata and its derived hash), and with `"0"` all four decline
to route and fall back to the pinhole vote exactly as before the flip:

| entry | unset | `"0"` |
| --- | --- | --- |
| KerryPark480 | routed, 36/48 posed at f = 153.2 | not routed, 13/48 at f = 144.0 |
| KerryPark360 | routed, 13/82 at f = 276.6 | not routed, 12/82 at f = 433.7 |
| OmniCoast | routed, 8/88 at f = 584.0 | not routed, 13/88 at f = 777.1 |
| OmniTemple1 | routed, 14/80 at f = 522.3 | not routed, 11/80 at f = 1139.9 |

The `"0"` column is also the clearest statement of what the flip is worth: those
focals are the pinhole votes, 4–118% away from each capture's own equidistant
focal, on a solve that knows it cannot model the capture.

**Gate numbers.**

- **Full suites.** `cargo fmt` applied; `cargo clippy --workspace --all-targets`
  and `pixi run doc` clean; `cargo test --workspace` green on every target
  (including `sfm-explorer --test ui_basic`, which Phase 5 recorded as failing —
  it needs a real focused window and fails whenever the machine is busy; it
  passes on an idle one). `pixi run -e test test`: **2106 passed, 1 skipped**
  (2101 before, plus the five new `pixel_radius_to_world_batch` binding tests).
  `ruff check` and `ruff format --check` clean on every touched file. Two
  pre-existing `ruff check` findings elsewhere in `scripts/` (`exp_hier_ba.py`
  F841, `exp_plus_descent_localize_compare.py` F541) are untouched by this
  phase.
- **`scripts/check_fisheye_seed_primitives.py` exits 0**, with a Phase-6 suite:
  `FeatureSize` sizes a 55°-off-axis patch by `σ·R/f` under the equidistant
  model and by `σ·|z|/f` under a pinhole (0.230769 vs 0.132364 on the same
  geometry); the far-field uid mask survives a simulated collapse rewrite that
  defeats the position-keyed map outright; and the routing override is
  tri-state.
- **Full-fleet A/B, three arms**, all 40 entries, artifacts and logs captured per
  arm: base (plan HEAD) → A (Item 1, `SFMTOOL_VOTE_BAND=0`) → B (Items 1 + 2).
  A vs base **40/40 identical** (and byte-parity clean 40/40); B vs A **37/40
  identical**, with KerryPark480 the only entry whose geometry moves and two
  entries (`20240614_203547691`, `20250906_211742965`) gaining the
  `vote_contradiction` flag beside the `census_guard` they already carried —
  same decision, same artifact, both reasons now recorded. Byte-parity B vs A is
  **39/40 clean**, KerryPark480 the sole genuine mover. The fisheye entries ran
  on the fisheye branch in every arm (`SFMTOOL_FISHEYE_SEED=1`), so the arms
  measure the changes and not the flip. The seed-writer unification landed after
  arm B and is covered by its own byte-identity proof (above) on one pinhole and
  one fisheye entry rather than a fourth fleet pass.
- **Canonical protection.** Fresh `fisheye-p6-canon-backup` md5-manifest dir over
  all 40 workspaces (never reused), with a pre-run `sfmr/` + `sfmr/seed-rounds/`
  listing; **40/40 restored byte-exact**, every Phase-6 round snapshot pruned,
  and **40/40 verified diff-clean** against the listing afterwards. OmniHilltop
  is appended to `wslist.txt` only after that check, so it carries its own first
  canonical rather than reading as a dirty workspace.
- **Explorer artifacts staged** — `C:/DataSets/workspace-prep/
  phase6-explorer-inspection/` holds before/after `seed-final.sfmr` for all four
  fisheye entries plus a README naming what moved and what to judge. Human
  inspection is NOT done.

**What remains open after this plan.**

- **The multi-hypothesis / static-vs-non-static extraction campaign.** Scoped as
  its own experiment since Phase 4 and unchanged: the Kerry entries' clouds are
  false-parallax dominated, and the coverage-complement experiment showed the
  complement support solves better on every measure taken. Until that lands,
  those two entries' point counts are machinery checks and they cannot be GT
  candidates.

  > _Status (2026-08-13): Landed as a pipeline mechanism — the seed stage is a
  > hypothesis loop (`specs/core/geometry/seed-hypothesis-loop.md`): each committed
  > hypothesis claims image area at its retained members' keypoints and the next
  > one is explored on the coverage complement, admitted as a
  > `restrict_cluster_ids` selection of the stage's own handle. The external
  > `SFMTOOL_SEED_EXCLUDE_MASKS` hook that carried the experiment is retired
  > with it. KerryPark480 commits three hypotheses, two of which qualify — the
  > 36-image 66.0% release that shipped before, and a 15-image 53.3% one on the
  > complement support; the arbitration keeps the first, so the artifact is
  > unchanged and the second is now recorded rather than invisible. Whether
  > either is a GT candidate is still a human-inspection question._
  >
  > _Status (2026-08-14): The loop is revised, against the same spec file, after
  > a 41-entry census of the 08-13 mechanism (`workspace-prep/fleetv5-parity/`)
  > measured it enumerating FRAME WINDOWS rather than worlds. Detector-scale
  > discs stamped on posed frames only claimed 0.2–3.6% of frame area, so 40 of
  > 41 complements retained 89–99.8% of the admission; every entry committed at
  > least two hypotheses (max 24), the complement passes ate 51% of fleet wall,
  > and the median posed-set Jaccard between a capture's own hypotheses was
  > 0.000 — with seven entries shipping a re-roll that outranked a healthy h0 on
  > raw inlier fraction. Three mechanism changes. The claim is now an adaptive
  > OCCUPANCY GRID — cell size the image's median nearest-neighbour keypoint
  > spacing among the retained members — stamped TRANSITIVELY in every image a
  > retained cluster's members appear in, posed or not. A complement is explored
  > only when it is MATERIAL (retains under half the previous admission), with
  > one rescue look past an untrusted hypothesis. And a qualified challenger
  > displaces the earliest qualified incumbent only when the two are DISTINCT
  > (at least two shared posed images and above 5° median relative-rotation
  > disagreement there), which is also the new condition on
  > `multiple_hypotheses`. Every committed hypothesis's release now writes a
  > release-grade `.sfmr` under `sfmr/seed-hypotheses/<stamp>-h<k>.sfmr` on
  > every run, winner included. Fleet effect
  > (`workspace-prep/hyprev-20260814/`): hypotheses committed 164 → 65, max per
  > entry 24 → 2, `multiple_hypotheses` fires nowhere, one entry ships a non-h0
  > hypothesis (DnDTabletop, a rescue past an `edge_scan` h0), and 40 of 41
  > payloads are byte-identical to the pre-loop tree. Machine-normalized fleet
  > wall 26.9 min against 26.5 pre-loop and 39.3 for the first form._
  >
  > _KerryPark480 stays the only capture whose complement is material (27.3%
  > retained; the next entry is 82.2%), and it still commits two qualifying
  > hypotheses — but they share exactly TWO posed images and disagree by 3.7°
  > there, under the 5° bar, so the arbitration calls them one world. Two
  > shared images is one image pair, so that "median disagreement" is a single
  > measurement, and 3.7° sits inside the entry's own seed pose noise (mean
  > 1.74°, max 6.70° against the reference). The one capture the distinctness
  > test exists for is the one place it has almost no evidence to work with._
  >
  > _Cost of the bigger claim: both genuine rescues of the 08-13 census are
  > lost. `20240614_225938434` and `20240915_073131082` fail on coverage reach
  > alone (70% → 59% and 62% → 49% against the 60% bar) because reach is
  > measured on the hypothesis's OWN admission, so a claim that removes more
  > clusters removes covisibility edges and deflates it. Both revert exactly to
  > the pre-loop h0 payload, so nothing is worse than before the loop — the loop
  > just no longer improves them. Measuring reach on the CAPTURE-level
  > covisibility, the way the focal vote is already held capture-level, is the
  > obvious follow-up and needs its own fleet arm (`reach_of` also drives the
  > thinning ladder, so it changes which seeds later hypotheses find)._
  >
  > _Status (2026-08-14, second layer): the reach follow-up landed, together
  > with a COMBINATION stage and — the part that belongs in this file — a
  > correctness fix to the fisheye half of the hypothesis artifacts. The
  > artifacts were being written with the BOOTSTRAP's camera context still at
  > its `SIMPLE_PINHOLE` default: `set_camera_context` had exactly one caller,
  > `_write_seed_sfmr`, so the product was right while `seed_snap` and
  > `write_hypothesis_sfmr` reached `seed_snapshot` -> `dense_structure` ->
  > `save_sfmr` uncontexted and densified an equidistant solve through a
  > pinhole map. `bootstrap_module()` now resolves the bootstrap with
  > workspace AND context installed and every writer goes through it. The
  > effect is not a stamp correction — it is a structure correction, because
  > the pinhole densification's cheirality and reprojection culls keep only
  > the near-axis core the two maps agree on:_
  >
  > | entry, h0 artifact | before | after |
  > |---|---|---|
  > | `OmniHilltop` | `SIMPLE_PINHOLE`, **12** pts, 4 imgs | `EQUIDISTANT_FISHEYE`, **11558** pts, 13 imgs |
  > | `OmniCoast` | `SIMPLE_PINHOLE`, 84 pts, 6 imgs | `EQUIDISTANT_FISHEYE`, 6764 pts, 8 imgs |
  > | `OmniTemple1` | `SIMPLE_PINHOLE`, 166 pts, 13 imgs | `EQUIDISTANT_FISHEYE`, 8583 pts, 14 imgs |
  > | `KerryPark480` | `SIMPLE_PINHOLE`, 549 pts, 35 imgs | `EQUIDISTANT_FISHEYE`, 4512 pts, 36 imgs |
  > | `KerryPark360` | `SIMPLE_PINHOLE`, 157 pts, 10 imgs | `EQUIDISTANT_FISHEYE`, 1539 pts, 13 imgs |
  >
  > _Own-track reprojection of the new artifacts (each observation's `.sift`
  > keypoint through `track_feature_indexes`, the stored points through the
  > stored poses) is **0.32–1.10 px read as equidistant** and 63–493 px read as
  > pinhole, on all six fisheye artifacts — the solve's own structure,
  > correctly stamped. Every fisheye hypothesis stack written before this fix
  > (`hypstack-20260814/`, and every `<ws>/sfmr/seed-hypotheses/hyprev-*.sfmr`
  > on a fisheye capture) shows a cloud that was never solved and should not be
  > read. Fleet arm: `workspace-prep/hypcombine-20260814/`._
  >
  > _On the fisheye entries the other two changes are quiet:
  > `KerryPark480` is the only one that moves at all (h1 is judged non-distinct,
  > so its 7 frames are combination fuel for h0; 2 resect, posed 36 -> 38,
  > inlier fraction 66.0% -> 64.0%, released focal unchanged, no new flag, so
  > keep-best has nothing to revert and 891 -> 864 finalized points ship).
  > `KerryPark480` h1's capture-level reach is 100% against 60% on its own
  > admission — the largest deflation on the fleet — but it qualified either
  > way. `KerryPark360`, `OmniCoast`, `OmniTemple1` and `OmniHilltop` are
  > byte-identical products; only their artifacts changed._
- **`max_shift_px` in image px against a grid-px search** (Phase 5's item 5,
  re-measured there): material on OmniTemple1 (+11% relative peripheral drop),
  absent on the Kerry entries, and changing it touches the pinhole branch.
  Carried forward unchanged.
- **A member-extent-consistency gate.** Item 1's measurement gives the missing
  half: the per-view world size that meets a pixel budget is now stated per
  model, so "do this track's members agree on the point's extent" is finally a
  well-posed question across a fisheye track. Nothing implements it.
- **The census's structure dependence.** Item 2 works around it with an
  independent-measurement veto rather than fixing it. A census that scored
  candidates against evidence the candidate's own structure cannot bias would
  not need the veto.
- **The vote candidate on the pinhole branch is the RAW vote** while the band it
  is tested against is the bias-corrected one. That asymmetry predates this
  phase and is harmless for a veto (which only removes the other candidate), but
  it means a contradicted pinhole entry falls back to a focal known to run ~10%
  low. Not exercised by any fleet entry today.
- **The rest of `save_sfmr`'s frame construction is still hand-written.** The
  SIZING is unified (Item 1), but the writer still builds the tilt solve, the
  obliquity cap and the in-plane basis in Python against a central-differenced
  projection Jacobian. Those have no core counterpart to unify against yet;
  `PatchNormal` is the natural home if one is ever wanted.
- **Human Explorer inspection of the Phase-5 and Phase-6 sets** is prepared, not
  done.

## Restriction as a pipeline stage

Narrowing the clusters to the seed images is a stage of the seed pipeline, with
a file artifact — `matches/seed-restricted.matches` — and everything downstream
reads it.

The stage's input is the cluster selection stage 1 solved on (the derivation
`exp_fast_seed.load_clusters` admitted: reference/kept members, the optional
thinning restriction, the span filter). Once the seed images are chosen, the
stage derives that selection again restricted to them, at `MIN_SPAN_BA`, and
writes the result. `exp_pinhole_bootstrap.load_clusters(preselected=True)`
reshapes the written file as it stands: the file *is* the admission, and
re-selecting it would drop every cluster whose reference member fell outside the
restriction, since a derived file records those as the absent-reference
sentinel and the selection reads that as unrefinable.

Cluster ids downstream are the restricted file's own. Nothing after the stage
names a cluster of the workspace's matches file, and the derived file records no
cross-numbering correspondence. Nothing measured per cluster before the stage
crosses it either: the one attribute that used to (the rotation initializer's
certified far field, carried by a survival mask and checked against the file the
crate wrote) is gone with the channel it fed, so the stage is a select and a
save.

Because the stage restricts stage 1's selection rather than the source file, a
run with load-time image thinning (`SFMTOOL_SEED_IMAGES`) now finalizes on the
clusters stage 1 itself admitted; a cluster the thinning made
reference-less is absent from the seed's cluster set instead of being re-admitted
by a second derivation of the source.

> _Status (2026-08-13): Load-time thinning is deleted. A 41-entry fleet A/B
> (workspace-prep/prethin-inspection) measured 16/40 firing entries failing to
> seed at all and a −61% median point count on the survivors, with focal
> accuracy worse, not better.  The mechanism is superlinear observation loss:
> keeping 13.5% of images keeps only 0.6–2.2% of observations, because
> clusters need two members on kept images and a maximally-spread subset is
> the worst case for that — no target value escapes the curve.  The thinned
> pair graph also degraded the focal vote (worst −75%).  Dense-video scaling
> belongs upstream: thin frames before matching, so the coarse level gets
> natively-built clusters.  The paragraph above stands as a record of the
> interaction while the knob existed._

## Far-field side channel removed

The seed pipeline no longer has a second route to the infinity label. The
rotation initializer's H-certified far field was carried out of stage 1, across
the restriction stage, bound to the finalization's per-point uids and forced at
infinity behind a bearing veto under `SFMTOOL_FAR_BRIDGE=1`. All of that is
deleted: `far_cluster_indexes` off the kernel's output and its binding,
`far_clusters` off the seed dict and `_finalize_seed`, `_far_point_uids`,
`_force_points_at_infinity`, the forcing site and its trace stage, the
restriction's `restriction_survivors` / `carry_cluster_ids` carry, and the
opt-in flag itself.

**What the ablation measured** (`workspace-prep/farablate-inspection/`, on the
three of 41 fleet entries whose forced count is non-zero). The channel's entire
fleet effect is **15 forced points**: 4 on `20250712_195736354`, 1 on
`20250907_000240907`, 10 on `KerryPark480`. The two arms are identical up to the
force site — the pre-BA native classify demotes exactly the same uid set in
both. Of the 15, **14 are better explained finite by their own observations**
(bearing fit against finite fit at decision time), and the remaining one
(uid 3802 on `20250712_195736354`, 0.80 px against 2.32 px) is a single point.
The channel's largest effect is not the label at all: of `KerryPark480`'s 10,
five die in the off arm at the post-BA reprojection cull or the late
localizability re-cull and all ten survive in the on arm — the infinity label
exempts them from gates the finite population has to pass, so what the channel
mostly buys is **quality-gate bypass**, not evidence. Reprojection and focal
move within noise either way (0.747 vs 0.750 px; 138.3 px both).

The native classifier keeps its own demotion path, untouched: it reads the
triangulation's observability diagnostics, it is checked by the same bearing
veto, and it is the only thing that labels a point at infinity now.

**Follow-up lead.** The veto on the native classifier's proposals is one-sided —
it asks only whether the bearing model fits, never whether it fits BETTER than
the finite alternative. The ablation's own tables show the two questions come
apart (points whose bearing fit clears the noise floor while their finite fit is
tighter still). A comparative demotion conjunct — demote only when the bearing
fit beats the finite fit — is the natural next test, and the native path is
where it would matter: its demotions number in the hundreds per entry
(70 / 508 / 72 shipped at infinity on the three ablation entries) against this
channel's 15.

## Sequencing note

Phases 1–2 are one campaign (geometry), 3 is small once 1–2 exist
(scan is band + rescale; the BA extension is the only kernel work),
4 is the risk concentration (photometric machinery), 5–6 are
integration. The natural experiment vehicle for 1–3 is a
`SFMTOOL_FISHEYE_SEED=1` env gate on the fisheye branch, promoted to
default-on-confirmed-verdict when Phase 5's gate passes.

## The curvature rung (2026-08-14)

Principle 2's deferral ladder — `SimpleRadialFisheye(k1=0) → SimpleRadialFisheye(k1
free)` — is now taken, once, at the very end of the finalization. Stage 1 is
untouched: the seed dict, the scan, the vote and the release all stay
`EQUIDISTANT_FISHEYE`, and only the finalization promotes.

**Why.** Pure `r = f·θ` cannot express a real lens's θ³ curvature. Held to it,
the adjustment buys the residual field back with GEOMETRY — a finite dome that
pulls sky and horizon off infinity, which is what the KerryPark480 inspection
was looking at. Releasing the one radial coefficient gives that field a
parameter to land on, and lets the far structure walk back out to infinity.

**Mechanism.** After the accepted finalization candidate (arbitration winner,
frozen vote, or guarded release), the camera is promoted to
`SIMPLE_RADIAL_FISHEYE(k1 = 0)` — the same map bit for bit at `k1 = 0`,
projection and pixel Jacobian alike — and the BA is continued from that
candidate's own poses and points with `opt_k1` on. The focal is released
alongside it **only where the accepted candidate was itself a free-focal
solve**: where the finalization deliberately froze `f` (the vote candidate, a
flagged seed's structure-free focal, the basin guard's refit, `fixed` mode),
the rung does not get to re-open that decision. The basin guard on a released
`f` is unchanged.

Two guards refuse the step. The monotonicity guard (mirrored from the BA
kernel's own): `θ_d = θ(1 + k1θ²)` must stay strictly increasing over the field
the observations occupy, or the projection folds two incidence angles onto one
pixel radius. And keep-best, measured against a CONTROL that runs the same
continuation with `k1` fixed at zero — same start, same schedule, same trims and
retriangulations, one parameter fewer. The rung ships only when it beats that
control AND does not degrade the pre-rung result, on both the median and the
`θ > 60°` peripheral reprojection. A refusal returns the pre-rung result, so it
is exactly the old behaviour — the three refusing entries below come out
content-identical to the rung-off arm.

Comparing against the pre-rung result alone would have been wrong: the
continuation restarts the trim schedule at `loss_scale = 5`, which moves the
residual field on its own (KerryPark480's `k1 = 0` control lands at 1.62 px
median against the pre-rung 0.85 px). The control is what isolates the
curvature.

**Fleet A/B** (`workspace-prep/k1rung-inspection/{before,after}`, five fisheye
entries, one round stamp per arm; the shipped file's own reprojection field,
measured under the model each arm shipped):

| entry | k1 | reproj median | θ > 60° | at infinity | points |
|---|---|---|---|---|---|
| KerryPark480 | **+0.01483** | 0.650 → 0.544 | 0.744 → 0.591 | 111 → 130 | 864 → 1140 |
| KerryPark360 | **+0.00875** | 0.643 → 0.566 | 0.654 → 0.610 | 5 → 7 | 493 → 488 |
| OmniCoast | refused (−0.0218) | 0.760 | 1.087 | 2 | 2566 |
| OmniTemple1 | refused (+0.0163) | 0.915 | 1.086 | 4 | 2160 |
| OmniHilltop | refused (−0.0108) | 0.539 | 0.682 | 9 | 3148 |

The two Kerry entries are the fisheye rig captures with a known best-fit
equidistant focal; both recover a small POSITIVE curvature, both drop their
peripheral reprojection, and KerryPark480 — the entry with the visible dome —
gains 19 points at infinity and 276 points overall (the better fit survives the
post-BA cheirality and reprojection culls that the dome geometry was failing).
No focal moved: all five entries reached the rung with `f` frozen by the vote
arbitration, so the `opt_f + opt_k1` half of the staged release is untested on
this fleet.

The three Omni refusals are not one story. OmniCoast's release is a real
divergence (median 0.67 → 2.08 px against its own control at 0.77): the extra
DOF found a different basin that is better in the robust cost over the kept set
and worse in the raw field — exactly what keep-best exists for. OmniTemple1 and
OmniHilltop are marginal: their medians improve on the control (1.012 → 0.869,
0.493 → 0.478) while their periphery does not (1.270 → 1.379, 0.662 → 0.711),
so the θ³ term is not what their residual field wants. Whether the peripheral
conjunct is the right bar on entries whose median gain is that large is the open
question the next pass should answer.

**The model inverse had to be fixed first.** `pixel_to_ray` for
`SIMPLE_RADIAL_FISHEYE` ran the equidistant family's wide-angle blend, which
past `r_d = 90°` hands back the identity `θ = r_d` ray and so DROPS `k1` exactly
where `k1·θ³` is largest — a 105° rim at `k1 = 0.02` came back 6° off. The BA's
retriangulation and direction re-estimation read that inverse, so the rung was
unusable until it went: a synthetic recovery test that should have converged
exactly left 28 of 120 points at 400 px. With the blend removed for the
one-coefficient model (the Newton recovery is the exact inverse there), the same
test recovers `k1` to 1e-11 and every residual to 1e-9 px.
