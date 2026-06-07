# Points at infinity: a unified homogeneous point model for `.sfmr`

**Status:** Draft proposal. Reworks the `.sfmr` point representation into
a single homogeneous collection so that points at infinity are
first-class, a format **version 2** change.

[`sfmr-file-format.md`]: ../formats/sfmr-file-format.md

## Motivation

A feature track on distant content — a skyline, a ridge, a far building
— is matched into the SfM solve like any other, but the 3-D point it
triangulates to is poorly conditioned in depth. The rays from the cameras are nearly parallel,
so the depth is essentially unconstrained: the point lands at *some*
arbitrary far coordinate, its depth wanders during bundle adjustment, or the viewing-angle point
filter culls it. Either way the track's information is wasted.

The root issue is representational. A point at infinity is a
**direction, not a location**: a homogeneous point with `w = 0`.
Storing it as a finite `(x, y, z)` is lossy and actively misleading: the
coordinate's radial component is meaningless, but every consumer treats
it as real. If the `.sfmr` represented points homogeneously, every algorithm could
operate on a more appropriate representation of the data. Spatial filters are the
clearest case: the isolated-points filter scores a point by the
distance to its neighbours — meaningless for an ill-conditioned point
whose finite coordinate is an arbitrary far guess, but a non-question
once that point is `w = 0`.

## The unified homogeneous model

Every point (finite or at infinity) is one homogeneous coordinate
`(x, y, z, w)`:

- `w ≠ 0` — a finite point at Euclidean position `(x/w, y/w, z/w)`.
- `w = 0` — a point at infinity; `(x, y, z)` is a direction in the world
  frame.

`w` is the kind: the representation is self-describing, with no separate
flag. A point at infinity is the `w → 0` limit, not a special case
bolted on. A track is a
track regardless of `w`; the finite/infinite distinction is a property
of the point it references, never of the observation.

This is a deliberate format break rather than an additive optional
section. The alternative (a separate side-collection for infinity
points) would mean every consumer handles two point collections
indefinitely; the homogeneous model keeps one collection, which a
solver that handles points at infinity consumes directly. The format
goes to **version 2**.

### `w` normalisation

`w` is a homogeneous coordinate, so `(x, y, z, w)` and
`(λx, λy, λz, λw)` denote the same point for any `λ ≠ 0`. The format
permits any such scale; it does **not** mandate a canonical one.

The **recommended normalised form** sets two conventions:

- finite points (`w ≠ 0`) are divided through by their own `w`, so
  `w = 1`;
- infinity points (`w = 0`) store a unit-length direction in `(x, y, z)`.

The first is chosen for compression: a `w` column that is all `1`s and
`0`s is a near-constant run that zstd collapses to almost nothing, where
a column of arbitrary scales compresses poorly. The second is the natural canonical form for a
direction. The writer in this codebase emits this form.

Because the format does not *require* the normalised form, a consumer
that relies on `w ∈ {0, 1}` (or on unit-length directions) must
normalise on read. It cannot assume an arbitrary v2 file, possibly
produced by other tooling, is already normalised.

## File representation (`.sfmr` v2)

The v1 `Points3D` arrays become a single homogeneous collection:

| v1 | v2 | change |
|----|----|--------|
| `positions_xyz` `(P,3)` f64 | `positions_xyzw` `(P,4)` f64 | homogeneous; `w = 0` is a point at infinity, `w ≠ 0` a finite point at `(x/w, y/w, z/w)` |
| `colors_rgb` `(P,3)` u8 | `colors_rgb` `(P,3)` u8 | unchanged |
| `reprojection_errors` `(P,)` f32 | `reprojection_errors` `(P,)` f32 | unchanged — a `w = 0` point still projects (rotation + intrinsics only), so its reprojection error stays well-defined |
| `estimated_normals_xyz` `(P,3)` f32 | `estimated_normals_xyz` `(P,3)` f32 | `NaN` rows where `w = 0` — a direction has no surface normal |

Tracks become a single observation list over the unified collection:

| v1 | v2 | change |
|----|----|--------|
| `image_indexes` `(M,)` u32 | `image_indexes` `(M,)` u32 | unchanged |
| `feature_indexes` `(M,)` u32 | `feature_indexes` `(M,)` u32 | unchanged |
| `points3d_indexes` `(M,)` u32 | `point_indexes` `(M,)` u32 | renamed; indexes the unified collection |
| `observation_counts` `(P,)` u32 | `observation_counts` `(P,)` u32 | unchanged shape; now over all points |

Metadata: `version` becomes `2`; `points3d_count` becomes
`point_count`. The count of infinity points (rows with `w = 0`) is
derivable from the points array, but is also stored in the metadata as
`infinity_point_count` so a consumer can read the finite/infinity split
cheaply, without decompressing that array.

### Versioning and migration

v2 changes the meaning of `positions_xyz` and renames
`points3d_indexes`, so it is not backward compatible, hence the bump.

Migration is mechanical and lossless. A v1 file upgrades to the v2
in-memory model by appending a `w = 1` column to `positions_xyz` and
renaming `points3d_indexes`; it carries no infinity points until a
classification pass runs. The reader in this
codebase accepts **both v1 and v2**, upgrading v1 on read so the rest of
the codebase only ever sees the unified model; the writer emits **v2
only**. v2 files are not readable by pre-v2 tooling, the accepted cost
of the bump. Tests regenerate their `.sfmr` files from the image datasets on each
run, so there are no checked-in v1 fixtures that need migrating.

## Classification — which tracks are at infinity

A point at infinity has a maximum viewing angle of zero: its
observation rays are exactly parallel. More usefully, a point's depth
is at infinity whenever the depth its rays *do* imply is buried in
measurement noise.

Two cameras observing a point at triangulation angle `α` resolve its
depth to a *relative* precision

```
Δz / z  ≈  noise / (α_max · f_max)
```

where `α_max` is the maximum viewing angle (the largest angle between
any pair of observation rays, from `viewing_angle.rs`) and `f_max` is
the largest focal length (pixels) among the observing cameras. The
product `α_max · f_max` is the **parallax signal in pixels**: how far
the point's projection shifts across the widest-baseline camera pair as
its depth ranges from the triangulated estimate out to infinity. When
that signal drops to the level of the measurement `noise`, the depth is
uncertain by 100% or more — the data cannot tell the finite estimate
from infinity. So the cut is:

```
α_max · f_max  <  noise        ⇒  w = 0   (else w = 1)
```

`noise` is **per point**, not a universal constant. The measurement
noise of a track is best estimated by the track's own RMS reprojection
error `e_reproj` — it captures every residual source for that track
(keypoint localisation, faint mismatches, camera-model error), not just
an idealised SIFT-localisation floor. A track with `e_reproj = 3 px`
genuinely has its depth lost below a 3 px parallax signal, even though a
fixed 1 px cut would wrongly keep it finite.

```
noise  =  max(e_reproj, ε_floor)
```

The floor `ε_floor` (≈ 1 px, the SIFT keypoint localisation noise)
guards the short-track case. A 2- or 3-view track is triangulated to
fit its handful of observations almost exactly *regardless of depth
conditioning*, so its `e_reproj` is spuriously small and under-states
the true measurement noise. Flooring `noise` at `ε_floor` keeps a
poorly-conditioned short track from masquerading as well-determined.
`ε_floor` is the one tunable parameter of the classifier; `e_reproj` is
read from the reconstruction.

Equivalently the per-point cut is an angle `α_∞ = noise / f_max`.

The two conversions are not inverses, and round-tripping is not
lossless: finite → infinity drops a depth the data never pinned down,
while infinity → finite has to *supply* a depth the data cannot give.

**Finite → infinity** (`w ≠ 0 → 0`): discard `(x, y, z)`, set `w = 0`,
and store the direction `d = normalise(Σ R_i b_i)`, the mean of the
observation bearings rotated into the world frame. The bearings are
consistently signed (each points from its camera toward the feature),
so the summed mean is well defined. The direction is well conditioned
exactly where the depth was not, so nothing the data determined is
lost. Where the viewing-angle point filter today *removes* such tracks,
this *reclassifies* them and keeps the information.

**Infinity → finite** (`w = 0 → 1`): there is no depth to recover. A
`w = 0` point's rays are parallel to within feature noise, so
triangulating them returns pure noise. The conversion therefore does
**not** triangulate; it *materialises* the point along its stored
direction `d`
(from a reference origin such as the camera-cloud centre) and sets
`w = 1`. It exists only for consumers that cannot represent `w = 0`:
COLMAP export, a finite-only solver or viewer.

Each observing camera's **pixel differential** (the angle a pixel
subtends along its ray to the point) fixes a distance beyond which the
materialised
point's parallax falls below one pixel, leaving it indistinguishable
from the `w = 0` point in that camera. The placement distance is the largest of these
per-ray distances — far enough to be faithful in every camera, no
farther (a needlessly large coordinate only costs floating-point
precision).

## Converting COLMAP solutions to `.sfmr` v2

COLMAP's `points3D.bin` stores every point as a finite `(x, y, z)`.
Converting a COLMAP solution to `.sfmr` v2 carries every point and track
over unchanged, then applies the classification above: each point is
labelled `w = 0` or `w = 1` by the threshold, and every `w = 0` point's
coordinate is replaced with its bearing-mean direction.

## API and consumers

- `sfmr-format`: `SfmrData` carries `positions_xyzw` and a unified track
  list; `read.rs` accepts v1 (upgrade on read) and v2; `write.rs` emits
  v2 in the recommended normalised form; `verify.rs` validates that
  `w = 0` rows have a non-zero direction, that no `positions_xyzw`
  coordinate is NaN or infinite, that `infinity_point_count` matches the
  data, and the track CSR.
- PyO3 bindings + `SfmrReconstruction`: accessors expose `w` (or a
  derived `is_at_infinity` mask) alongside the existing point/track
  accessors; one tracks view spans all points.
- `SfmrReconstruction` (Rust core) owns the two conversions as methods:
  `classify_points_at_infinity` (finite → infinity, per the cut above)
  and `materialize_points_at_infinity` (infinity → finite). Both return
  a new reconstruction and are exposed through the PyO3 bindings.
- Rotation-only refinement reads the `w = 0` points directly.
- `xform`: an SE(3) similarity transform rotates a `w = 0` direction and
  renormalises it, leaving translation and scale to act on finite points
  only; point filters whose score is undefined for a direction
  (reprojection error, triangulation angle, neighbour distance) pass
  infinity points through untouched, while filters scoring track length or
  feature size — well defined regardless of `w` — score them normally;
  bundle adjustment materialises infinity points, refines, then
  reclassifies.
- `sfm-explorer`: reads `positions_xyzw`; renders a `w = 0` point as a
  direction on the far view sphere rather than at a finite coordinate.
- `sfmr-colmap`: on import, classify per
  [Converting COLMAP solutions](#converting-colmap-solutions-to-sfmr-v2);
  on export, drop or materialise `w = 0` points as far landmarks.

## Format / module layout

| Layer | Location | Notes |
|-------|----------|-------|
| Format types | `crates/sfmr-format/src/types.rs` | `positions_xyzw`, unified track arrays on `SfmrData` |
| Read / write / verify | `crates/sfmr-format/src/{read,write,verify}.rs` | v1 upgrade-on-read; v2-only write; `w` validation |
| Format spec | [`sfmr-file-format.md`] | Document v2 |
| Bindings | `crates/sfmtool-py/src/py_sfmr_reconstruction.rs` | `w` / `is_at_infinity` + unified track accessors; the two conversion methods |
| Conversions | `crates/sfmtool-core/src/infinity/convert.rs` | `classify_points_at_infinity` / `materialize_points_at_infinity` on `SfmrReconstruction` |
| COLMAP interop | `src/sfmtool/_colmap_io.py` | Classify on import by default (`--no-detect-infinity` to skip) |
| GUI viewer | `crates/sfm-explorer` | Render `w = 0` points as directions on the view sphere |
