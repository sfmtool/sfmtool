# Measurement Endpoint

A measurement in an `.sfmr` file is a real-world distance between two
**endpoints**, and each endpoint is a pixel in one image of the reconstruction
rather than a 3D point ([formats/sfmr-file-format.md](../../formats/sfmr-file-format.md)
§ 10 "Measurements"). Comparing the measurement with the reconstruction, which
is how its scale is set and how its distortion is checked, needs each endpoint
as a 3D position, and this operation gives it. An observation of a point in a
patch-based reconstruction carries a **square**: the point's patch, slid across
the line of sight so that its centre lies on the keypoint's ray, and projected
into the photograph. When the endpoint's pixel lies inside the square of an
observation in its image, the pixel is a known position `(s, t)` on that
point's patch, a texel of every render of the patch, and the patch says where
it is in 3D. The operation finds the observations whose squares hold the
pixel, picks one, and returns the texel and the 3D point it stands for. When
no observation's square holds the pixel, it refuses and gives the reason in a
sentence a person can act on; in SfM Explorer the usual remedy is **Create
Track Here** at the endpoint's pixel, which builds a track with an observation
centred on it.

The candidates are only the observations in the endpoint's own image. Patches
with no observation in the image are not candidates. Usually exactly one
square holds the pixel, and that observation is the answer with no ranking: no
stored reading is read and no cost is computed. Where several hold it, each
candidate in the front layer gets a cost in the manner of the reference-view
rule ([core/patch/reference-view.md](../patch/reference-view.md)), read from
the observation's own stored readings, which prefers smaller patches with
finer resolution and sharper texture holding the pixel near their centre.
Everything is read from stored data, with no photograph, so the operation is
cheap enough to run again after every edit. The form of the cost and its
constants are still to be measured, with the questions that remain open, in
[drafts/measurement-endpoints.md](../../drafts/measurement-endpoints.md).

## Rust API

In [`reconstruction::measurement_endpoint`](../../../crates/sfmtool-core/src/reconstruction/measurement_endpoint.rs),
with its tests in
[`measurement_endpoint/tests.rs`](../../../crates/sfmtool-core/src/reconstruction/measurement_endpoint/tests.rs).
It takes an `EditedReconstruction`, as the bench's operations do, so that SfM
Explorer can resolve endpoints against the reconstruction it is editing
without materialising it; the bindings wrap an `SfmrReconstruction` in one
(`EditedReconstruction::new`).

```rust
pub enum EndpointPosition {
    Finite { position: [f64; 3] },      // scene units
    AtInfinity { direction: [f64; 3] }, // a unit direction; no distance to it
}
impl EndpointPosition { pub fn finite(&self) -> Option<[f64; 3]>; }

/// One observation in the endpoint's image whose square holds the pixel.
pub struct EndpointCandidate {
    pub point: u32,                 // live index in the edited reconstruction
    pub observation: usize,         // index within the point's own track
    pub patch_coords: [f64; 2],     // (s, t) on the point's patch, in [-1, 1]²
    pub position: EndpointPosition, // the 3D point that texel stands for
    pub depth: f64,                 // along the pixel's unit ray; ∞ at infinity
    pub standing: CandidateStanding,
}

pub enum CandidateStanding {
    Alone,                          // the only square: no ranking was made
    Ranked(EndpointCost),           // one of several, in the front layer
    Excluded(EndpointExclusion),    // not eligible
}

pub struct EndpointCost {
    pub localization: f64,      // ℓ: scene units, or degrees at infinity
    pub localization_read: bool,// whether ℓ is the observation's own reading
    pub capped_radius: f64,     // the step on a reading at its cap
    pub offset: f64,            // O
    pub angle: f64,             // A
    pub agreement: f64,         // Z
    pub cost: f64,              // the product
}
impl EndpointCost { pub fn ln_parts(&self) -> [(EndpointTerm, f64); 5]; }
pub enum EndpointTerm { Localization, CappedRadius, Offset, Angle, Agreement }
pub enum EndpointExclusion { BackFacing, BehindFrontLayer }

pub struct EndpointResolution {
    pub chosen: usize,                      // into `candidates`
    pub candidates: Vec<EndpointCandidate>, // ascending point, then observation
}
impl EndpointResolution {
    pub fn chosen(&self) -> &EndpointCandidate;
    pub fn position(&self) -> EndpointPosition;
}

pub enum EndpointRefusal {                  // Display: a sentence naming the remedy
    NoSuchImage { image: u32, image_count: usize },
    NoPose { image: u32 },
    PixelOutsideImage { image: u32, pixel: [f64; 2] },
    NoPatchFrames,
    NoObservations { image: u32 },
    NoCoveringSquare { image: u32, pixel: [f64; 2] },
    AllExcluded { candidates: Vec<EndpointCandidate> },
}

pub struct MeasurementEndpointOptions {     // impl Default: § "Parameters"
    pub layer_depth_factor: f64,
    pub offset_weight: f64,
    pub placeholder_normal_factor: f64,
    pub reference_view: ReferenceViewOptions,
}

pub fn resolve_measurement_endpoint(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
    options: &MeasurementEndpointOptions,
) -> Result<EndpointResolution, EndpointRefusal>;

pub struct ResolvedMeasurement {
    pub endpoints: [Result<EndpointResolution, EndpointRefusal>; 2],
    pub scene_distance: Option<f64>, // both endpoints finite
}
pub fn resolve_measurements(
    edited: &EditedReconstruction,
    options: &MeasurementEndpointOptions,
) -> Vec<ResolvedMeasurement>;
```

**Why every candidate is returned.** Where the pick is wrong, the person
correcting it needs to see which other squares held the pixel and why each one
lost. With one candidate the list has one entry, `Alone`. This is what the
reference-view rule's `ReferenceChoice` does for views, and `ln_parts` splits
a cost so that a caller can name the term that cost a candidate most against
the pick, as `ReferenceStanding::costliest_term` does.

**Why a refusal is an error and not an empty resolution.** Every caller has to
say something to the person when an endpoint has no position, and each refusal
says something different: an unposed image is fixed by resecting it, a
reconstruction with no patch frames by converting it with
`sfm xform --to-embedded-patches`, a pixel no square holds by Create Track
Here. The refusal's sentence names the remedy.

**At infinity is a position, not a refusal.** A pixel on the sky can lie in the
square of a point at infinity; that is a real answer, and it says the
measurement cannot be compared with anything. `resolve_measurements` gives such
a measurement no `scene_distance` and does not refuse its endpoint.

**One candidate is not ranked.** Most pixels on a surface lie in exactly one
observation's square, since the squares of one image's observations rarely
overlap. Then that observation is the answer and its standing is `Alone`: the
front layer is not formed, no stored reading is read and no cost is computed.
The only test it still passes is the back-facing one, a single dot product, so
that a square seen from behind is refused (`AllExcluded`) rather than used.

**Cost of a call.** One endpoint visits every live point's observations once
to find those in its image, a few dozen floating-point operations for each in
the image, and reads the stored readings only where two or more squares hold
the pixel: well under a second on the largest reconstructions in `test-data`,
so it takes no `&Progress`. A file holds a handful of measurements.

An example call, from the viewer after an edit:

```rust
let opts = MeasurementEndpointOptions::default();
for (i, m) in resolve_measurements(&edited, &opts).iter().enumerate() {
    match (&m.endpoints[0], &m.endpoints[1], m.scene_distance) {
        (Ok(_), Ok(_), Some(d)) => println!("measurement {i}: {d:.4} scene units"),
        (Err(e), _, _) | (_, Err(e), _) => println!("measurement {i}: {e}"),
        _ => println!("measurement {i}: an endpoint is at infinity"),
    }
}
```

## From the pixel to a texel

An observation `j` of point `p` in image `i` fixes its square exactly
([formats/sfmr-file-format.md](../../formats/sfmr-file-format.md) § "The
embedded-patch feature relationship"): the point's frame `u`, `v`, outward
normal `n = normalize(u × v)`, centred on the **anchor** `A_j`, the point on
the keypoint's ray at the point's distance across the line of sight. The
square is `{ A_j + s·u + t·v : (s, t) ∈ [−1, 1]² }`, the patch as
`OrientedPatch::seen_at_keypoint` gives it through the image's own intrinsics
(`ImageSet::image_intrinsics`). This is the square every render of the
observation draws, and the one its stored readings were taken on.

With the pixel's ray `x = o + λ d` (`o` the camera centre, `d` the unit ray
`ImageSet::world_ray` casts), the operation solves

```
λ d − s u − t v = A_j − o
```

for `(λ, s, t)`. The square holds the pixel where `λ > 0` and
`(s, t) ∈ [−1, 1]²`. Since the square's projection into image `i` is the
observation's render, `(s, t)` is exactly the texel of that render that shows
the pixel, in the grid every reading of the observation is in. Where the
observation has no anchor (a keypoint ray at or past a right angle to the line
of sight), or no inline keypoint (a `sift_files` reconstruction), the square
is centred on the point itself, `A_j = X_p`, the fallback the format names and
the renders use. For a point at infinity the square is the set of directions
`p + s u + t v`, with `p` the anchored unit direction, and the same system is
solved with `o = 0`: the pixel's direction is a positive multiple of one of
them.

**The texel names a 3D point.** `(s, t)` is a position on the point's patch,
the same in every view of it. The endpoint's position is that place on the
point's own patch, `X_p + s·u + t·v`: the patch as triangulated from all its
views, not shifted by this view's keypoint. It is not the place on this
observation's square, `A_j + s·u + t·v`, which lies on the pixel's own ray.
The two differ by `A_j − X_p`, this observation's reprojection residual
carried across the line of sight. That residual is a property of one view's
keypoint, not of the surface, and two endpoints resolved through observations
in different images would otherwise each carry their own view's residual into
the distance between them. For a point at infinity the position is the unit
direction of `p + s·u + t·v` on the point's own patch.

A point with no patch frame has no square and is not a candidate. A
reconstruction with no patch frames at all is refused with `NoPatchFrames`,
and an image no live point is observed in with `NoObservations`, rather than
`NoCoveringSquare`, since each remedy differs.

## Occlusion at an edge

This section and the next apply only where two or more squares hold the pixel.

A square reaches past its keypoint by the patch's half-size, so near an
occluding edge the square of a point on the far surface, observed beside the
edge, can reach behind the near surface the person clicked on. Only the
surface nearest the camera is seen at the pixel, so the ranking is made within
the **front layer**: the candidates whose depth along the pixel's ray is within
`layer_depth_factor` times the nearest candidate's patch half-size
(`max(|u|, |v|)`) of the nearest depth. A candidate behind it is excluded with
`BehindFrontLayer`. A finite square whose normal faces away from the camera
(`n · d ≥ 0`) is excluded with `BackFacing` and does not set the layer; the
square of a point at infinity is never back-facing, as
`OrientedPatch::is_front_facing` reads it. A square of a point at infinity is
in the front layer only when no finite square that faces the camera holds the
pixel. Where every candidate is back-facing, the endpoint is refused with
`AllExcluded`.

## The cost

Within the front layer, each candidate gets a cost and the lowest is picked;
a tie goes to the nearer depth, then to the lower point index. As in the
reference-view rule ([core/patch/reference-view.md](../patch/reference-view.md)
§ "The rule"), the cost is one length multiplied by factors, each `1` where it
has nothing to say, so that a factor of `2` reads as "this candidate counts as
if it placed the endpoint twice as loosely":

```
cost = ℓ · S · O · A · Z
```

The terms read **the candidate observation's own stored reading**
([formats/sfmr-file-format.md](../../formats/sfmr-file-format.md) § "Observation
readings"), which was taken on exactly the square the pixel was mapped
through.

- **`ℓ`, the localization length.** How precisely this observation fixes its
  square on the surface. The square follows the keypoint, so an error in the
  keypoint moves the texel the pixel maps to by the same number of grid px.
  The observation's ZNCC self-similarity ellipse (`stored_ellipse` of its row)
  says how far its render can slide over itself and still match; `ℓ` is its
  semi-major axis mapped onto the patch (`SelfSimilarityEllipse::on_patch`),
  whose grid px are `2|u| / R` and `2|v| / R` in scene units, `R` the
  readings' resolution. This is the term that prefers a smaller patch with
  finer resolution: at the same `R`, a patch half the size has grid px half
  the size, so the same reading fixes the texel twice as closely. And it
  prefers the observation whose render holds sharper texture. An observation
  with no reading takes the reference-view rule's `unread_radius` grid px, a
  circle. Where the reconstruction stores no readings, `R` is the stored
  bitmaps' side, and where it stores neither, `1`, a grid px then being the
  patch's whole width; every candidate of a call is read at the same `R`. For
  a point at infinity `on_patch` reads the axis as an angle in degrees, so
  `ℓ` is an angle; such a candidate is only ever ranked against other points
  at infinity.
- **`S`, the step on a radius at the cap.** A semi-major axis at or above the
  readings' `max_radius` is a lower bound and takes the reference-view rule's
  `capped_radius_factor`, as there; otherwise `1`.
- **`O`, the offset factor.** `1 + offset_weight · ρ²`, with
  `ρ = max(|s|, |t|)` the texel's distance from the patch centre in
  half-sizes. A patch is a plane fitted to the surface around its centre, and
  how far the surface departs from that plane, through curvature or an error in
  the normal, grows with the distance from the centre. A small patch that holds
  the pixel near its edge can lose here to a larger one that holds it near its
  centre. Where the point's stored `normal_confidence` is `0`, a placeholder
  normal, the weight is multiplied by `placeholder_normal_factor`, since a
  guessed normal tilts the plane the texel is read from.
- **`A`, the angle factor.** `1 + angle_weight · tan θ`
  (`ReferenceViewOptions::angle_factor`), with `θ` the viewing angle of the
  square in the current geometry (`viewing_angle`). A square seen obliquely
  is foreshortened, so one image pixel spans more of the patch along its tilt,
  and the texel a pixel maps to is less certain.
- **`Z`, the agreement factor.** The reference-view rule's agreement penalty
  (`ReferenceViewOptions::agreement_penalty`) on the shortfall `1 − z` of the
  observation's blur-matched ZNCC `z` against the point's stored bitmap: `1`
  inside its band, growing below it, and `1` where the score was not read. An
  observation that does not look like its point is a poor witness of where
  the point's texels fall in this image. The shortfall is read below `1`
  rather than below the best candidate's score, as the rule does for views,
  because the candidates are different points scored against different
  bitmaps, and the blur-matched score already leaves a sharp view unpenalised
  for detail the bitmap lacks.

The cost has no term for how well the track fixes the point's own position,
which the texel's 3D point inherits; whether it needs one is open in
[drafts/measurement-endpoints.md](../../drafts/measurement-endpoints.md). Nor is
the observation's zoom (grid px per photograph px) a term: how many grid px one
photograph pixel covers is set by the camera, the surface and the angle, and
is the same for every square on one surface at one pixel; what differs between
candidates is the size of their grid px, which `ℓ` already reads.

## Parameters

Fields of `MeasurementEndpointOptions`. The values are starting values, to be
measured by the harness in
[drafts/measurement-endpoints.md](../../drafts/measurement-endpoints.md).

| Field | Default | Reason |
|---|---|---|
| `layer_depth_factor` | `1.0` | Patches of one surface agree in depth to within about their own size. |
| `offset_weight` | `1.0` | A texel at the patch edge counts as twice as loose as one at its centre. |
| `placeholder_normal_factor` | `4.0` | A placeholder normal can be off by tens of degrees. |
| `reference_view` | `ReferenceViewOptions::default()` | The cost reads its `angle_weight`, `capped_radius_factor`, `unread_radius`, `agreement_band` and `agreement_scale`: the same readings, read the same way as the rule reads them. Its other fields do not apply. |

## Python bindings

`SfmrReconstruction.resolve_measurement_endpoint(image, xy, **options)` and
`SfmrReconstruction.resolve_measurements(**options)`, in
[`reconstruction/measurement_endpoint.rs`](../../../crates/sfmtool-py/src/reconstruction/measurement_endpoint.rs).
The options are keywords named as the fields above, with the reference-view
fields flat (`angle_weight`, `capped_radius_factor`, `unread_radius`,
`agreement_band`, `agreement_scale`). A resolution is a dict of `chosen` and
`candidates`, each candidate a dict of its fields, with `position` and
`direction` (one of them `None`), `standing` as `"alone"`, `"ranked"` or
`"excluded"`, `exclusion` and `cost`. The single endpoint raises `ValueError`
with the refusal's sentence; the batch gives each measurement's `endpoints`
(`None` where refused), `refusals` (the sentence, `None` where resolved) and
`scene_distance`. Each call wraps a copy of the reconstruction in an
`EditedReconstruction`. The logic stays in Rust.

## Testing

In [`measurement_endpoint/tests.rs`](../../../crates/sfmtool-core/src/reconstruction/measurement_endpoint/tests.rs),
on synthetic reconstructions of two posed cameras and a few patches:

- A pixel in exactly one square resolves to it as `Alone`, and resolves the
  same on the reconstruction with its readings stripped.
- An observation's own keypoint resolves to texel `(0, 0)`, and a pixel off it
  to the texel its render shows there, at `X_p + s·u + t·v` on the point's own
  patch and not on the slid square.
- A pixel held by squares of equal patches resolves to the one it is nearest
  the centre of.
- A small patch holding the pixel at its centre beats a large one; the large
  one wins once the pixel is near the small one's edge.
- Two observed squares on stacked surfaces: the far one is `BehindFrontLayer`
  though its cost is the lower, and wins once the layer reaches it.
- A patch whose point has no observation in the image is no candidate; a
  deleted point is no candidate.
- Each refusal, with its sentence.
- A pixel in the square of a point at infinity resolves to `AtInfinity`, and a
  measurement with such an endpoint has no `scene_distance`.
- A similarity transform keeps every texel, moves every position with it, and
  scales `scene_distance` and every cost by its scale.

[`tests/rust_bindings/reconstruction/test_measurement_endpoint_rust_bindings.py`](../../../tests/rust_bindings/reconstruction/test_measurement_endpoint_rust_bindings.py)
checks the bindings on the seoul_bull ground truth.
