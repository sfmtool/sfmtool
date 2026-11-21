# Measurement endpoints

**Status:** Draft

Decided: the candidates for an endpoint are the **observations in the
endpoint's own image** whose square covers the endpoint's pixel. Each
observation fixes its patch's square in that image exactly, so the pixel is a
known grid coordinate `(s, t)` on each candidate's patch, and the endpoint is
resolved to that texel of one patch. The endpoint's 3D position is that texel's
place on the point's own patch, `X_p + s·u + t·v`. Patches with no observation in the image
are not candidates. Usually exactly one square holds the pixel, and that
observation is the answer with no ranking: no readings are read and no cost is
computed. Where several hold it, the pick is not simply the nearest: each
candidate gets a cost in the manner of the reference-view rule, read from the
observation's own stored readings, which prefers smaller patches with finer
resolution and sharper texture. Everything is read from stored data, with no
photograph, so it is cheap enough to run again after every edit. Open: the
form of the cost and its constants, to be settled by measuring them in the
harness in § "Validation" (§ "Open questions").

A measurement in an `.sfmr` file is a real-world distance between two
**endpoints**, and each endpoint is a pixel in one image of the reconstruction
rather than a 3D point ([formats/sfmr-file-format.md](../formats/sfmr-file-format.md)
§ 10 "Measurements"). Comparing the measurement with the reconstruction, which
is how its scale is set and how its distortion is checked, needs each endpoint
as a 3D position. This draft proposes the operation that gives it. An
observation of a point in a patch-based reconstruction carries a square: the
point's patch, moved so that its centre lies on the keypoint's ray, and
projected into the photograph. When the endpoint's pixel lies inside the
square of an observation in its image, the pixel is a known position on that
point's patch, and the patch says where it is in 3D. The operation finds the
observations whose squares hold the pixel, picks the one best placed to
say where the surface is, and returns the patch position and the 3D point it
stands for. When no observation's square holds the pixel, it refuses and gives
the reason in a sentence a person can act on. In SfM Explorer the usual remedy
is **Create Track Here** at the endpoint's pixel, which builds a track with an
observation centred on it.

It is the first step of the measurement work in SfM Explorer. The panel that
lists measurements, the lines drawn between endpoints, the Image Detail menu
entries that place them and the edit that scales a reconstruction to them all
read their positions from this operation, and are left to later drafts. Because
an endpoint resolves to a texel of a patch, the viewer can also draw it in every
other image that observes the same point, at that texel of the point's square
there.

## Rust API

In `sfmtool-core`, a module `reconstruction::measurement_endpoint`, beside the
`MeasurementsData` it reads. It takes an `EditedReconstruction`, as the bench's
operations do, so that SfM Explorer resolves endpoints against the
reconstruction it is editing without materialising it; the CLI and the
bindings wrap an `SfmrReconstruction` in one (`EditedReconstruction::new`).

```rust
/// Where one endpoint of a measurement lies.
pub enum EndpointPosition {
    /// A point in scene units.
    Finite { position: [f64; 3] },
    /// The texel lies on a patch of a point at infinity: a unit direction.
    /// No distance can be measured to it.
    AtInfinity { direction: [f64; 3] },
}

/// One observation in the endpoint's image whose square holds the pixel.
pub struct EndpointCandidate {
    pub point: u32,                 // live index in the edited reconstruction
    pub observation: usize,         // index within the point's own track
    pub patch_coords: [f64; 2],     // (s, t) on the point's patch, in [-1, 1]²
    pub position: EndpointPosition, // the 3D point that texel stands for
    pub depth: f64,                 // along the pixel's ray; ∞ at infinity
    pub standing: CandidateStanding,
}

pub enum CandidateStanding {
    /// The only square that holds the pixel: no ranking was made.
    Alone,
    /// One of several, in the front layer, with its cost.
    Ranked(EndpointCost),
    /// One of several, not eligible.
    Excluded(EndpointExclusion),
}

pub struct EndpointCost {
    pub localization: f64,  // ℓ, scene units: § "The cost"
    pub offset: f64,        // O
    pub angle: f64,         // A
    pub agreement: f64,     // Z
    pub triangulation: f64, // G
    pub cost: f64,          // ℓ · O · A · Z · G, scene units
}
impl EndpointCost { pub fn ln_parts(&self) -> [(EndpointTerm, f64); 5]; }
pub enum EndpointTerm { Localization, Offset, Angle, Agreement, Triangulation }

pub enum EndpointExclusion { BackFacing, BehindFrontLayer }

pub struct EndpointResolution {
    pub chosen: usize,                      // into `candidates`
    pub candidates: Vec<EndpointCandidate>, // every observation whose square holds the pixel
}

pub enum EndpointRefusal {
    NoSuchImage { image: u32, image_count: usize },
    NoPose { image: u32 },
    PixelOutsideImage { image: u32, pixel: [f64; 2] },
    NoPatchFrames,                   // the reconstruction stores none
    NoObservations { image: u32 },   // no point is observed in the image
    NoCoveringSquare { image: u32, pixel: [f64; 2] },
    AllExcluded { candidates: Vec<EndpointCandidate> },
}

pub struct MeasurementEndpointOptions { /* § "Parameters"; impl Default */ }

pub fn resolve_measurement_endpoint(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
    options: &MeasurementEndpointOptions,
) -> Result<EndpointResolution, EndpointRefusal>;

/// Both endpoints of measurement `index`, and the distance between them in
/// scene units where both are finite.
pub struct ResolvedMeasurement {
    pub endpoints: [Result<EndpointResolution, EndpointRefusal>; 2],
    pub scene_distance: Option<f64>,
}
pub fn resolve_measurements(
    edited: &EditedReconstruction,
    options: &MeasurementEndpointOptions,
) -> Vec<ResolvedMeasurement>;
```

**Why every candidate is returned.** Where the pick is wrong, the person
correcting it needs to see which other squares held the pixel and why each one
lost. With one candidate the list has one entry, `Alone`. This is what the reference-view rule's `ReferenceChoice` does for views,
and `ln_parts` names the term that cost a candidate most, as
`ReferenceStanding::costliest_term` does.

**Why a refusal is an error and not an empty resolution.** Every caller has to
say something to the person when an endpoint has no position, and each refusal
says something different: an unposed image is fixed by resecting it, a
reconstruction with no patch frames by converting it to embedded patches, a
pixel no square holds by Create Track Here. The refusal's sentence names the
remedy.

**At infinity is a position, not a refusal.** A pixel on the sky can lie in the
square of a point at infinity; that is a real answer, and it says the
measurement cannot be compared with anything. `resolve_measurements` gives such
a measurement no `scene_distance` and does not refuse its endpoint.

**One candidate is the common case, and it is not ranked.** Most pixels on a
surface lie in exactly one observation's square, since the squares of one
image's observations rarely overlap. Then that observation is the answer and
its standing is `Alone`: the front layer is not formed, no stored reading is
read and no cost is computed. The only test it still passes is the back-facing
one, a single dot product, so that a square seen from behind is refused rather
than used. The cost exists to choose between squares, and with one square
there is nothing to choose.

**Cost of a call.** One endpoint visits the observations in its image once, a
few dozen floating-point operations each, and reads the stored readings only
where two or more squares hold the pixel: well under a second on the largest
reconstructions in `test-data`, so it takes no `&Progress`. A file holds a
handful of measurements.

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
([formats/sfmr-file-format.md](../formats/sfmr-file-format.md) § "The
embedded-patch feature relationship"): the point's frame `u`, `v`, outward
normal `n = normalize(u × v)`, centred on the **anchor** `A_j`, the point on
the keypoint's ray at the point's distance across the line of sight. The
square is `{ A_j + s·u + t·v : (s, t) ∈ [−1, 1]² }`, projected into image `i`
through its own intrinsics (`ImageSet::image_intrinsics`). This is the square
every render of the observation draws, and the one its stored readings were
taken on.

The endpoint's pixel `q` is a point of that square where its ray meets the
square's plane. With `q`'s ray `x = o + λ d` (as
`analysis::point_or_bearing::observation_ray` casts it):

```
λ = n · (A_j − o) / (n · d),   x = o + λ d,   (s, t) from  x − A_j = s u + t v
```

The square holds `q` where `λ > 0` and `(s, t) ∈ [−1, 1]²`. Since the square's
projection into image `i` is the observation's render, `(s, t)` is exactly the
texel of that render that shows `q`, in the grid every reading of the
observation is in. Where the observation has no anchor (a keypoint ray at or
past a right angle to the line of sight), the square is centred on the point
itself, `A_j = X_p`, the fallback the format names and the renders use. For a
point at infinity, the square is a region of the sphere of directions and `(s,
t)` is read the same way on it (`d / (d · p) − p = s u + t v`, with `p` the
point's direction).

**The texel names a 3D point.** `(s, t)` is a position on the point's patch,
the same in every view of it, and the bitmap's texel there is the patch's
appearance at that place. The endpoint's position is that place on the point's
own surfel, `X_p + s·u + t·v`: the patch as triangulated from all its views,
not shifted by this view's keypoint. It is not the place on this observation's
square, `x = A_j + s·u + t·v`, which lies on the pixel's own ray. The two
differ by `A_j − X_p`, this observation's reprojection residual carried across
the line of sight. That residual is a property of one view's keypoint, not of
the surface, and two endpoints resolved through observations in different
images would otherwise each carry their own view's residual into the
distance between them.

A point with no patch frame has no square and is not a candidate. A
reconstruction with no patch frames at all is refused with `NoPatchFrames`, and
an image with no observations with `NoObservations`, rather than
`NoCoveringSquare`, since each remedy differs.

## Occlusion at an edge

This section and the next apply only where two or more squares hold the pixel.

Two observations in one image can both hold the pixel. A square reaches past
its keypoint by the patch's half-size, so near an occluding edge the square of
a point on the far surface, observed beside the edge, can reach behind the
near surface the person clicked on. Only the surface nearest the camera is
seen at the pixel, so the ranking is made within the **front layer**: the
candidates whose depth along `q`'s ray is within `layer_depth_factor` times the
nearest candidate's patch half-size (`max(|u|, |v|)`) of the nearest depth. A
candidate behind it is excluded with `BehindFrontLayer`. A square whose normal
faces away from this camera (`n · d ≥ 0`) is excluded with `BackFacing`. A
square of a point at infinity is in the front layer only when no finite square
holds the pixel.

## The cost

Where two or more squares hold the pixel, within the front layer, each candidate gets a cost in scene units, and the
lowest cost is chosen. As in the reference-view rule
([core/patch/reference-view.md](../core/patch/reference-view.md) § "The
rule"), the cost is one length multiplied by factors, each `1` where it has
nothing to say, so that a factor of `2` reads as "this candidate counts as if
it placed the endpoint twice as loosely":

```
cost = ℓ · O · A · Z · G
```

The terms read **the candidate observation's own stored readings**
([formats/sfmr-file-format.md](../formats/sfmr-file-format.md) § "Observation
readings"), which were taken on exactly the square the pixel was mapped
through.

- **`ℓ`, the localization length.** How precisely this observation fixes its
  square on the surface, in scene units. The square follows the keypoint, so an
  error in the keypoint moves the texel `q` maps to by the same number of grid
  px. The observation's ZNCC self-similarity ellipse says how far its render
  can slide over itself and still match: its semi-major axis in grid px,
  mapped onto the patch (`SelfSimilarityEllipse::on_patch`), whose grid px are
  `2|u| / R` and `2|v| / R` in scene units. This is the term that prefers a
  smaller patch with finer resolution: at the same patch resolution `R`, a
  patch half the size has grid px half the size, so the same reading fixes the
  texel twice as closely. And it prefers the observation whose render holds
  sharper texture. A reading at the cap of its range is a lower bound and
  takes the reference-view rule's step, `capped_radius_factor`. An
  observation with no reading takes `unread_radius` grid px, the
  reference-view rule's value for the same case.
- **`O`, the offset factor.** `1 + offset_weight · ρ²`, with
  `ρ = max(|s|, |t|)` the texel's distance from the patch centre in
  half-sizes. A patch is a plane fitted to the surface around its centre, and
  how far the surface departs from that plane, through curvature or an error in
  the normal, grows with the distance from the centre. A small patch that holds
  the pixel near its edge can lose here to a larger one that holds it near its
  centre. Where the point's normal carries no data-derived support
  (`normal_confidence` stored and `0`), the weight is multiplied by
  `placeholder_normal_factor`, since a guessed normal tilts the plane the texel
  is read from.
- **`A`, the angle factor.** `1 + angle_weight · tan θ`, with `θ` the
  observation's viewing angle in the current geometry, the reference-view
  rule's angle factor. A square seen obliquely is foreshortened, so one image
  pixel spans more of the patch along its tilt, and the texel a pixel maps to is
  less certain.
- **`Z`, the agreement factor.** The observation's blur-matched ZNCC against
  the point's stored bitmap, read through the reference-view rule's agreement
  penalty: `1` inside its band, growing below it. An observation that does not
  look like its point is a poor witness of where the point's texels fall in
  this image.
- **`G`, the triangulation factor.** How well the track fixes the point's own
  position, which the texel's 3D point inherits. Candidates:
  `1 + triangulation_weight / tan(max_ray_angle)`, from the track's largest
  angle between viewing rays, or the track table's `inverse_depth_z`. Open
  (question 2).

A tie on the cost goes to the nearer depth and then to the lower point index.

The observation's zoom (grid px per photograph px) is not a separate term. How
many grid px one photograph pixel covers is set by the camera, the surface and
the angle, and is the same for every square on one surface at one pixel; what
differs between candidates is the size of their grid px, which `ℓ` already
reads. It is reported on the candidate, beside the readings, for the person to
see.

## Parameters

Fields of `MeasurementEndpointOptions`, with starting values to be measured
(§ "Validation"):

| Field | Starting value | Reason |
|---|---|---|
| `layer_depth_factor` | `1.0` | Patches of one surface agree in depth to within about their own size. |
| `offset_weight` | `1.0` | A texel at the patch edge counts as twice as loose as one at its centre. |
| `placeholder_normal_factor` | `4.0` | A placeholder normal can be off by tens of degrees. |
| `angle_weight` | `ReferenceViewOptions::default().angle_weight` (0.25) | The same obliquity effect as in the reference-view rule. |
| agreement band and scale | the reference-view rule's `agreement_band` and `agreement_scale` | The same score, read the same way. |
| `capped_radius_factor`, `unread_radius` | the reference-view rule's defaults | The same readings, read the same way. |

## Validation

A leave-one-out harness on the ground-truth reconstructions, which are in
metres, in the manner of the track-at-pixel harness
([track-at-pixel.md](track-at-pixel.md)):

1. Take a point with a patch, and one of its observations: image `i`,
   keypoint `k`.
2. Resolve `(i, k)` on the reconstruction with that point deleted, so that the
   answer has to come from the squares of neighbouring observations in image
   `i`.
3. The error is the distance from the resolved position to the deleted
   point's position, in metres.

Over many points this gives the error of each rule: the nearest square only,
the cost above, and each factor removed in turn. The constants are fitted where
error falls, and the cost is checked against the error it predicts. Where the
cost tracks the error closely, it can be calibrated into an uncertainty, and
SfM Explorer can show a measurement's reconstructed distance with an error bar.
The harness computes the cost for every resolved endpoint, `Alone` ones
included, so that the calibration covers the common case too; the operation
itself does not.

`dino_dog_toy_ground_truth_v01` was scaled from four tape-measured distances.
Entered as measurements, they are a check of the whole chain, from endpoints
placed by a person to a scale factor.

## Python bindings

`SfmrReconstruction.resolve_measurement_endpoint(image, xy, **options)` and
`SfmrReconstruction.resolve_measurements(**options)`, returning dicts of the
fields above, with a refusal raised as `ValueError` for the single endpoint
and given as the refusal's sentence per endpoint for the batch. The logic stays
in Rust.

## Testing

- A pixel in exactly one square resolves to it with standing `Alone`, and
  reads no observation reading: it resolves the same on a reconstruction with
  the readings stripped.
- An observation's own keypoint resolves to texel `(0, 0)` of its point's
  patch, and a pixel offset from it to the texel its render shows there.
- A pixel held by squares of equal patches resolves to the one it is nearest
  the centre of.
- A pixel inside a patch's projection whose point has no observation in the
  image has no candidate; with no other square the endpoint is refused with
  `NoCoveringSquare`.
- Two observed squares on stacked surfaces: the far one is `BehindFrontLayer`,
  whichever has the lower cost.
- A small patch holding the pixel at its centre beats a large one; the large
  one wins once the pixel is near the small one's edge.
- Each refusal, with its sentence.
- A pixel in the square of a point at infinity resolves to `AtInfinity`, and a
  measurement with such an endpoint has no `scene_distance`.
- A similarity transform of the reconstruction leaves every texel `(s, t)` the
  same, transforms every resolved position by it, and scales `scene_distance`
  and every cost by its scale.

## Open questions

1. **The form of the cost.** A product, as the reference-view rule chose, or
   lengths added in quadrature, which is how independent errors combine and
   would make the cost an error estimate from the start. The harness decides.
   If the cost becomes an uncertainty a caller shows, the operation would
   compute it for an `Alone` candidate too, as an option the viewer turns on
   when it draws error bars.
2. **The triangulation factor.** `max_ray_angle`, `inverse_depth_z`, or
   nothing: bundle adjustment may fix every surviving point in a ground truth
   well enough that the term never decides a pick.
3. **Blending.** Whether to take a weighted mean of the front layer's
   positions instead of one pick. A mean would be smoother as an endpoint is
   dragged across square borders. A pick is what a person can inspect and
   correct, and what the reference-view rule does.
4. **Storing the resolution.** Whether a measurement should record the point
   and texel its endpoint resolved to, so that a later edit cannot move it
   silently. This draft resolves on every read and stores nothing, which keeps
   the format as it is.
