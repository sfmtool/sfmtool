# Seed Candidate Evaluation

When the seed's hypothesis loop commits its candidate set for a capture,
every committed member is measured by a fixed battery of evaluation
channels before release. Channels are per-member, with per-frame or
per-image detail wherever the mechanism localizes, and are stored with the
member's hypothesis record so a later selection pass can rank, refuse, and
trim members without re-deriving evidence.

## Principles

- **Gauge-free.** Pose comparisons are pairwise relative-rotation and
  relative-translation-direction deltas over frame pairs; absolute poses
  are never compared across independently fit geometry.
- **Camera-model-generic.** Pixels unproject through the member's own
  camera model. Depth along a ray is ray range (never a coordinate axis);
  field validity is `theta <= theta_max` (never a half-space test).
- **Data-derived thresholds.** Floors and gates are quantiles of the
  relevant population (the capture's own readings, or a maintained fleet
  distribution), not fixed constants.
- **Measurability is a value.** Every channel reports whether it could be
  measured; an unmeasurable reading is recorded with its reason and is
  never silently dropped. A reading below a conditioning floor is a
  non-measurement, not a disagreement.
- **Frame identity is the full relative path.** Basenames collide across
  rig sensor directories.
- **Evaluation never alters reconstruction output.** Released `.sfmr`
  files are byte-identical with evaluation enabled or disabled; only the
  release manifest gains fields.

## Channels

### Fit (existing)

Inlier fractions at fixed pixel radii, median reprojection error, and
per-frame observation counts, as already emitted by the release records.

### Hold-out self-resection

For each frame `f` of a member:

1. Collect the points `f` observes that retain at least two supporting
   observations on other member frames, and re-triangulate them from
   those other frames only. (The stored positions are contaminated for
   this purpose: `f` helped triangulate them.)
2. Robustly resect `f` (trimmed PnP) against the re-triangulated points
   using `f`'s stored observations and camera.
3. Compare the resected pose to the committed pose.

Per-frame outputs: rotation delta (degrees); translation delta as a
fraction of the member's median camera-to-structure distance; resection
inlier fraction and support count; re-triangulation depth agreement
(Spearman correlation of re-triangulated versus stored depths at `f`, and
median and worst absolute log depth deviation); and support conditioning
(each re-triangulated point's maximum pairwise triangulation angle over
its supporting frames). A frame whose surviving support subtends only
small angles is conditioning-limited: its pose deltas are reported but not
gate-eligible.

Member channels: worst and median pose deltas over measurable frames,
gated by an inlier-fraction floor (population quantile); depth-agreement
rho at p10 and median; log depth deviation median and worst; and the
identity of the worst frame.

### Non-member resection

Rank the capture's images outside the member by two-dimensional match
connectivity to member images and take the top `k` (small, fixed). For
each held-out image:

1. Build 2D-3D correspondences: held-out keypoint, to member keypoint via
   the match graph, to that member observation's stored 3D point. (Stored
   points are legitimate here: the held-out image never touched the fit.)
   An image connected to several members receives correspondences from
   all of them.
2. Robust PnP with the member's camera model.
3. Independently estimate the two-view relative pose between the held-out
   image and its best-connected member image from the raw matches alone
   (essential matrix on unprojected unit rays), and compare it with the
   PnP-implied relative pose. Structure cannot fake agreement between
   both estimates.

Per-image outputs: resection inlier fraction and support count; PnP
versus two-view relative-rotation delta and translation-direction delta;
and the pair's recovered parallax expressed in units of the epipolar
estimator's own consensus bound (pixel bound over equivalent focal).
Pairs below a parallax bar in that unit are conditioning-limited.

Member channels: median and worst of the per-image outputs; the count of
resectable non-member images. Zero resectable images is itself a reading.

### Settling probe

A short bundle adjustment of the whole member: staged iteration schedule,
intrinsics frozen, poses and points free. (A single late-stage round
trims re-triangulated points before the solve reaches them; the schedule
must open with permissive rounds.) Outputs: gauge-free per-pair
relative-rotation and translation-direction deltas between the pre- and
post-refit poses, aggregated at median, p90, and worst; and the residual
medians before and after. A residual that grows marks a diverging refit;
a diverging member's settling channels are read from worst aggregates,
never medians.

### Warp epipolar consistency

The cluster file the capture was matched on stores, per cluster member,
that member's absolute affine shape `S`: the 2x2 map from the detector's
canonical unit frame onto the member's image pixels, as the patch
refinement fitted it. Two observations `i` and `j` of one track therefore
carry a measured pixel-to-pixel warp

```
A = S_j · S_i⁻¹
```

anchored at the stored keypoints the shapes were fitted around, never at a
reprojection.

Of the warp's four numbers the member's relative pose fixes exactly two.
Differentiating the epipolar constraint along the correspondence gives

```
Aᵀ l_j + l_i = 0        l_i = J_iᵀ Eᵀ r_j,  l_j = J_jᵀ E r_i
```

with `J` the pixel-to-ray Jacobian of the member's own camera, `r` the
observation's ray and `E = [t_rel]ₓ R_rel` the pair's essential matrix:
`A` maps the epipolar direction at the point in image `i` onto the
epipolar direction in image `j`, and fixes the scale across that
direction. The residual is `‖Aᵀ l_j + l_i‖`, normalized by the
magnitudes it is formed from. The remaining two numbers (the scale
along the epipolar direction, and the shear) are set by the point's
unknown tangent-plane normal and are not read.

The same two numbers are also read in the tangent chart, which is where
the normalization by expected warp magnitude is available. At each
observation take the unit ray, an orthonormal tangent basis `B` of the
sphere at that ray, and `P = Bᵀ J / |r|`, so a pixel displacement becomes
an angular one; every member is then read on its own camera model rather
than on a perspective stand-in. With `R_rel` the relative rotation,
`W0 = B_jᵀ R_rel B_i` the pose-only warp, `a = B_jᵀ R_rel u_i` the
epipolar direction in `j` (`|a| = sin(vergence)`), and the measured warp
carried into angular units and divided by the pair's ray-range ratio,
the component of the difference perpendicular to `a` is the residual and
`‖W0‖` is its normalizer.

The pair's **vergence** is the angle its two rays subtend at the point.
Below a vergence floor the normal-free content of a warp constrains
almost nothing and the pair is conditioning-limited: its residual is
reported and is not gate-eligible. The floor and the count of pairs
above it ship with the channel.

Per-frame outputs: over the pairs the frame takes part in, the residual
median and p90, the same restricted to conditioned pairs, and the
frame's median vergence. Member outputs: median, p90 and worst over
pairs, the conditioned forms of each, the worst frame by frame median,
and the measurability census (observations that resolved to a cluster
member carrying a non-degenerate shape, pairs formed, points covered).

### Stranger-surface membership

For each finite point, scan its nearest finite neighbours in 3D and keep
those whose observing-frame set does not intersect the point's own: the
strangers, which contributed nothing to where the point was placed. Fit
a robust plane (total least squares, then redescending IRLS passes on
the orthogonal residual, with the robust scale floored at a fraction of
the neighbourhood's own spacing) through the nearest of them alone, and
read the point's off-plane distance in units of the neighbourhood's
median nearest-neighbour spacing. The plane never sees the point, so a
point that sits off the surface is measured against the surface rather
than against a plane it helped define; the reading is a ratio of lengths
measured in the same neighbourhood and is therefore invariant to any
similarity transform of the cloud.

A point with fewer than a support floor of strangers among the scanned
candidates is **unmeasurable**: it is counted with its reason, and the
measurable fraction ships with the channel. A member whose covisibility
is complete, every point seen by every frame, has no strangers
anywhere and declines the channel entirely, which is a reading about the
member, not a failure.

Two derived readings accompany the residual: the same median restricted
to the points whose stranger neighbourhood is no wider than the member's
own median stranger locality, and the ratio of the stranger residual to
the ordinary k-nearest-neighbour residual at the same point. Per-frame
outputs are the frame's median and p90 over the points it observes;
member outputs are the point-level median and p90, the two derived
readings, the median and worst of the per-frame medians with the frame
named, and the measurability census.

### Local surface variation

For each finite point, `λ3 / (λ1 + λ2 + λ3)` of the covariance of the
point together with its `k` nearest finite neighbours in 3D, with
`λ1 ≥ λ2 ≥ λ3`. The reading is a ratio of eigenvalues and so is
gauge-free; it says how far the neighbourhood departs from a surface,
whatever the surface's orientation or scale.

Its absolute level is a property of the scene: a foliage capture is
more volumetric than a wall capture at equal correctness, so the member
channel is the median over frames of each frame's median over the points
that frame observes, and it is read capture-relative.

The plain leave-one-out plane residual of the same k-nearest-neighbour
set is reported alongside it, per frame and per member, because the
frame spread of that residual is what localizes a defect confined to a
few frames.

### Range-vetted surface residual

Per frame, take each observation's `k` nearest keypoint neighbours in
image space whose points are finite, and vet the pairs on range: keep a
neighbour only when the two points' ranges from that frame's centre
agree to within the frame's own median relative range difference. The
bar is a within-frame quantile, and it separates neighbours that lie
next to each other on a surface from neighbours that lie one behind the
other along a viewing ray. Fit the robust plane through the surviving
neighbours and read the point's off-plane distance in units of their
median spacing, exactly as for the stranger neighbourhood.

An observation left with fewer than a support floor of vetted
neighbours is unmeasurable; the measurable fraction and the frame's
vetting bar ship with the channel. The reading consults the member's
own poses through the ranges, which is what makes it sensitive to a
cloud whose depth ordering is wrong while its neighbourhoods still look
locally plausible. Per-frame outputs are the frame's median and p90;
member outputs are the point-level median and p90 over the per-point
readings averaged across the frames that see the point, and the median
frame vetting bar.

### Focal-vote consistency

The capture census's focal vote already exists at seed time. The channel
is the member's released equivalent focal versus the vote, as a signed
fraction. No gate is applied at generation time.

### Per-frame support and coherence (existing)

Near-support ratios, per-frame depth-scale coherence (each frame's median
depth ratio against the member median), and per-frame observation floors,
as already emitted.

### Peer corroboration (existing)

Minimum and maximum rotation disagreement with sibling members over
shared frames.

## Rotation-only members

A committed member may claim **bearing without range**: one rotation per
frame, every camera centre at one point, and one direction per cluster the
model explains. Such a member is measured and judged like any other. The
channels below are the rotation-only forms of the ones above; the surface
channels have no rotation-only form and are recorded unmeasurable with the
reason `no finite structure`.

A rotation-only member's membership is its model's own inlier set at the
pipeline's pixel bar, so the fit channel reads the member's whole admission
over its posed frames: a fit measured only over the observations the bar kept
reports the bar.

Every rotation-only member also carries a relaxed sibling
([seed-relaxation.md](seed-relaxation.md)): a finite member built from the
observations its rotation model refused. The sibling is a finite-family
member and is measured on the finite channels, not on the ones below.

### Hold-out rotation resection

For each frame: re-derive the directions it observes from its **other**
observing frames alone (the normalized mean of their rotated rays), refit the
frame's rotation against them, and compare with the committed rotation. The
world frame the delta is read in is the one those other frames define, so no
absolute pose is compared across independently fit geometry.

Per-frame outputs: rotation delta; fit inlier fraction and support count; the
angular deviation between re-derived and stored directions; and the support's
own angular spread. A frame whose support directions lie in a narrow cone
leaves the rotation about that cone's axis undetermined and is
conditioning-limited. Both floors (an inlier floor and a spread bar) are
quantiles of the capture's own per-frame readings and ship with the channel.

### Non-member rotation resection

Rank the capture's images outside the member by match connectivity and fit the
top `k` as rotations against the member's stored directions. The two-view
witness is the pair's own relative rotation from the raw matches alone: under
a rotation-only model the pairwise image map is a rotation of unit rays, so
the witness is that fit and not an essential matrix, which a model with no
baseline has no geometry to support. Compare the resection-implied relative
rotation with the witness. An image the member cannot explain as a turn of
its own directions is a non-measurement carrying that reason.

### Rotation-only settling

Alternate: each direction becomes the normalized mean of its own rotated
rays; each rotation is refitted against the directions. There is no scale and
no baseline to trade, so every step is closed form. Read gauge-free as the
between-frame relative-rotation change, aggregated at median, p90 and worst,
plus the residual angle before and after over the observations that are not
their cluster's own reference ray (a reference ray's residual is zero by
construction). A residual that grows past the member's own consensus bound
marks a diverging refit.

### Warp consistency, full form

Under a pure rotation a track's pairwise image map carries no surface term:
the two views' range ratio is one and there is no tangent-plane normal to
absorb anything, so the relative rotation and the camera model predict **all
four** numbers of the measured warp `A = S_j S_i⁻¹`. The residual is the whole
departure from that prediction, normalized by the pose-only warp's magnitude.
No vergence floor applies, because there is no vergence. Per frame and per
member.

### Cycle consistency

Every edge of the member's covisibility graph — a pair of posed frames
sharing at least a floor of points — carries a relative rotation
**measured from that pair's own shared rays alone**, by trimmed Kabsch on
the unprojected unit rays of the points they share. Composing the member's
own per-frame rotations around a cycle returns the identity by
construction and says nothing; composing these pairwise measurements
returns the identity only if they are one consistent field, which is what
a pure rotation implies and what parallax, a mis-associated track or a
frame fitted to the wrong points breaks.

The cycles read are a spanning tree's fundamental basis — each non-tree
edge together with the tree path that closes it — which spans the graph's
whole cycle space, plus the longest such cycle, because a long walk
accumulates what a triangle can hide. The tree is grown over the
best-shared edges first, so its paths are the best-measured ones.

Per cycle: its length and the residual angle of the composition. Member
outputs: median, p90 and worst residual, the median cycle length, the
largest cycle with its own residual, and the census of edges fitted. Per
frame: the worst residual over the cycles the frame takes part in, which
says where to look. A cycle residual implicates every edge of its cycle,
so the attribution names no single frame as the defect.

The reading is gauge-free and internal: no absolute pose, no referee, and
nothing outside the member.

### Exact photometric witness

Under a pure rotation the pairwise image map carries no surface term at
all: a pixel in `i` unprojects to a ray through the member's camera, the
relative rotation turns it, and the camera projects it into `j`. The map
is therefore fully predicted, and it can be checked against the images
themselves rather than against a fitted warp.

At each stored keypoint of a point two frames share, a window is laid over
image `i` whose extent is that observation's own affine shape — the
largest singular value of the map from the detector's canonical frame onto
this image's pixels — carried into image `j` through the member's camera
model and its relative rotation, and the two samplings are compared by
zero-mean normalized cross-correlation. The window is anchored at the
**stored keypoint**, never at a reprojection: the question is whether the
model explains the content the member matched.

The reading is the photometric disagreement, `1 - ZNCC`, so it grows with
the defect. A sample that leaves either image, a mapped ray that leaves
the camera's field, and a window with no contrast in either view are
non-measurements counted with their reason. Per pair, per frame and per
member: median and p90 disagreement with the correlation medians beside
them, plus the measurability census.

### Per-frame support

The observations the member holds on each posed frame, and that count against
the member's own median frame. A frame the member posed on almost nothing is
not a frame the member measured: whatever its rotation reads, it reads on too
little to be a reading. Member outputs: the most starved frame with its count,
and the worst starvation ratio with the frame named. The ratio is taken inside
one member, so it carries no capture scale.

The count is read per frame against the family's **support floor**, and a
frame below it is named. A named frame is cut, and the cut is taken on a
member no channel accuses exactly as on one every channel does: the count
says the member did not measure the frame, which is not a claim about the
frame's geometry and is not answered by one. Where nothing else accuses the
member, the named frames are the whole cut — a pose-shaped reading past a
fleet frame bar is a rank inside a sound member's own spread, and no frame is
cut on a rank.

The count travels with the frame: any reading that names a frame ships the
observations the member holds there beside it, so a worst frame the member
posed on a handful of observations is not read as the same object as one it
posed on a hundred.

### Parallax residue

A point claimed at infinity is an approximation: it holds over a
narrow-parallax subset of the capture and stops holding over a wider one. A
point that stops holding is a candidate for **graduation** to a finite depth,
not a defect of the member, which claims orientations. So this channel
measures the departure and never accuses: its readings are graduation
evidence.

Per point, over the frames that observe it in the member's whole admission:
the median residual against the rotation-only prediction, in units of the
member's own admission bound, which is the largest residual its own model
kept. A point past one bound is **parallax-bearing**. Member outputs: how many
points were measured, how many are parallax-bearing and what share, the
quantiles of the per-point residue, and the parallax-bearing points
themselves, loudest first, by the member's own point identity. The share
classifies the member's field as carrying no parallax-bearing points, some, or
a majority of them. Per frame: the same census restricted to the points that
frame observes.

The residual **field** is read beside it, because a departure that is baseline
has a shape a departure that is noise does not: each residual vector points
along the line from the frame's epipole through the observation, with a
magnitude set by the baseline over the point's range.

Per frame, over the whole admission: fit the epipole that best explains the
residual directions (least squares on `cross(residual, position − epipole) =
0`, using the frame's residuals above its own median magnitude, since a short
residual carries no direction), then read the median alignment between each
residual and its radial direction. Reported alongside: the residual
anisotropy, the median and p90 residual magnitude, and the share of the
frame's admitted observations the model refused. Isotropic residuals read at
the channel's own null level, the mean absolute cosine of a uniform angle; a
frame carrying baseline reads far above it. The null level ships with the
channel.

### Arbitration and coverage

Where a finite member and a rotation-only member cover the same frames, both
records are stored side by side, so the selection pass arbitrates between them
on their channels rather than on which model was fit. Coverage counts the
frames of every judged surviving member of either kind. A rotation-only
member's relaxed sibling is stored the same way, and arbitrates as the
finite-family member it is.

## Storage

Channels attach to each member's entry in the release manifest, with
per-frame and per-held-out-image detail nested under it. Frame and image
references use full relative paths. Conditioning and measurability fields
accompany every gated channel, and a refusal that names a frame carries
that frame's own support count.

Every member carries a block for every channel of both families. A finite
member records the rotation-only channels unmeasurable, and a
rotation-only member records the finite-only ones unmeasurable with the
reason `no finite structure`, so a reader of one record never has to know
which family it came from to know what was asked of it.

## Gating principles

A gate is a bar on a channel plus the population the bar was taken over.
The following hold for every gated channel.

- **Two readings, one fired jointly.** Each gated channel carries a
  noise floor (a quantile of the fleet's good-member readings, which
  is that channel's noise scale) and a capture-relative reading, the
  member's value over its own capture's median. A capture-relative
  reading fires only when the absolute reading also clears the noise
  floor. Normalizing by a capture median guarantees some member is the
  loudest, so the ratio alone accuses a member of a capture that carries
  no defect at all.
- **A rank is not a magnitude.** Both readings are ranks: the absolute
  bar is a quantile of the fleet's members and the capture-relative
  reading is a ratio against the capture's own median, and a population
  of sound members still has a loudest one. Every gated channel therefore
  carries a **refusal floor** beside its bars, the magnitude those ranks
  are read next to: the stricter quantile of the channel's whole
  gate-eligible population, taken over that whole population and not
  over the members no bar accuses, because a population defined by the
  bar cannot exonerate a member the bar accuses. A reading inside that
  spread refuses nothing, whichever reading fired. The floor is drawn
  per camera-model family on the same conditions the bar is, and a
  reading taken against the fleet's floor records that it was. Which
  channels carry a floor is stated per channel, alongside which readings
  of it may accuse; every channel of the rotation-only family carries
  one.
- **A count of the evidence is floored, not ranked.** A reading that
  counts what a measurement rests on, rather than saying what the
  measurement found, is read against a **support floor**: the low decile
  of the same counts over the fleet's own readings of that channel, drawn
  per camera-model family on the conditions above. Below the floor there
  was too little to measure. That is a statement about the evidence and
  not an accusation, so a support floor never refuses a member, is never
  read as a rank against one, and stands whatever else the member reads.
- **An absolute bar is drawn per camera-model family where the family
  reads differently.** A fleet holding several lens families is several
  populations, and one bar over all of them is the largest family's bar
  applied to the rest. Each family's bar and noise floor are taken at the
  same quantile over that family's readings alone. A family's own bar
  stands when the family holds enough readings for the quantile to be an
  interpolation between two of them rather than their maximum, and when
  the whole-fleet bar falls outside the family bar's own sampling
  interval; otherwise the fleet bar stands and every reading taken
  against it records that. A capture-relative reading is a ratio inside
  one capture, which is already inside one family, and is not split.
- **A majority-defective capture has no usable median.** When the
  capture's median member fails the absolute gates, the capture-relative
  readings of every one of its members are non-measurements with that
  reason: the denominator is itself a defective member. The capture is
  then judged on absolute readings alone.
- **Hard-weld channels gate at a stricter quantile.** A defect confined
  to two frames of a dozen is diluted to nothing by a member-level
  median, and the channels that see it are worst-over-median frame
  spreads whose bulk is noise. Those gate at the top quantile of their
  population; member-median channels gate at the ordinary one. Both
  quantiles ship with the gate record.
- **Conditioning is respected, never re-derived.** A reading taken below
  a conditioning floor the generation pass recorded (a vergence floor,
  a triangulation-angle bar, a parallax bar, a support floor) is a
  non-measurement and can never refuse.

## Selection-pass consumption

The selection pass consumes the channels as follows:

- **Rank** members within the capture by the fit channels together with
  the non-member translation-direction delta and the depth-agreement rho.
- **Refuse** a member only on defect evidence: a gated worst-channel past
  its population gate with healthy conditioning, a diverging refit, or a
  focal-vote outlier. Redundancy is not defect evidence.
- **Corroborating channels never refuse alone.** The coherence and
  support channels (per-frame depth-scale coherence, near-support ratio)
  and the focal-vote deviation qualify a verdict that other evidence
  already carries; none of them is a sufficient ground for a refusal on
  its own, and none of them names a frame.
- **Trim** instead of refusing when the defect localizes to named frames:
  drop those frames and the points that lose support. Every frame cut must
  be named by a pose-shaped per-frame channel — a hold-out pose delta, a
  settling delta, a per-frame warp or surface residual — or by a per-frame
  support count below its floor, which names a frame by counting what the
  member holds there rather than by inferring anything about its geometry. A
  member-wide reading cannot be repaired by dropping frames, and a
  coherence channel cannot name a frame to cut.
- **A trim may not fragment a member.** Every pair of surviving frames the
  member tied together through shared structure must still be tied together
  in the core, which the loss of points below two supporting observations is
  what can break. The test is a comparison and not an absolute: a member
  whose frames are held together by something other than shared structure is
  in as many pieces after the cut as before, and a cut that changes nothing
  about how its frames are tied has broken nothing. A cut that names no
  frame the member still holds is not a trim at all.
- **A trimmed core is re-evaluated as a member.** The channels are
  re-derived over the frames the core kept, at the conditioning floors the
  generation pass recorded for the capture, and the readings are put to the
  same member gates the whole member faced. A reading taken before the cut
  describes geometry the core no longer holds and cannot answer for it. Where
  defect evidence drove the cut the trim stands only if the core passes;
  otherwise the verdict is a refusal, and the frames the trim would have cut
  are recorded with it. Where a support count alone named the frames there is
  no accusation to fall back on, so a core that does not stand leaves the
  member exactly as it was, with the cut recorded and not taken. A channel
  the capture supplies rather than the member (a held-out image's verdict) is
  unmeasurable on a core, which is a reading about what may be asked of a
  core rather than a reading about this one.
- **Every model family is judged on its own channels.** A rotation-only
  member is ranked and refused on the rotation-only family exactly as a finite
  member is on the finite one; it is never passed through unjudged, and it is
  never trimmed on a coherence reading. Gates are quantiles of the family's
  own population, because only that family produces the reading.
- **A rotation-only member is refused on its orientations and its
  support, never on its parallax.** The grounds are the hold-out rotation
  resection, the rotation-only settling probe, the full-form warp, the
  cycle residual, the photometric witness, the non-member witness, and the
  member's support census. Each of them is read past its refusal floor
  before its rank is read at all, so a member that is merely its capture's
  or its fleet's loudest is not a member with a defect. The per-frame
  support floor is not among the grounds: it cuts the frames it names and
  leaves the member standing on the rest. The parallax residue is recorded
  and consulted by whatever graduates points to finite depths; it is not
  defect evidence, because a point that leaves infinity leaves the member's
  orientations where they were.
- **Coverage** of the capture is computed from the surviving set of both
  families, never inferred from quality; complementary members (local
  density, loop closure, and the far field) are all kept. Where a surviving
  finite member and a surviving rotation-only member cover the same frames,
  the pass reports the overlap with both sets of channels rather than
  choosing between the models.
