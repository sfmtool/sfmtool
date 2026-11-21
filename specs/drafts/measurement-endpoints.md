# Measurement endpoints: validation and open questions

**Status:** Draft

Amends [core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md),
which resolves a measurement endpoint, a pixel in one image, to a texel of the
patch of an observation in that image whose square holds the pixel, and to the
3D position that texel stands for. Where several squares hold the pixel, that
spec ranks the front layer by a cost `ℓ · S · O · A · Z` whose form and
constants are starting values. Decided: the candidates, the texel, the
position on the point's own patch, the front layer, the refusals, and that an
`Alone` candidate is not ranked. Open: the form of the cost and its constants,
to be settled by measuring them in the harness below, and the questions in
§ "Open questions".

## Validation

A leave-one-out harness on the ground-truth reconstructions, which are in
metres, in the manner of the track-at-pixel harness
([track-at-pixel.md](track-at-pixel.md)):

1. Take a point with a patch, and one of its observations: image `i`,
   keypoint `k`.
2. Resolve `(i, k)` on the reconstruction with that point deleted
   (`EditedReconstruction::delete_point`), so that the answer has to come from
   the squares of neighbouring observations in image `i`.
3. The error is the distance from the resolved position to the deleted
   point's position, in metres.

Over many points this gives the error of each rule: the nearest square only,
the cost, and each factor removed in turn. The constants are fitted where
error falls, and the cost is checked against the error it predicts. Where the
cost tracks the error closely, it can be calibrated into an uncertainty, and
SfM Explorer can show a measurement's reconstructed distance with an error bar.
The harness computes the cost for every resolved endpoint, `Alone` ones
included, so that the calibration covers the common case too; the operation
itself does not.

`dino_dog_toy_ground_truth_v01` was scaled from four tape-measured distances.
Entered as measurements, they are a check of the whole chain, from endpoints
placed by a person to a scale factor.

## Open questions

1. **The form of the cost.** A product, as the reference-view rule chose and
   the operation uses, or lengths added in quadrature, which is how
   independent errors combine and would make the cost an error estimate from
   the start. The harness decides. If the cost becomes an uncertainty a caller
   shows, the operation would compute it for an `Alone` candidate too, as an
   option the viewer turns on when it draws error bars.
2. **A triangulation factor.** The cost has no term for how well the track
   fixes the point's own position, which the texel's 3D point inherits.
   Candidates for a factor `G`: `1 + triangulation_weight / tan(max_ray_angle)`,
   from the track's largest angle between viewing rays, or the track table's
   `inverse_depth_z`; or nothing, since bundle adjustment may fix every
   surviving point in a ground truth well enough that the term never decides a
   pick.
3. **Blending.** Whether to take a weighted mean of the front layer's
   positions instead of one pick. A mean would be smoother as an endpoint is
   dragged across square borders. A pick is what a person can inspect and
   correct, and what the reference-view rule does.
4. **Storing the resolution.** Whether a measurement should record the point
   and texel its endpoint resolved to, so that a later edit cannot move it
   silently. The operation resolves on every read and stores nothing, which
   keeps the format as it is.
