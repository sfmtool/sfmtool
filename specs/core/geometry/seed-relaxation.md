# Seed Relaxation

A rotation-only member claims bearing without range: one rotation per frame,
every camera centre at one point, and one direction per cluster its model
explains. The observations that model REFUSED are the near points, and over a
frame pair they carry the pair's baseline. The relaxation turns those refused
rows into camera centres and finite depths, fills the result in from the source
clusters the member's admission never held, reads a lens on it, and ships it as
a member beside the original.

The rung runs on every committed rotation-only member, at group scope and at
capture scope alike. It changes nothing about the member it reads: the
rotation-only entry, its release file and its channels stand, and the relaxed
member is a new entry.

## Release the lens on the bearings

A pure rotation between two frames maps rays to rays, with no depth in the map,
so the bearings a rotation-only member already carries are by themselves
evidence about the lens. Reading them costs one adjustment with every point
marked at infinity, which freezes every translation and leaves the rotations,
the focal and a radial spline in the solve.

The reading is taken only where the member's base chart is the chart it
corrects. A fisheye chart is a nominal design target a real lens misses by
shape no focal absorbs, and a release over that field corrects a chart that is
wrong by construction. A perspective chart is what a rectilinear lens is built
to be: such a member is either nearly right already, or wrong in a way its
admission (mostly refused rows, which is to say parallax) cannot tell apart
from a focal error, and the release would shorten the focal to absorb the
parallax. So the gate is the base chart.

Where the gate opens, the member's base camera is promoted to its base's
radial-spline model with an all-zero spline. That promotion is the same map bit
for bit; what it does is allocate the coefficient slots the release needs. The
solve releases the focal and the spline together, since under the
centre-anchored gauge the spline cannot express a central-scale correction. On
success the released camera and the refined rotations are both adopted, and
every cached ray and the member's angular bound are re-derived through the new
camera. A solve that raises, or that comes back with no finite residual, is a
refusal: the base camera stands and the record says so.

## Read baselines from the refused rows

The graph is the member's covisibility graph over its WHOLE admission: its own
inlier rows plus the observations the rotation model refused. A pair fit that
reads only the inlier set has no parallax residue to anchor on. Two frames are
an edge when they share at least the rotation channel's own floor of clusters.

With the member's rotations held, the baseline `b = c_j - c_i` is coplanar with
every point's two world rays: `b . (u_i x u_j) = 0`. The normal `u_i x u_j` has
norm `sin(parallax angle)`, so a row's weight is literally how much baseline
that point saw, and a row inside the member's own angular consensus bound
carries none: inside the bound the two rays are the same ray. Those rows are
DROPPED rather than down-weighted. An edge with fewer than three rows past the
bound states no direction and is not an edge.

The direction is the null space of the retained rows, refitted on its own best
fraction for the trim rounds the pipeline's rotation fits use. Its sign is
fixed by cheirality: with one centre at the origin and the other at `+d`, the
direction that puts more points in front of both cameras is the one kept.

Each edge is trusted in proportion to how much baseline its rows saw and how
cleanly one direction explains them: the count of rows past the bound,
discounted by the null space's own conditioning and by the cheirality majority.
All three come out of the edge's own solve, and the whole graph is one call of
[the baseline-direction operation](baseline-direction.md).

## Read the relative scale off the depths

A direction fixes a pair's geometry only up to how long its baseline is, so the
two-view depths the pair's own solve implies are in units of THAT baseline. A
cluster two edges both see from the same frame therefore carries two depths for
one world distance, and their ratio is the ratio of the two baselines. That is
the relative scale the directions cannot state, and it is read off the same
rows the direction was read from: the rows past the member's angular bound that
triangulate in front of both cameras.

The whole graph is one fit of `log z(edge, frame, cluster) = D(frame, cluster)
- x(edge)`, with `x` the log baseline length and `D` the log world depth. There
are as many `D` as clusters seen, and they are eliminated rather than solved
for -- each sits at its own weighted mean over the rows that saw it -- which
leaves a system in the edges alone whose operator is a pass over the rows and
never a matrix. Rows are reweighted for a fixed number of rounds by the inverse
of their own residual, floored at the median residual, so a wild depth stops
carrying the fit. An edge that shares no cluster with any other edge states
nothing relative and is left without a length rather than given one.

## Average the translations

The centres minimize `sum_ij w_ij || P_ij (c_j - c_i) ||^2 + a_ij (d_ij . (c_j
- c_i) - s L_ij)^2` -- the part of each baseline that is NOT along its measured
direction, and the part that is against the length the depths measured -- under
two gauges: the scale gauge `sum_ij w_ij d_ij . (c_j - c_i) = sum_ij w_ij` and
the shift gauge `sum_j c_j = 0`. Those are exactly the two freedoms the
pairwise readings cannot see, so fixing them adds nothing and removes the
collapse to a point. The lengths are RELATIVE, so the scale `s` that turns them
into distances is an unknown of the fit and is eliminated: the length half
states proportions and never a gauge. An edge without a length has `a_ij = 0`
and constrains only its direction.

Each half is reweighted for a fixed number of rounds by the inverse of that
edge's own residual in that half, so an edge whose direction or whose length
the rest of the graph contradicts stops carrying the solve. The floor of each
reweighting is the median residual of its own half, which is the graph's own
scale rather than a number set here.

The averaging runs over the largest connected component of the graph the
baselines describe, and every frame in that component is placed at once: no
frame is chained onto a single neighbour, so a short baseline cannot set the
scale of everything past it.

### The constellation is what the form sends to zero

The objective is a quadratic form on the stacked centres, and the true centres
send it to zero: at them every baseline's perpendicular part vanishes and its
along part is exactly its relative length. So the solve is `argmin x' B x`
under the scale gauge, read off the form's own spectrum, rather than a linear
system posed against `B`. Where the form is positive definite the two are the
same answer. Where it is singular they are opposites, because the
pseudo-inverse returns the part of the answer ORTHOGONAL to the null space,
which is the one part the measurement does not carry.

How many directions the form leaves null is the graph's own reading, counted at
the numerical rank tolerance and nothing set here. One of them is the
constellation. A graph whose baselines all point the same way, which is what a
camera moving along a straight line produces, leaves one more per free spacing:
the perpendicular part is empty for every arrangement of the cameras along the
line, so only the total scale is fixed by the gauge, and six exactly colinear
directions leave five null directions where a general graph leaves one. The
relative lengths remove exactly those, because moving a camera along the line
changes the along part even when it leaves the perpendicular part alone.

**The null space is the constellation only when the whole of it is shared.** A
frame whose edges do not between them fix where it sits -- one edge and no
length for it is enough -- moves freely at no cost, so it owns a null direction
of its own, and a constellation read off a null space containing one would put
that frame at no distance in particular. Which frames those are is read off the
projector onto the null space, which needs no basis: the trace of a frame's own
three-by-three block of it is how many null dimensions are that frame moving,
and a frame owning more than half of one is not part of a constellation. So the
null space states the centres only when it is one dimension no frame owns half
of; otherwise the centres are the range solution, which leaves a loose frame
where the measurement leaves it rather than at a distance nothing measured.

Every member records what its own form said: the largest eigenvalue, the two
smallest as fractions of it, their ratio, how many directions were left free,
how many frames were loose and whether the centres were read off the null space
-- once for the directions alone and once with the lengths in -- beside how
many edges stated a length at all. A graph that states no length and still
leaves a direction free does not determine the constellation, and its frames
are left unplaced with that reason recorded rather than placed at a spacing the
solve's own arithmetic chose.

## Fix the orientation

Pairwise directions determine the constellation up to the point reflection
`c -> -c`. That reflection is exact: the angular least-squares estimate is
linear in the centres and its matrix does not contain them, so negating every
centre negates every point and every depth and changes nothing else. One pass
therefore describes both orientations, and the second is arithmetic that is
already known.

Structure sitting in FRONT of the cameras is the one physical statement that
separates them. The bit is read as a cheirality vote over the observations of
every cluster two placed frames see, each cluster weighted by its own widest
ray pair: a cluster's cheirality statement is worth exactly the parallax it was
measured with, and a cluster inside the member's bound is a bearing whose depth
sign is a coin toss. A negative vote reflects the constellation. The vote per
observation and the unweighted front fraction's distance from a half are both
recorded beside the verdict.

## Graduate and adjust

Every cluster the placed frames see is estimated by angular least squares and
admitted as FINITE when its own widest pair of rays subtends more than the
member's angular bound and it lands in front of every camera that sees it. The
rest stay bearings.

Any frame the graduated structure can resect is then placed, rotation locked,
and the structure re-estimated; the adjustment follows, over the member's whole
admission on the placed frames, with the at-infinity points marked so they stay
directions and contribute rotation only.

A second round repeats the graduation and the adjustment, because a point the
first round could not place may clear the bar once the centres have moved.
Restarting a converged adjustment at the loosest stage would re-admit
everything the first round trimmed, so the second round runs only the stages
whose trim bound is at or below the residual the state actually carries -- its
own 99th percentile -- and never fewer than the final stage.

The round KEPT is the one whose own admission it reprojects through best. A
later round adds frames and points, but it can also chase a re-estimation the
geometry did not support, so the choice is a reading and not a preference.

## Fill in clusters by radius

A seed candidate is built from a small number of large-radius clusters, and
every stage above reads only that admission. The clusters it left behind are
not worse evidence, they are FINER evidence: a smaller feature localizes
better, and the reason it was dropped is that the bootstrap needed an unaliased
basin.

The identity between a member observation and a source-file member is exact and
needs no geometry: both carry the image index and the feature index, so a
member row is a file row by `(image, feature)`. A CANDIDATE is a source cluster
that at least two placed frames see and that the member's admission never held.
Its radius is the seed's own reading, the refine radius times the mean of the
stored affine's two column norms, a cluster taking its widest member's, so
"radius" means here what it meant when the admission was drawn.

Candidates are banded by radius in units of the member's own admission floor,
the smallest radius its admission holds. The bands are octaves down from that
floor, and the top band is open above it: a member's admission is a group-local
top-N over its own images rather than a capture-wide radius bar, so a cluster
coarser than the floor can be outside it, and it is the coarsest evidence
there is. How many octaves the grid runs to is a fleet-derived constant, the
pooled first percentile of candidate radius over the floor, and it ships with
its derivation so a later fleet re-derives it rather than re-tuning it. A
per-member percentile would change the band count per member and make two runs
of the same member incomparable.

The join, the candidate rule, the radius reading and the band each candidate
falls in are one operation:
[source clusters](../analysis/source-clusters.md), which the relaxation calls
once per member with the band grid it derived from that constant.

A band admits its WHOLE population: every candidate whose radius falls between
its edges. There is no count to choose, because the population a band stands
for is the band. A count admits a fraction of a band instead, and which
fraction differs per member, since a member's supply within one octave is a
property of its own capture: the same number is one member's whole band and a
small part of another's, so what it keeps is a sub-band nothing states. What
bounds the work is the member's own candidate supply and the number of bands.
A caller that has to bound it by something else may set an absolute count per
band; the band is then taken coarsest first with the cluster id breaking ties,
which is the order the whole band is admitted in anyway, so the result stays a
stable function of the file.

Per band, in order from coarsest to finest: the band's clusters are estimated
at the current poses and lens, admitted finite under the same three rules the
graduation uses plus the adjustment's own final trim bound (a point the
adjustment would throw away is not admitted), and then one adjustment runs over
the extended state with the LENS HELD. Holding the lens is the point of running
it between the bands: the band that just joined is absorbed by the geometry and
not by the camera, the next band is resected against refined structure, and the
lens is asked once, at the end, on all of the evidence.

The added rows join the member's admission and stay OUTSIDE its membership, so
the member's own inlier set -- which every reference-free reading of the
rotation-only member is taken on -- is unchanged. Their cluster ids are the
extended member's own and continue after the member's; they are not ids of the
source selection.

## Release the lens when the finite count clears the settling bar

The late release is one adjustment over the filled-in state with the focal and
the spline both open. By then the near points have depth and the parallax is
explained by the geometry instead of being pushed into the focal.

On the fisheye chart it always runs: that family's released focal is settled
from the first reading. On the perspective chart it runs only when the filled
state's finite-point count clears the SETTLING BAR -- a fleet-derived count
below which a member's released focal is still moving from band to band by more
than the focal vote's own spread. The bar is a population statement about how
much finite evidence a lens reading needs, and it ships with its derivation
beside the ring constant.

The knot count is per chart: the fisheye chart's is wider than the seed's own,
the perspective chart's is the seed's. Before the solve, the current lens is
RE-EXPRESSED on a spline of that knot count. The spline correction enters the
map linearly, so the coefficients that make a k-knot camera reproduce a given
map are one linear least squares over the basis, and the basis columns are read
from the model itself. A camera re-expressed this way starts where it ended
rather than at the base model, and a base camera re-expressed at any knot count
gives exactly zeros.

A release that raises, that finds no spline domain, or that comes back with no
finite residual is a refusal: the previous camera and state stand.

## Re-estimate every point

An adjustment trims observations, and a point whose observations were all
trimmed stops moving while the poses and the lens go on; a point that graduated
on rays that do not cross carries a depth nobody measured. Both are re-read in
one pass at the final poses and the final lens, before anything is written.

Every point is re-estimated by least-squares midpoint over its own observing
rays. The angular floor is the pixel bar carried through the equivalent focal
of the camera in use, so the bound moves with the lens the state settled under.
A point whose widest ray pair does not clear that floor becomes a bearing, so
does one that lands behind a camera that sees it, and so does a single-view
point -- whose direction is nonetheless measured, and is its own observation's
ray.

The pass, and the per-ring estimation of the fill-in, are the point
estimation operation with the relaxation's settings
([point-estimation.md](../reconstruction/point-estimation.md)).

## Report the runaway frames

Per frame, the distance to its nearest other centre, read WITHIN its own
sensor, divided by the member's own median of that distance. On a rig the
nearest camera to a left frame is its own right frame a few centimetres away,
which is the rig's baseline and not the capture's sampling, so a spacing read
across sensors would report the wrong quantity; on a single-camera capture the
two readings are the same. A frame at 1.0 sits where its neighbours do; a frame
far above it is somewhere else.

Nothing is trimmed on this reading. It is recorded, per frame and in aggregate,
and the decision is the reader's.

## Product

A relaxed member is committed through the same seam every hypothesis passes
through, so it gets a manifest entry, a release file beside the others and its
arrays in the members sidecar. Its model is `relaxed`, it names the
rotation-only member it came from, and it copies that member's scope. The
rotation-only entry gains a back-pointer to it, or, where the chain refused,
the reason.

The release carries the relaxed camera, the placed poses, the finite points and
the bearings, with the bearings stated homogeneously at `w = 0`. Its metadata
records the released and chart focals, the spline and its domain, the early
release's verdict, the orientation and the two readings behind it, the fill-in
per band, the late release's verdict with its knot count, the finite and
at-infinity point counts, and the runaway aggregates. The writer keeps a point
only where at least two of its observations survive, so a single-view bearing
from the re-estimation is not in the file; the record states how many those
were.

The relaxed member is measured by the candidate battery on the FINITE channels,
beside its rotation-only source, so a selection pass arbitrates between the two
on their channels rather than on which model was fit.

## Determinism and options

Nothing in the chain samples, and no stage reads a clock into a record: the
same member and the same source file produce the same arrays, bit for bit. Every
ordering is stable -- edges in frame order, candidates by radius with the
cluster id breaking ties, bands coarsest first.

The rung has a kill switch, and with it off every product is byte-identical to
a run from before the rung existed. An absolute count per band, the fisheye
knot count and a per-stage trace are the only other options; the band grid and
the settling bar are fleet-derived constants and are not tunable.
