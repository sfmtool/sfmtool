# Depth layers

The scene near one pixel of a photograph can hold surfaces at very different
distances: a railing in front, the wall the pixel is on, a skyline behind.
When the other photographs are searched for 3D points near that pixel, the
points found can lie on any of those surfaces, so they are hypotheses about
the pixel's distance, not one estimate. **Depth layers** sort them out. Each
candidate point carries the stretch of distances along the pixel's ray that
its photographs allow; candidates whose stretches overlap form one layer, and
the layers are put in order of distance, nearest first. Each layer is then
checked against the pixel itself: the small square of the photograph around
the pixel is projected into every other photograph as if it lay at the
layer's distance, and compared with what those photographs show there. The
layer where it matches best, weighed with how close to the pixel its points
were found, ranks first, and every layer gets a confidence that the pixel is
on it. Nothing is written to the reconstruction.

The layers are the fourth part of the anchor finder in the track-at-pixel
harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`_layers`, `_layer_reads`, `layer_evidence` and `_rank_layers`) moved into
core ([the plan](../../drafts/nearby-tracks.md)). Their inputs are the
candidates the [matching sources](nearby-sources.md) and the
[far-field sweep](far-field-sweep.md) find, with the
[distance ranges](distance-range.md) of their sightings; the patch reads are
the far-field sweep's `read_patch_along_ray`.

## Rust API

In [`bench/nearby/layers.rs`](../../../crates/sfmtool-core/src/bench/nearby/layers.rs),
re-exported from `sfmtool_core::bench`, and bound as
`sfmtool._sfmtool.bench.depth_layers`.

```rust
pub fn depth_layers(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,               // the blurred grey images, cached across queries
    image: u32,                      // the queried image
    pixel: [f64; 2],                 // the queried pixel
    candidates: &[LayerCandidate<'_>],
    options: &DepthLayerOptions,
) -> Result<DepthLayers, DepthLayerError>; // NoSuchImage | GreyMismatch

pub struct LayerCandidate<'a> {
    pub source: NearbySource,               // Points | Clusters | Guided | Constellation | FarField
    pub sightings: &'a [(u32, [f64; 2])],   // only the images are read
    pub distance_px: f64,                   // from the queried pixel
    pub max_ray_angle_deg: f64,
    pub range: [f64; 2],                    // its distance range, as the caller gave it
    pub class: RangeClass,                  // bounded, far; usable if either
}
impl<'a> LayerCandidate<'a> {
    pub fn from_candidate(c: &'a NearbyCandidate, range: [f64; 2], class: RangeClass) -> Self;
    pub fn from_far_field(r: &'a FarFieldReading, range: [f64; 2], class: RangeClass) -> Self;
}

pub struct DepthLayerOptions {
    pub evidence: bool,           // read and rank the layers (true)
    pub rank_by: LayerRankBy,     // Key (default) | Score
    pub radius_px: f64,           // the patch's half-width (8)
    pub samples: usize,           // distances read across each layer (5)
}

pub struct DepthLayers {
    pub support: Vec<usize>,      // per candidate, in the order given
    pub layers: Vec<DepthLayer>,  // nearest first
}

pub struct DepthLayer {
    pub range: [f64; 2],          // the union of the members' ranges
    pub members: Vec<usize>,      // indexes into the candidates, nearest range first
    pub nearest_px: f64,
    pub max_views: usize,
    pub ranking: Option<LayerRanking>, // with `evidence`
}

pub struct LayerRanking {
    pub evidence: LayerEvidence,
    pub score: f64,
    pub key: f64,
    pub rank: usize,              // 1 is the best-supported
    pub confidence: f64,
}

pub struct LayerEvidence {
    pub n_candidates: usize, pub n_independent: usize, pub n_images: usize,
    pub max_views: usize, pub max_ray_angle_deg: f64, pub nearest_px: f64,
    pub at_pixel: bool, pub sources: Vec<NearbySource>, pub weight: f64,
    pub photo: f64, pub photo_middle: f64, pub photo_both: f64,
    pub votes: usize, pub votes_all: usize,
}
```

The constants of the key and the confidence are public: `KEY_NEAREST`,
`CONF_BIAS`, `CONF_MARGIN`, `CONF_VOTES`, `CONF_SUPPORT` and `CONF_NEAREST`.

**Why a `LayerCandidate` and not the candidates themselves.** The layers read
six things of a candidate, and two kinds of value carry them: a matching
source's `NearbyCandidate` and a far-field sweep's `FarFieldReading`. A small
struct that borrows the sightings lets a caller put both kinds in one list
without copying, through `from_candidate` and `from_far_field`, and lets the
Python binding build one from the harness's anchor dicts. Its `source` is a
`NearbySource`, which has a `FarField` variant for this, so a layer can name
every source among its members.

**Why the range and class are the caller's.** A candidate's range depends on
what it is: a matching source's is its sightings'
[`distance_range`](distance-range.md) at its point's distance, a far-field
reading's is the stretch between its neighbouring disparities, and a moved
far-field reading's is its sightings' again. The class depends on the
reconstruction's camera spread and the finder's thresholds. The caller that
ran the sources has all of this already, and the layers compare ranges
without caring how they were found.

**Why support is returned beside the layers.** Support is a per-candidate
count that reads the same ranges and image sets the grouping does, and the
finder reports it with each candidate; computing it here keeps the ranges
compared in one place.

**Why `ranking` is an `Option`.** With `evidence` off, the layers are only
grouped, which needs no photograph: that is what a caller deciding whether to
run another source needs to know (how many layers there are), and it costs
nothing. With it on, every layer is ranked, so the evidence, score, key, rank
and confidence arrive together or not at all.

```rust
use sfmtool_core::bench::{
    camera_spread, classify_range, depth_layers, DepthLayerOptions, GreyImages,
    LayerCandidate, RangeOptions,
};

let spread = camera_spread(&views);
// `found`: the Vec<NearbyCandidate> the matching sources returned.
let ranges: Vec<[f64; 2]> = found
    .iter()
    .map(|c| c.range(&views, 1.0))
    .collect::<Result<_, _>>()?;
let candidates: Vec<LayerCandidate<'_>> = found
    .iter()
    .zip(&ranges)
    .map(|(c, &range)| {
        let class = classify_range(range, spread, &RangeOptions::default());
        LayerCandidate::from_candidate(c, range, class)
    })
    .collect();
let grey = GreyImages::new(views.len());
let layers = depth_layers(&views, &grey, image, pixel, &candidates, &DepthLayerOptions::default())?;
let first = layers.layers.iter().find(|l| l.ranking.as_ref().unwrap().rank == 1);
```

## Theory

**Support.** A candidate is usable when its range is bounded or far. Two
usable candidates support each other when their ranges overlap and neither's
images are all among the other's: two readings of one structure that do not
rest on the same photographs. A candidate's support is how many others
support it. A cluster and a point seen in the same four photographs agree
trivially; a cluster in images 1, 2, 3 and a guided match in 1, 4, 5 that put
the point at the same distance are two pieces of evidence.

**Grouping.** The usable candidates are taken nearest range first, and each
joins the last layer when its near end is within that layer's range, which
its far end then widens, or starts a new layer. The layers are therefore
nearest first and their ranges do not overlap. Candidates with the same near
end keep the order they were given in.

**The evidence reads.** The pixel's patch, an 11 by 11 grid of `radius_px`
around it, is read in every other image at `samples` distances across each
layer's range, evenly spaced in inverse distance from the far end in; for a
layer with no far end, from infinity in. Each read gives the ZNCC of the whole
grid and of its middle 5 by 5 from the same samples. An image's reading at a
layer is its best whole-patch reading over the layer's distances, and its
middle reading is the one at that same distance. An image the patch does not
land in reads `-1`.

**The evidence.** From the members: how many (`n_candidates`), how many
different image sets they rest on, not counting a set wholly within another
(`n_independent`), how many images between them, the most views and widest
ray angle of one member, the nearest member's distance from the pixel,
whether one is within a pixel of it (`at_pixel`), their sources, and their
**weight**, the sum of `log2(1 + views)` times `exp(-distance_px / 20)`. From
the reads: `photo`, the mean of the three best images' whole readings;
`photo_middle`, the same for the middle over the images the whole reads in;
`photo_both`, the same for the lesser of the two, image by image. An image
**votes** for a layer when it reads the whole patch 0.7 or better there, 0.05
better than at any other layer, and the middle 0.7 or better; `votes_all`
counts without the middle's condition. Short-baseline images read every layer
alike and do not vote.

**Score, key and rank.** The `score` is `(photo + photo_both) / 2`: the whole
patch's reading, and the lesser of the whole and the middle, so a match the
middle does not share counts half. The **key** is

    key = score + photo_middle - KEY_NEAREST * ln(1 + nearest_px)

the patch weighted toward its middle, which is the pixel's own neighbourhood,
less a little for a layer whose candidates were all found away from the
pixel. The layers are ranked by the key, highest first, or by the score with
`LayerRankBy::Score`; a tie keeps the layers' order.

**Confidence.** Each layer's confidence is the logistic

    z = CONF_BIAS + CONF_MARGIN * margin + CONF_VOTES * ln(1 + votes)
        + CONF_SUPPORT * ln(1 + weight) - CONF_NEAREST * ln(1 + nearest_px)
    confidence = 1 / (1 + exp(-z))

with `margin` the layer's key less the best other layer's key, and 1 for a
lone layer. It was fitted as the chance that the first-ranked layer is the
pixel's; a layer ranked below has a negative margin and a lower confidence. A
lone layer can be wrong with nothing ranked above it, which is what the
confidence lets a later step refuse.

**Where the constants came from.** Both were fitted on the harness's rows of
both ground truths (seoul_bull's 1277 queries and Kerry Park's 3903), full and
empty passes, with every source run and with the default stopping rule. The
key was fitted as a ranking over every layer feature on one ground truth and
tested on the other: three features carried it, and the middle's reading and
the nearest candidate's distance took the same weight fitted on either. It
ranks the pixel's layer first 94 to 95% of the time where there are several
layers, against 93 to 94% for the score alone and about 39% for a layer
chosen at random. A support term in the key ranked better with every source
run and worse with the default stopping rule, which stops after the
reconstruction's own points: their many views give their layers high support
whichever surface they are on. It is left out of the key and kept in the
confidence. The confidence tells a right first-ranked layer from a wrong one
with an area under the ROC curve of 0.88 to 0.93 when fitted on one ground
truth and tested on the other, against 0.75 to 0.78 for the first layer's
score. The far-field readings' own metrics added nothing to either once
support and votes were in. The harness's
[README](../../../scripts/track_at_pixel/README.md) (§ "Ranges, support and
layers" and the ranking tables after it) records the runs.

## Implementation notes

**One read for every layer.** The distances of every layer are read in one
`read_patch_along_ray` call over every image but the queried one, so the
queried patch is sampled once. An image's best reading at a layer is taken
with a strict `>` in the order of the distances, so of two equal readings the
further one wins, and the middle reading comes from the same distance as the
whole's; the harness does the same.

**The distances are `numpy.linspace`'s.** The inverse distances are
`i * step + lo` with the last set to `hi` exactly, as `numpy.linspace` makes
them, so the harness and core read the same planes.

**The ZNCCs are `f64`** where the harness's Python read in `f32`, sampling
at `f32` coordinates. Over both ground truths, full and empty passes, with
every source run and with the default stopping rule, the two give the same
support, layers, members, votes and ranks, and the harness's summaries are
identical. The scores, keys and confidences differ in the last bits of a
32-bit float: at most about 3e-6 on seoul_bull, and up to 6e-5 on Kerry Park,
where the largest are layers very near the queried camera, under a tenth of a
unit away. Where an image reads two of a layer's distances equally to within
that noise, the two can pick different distances for its middle reading; the largest such case
moved one layer's key by 7e-4 and its confidence by 3e-4, and no rank.

## Parameters

Defined in `DepthLayerOptions::default` and the constants in
[`layers.rs`](../../../crates/sfmtool-core/src/bench/nearby/layers.rs); the
harness names are in parentheses.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `evidence` (`layer_evidence`) | `true` | Read and rank the layers |
| `rank_by` (`layer_rank`) | `Key` (`"evidence"`) | Rank by the key, or by the score (`"score"`) |
| `radius_px` (`layer_radius_px`) | `8.0` | The patch's half-width, in px of the queried image |
| `samples` (`layer_samples`) | `5` | Distances read across each layer's range |
| `KEY_NEAREST` | `0.05` | The key's weight on `ln(1 + nearest_px)` |
| `CONF_BIAS` | `-2.0` | The confidence's constant term |
| `CONF_MARGIN` | `2.0` | Its weight on the margin over the best other layer |
| `CONF_VOTES` | `0.47` | Its weight on `ln(1 + votes)` |
| `CONF_SUPPORT` | `1.5` | Its weight on `ln(1 + weight)` |
| `CONF_NEAREST` | `0.3` | Its weight against `ln(1 + nearest_px)` |

The vote's thresholds (0.7, 0.05), the pixel's reach for `at_pixel` (1 px) and
the weight's fall-off (20 px) are private constants of the same file.

## Python bindings

`bench.depth_layers(edited, images, image, pixel, anchors, *, options=None)`
takes the harness's anchor dicts, reading `source` (`"tracks"`,
`"clusters"`, `"guided"`, `"constellation"` or `"farfield"`), `views`
(`[image, x, y]` rows), `n_views`, `distance_px`, `max_ray_angle_deg`,
`range`, `bounded` and `far`, and writes nothing to them. `images` is a list of
photographs or an `ImagePyramidSet`, which also keeps the grey images between
calls. `options` overrides `evidence`, `rank_by` (`"evidence"` or `"score"`),
`radius_px` and `samples`; an unknown key, an unknown source or an anchor
whose `n_views` is not its number of views is a `ValueError`.

It returns `{"support": [...], "layers": [...]}` with each layer a dict of the
harness's keys: `range`, `anchors`, `nearest_px`, `views` (the most views of a
member) and, with the evidence, `evidence`, `score`, `key`, `rank` and
`confidence`. The evidence keeps the harness's names: `n_anchors`,
`max_ray_angle`, `support` for the weight and `photo_mid` for the middle.

```python
found = bench.depth_layers(edited, pyramids, image, pixel, anchors)
for a, s in zip(anchors, found["support"]):
    a["support"] = s
first = next(L for L in found["layers"] if L["rank"] == 1)
```

The harness's `find_anchors` calls it by default (`layers_impl="rust"`); with
`layers_impl="python"` it runs its own reference.

## Testing

[`layer_tests.rs`](../../../crates/sfmtool-core/src/bench/nearby/layer_tests.rs)
decides the layers on the synthetic capture of a textured plane: two
candidates at and in front of the plane form two layers nearest first, and the
plane's ranks first with the higher confidence and every other image voting
for it; overlapping ranges merge into one layer with the union of their
ranges; support needs different photographs and a usable, overlapping range;
equal near ends keep the order given; a far layer is read from infinity in; a
clear winner is more confident than a close one, and the key and confidence
follow their constants; and the refusals.
[`tests/rust_bindings/test_depth_layers_rust_bindings.py`](../../../tests/rust_bindings/test_depth_layers_rust_bindings.py)
checks the binding's support, grouping, evidence and ranking against the
harness's own Python on the seoul_bull fixture, the evidence from reads made
with the `read_patch_along_ray` binding at the harness's distances; its keys,
a far-field anchor, the options and the refusals; and that binding's shapes
and refusals. Parity with the harness is measured by running it with
`layers_impl=rust` and `layers_impl=python`.

## Non-goals

The layers do not build or fit tracks, and do not decide what to do with a
low confidence; the nearby-tracks finder and the steps after it do. They do not
merge two layers whose ranges do not overlap, however close.
