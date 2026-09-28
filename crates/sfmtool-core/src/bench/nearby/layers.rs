// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Depth layers: the candidates near a pixel grouped by the distances along
//! its ray their sightings allow, each group read with the pixel's own patch
//! and ranked by how well the photographs say the pixel is on it.
//!
//! `specs/core/bench/depth-layers.md` is the design, and where the ranking's
//! constants came from.

use crate::patch::normal_refine::ProjectedImage;

use super::candidate::NearbyCandidate;
use super::far_field::FarFieldReading;
use super::grey::GreyImages;
use super::patch_read::{read_patch_along_ray, RayPatch};
use super::range::RangeClass;
use super::NearbySource;

/// How much the key takes off for a layer found away from the pixel, per unit
/// of `ln(1 + nearest_px)`.
pub const KEY_NEAREST: f64 = 0.05;
/// The confidence's constant term, before the logistic.
pub const CONF_BIAS: f64 = -2.0;
/// The confidence's weight on the layer's margin over the best other layer.
pub const CONF_MARGIN: f64 = 2.0;
/// The confidence's weight on `ln(1 + votes)`.
pub const CONF_VOTES: f64 = 0.47;
/// The confidence's weight on `ln(1 + weight)`, the members' weight.
pub const CONF_SUPPORT: f64 = 1.5;
/// How much the confidence takes off per unit of `ln(1 + nearest_px)`.
pub const CONF_NEAREST: f64 = 0.3;

/// The reading an image must reach at a layer to vote for it, whole and, for
/// [`LayerEvidence::votes`], in the middle.
const VOTE_MIN_ZNCC: f64 = 0.7;
/// How much better an image must read at a layer than at any other to vote
/// for it.
const VOTE_MARGIN: f64 = 0.05;
/// A member within this many px of the pixel is a reading of the pixel's own
/// distance.
pub(super) const AT_PIXEL_PX: f64 = 1.0;
/// The distance from the pixel, in px, over which a member's weight falls by
/// a factor of `e`.
const WEIGHT_FALLOFF_PX: f64 = 20.0;

/// A candidate as the depth layers read it: what found it, the images it rests
/// on, where it sits relative to the pixel, and its range with what the range
/// says.
///
/// The layers read nothing else of a candidate, so a matching source's
/// [`NearbyCandidate`] and a far-field sweep's [`FarFieldReading`] both become
/// one, through [`Self::from_candidate`] and [`Self::from_far_field`], and a
/// caller holding either kind passes them in one list. The range is the
/// caller's because it depends on the reading: a matching source's candidate
/// has its sightings' [`super::distance_range`], a far-field reading the range
/// the sweep gave it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerCandidate<'a> {
    /// What found it.
    pub source: NearbySource,
    /// Its sightings, `(image, pixel)`. Only the images are read: how many,
    /// and which.
    pub sightings: &'a [(u32, [f64; 2])],
    /// How far it sits from the pixel asked about, in the queried image, in px.
    pub distance_px: f64,
    /// The widest angle between two of its sightings' rays, in degrees.
    pub max_ray_angle_deg: f64,
    /// The distances along its pixel's ray its sightings allow, `[near, far]`.
    pub range: [f64; 2],
    /// Whether [`Self::range`] is bounded or far ([`super::classify_range`]).
    pub class: RangeClass,
}

impl<'a> LayerCandidate<'a> {
    /// A matching source's candidate, with the range it was given and its
    /// class.
    pub fn from_candidate(
        candidate: &'a NearbyCandidate,
        range: [f64; 2],
        class: RangeClass,
    ) -> Self {
        Self {
            source: candidate.source,
            sightings: &candidate.sightings,
            distance_px: candidate.distance_px,
            max_ray_angle_deg: candidate.max_ray_angle_deg,
            range,
            class,
        }
    }

    /// A far-field reading, with the range it was given (its
    /// [`FarFieldReading::range`], or its sightings' range for a moved
    /// reading) and its class.
    pub fn from_far_field(
        reading: &'a FarFieldReading,
        range: [f64; 2],
        class: RangeClass,
    ) -> Self {
        Self {
            source: NearbySource::FarField,
            sightings: &reading.views,
            distance_px: reading.distance_px,
            max_ray_angle_deg: reading.max_ray_angle_deg,
            range,
            class,
        }
    }

    /// How many images see it.
    pub fn n_views(&self) -> usize {
        self.sightings.len()
    }

    /// The images it rests on, sorted, each once.
    fn images(&self) -> Vec<u32> {
        let mut images: Vec<u32> = self.sightings.iter().map(|s| s.0).collect();
        images.sort_unstable();
        images.dedup();
        images
    }
}

/// How the layers are put in rank order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerRankBy {
    /// By [`LayerRanking::key`]: the patch reading, its middle and the
    /// members' nearness to the pixel.
    #[default]
    Key,
    /// By [`LayerRanking::score`], the patch reading alone.
    Score,
}

impl std::str::FromStr for LayerRankBy {
    type Err = String;

    /// The harness's spelling: `"evidence"` for [`Self::Key`], `"score"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "evidence" | "key" => Ok(Self::Key),
            "score" => Ok(Self::Score),
            other => Err(format!(
                "unknown layer rank {other:?} (expected evidence|score)"
            )),
        }
    }
}

/// What the depth layers run with. The defaults are the harness's.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthLayerOptions {
    /// Read each layer's evidence and rank the layers; without it the layers
    /// are only grouped (harness `layer_evidence`).
    pub evidence: bool,
    /// What the rank orders by (harness `layer_rank`).
    pub rank_by: LayerRankBy,
    /// The half-width of the pixel's patch the layers are read with, in px of
    /// the queried image (harness `layer_radius_px`).
    pub radius_px: f64,
    /// How many distances across each layer's range the patch is read at
    /// (harness `layer_samples`).
    pub samples: usize,
}

impl Default for DepthLayerOptions {
    fn default() -> Self {
        Self {
            evidence: true,
            rank_by: LayerRankBy::Key,
            radius_px: 8.0,
            samples: 5,
        }
    }
}

/// The candidates grouped into depth layers, and each candidate's support.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DepthLayers {
    /// Per candidate, in the order given, how many other candidates support
    /// it: both usable, their ranges overlapping, and neither's images all
    /// among the other's. Zero for a candidate that is not usable.
    pub support: Vec<usize>,
    /// The layers, nearest first.
    pub layers: Vec<DepthLayer>,
}

/// A range of distances along the pixel's ray where some of the candidates
/// put the scene, and the candidates that put it there.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthLayer {
    /// The union of the members' ranges, `[near, far]`.
    pub range: [f64; 2],
    /// The members, as indexes into the candidates, nearest range first.
    pub members: Vec<usize>,
    /// The smallest of the members' distances from the pixel, in px.
    pub nearest_px: f64,
    /// The most images one member is seen in.
    pub max_views: usize,
    /// The layer's evidence and its place in the ranking, when
    /// [`DepthLayerOptions::evidence`] is on.
    pub ranking: Option<LayerRanking>,
}

/// How a layer ranks among the others.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRanking {
    /// What the layer rests on.
    pub evidence: LayerEvidence,
    /// The pixel's own patch read at the layer: the mean of the whole patch's
    /// reading and the lesser of the whole's and the middle's,
    /// `(photo + photo_both) / 2`.
    pub score: f64,
    /// [`Self::score`] plus the middle's reading, less [`KEY_NEAREST`] times
    /// `ln(1 + nearest_px)`: what [`LayerRankBy::Key`] ranks by.
    pub key: f64,
    /// The layer's place, 1 for the best-supported; ties keep the layers'
    /// order.
    pub rank: usize,
    /// The chance, from the logistic fitted on the harness's rows, that the
    /// pixel is on this layer: for the first-ranked layer that it is right,
    /// and lower for the others, whose margin is negative.
    pub confidence: f64,
}

/// What supports one layer: its members, and how the pixel's own patch reads
/// there in every other image.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerEvidence {
    /// How many members.
    pub n_candidates: usize,
    /// How many different sets of images the members rest on, not counting a
    /// set that is all within another.
    pub n_independent: usize,
    /// How many images the members see between them.
    pub n_images: usize,
    /// The most images one member is seen in.
    pub max_views: usize,
    /// The widest ray angle of a member, in degrees.
    pub max_ray_angle_deg: f64,
    /// The smallest of the members' distances from the pixel, in px.
    pub nearest_px: f64,
    /// Whether a member is within a pixel of the pixel.
    pub at_pixel: bool,
    /// The sources the members come from, each once, by name.
    pub sources: Vec<NearbySource>,
    /// The members' weight: the sum over them of `log2(1 + views)` times
    /// `exp(-distance_px / 20)`, more for a member seen in more images and
    /// found nearer the pixel (harness `support`).
    pub weight: f64,
    /// The pixel's patch read at the layer: the mean of the three best images'
    /// whole-patch ZNCC, each image at the layer's distance it reads best at;
    /// `-1` when no image reads it.
    pub photo: f64,
    /// The same for the middle of the patch, over the same reads.
    pub photo_middle: f64,
    /// The same for the lesser of the whole and the middle, image by image.
    pub photo_both: f64,
    /// How many images vote for the layer: they read the whole patch 0.7 or
    /// better there, 0.05 better than at any other layer, and the middle 0.7
    /// or better.
    pub votes: usize,
    /// The votes without the middle's condition.
    pub votes_all: usize,
}

/// Why the depth layers could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum DepthLayerError {
    /// The queried image is not one of the views.
    NoSuchImage {
        /// The image asked about.
        image: u32,
        /// How many images the views hold.
        image_count: usize,
    },
    /// The grey-image cache is not for the views' images.
    GreyMismatch {
        /// How many images the cache is for.
        grey: usize,
        /// How many views there are.
        image_count: usize,
    },
}

impl std::fmt::Display for DepthLayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not one of the reconstruction's {image_count} images"
            ),
            Self::GreyMismatch { grey, image_count } => write!(
                f,
                "the grey images are for {grey} images, but there are {image_count} views"
            ),
        }
    }
}

impl std::error::Error for DepthLayerError {}

/// Whether ranges `a` and `b` share a distance.
fn overlap(a: [f64; 2], b: [f64; 2]) -> bool {
    a[0] <= b[1] && b[0] <= a[1]
}

/// Whether every element of sorted `a` is in sorted `b`.
fn is_subset(a: &[u32], b: &[u32]) -> bool {
    a.iter().all(|x| b.binary_search(x).is_ok())
}

/// Group `candidates` into depth layers near `pixel` in `image`, read each
/// layer with the pixel's own patch, and rank them.
///
/// A candidate is usable when its range is bounded or far. The usable ones are
/// taken nearest range first (ties in the order given) and each joins the
/// last layer when its near end is within that layer's range, which it then
/// widens, or starts a new one; so the layers are nearest first and their
/// ranges do not overlap. Each candidate's support counts the other usable
/// candidates whose ranges overlap its own and whose images are not all among
/// its own, nor its among theirs: two readings that do not rest on the same
/// photographs.
///
/// With [`DepthLayerOptions::evidence`], the pixel's patch of
/// [`DepthLayerOptions::radius_px`] is read ([`read_patch_along_ray`]) in every
/// other image at [`DepthLayerOptions::samples`] distances across each layer's
/// range, evenly spaced in inverse distance, from infinity in for a layer with
/// no far end. Each image's reading at a layer is its best over those
/// distances, and that image's middle reading is taken from the same distance.
/// From those and the members come the [`LayerEvidence`], the score, the key,
/// the rank and the confidence.
///
/// `views` and `grey` hold one entry per image of the reconstruction.
pub fn depth_layers(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    image: u32,
    pixel: [f64; 2],
    candidates: &[LayerCandidate<'_>],
    options: &DepthLayerOptions,
) -> Result<DepthLayers, DepthLayerError> {
    let image_count = views.len();
    if image as usize >= image_count {
        return Err(DepthLayerError::NoSuchImage { image, image_count });
    }
    if grey.len() != image_count {
        return Err(DepthLayerError::GreyMismatch {
            grey: grey.len(),
            image_count,
        });
    }
    let images: Vec<Vec<u32>> = candidates.iter().map(LayerCandidate::images).collect();
    let support = support_counts(candidates, &images);
    let mut layers = group_layers(candidates);
    if options.evidence && !layers.is_empty() {
        let reads = layer_reads(views, grey, image, pixel, &layers, options);
        let evidence: Vec<LayerEvidence> = (0..layers.len())
            .map(|n| layer_evidence(candidates, &images, &layers[n], &reads, n))
            .collect();
        for (ranking, layer) in rank_layers(evidence, options.rank_by)
            .into_iter()
            .zip(&mut layers)
        {
            layer.ranking = Some(ranking);
        }
    }
    Ok(DepthLayers { support, layers })
}

/// Each candidate's support.
fn support_counts(candidates: &[LayerCandidate<'_>], images: &[Vec<u32>]) -> Vec<usize> {
    candidates
        .iter()
        .enumerate()
        .map(|(a, ca)| {
            if !ca.class.usable() {
                return 0;
            }
            candidates
                .iter()
                .enumerate()
                .filter(|&(b, cb)| {
                    b != a
                        && cb.class.usable()
                        && overlap(ca.range, cb.range)
                        && !is_subset(&images[a], &images[b])
                        && !is_subset(&images[b], &images[a])
                })
                .count()
        })
        .collect()
}

/// The usable candidates grouped by overlapping ranges, nearest first, without
/// their evidence.
fn group_layers(candidates: &[LayerCandidate<'_>]) -> Vec<DepthLayer> {
    let mut usable: Vec<usize> = (0..candidates.len())
        .filter(|&k| candidates[k].class.usable())
        .collect();
    // A stable sort, so candidates with the same near end keep their order.
    usable.sort_by(|&a, &b| {
        candidates[a].range[0]
            .partial_cmp(&candidates[b].range[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut layers: Vec<DepthLayer> = Vec::new();
    for k in usable {
        let c = &candidates[k];
        match layers.last_mut() {
            Some(last) if c.range[0] <= last.range[1] => {
                if c.range[1] > last.range[1] {
                    last.range[1] = c.range[1];
                }
                last.members.push(k);
            }
            _ => layers.push(DepthLayer {
                range: c.range,
                members: vec![k],
                nearest_px: 0.0,
                max_views: 0,
                ranking: None,
            }),
        }
    }
    for layer in &mut layers {
        layer.nearest_px = layer
            .members
            .iter()
            .map(|&k| candidates[k].distance_px)
            .fold(f64::INFINITY, f64::min);
        layer.max_views = layer
            .members
            .iter()
            .map(|&k| candidates[k].n_views())
            .max()
            .unwrap_or(0);
    }
    layers
}

/// The pixel's patch read at each layer, `[layer][image]` over every image but
/// the queried one in order: the whole patch's reading, the best over the
/// layer's distances, and the middle's at that distance; `-1` where unread.
struct LayerReads {
    whole: Vec<Vec<f64>>,
    middle: Vec<Vec<f64>>,
}

/// The distances across `range` the patch is read at: `samples` of them, even
/// in inverse distance from the far end in, as `numpy.linspace` spaces them.
pub(super) fn layer_distances(range: [f64; 2], samples: usize) -> Vec<f64> {
    let [near, far] = range;
    let lo = if far.is_finite() { 1.0 / far } else { 0.0 };
    let hi = if near > 0.0 { 1.0 / near } else { lo };
    let inverse: Vec<f64> = match samples {
        0 => Vec::new(),
        1 => vec![lo],
        n => {
            let step = (hi - lo) / (n - 1) as f64;
            let mut v: Vec<f64> = (0..n)
                .map(|i| {
                    if step == 0.0 {
                        lo
                    } else {
                        i as f64 * step + lo
                    }
                })
                .collect();
            v[n - 1] = hi;
            v
        }
    };
    inverse
        .into_iter()
        .map(|v| if v == 0.0 { f64::INFINITY } else { 1.0 / v })
        .collect()
}

fn layer_reads(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    image: u32,
    pixel: [f64; 2],
    layers: &[DepthLayer],
    options: &DepthLayerOptions,
) -> LayerReads {
    let others: Vec<u32> = (0..views.len() as u32).filter(|&i| i != image).collect();
    let mut distances = Vec::new();
    let mut owner = Vec::new();
    for (n, layer) in layers.iter().enumerate() {
        for t in layer_distances(layer.range, options.samples) {
            distances.push(t);
            owner.push(n);
        }
    }
    let mut whole = vec![vec![-1.0; others.len()]; layers.len()];
    let mut middle = whole.clone();
    let patch = RayPatch {
        image,
        pixel,
        radius_px: options.radius_px,
    };
    if let Some(read) = read_patch_along_ray(views, grey, &patch, &distances, &others, false) {
        for (k, &n) in owner.iter().enumerate() {
            for v in 0..others.len() {
                let w = read.whole[[k, v]];
                if w > whole[n][v] {
                    whole[n][v] = w;
                    middle[n][v] = read.middle[[k, v]];
                }
            }
        }
    }
    LayerReads { whole, middle }
}

/// The mean of the three largest of `values`, `-1` when there are none.
fn best3(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    values.truncate(3);
    if values.is_empty() {
        -1.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn layer_evidence(
    candidates: &[LayerCandidate<'_>],
    images: &[Vec<u32>],
    layer: &DepthLayer,
    reads: &LayerReads,
    n: usize,
) -> LayerEvidence {
    let members = &layer.members;
    // The members' image sets, each once.
    let mut sets: Vec<&Vec<u32>> = members.iter().map(|&k| &images[k]).collect();
    sets.sort();
    sets.dedup();
    let n_independent = sets
        .iter()
        .filter(|a| {
            !sets
                .iter()
                .any(|b| a.len() < b.len() && is_subset(a.as_slice(), b.as_slice()))
        })
        .count();
    let mut seen: Vec<u32> = sets.iter().flat_map(|s| s.iter().copied()).collect();
    seen.sort_unstable();
    seen.dedup();

    let mine = &reads.whole[n];
    let middle = &reads.middle[n];
    let readable: Vec<bool> = mine.iter().map(|&w| w > -1.0).collect();
    let mut votes = 0;
    let mut votes_all = 0;
    for v in 0..mine.len() {
        let others = (0..reads.whole.len())
            .filter(|&m| m != n)
            .map(|m| reads.whole[m][v])
            .fold(f64::NEG_INFINITY, f64::max);
        let others = if reads.whole.len() > 1 { others } else { -1.0 };
        if readable[v] && mine[v] >= VOTE_MIN_ZNCC && mine[v] >= others + VOTE_MARGIN {
            votes_all += 1;
            if middle[v] >= VOTE_MIN_ZNCC {
                votes += 1;
            }
        }
    }
    let over_readable = |f: &dyn Fn(usize) -> f64| {
        best3(
            (0..mine.len())
                .filter(|&v| readable[v])
                .map(f)
                .collect::<Vec<_>>(),
        )
    };

    let mut sources: Vec<NearbySource> = members.iter().map(|&k| candidates[k].source).collect();
    sources.sort_by_key(|s| s.name());
    sources.dedup();

    LayerEvidence {
        n_candidates: members.len(),
        n_independent,
        n_images: seen.len(),
        max_views: layer.max_views,
        max_ray_angle_deg: members
            .iter()
            .map(|&k| candidates[k].max_ray_angle_deg)
            .fold(0.0, f64::max),
        nearest_px: layer.nearest_px,
        at_pixel: members
            .iter()
            .any(|&k| candidates[k].distance_px <= AT_PIXEL_PX),
        sources,
        weight: members.iter().fold(0.0, |sum, &k| {
            let c = &candidates[k];
            sum + (1.0 + c.n_views() as f64).log2() * (-c.distance_px / WEIGHT_FALLOFF_PX).exp()
        }),
        photo: over_readable(&|v| mine[v]),
        photo_middle: over_readable(&|v| middle[v]),
        photo_both: over_readable(&|v| mine[v].min(middle[v])),
        votes,
        votes_all,
    }
}

/// The score, key, rank and confidence of each layer, from its evidence.
pub(super) fn rank_layers(evidence: Vec<LayerEvidence>, by: LayerRankBy) -> Vec<LayerRanking> {
    let mut out: Vec<LayerRanking> = evidence
        .into_iter()
        .map(|e| {
            let score = 0.5 * (e.photo + e.photo_both);
            let key = score + e.photo_middle - KEY_NEAREST * e.nearest_px.ln_1p();
            LayerRanking {
                evidence: e,
                score,
                key,
                rank: 0,
                confidence: 0.0,
            }
        })
        .collect();
    let value = |r: &LayerRanking| match by {
        LayerRankBy::Key => r.key,
        LayerRankBy::Score => r.score,
    };
    let values: Vec<f64> = out.iter().map(value).collect();
    let mut order: Vec<usize> = (0..out.len()).collect();
    // A stable sort, highest first, so ties keep the layers' order.
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, &n) in order.iter().enumerate() {
        out[n].rank = rank + 1;
    }
    for n in 0..out.len() {
        let rest = (0..values.len())
            .filter(|&m| m != n)
            .map(|m| values[m])
            .fold(f64::NEG_INFINITY, f64::max);
        let margin = if values.len() > 1 {
            values[n] - rest
        } else {
            1.0
        };
        let e = &out[n].evidence;
        let z = CONF_BIAS
            + CONF_MARGIN * margin
            + CONF_VOTES * (e.votes as f64).ln_1p()
            + CONF_SUPPORT * e.weight.max(0.0).ln_1p()
            - CONF_NEAREST * e.nearest_px.ln_1p();
        out[n].confidence = 1.0 / (1.0 + (-z).exp());
    }
    out
}
