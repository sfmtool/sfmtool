// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which other images hold the patch around a pixel, and where in them it sits.
//!
//! Given the SIFT features inside a small radius of one pixel in one image, the
//! query looks each of them up in a descriptor index, groups the hits by the
//! image they came from, and keeps the images whose correspondences agree on a
//! single affine warp. The answer is per image: the warp, how many
//! correspondences voted for it, and which ones they were.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use sfmtool_kdf_format::KdfScalar;
use sfmtool_sift_format::SiftError;

use super::distance::ForestScalar;
use super::neighbor_index::NeighborIndex;
use super::{FeatureGeometry, FeatureOrigin, KdfError, LazyKdForest};

#[cfg(test)]
mod tests;

/// How many feature IDs one pass of the origin scan resolves at a time.
///
/// The origin table is indexed by corpus feature ID, never by image, so finding
/// one image's features means reading all of them. Resolving the whole table in
/// one call would allocate an origin per corpus feature; a chunk keeps that
/// bounded while still reading each origin block exactly once.
const ORIGIN_SCAN_CHUNK: usize = 1 << 16;

/// How the affine a candidate image is reported with is fitted to its consensus.
///
/// Three points are how a model is *found* -- the sample size is what RANSAC's
/// cost is exponential in -- and a poor way to report one, because a model
/// passing exactly through three keypoints carries all three keypoints'
/// localisation noise. Once the consensus is chosen the whole of it can be
/// fitted, at the cost of one 3x3 solve per reported image, and which images are
/// found does not change.
///
/// Off, unweighted and weighted are three behaviours rather than one number:
/// encoding the first two as a sigma of zero and of infinity would put a
/// correctness condition on a float comparison. [`AffineRefit::None`] is kept
/// because the two forests' parity test and anyone diagnosing RANSAC itself want
/// the model as it was drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AffineRefit {
    /// Report the best three-point model as drawn.
    None,
    /// Least squares over the consensus, every inlier weighted alike.
    LeastSquares,
    /// Least squares with a Gaussian weight in the distance of an inlier's
    /// constellation position from [`Constellation::center`], the standard
    /// deviation being `sigma` times the constellation's radius about that
    /// centre -- the largest distance from it to any constellation position.
    ///
    /// The radius is measured rather than taken from the caller because the
    /// radius a caller *asked* for can be much larger than the disc its features
    /// actually fill, and a sigma proportional to an empty rim would flatten the
    /// weights towards [`AffineRefit::LeastSquares`] without anyone having
    /// chosen that. A constellation with no centre is fitted as
    /// `LeastSquares`, which is the honest reading of "no point matters more
    /// than another".
    CenterWeighted {
        /// Weight scale as a fraction of the constellation radius. At 0.5 the
        /// rim still weighs `exp(-2)`, about an eighth, so it constrains the
        /// linear part; at 0.25 it is all but discarded, which is the best warp
        /// at the centre and the worst over the disc.
        sigma: f64,
    },
}

/// Tunables for [`constellation_query`].
///
/// `k` is much larger than a descriptor matcher's. Most of a constellation
/// feature's nearest neighbours belong to images that do not contain the patch
/// at all, so a small `k` can leave the right image with no candidates to fit;
/// the model only has to survive a consensus test afterwards, so the cost of
/// carrying wrong candidates is far lower than the cost of missing the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstellationParams {
    /// Neighbours retrieved per constellation feature.
    pub k: usize,
    /// Per-query budget of distance evaluations in the forest traversal.
    pub max_leaf_checks: usize,
    /// Reprojection distance, in the candidate image's pixels, within which a
    /// correspondence agrees with a model.
    pub threshold_px: f64,
    /// Three-point samples drawn per candidate image.
    pub iterations: usize,
    /// Fewest correspondences an image needs before it is fitted at all.
    pub min_correspondences: usize,
    /// Keep at most one hit of each constellation feature in each candidate
    /// image, the nearest one. A feature's `k` neighbours may hold several
    /// features of one image, and only one of them can be that feature's match
    /// there; the rest are extra correspondences RANSAC has to outvote. On by
    /// default, since the hits it drops cannot be right and a small corpus
    /// produces many of them. A `same_image_ratio` below 1.0 implies this, and
    /// adds a test to it.
    pub one_hit_per_image: bool,
    /// Lowe's ratio test inside one (constellation feature, candidate image)
    /// cell. Below 1.0 it is on: the cell collapses to the feature's nearest
    /// hit in that image, and that hit survives only when its distance is less
    /// than `same_image_ratio` times the distance of the runner-up **in the
    /// same image**, so a feature that matches one spot of an image no better
    /// than it matches another contributes nothing there. A cell holding a
    /// single hit has no runner-up and is kept. 1.0 and above is off, which is
    /// the default: a cell keeps its nearest hit unconditionally, that being
    /// what `one_hit_per_image` does on its own.
    ///
    /// The ratio is a ratio of Euclidean distances. The forest reports squared
    /// ones, so it is applied squared.
    pub same_image_ratio: f32,
    /// Fewest inliers an image needs to be reported.
    pub min_inliers: usize,
    /// Widest change of scale a model may claim, as the geometric mean
    /// `sqrt(|det|)` of its 2x2 linear part. A model scaling the patch by more
    /// than this, or by less than its reciprocal, is refused unfitted. The
    /// default is permissive on purpose: two frames of one capture can
    /// legitimately differ by two or three times in scale, so it only refuses
    /// the absurd, and a caller who knows its baselines tightens it.
    pub max_scale: f64,
    /// How the reported affine is fitted to the consensus RANSAC chose.
    ///
    /// The default fits it to the whole consensus by least squares, weighted
    /// towards the patch centre, because that is where the caller applies the
    /// warp. It moves no image into or out of the answer and leaves
    /// [`ConstellationMatch::inliers`] alone.
    pub refit: AffineRefit,
    /// Base RNG seed; each candidate image draws from `seed + image_index`.
    pub seed: u64,
}

impl ConstellationParams {
    /// The defaults, as a constant, so the Python bindings can name a single
    /// field in a keyword default instead of repeating the number.
    pub const DEFAULT: Self = Self {
        k: 32,
        max_leaf_checks: 512,
        threshold_px: 8.0,
        iterations: 200,
        min_correspondences: 3,
        one_hit_per_image: true,
        same_image_ratio: 1.0,
        min_inliers: 8,
        max_scale: 4.0,
        refit: AffineRefit::CenterWeighted { sigma: 0.5 },
        seed: 0,
    };
}

impl Default for ConstellationParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Where a constellation's descriptors come from.
///
/// A caller that computed the descriptors holds vectors; a caller whose query
/// image is itself in the corpus holds IDs and would otherwise have to reopen a
/// `.sift` file to turn them back into vectors. Both reach the same search, and
/// the ID form reads the vectors through
/// [`NeighborIndex::resolve_descriptors`].
#[derive(Clone, Copy, Debug)]
pub enum ConstellationDescriptors<'a, S> {
    /// Flat `n * dim` row-major vectors, one row per constellation feature.
    Vectors(&'a [S]),
    /// Corpus feature IDs, one per constellation feature.
    FeatureIds(&'a [u32]),
}

/// The features inside the patch, in the query image.
#[derive(Clone, Copy, Debug)]
pub struct Constellation<'a, S> {
    /// Each feature's position in the query image, in that image's pixels.
    pub positions: &'a [[f32; 2]],
    /// The descriptors for those same features, in the same order.
    pub descriptors: ConstellationDescriptors<'a, S>,
    /// The query image's index in the corpus, when the corpus indexes it.
    /// Candidates from that image are dropped: an image matching itself is not
    /// an answer to "where else is this patch".
    pub image_index: Option<u32>,
    /// The pixel the patch is about, when there is one.
    ///
    /// It is a fact about the patch, like [`Self::positions`] are, and it is
    /// what [`AffineRefit::CenterWeighted`] weighs distances from: the caller
    /// applies the warp here, so this is where it should be most accurate. The
    /// two `*_at_pixel` / `*_from_keypoints` entry points pass their own centre
    /// through. A caller that assembled its positions some other way may have
    /// none, and `None` under `CenterWeighted` fits as
    /// [`AffineRefit::LeastSquares`].
    pub center: Option<[f32; 2]>,
}

/// One inlier correspondence of a candidate image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstellationCorrespondence {
    /// Position of this feature within the query constellation.
    pub query_index: u32,
    /// The corpus feature it matched.
    pub feature_id: u32,
    /// That feature's keypoint center in the candidate image.
    pub position: [f32; 2],
    /// That feature's 2x2 affine shape, as the `.sift` file stores it.
    pub affine_shape: [[f32; 2]; 2],
}

/// One candidate image and the warp that places the query patch in it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstellationMatch {
    /// The candidate's index in the corpus image table.
    pub image_index: u32,
    /// Row-major 2x3 affine taking query-image pixels to this image's pixels:
    /// `x' = affine[0][0] * x + affine[0][1] * y + affine[0][2]`, and likewise
    /// `y'` from `affine[1]`.
    ///
    /// It is [`ConstellationParams::refit`]'s fit to [`Self::inliers`], not the
    /// three-point model that selected them, so an inlier may sit a little
    /// outside `threshold_px` of this warp.
    pub affine: [[f64; 3]; 2],
    /// Correspondences agreeing within the pixel threshold with the three-point
    /// model that won: the consensus [`Self::affine`] was fitted to, and not a
    /// re-selection under it.
    pub inliers: usize,
    /// Correspondences this image had before the fit.
    pub correspondences: usize,
    /// The inliers themselves, in constellation order, enough to seed a patch
    /// cluster without a second lookup.
    pub inlier_correspondences: Vec<ConstellationCorrespondence>,
}

/// Where a corpus feature came from and where it sits in its own image.
///
/// The file-backed forest answers both from the `.kdf` itself; a forest loaded
/// eagerly answers neither, because loading rebuilds the trees and the corpus
/// and keeps no source tables, so a caller of the eager path supplies them as
/// [`ResidentSources`]. Making this an argument rather than something the query
/// digs out of the index is what keeps that gap visible.
pub trait FeatureSources {
    /// Corpus features these sources describe. IDs run `0..feature_count`.
    fn feature_count(&self) -> usize;

    /// Origins in request order, repeats included.
    fn resolve_origins(&self, feature_ids: &[u32]) -> Result<Vec<FeatureOrigin>, KdfError>;

    /// Keypoint centers and affine shapes in request order, repeats included.
    fn resolve_feature_geometry(
        &self,
        feature_ids: &[u32],
    ) -> Result<Vec<FeatureGeometry>, KdfError>;

    /// Corpus feature IDs of one image's features, keyed by each feature's row
    /// in that image's `.sift` file.
    ///
    /// This is the reverse of [`resolve_origins`](Self::resolve_origins), and
    /// it costs a pass over the whole origin table because nothing indexes it
    /// by image: origins are stored in corpus feature-ID order, and a subset
    /// corpus may hold any part of any image. The pass is chunked and reads
    /// each origin block once.
    fn image_feature_ids(&self, image_index: u32) -> Result<HashMap<u32, u32>, KdfError> {
        let total = self.feature_count();
        let mut map = HashMap::new();
        let mut ids: Vec<u32> = Vec::with_capacity(ORIGIN_SCAN_CHUNK.min(total));
        let mut start = 0;
        while start < total {
            let end = (start + ORIGIN_SCAN_CHUNK).min(total);
            ids.clear();
            ids.extend((start..end).map(|id| id as u32));
            for (offset, origin) in self.resolve_origins(&ids)?.into_iter().enumerate() {
                if origin.image_index == image_index {
                    map.insert(origin.image_feature_index, (start + offset) as u32);
                }
            }
            start = end;
        }
        Ok(map)
    }
}

impl<S: ForestScalar + KdfScalar> FeatureSources for LazyKdForest<S> {
    fn feature_count(&self) -> usize {
        self.len()
    }

    fn resolve_origins(&self, feature_ids: &[u32]) -> Result<Vec<FeatureOrigin>, KdfError> {
        Self::resolve_origins(self, feature_ids)?.ok_or_else(no_sources)
    }

    fn resolve_feature_geometry(
        &self,
        feature_ids: &[u32],
    ) -> Result<Vec<FeatureGeometry>, KdfError> {
        Self::resolve_feature_geometry(self, feature_ids)?.ok_or_else(no_sources)
    }
}

/// Source tables held in memory, for a corpus whose origins and geometry the
/// caller already has: a forest built from `.sift` files in this process, or one
/// reloaded from a `.kdf`, which keeps neither.
#[derive(Clone, Debug)]
pub struct ResidentSources {
    origins: Vec<FeatureOrigin>,
    geometry: Vec<FeatureGeometry>,
}

impl ResidentSources {
    /// Both tables are in corpus feature-ID order and must be the same length.
    pub fn new(
        origins: Vec<FeatureOrigin>,
        geometry: Vec<FeatureGeometry>,
    ) -> Result<Self, KdfError> {
        if origins.len() != geometry.len() {
            return Err(KdfError::ShapeMismatch(format!(
                "{} origins against {} geometry rows",
                origins.len(),
                geometry.len()
            )));
        }
        Ok(Self { origins, geometry })
    }
}

impl FeatureSources for ResidentSources {
    fn feature_count(&self) -> usize {
        self.origins.len()
    }

    fn resolve_origins(&self, feature_ids: &[u32]) -> Result<Vec<FeatureOrigin>, KdfError> {
        feature_ids
            .iter()
            .map(|&id| {
                self.origins
                    .get(id as usize)
                    .copied()
                    .ok_or_else(|| out_of_range(id))
            })
            .collect()
    }

    fn resolve_feature_geometry(
        &self,
        feature_ids: &[u32],
    ) -> Result<Vec<FeatureGeometry>, KdfError> {
        feature_ids
            .iter()
            .map(|&id| {
                self.geometry
                    .get(id as usize)
                    .copied()
                    .ok_or_else(|| out_of_range(id))
            })
            .collect()
    }
}

fn no_sources() -> KdfError {
    KdfError::InvalidQuery(
        "the corpus carries no SIFT sources, so its features have no image or geometry".into(),
    )
}

fn out_of_range(id: u32) -> KdfError {
    KdfError::InvalidQuery(format!("feature ID {id} is out of range"))
}

/// Rank the images that contain the query constellation.
///
/// Each constellation feature is looked up in `index`, every hit is attributed
/// to its source image through `sources`, and each image with enough
/// correspondences is fitted by three-point affine RANSAC. Images reaching
/// `min_inliers` are returned, most inliers first, ties in ascending image
/// index. The query's own image, when it names one, is never a candidate.
///
/// One feature's neighbour list can hold several features of one image, and
/// [`ConstellationParams::one_hit_per_image`] and
/// [`ConstellationParams::same_image_ratio`] cut each such group down to its
/// nearest member, or drop it outright when that member is not clearly nearer.
/// The first is on by default and the second is off, so by default a cell keeps
/// its nearest hit whatever its runner-up looks like.
///
/// Three points find each model and the consensus reports it: the affine handed
/// back is [`ConstellationParams::refit`]'s fit to the inliers of the winning
/// three-point model, while [`ConstellationMatch::inliers`] and
/// [`ConstellationMatch::inlier_correspondences`] remain that model's own
/// consensus, which is the set the fit was computed from.
///
/// Determinism is a requirement rather than a nicety here, because this is the
/// function a `.kdf`'s two access paths are compared through: given the same
/// neighbours, the resident and file-backed forests must produce identical
/// warps and identical inlier sets. So candidate images are fitted in ascending
/// index order, and **each one seeds its own generator from
/// `params.seed + image_index`** rather than drawing from one generator
/// threaded through the run. A shared generator would make each image's samples
/// depend on how many images preceded it, so adding, dropping or reordering a
/// candidate would silently change every later fit, and two paths handed
/// identical neighbours could still disagree. That disagreement would read as
/// an index bug.
pub fn constellation_query<S, I, F>(
    index: &I,
    sources: &F,
    query: &Constellation<'_, S>,
    params: &ConstellationParams,
) -> Result<Vec<ConstellationMatch>, KdfError>
where
    S: ForestScalar,
    I: NeighborIndex<S> + ?Sized,
    F: FeatureSources + ?Sized,
{
    let n = query.positions.len();
    let dim = index.dim();
    let owned;
    let descriptors: &[S] = match query.descriptors {
        ConstellationDescriptors::Vectors(vectors) => {
            if vectors.len() != n * dim {
                return Err(KdfError::ShapeMismatch(format!(
                    "{} descriptor values for {n} positions of dimension {dim}",
                    vectors.len()
                )));
            }
            vectors
        }
        ConstellationDescriptors::FeatureIds(ids) => {
            if ids.len() != n {
                return Err(KdfError::ShapeMismatch(format!(
                    "{} feature IDs for {n} positions",
                    ids.len()
                )));
            }
            owned = index.resolve_descriptors(ids)?;
            &owned
        }
    };
    if n == 0 || params.k == 0 {
        return Ok(Vec::new());
    }

    let (neighbors, distances) = index.search_batch_with_distances(
        descriptors,
        n,
        params.k,
        params.max_leaf_checks,
        None,
    )?;

    // Every hit, in encounter order, so the origin and geometry lookups below
    // ask for IDs in the order the corpus is most likely to hold them together.
    let mut hit_ids = Vec::with_capacity(neighbors.len());
    let mut hit_query = Vec::with_capacity(neighbors.len());
    let mut hit_dist = Vec::with_capacity(neighbors.len());
    for (slot, (&id, &dist)) in neighbors.iter().zip(distances.iter()).enumerate() {
        if id == u32::MAX || !dist.is_finite() {
            continue;
        }
        hit_ids.push(id);
        hit_query.push((slot / params.k) as u32);
        hit_dist.push(dist);
    }
    if hit_ids.is_empty() {
        return Ok(Vec::new());
    }

    let origins = sources.resolve_origins(&hit_ids)?;
    let mut kept_ids = Vec::with_capacity(hit_ids.len());
    let mut kept: Vec<(u32, u32)> = Vec::with_capacity(hit_ids.len());
    let mut kept_dist = Vec::with_capacity(hit_ids.len());
    for (slot, origin) in origins.iter().enumerate() {
        if query.image_index == Some(origin.image_index) {
            continue;
        }
        kept_ids.push(hit_ids[slot]);
        kept.push((origin.image_index, hit_query[slot]));
        kept_dist.push(hit_dist[slot]);
    }
    // Before the grouping rather than after it, because the cell a hit belongs
    // to is already known here and a dropped hit then costs no geometry read.
    if let Some(keep) = collapse_cells(&kept, &kept_dist, params) {
        let mut ids = Vec::with_capacity(kept_ids.len());
        let mut cells = Vec::with_capacity(kept.len());
        for (slot, &survives) in keep.iter().enumerate() {
            if survives {
                ids.push(kept_ids[slot]);
                cells.push(kept[slot]);
            }
        }
        kept_ids = ids;
        kept = cells;
    }
    if kept_ids.is_empty() {
        return Ok(Vec::new());
    }
    let geometry = sources.resolve_feature_geometry(&kept_ids)?;

    // Group before fitting: a model needs three correspondences, so an image
    // with fewer cannot win and is never sampled.
    let mut by_image: BTreeMap<u32, Vec<ConstellationCorrespondence>> = BTreeMap::new();
    for (slot, &(image, query_index)) in kept.iter().enumerate() {
        let row = geometry[slot];
        by_image
            .entry(image)
            .or_default()
            .push(ConstellationCorrespondence {
                query_index,
                feature_id: kept_ids[slot],
                position: row[0],
                affine_shape: [row[1], row[2]],
            });
    }

    // The centre and the disc it sits in are the refit's weighting, and both are
    // facts about the constellation rather than about a candidate, so they are
    // measured once. The radius is the largest distance from the centre to any
    // constellation position, including features whose hits all fell away: it
    // describes the disc the patch was taken from, not the consensus.
    let center = query.center.map(|c| [c[0] as f64, c[1] as f64]);
    let radius = center.map_or(0.0, |c| {
        query
            .positions
            .iter()
            .map(|p| (p[0] as f64 - c[0]).hypot(p[1] as f64 - c[1]))
            .fold(0.0f64, f64::max)
    });

    let mut matches = Vec::new();
    for (image, pairs) in by_image {
        if pairs.len() < params.min_correspondences.max(3) {
            continue;
        }
        let src: Vec<[f64; 2]> = pairs
            .iter()
            .map(|c| {
                let p = query.positions[c.query_index as usize];
                [p[0] as f64, p[1] as f64]
            })
            .collect();
        let dst: Vec<[f64; 2]> = pairs
            .iter()
            .map(|c| [c.position[0] as f64, c.position[1] as f64])
            .collect();
        let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(image as u64));
        let Some((affine, inliers)) = fit_affine_ransac(&src, &dst, params, &mut rng) else {
            continue;
        };
        if inliers.len() < params.min_inliers {
            continue;
        }
        // One fit per reported image, over the consensus already chosen. A
        // refusal reports the three-point model: the consensus that admitted
        // this candidate stands whatever the fit to it comes out as.
        let reported = refit_affine(&src, &dst, &inliers, center, radius, params).unwrap_or(affine);
        matches.push(ConstellationMatch {
            image_index: image,
            affine: reported,
            inliers: inliers.len(),
            correspondences: pairs.len(),
            inlier_correspondences: inliers.into_iter().map(|i| pairs[i]).collect(),
        });
    }

    // A stable sort over an ascending-index list leaves equal inlier counts in
    // image order, so a tie is broken by identity rather than by hash order.
    matches.sort_by_key(|m| std::cmp::Reverse(m.inliers));
    Ok(matches)
}

/// Which hits survive the per-cell collapse, or `None` when it is off.
///
/// A cell is one (constellation feature, candidate image) pair, and `cells`
/// names each hit's cell as `(image, query_index)`. Only the nearest hit of a
/// cell can be that feature's match in that image, so the cell keeps that one
/// and drops the rest; with [`ConstellationParams::same_image_ratio`] below 1.0
/// it keeps the nearest only when it is nearer than the runner-up **of the same
/// cell** by that factor, and a cell with one hit has no runner-up and is kept.
///
/// `distances` are the **squared** Euclidean distances the forest reports, so
/// the ratio, which is a ratio of Euclidean distances, is squared to meet them.
/// The comparison widens to `f64` so a squared `f32` ratio cannot round a
/// borderline cell the wrong way.
///
/// The nearest hit is the first slot holding the smallest distance, so two hits
/// at exactly one distance resolve to the earlier of them, which is the nearer
/// neighbour in the list the forest returned. The mask is in slot order, and
/// nothing about it depends on how the cells hash, so the survivors keep the
/// encounter order the rest of the query relies on.
fn collapse_cells(
    cells: &[(u32, u32)],
    distances: &[f32],
    params: &ConstellationParams,
) -> Option<Vec<bool>> {
    let ratio = (params.same_image_ratio < 1.0).then_some(params.same_image_ratio);
    if !params.one_hit_per_image && ratio.is_none() {
        return None;
    }
    // Cell -> the slot of its nearest hit, that distance, and the runner-up's.
    let mut best: HashMap<(u32, u32), (usize, f32, f32)> = HashMap::new();
    for (slot, (&cell, &distance)) in cells.iter().zip(distances).enumerate() {
        let entry = best.entry(cell).or_insert((slot, distance, f32::INFINITY));
        if distance < entry.1 {
            *entry = (slot, distance, entry.1);
        } else if slot != entry.0 && distance < entry.2 {
            entry.2 = distance;
        }
    }
    let limit = ratio.map(|r| (r as f64) * (r as f64));
    Some(
        cells
            .iter()
            .enumerate()
            .map(|(slot, cell)| {
                let &(nearest, near, runner_up) = &best[cell];
                if slot != nearest {
                    return false;
                }
                match limit {
                    Some(limit) if runner_up.is_finite() => {
                        (near as f64) < limit * (runner_up as f64)
                    }
                    _ => true,
                }
            })
            .collect(),
    )
}

/// Best affine fit and its inlier positions within `src`/`dst`.
///
/// Three correspondences determine an affine transform, so each trial draws
/// three and scores the rest by reprojection distance. Affine rather than a
/// homography because a small patch seen from a nearby viewpoint is well
/// approximated by one, and a three-point model reaches a clean sample in far
/// fewer trials than a four-point one.
fn fit_affine_ransac(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    params: &ConstellationParams,
    rng: &mut StdRng,
) -> Option<([[f64; 3]; 2], Vec<usize>)> {
    let n = src.len();
    if n < 3 {
        return None;
    }
    let threshold_sq = params.threshold_px * params.threshold_px;
    let mut best: Option<([[f64; 3]; 2], usize)> = None;
    for _ in 0..params.iterations {
        let pick = sample(rng, n, 3).into_vec();
        let Some(model) = solve_affine(
            [src[pick[0]], src[pick[1]], src[pick[2]]],
            [dst[pick[0]], dst[pick[1]], dst[pick[2]]],
            params.max_scale,
        ) else {
            continue;
        };
        let count = (0..n)
            .filter(|&i| residual_sq(&model, src[i], dst[i]) <= threshold_sq)
            .count();
        if best.is_none_or(|(_, previous)| count > previous) {
            best = Some((model, count));
        }
    }
    let (model, _) = best?;
    let inliers: Vec<usize> = (0..n)
        .filter(|&i| residual_sq(&model, src[i], dst[i]) <= threshold_sq)
        .collect();
    Some((model, inliers))
}

/// The affine transform through three correspondences, or `None` when it is not
/// a physically possible warp of the patch.
///
/// Three ways that happens, and all of them have to be refused. Collinear or
/// coincident *source* points determine no transform at all. Collinear or
/// coincident *destination* points determine one that collapses the whole patch
/// onto a line or a point, and that one is worse than useless: a few corpus
/// features hit repeatedly by different constellation features give every one
/// of those correspondences the same destination, so a collapsing model scores
/// every pair sharing a keypoint as an inlier and manufactures a consensus out
/// of an image that contains nothing. The third is a model no pair of cameras
/// could produce: the determinant of the 2x2 linear part is the signed area
/// ratio, so a *negative* one mirrors the surface, which two views of one piece
/// of surface cannot do, and `sqrt(|det|)` far from unity blows the patch up or
/// shrinks it past anything a change of viewpoint explains.
///
/// All three are read off the same 2x2 determinant, and all three are checked
/// here rather than at scoring time because a model of any of these shapes is
/// never worth scoring: a refused one is skipped, so it can neither win a trial
/// nor be reported.
fn solve_affine(src: [[f64; 2]; 3], dst: [[f64; 2]; 3], max_scale: f64) -> Option<[[f64; 3]; 2]> {
    // The three homogeneous source rows, whose determinant is twice the signed
    // area of their triangle: the degeneracy test and the inverse share it.
    let [a, b, c] = src;
    let det = a[0] * (b[1] - c[1]) - a[1] * (b[0] - c[0]) + (b[0] * c[1] - c[0] * b[1]);
    if det.abs() < 1e-9 {
        return None;
    }
    // Adjugate of [[ax, ay, 1], [bx, by, 1], [cx, cy, 1]], transposed in place.
    let inverse = [
        [b[1] - c[1], c[1] - a[1], a[1] - b[1]],
        [c[0] - b[0], a[0] - c[0], b[0] - a[0]],
        [
            b[0] * c[1] - c[0] * b[1],
            c[0] * a[1] - a[0] * c[1],
            a[0] * b[1] - b[0] * a[1],
        ],
    ];
    let mut affine = [[0.0f64; 3]; 2];
    for (axis, row) in affine.iter_mut().enumerate() {
        for (coefficient, weights) in row.iter_mut().zip(inverse.iter()) {
            *coefficient =
                (weights[0] * dst[0][axis] + weights[1] * dst[1][axis] + weights[2] * dst[2][axis])
                    / det;
        }
    }
    plausible_warp(&affine, max_scale).then_some(affine)
}

/// Whether a 2x3 model is a warp of a patch two cameras could produce.
///
/// One test with three jobs, read off the determinant of the 2x2 linear part:
/// not finite or near zero is a collapse of the patch onto a line or a point,
/// below zero is a reflection, and the square root is the geometric-mean scale,
/// which `max_scale` bounds either side of unity. It is applied to a three-point
/// model before it is ever scored and to a refitted one before it is reported,
/// so neither can be one of these.
fn plausible_warp(affine: &[[f64; 3]; 2], max_scale: f64) -> bool {
    let linear = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
    if !linear.is_finite() || linear < 1e-9 {
        return false;
    }
    let scale = linear.sqrt();
    scale <= max_scale && scale >= 1.0 / max_scale
}

/// The affine [`ConstellationParams::refit`] asks for over one candidate's
/// consensus, or `None` to report the three-point model as drawn.
///
/// `inliers` indexes `src` and `dst`, and is the consensus of the model RANSAC
/// chose. The fit minimises `sum w_i |A p_i + t - q_i|^2`, whose two rows share
/// one 3x3 normal matrix `sum w_i [p_i; 1][p_i; 1]^T` and differ only in the
/// right-hand side, so it is one solve reused twice.
///
/// Positions are taken relative to `center`, or to the inliers' centroid when
/// there is none, before the sums are formed, and the translation is carried
/// back afterwards: pixel coordinates reach the thousands and squaring them
/// uncentred spends digits the solve then needs. That also makes the normal
/// matrix's determinant meaningful against its own trace, which is how a
/// singular one -- inliers collinear in the query image -- is detected without
/// an absolute threshold in units of pixels to the fourth power.
///
/// `None` comes back for [`AffineRefit::None`], for a consensus too small or too
/// degenerate to fit, and for a fit that fails [`plausible_warp`]. Every one of
/// them means "report the three-point model"; none of them drops the candidate.
fn refit_affine(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    inliers: &[usize],
    center: Option<[f64; 2]>,
    radius: f64,
    params: &ConstellationParams,
) -> Option<[[f64; 3]; 2]> {
    // A centre-weighted fit with no centre to weigh distances from, or a disc of
    // no extent to scale them by, is a flat one: no point of it matters more
    // than another, which is exactly least squares.
    let weight_scale = match params.refit {
        AffineRefit::None => return None,
        AffineRefit::LeastSquares => None,
        AffineRefit::CenterWeighted { sigma } => match center {
            Some(_) if sigma > 0.0 && sigma.is_finite() && radius > 0.0 => Some(sigma * radius),
            _ => None,
        },
    };
    if inliers.len() < 3 {
        return None;
    }
    let origin = center.unwrap_or_else(|| {
        let n = inliers.len() as f64;
        let sum = inliers.iter().fold([0.0f64; 2], |acc, &i| {
            [acc[0] + src[i][0], acc[1] + src[i][1]]
        });
        [sum[0] / n, sum[1] / n]
    });

    // The normal matrix is symmetric, so five of its nine entries are the other
    // four; `rhs[axis]` is that axis's right-hand side.
    let (mut xx, mut xy, mut yy, mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut rhs = [[0.0f64; 3]; 2];
    for &i in inliers {
        let u = src[i][0] - origin[0];
        let v = src[i][1] - origin[1];
        let w = match weight_scale {
            Some(scale) => {
                let d = u.hypot(v) / scale;
                (-0.5 * d * d).exp()
            }
            None => 1.0,
        };
        xx += w * u * u;
        xy += w * u * v;
        yy += w * v * v;
        sx += w * u;
        sy += w * v;
        sw += w;
        for (axis, row) in rhs.iter_mut().enumerate() {
            let q = dst[i][axis];
            row[0] += w * u * q;
            row[1] += w * v * q;
            row[2] += w * q;
        }
    }

    // Cofactors of the symmetric [[xx, xy, sx], [xy, yy, sy], [sx, sy, sw]].
    let c00 = yy * sw - sy * sy;
    let c01 = sy * sx - xy * sw;
    let c02 = xy * sy - yy * sx;
    let determinant = xx * c00 + xy * c01 + sx * c02;
    // The matrix is positive semi-definite, so its determinant is at most the
    // cube of a third of its trace and vanishes exactly when the weighted
    // positions are collinear. Comparing the two is a conditioning test that
    // carries no unit and no assumption about how large a patch is.
    let trace = xx + yy + sw;
    if !determinant.is_finite() || determinant <= 1e-12 * trace * trace * trace {
        return None;
    }
    let inverse = [
        [c00, c01, c02],
        [c01, xx * sw - sx * sx, xy * sx - xx * sy],
        [c02, xy * sx - xx * sy, xx * yy - xy * xy],
    ];

    let mut affine = [[0.0f64; 3]; 2];
    for (row, b) in affine.iter_mut().zip(rhs) {
        for (coefficient, column) in row.iter_mut().zip(inverse.iter()) {
            *coefficient = (column[0] * b[0] + column[1] * b[1] + column[2] * b[2]) / determinant;
        }
        // Back out of the centred frame: `A (p - origin) + t` is `A p` plus a
        // translation the origin has been folded into.
        row[2] -= row[0] * origin[0] + row[1] * origin[1];
    }
    if !affine.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    plausible_warp(&affine, params.max_scale).then_some(affine)
}

fn residual_sq(model: &[[f64; 3]; 2], src: [f64; 2], dst: [f64; 2]) -> f64 {
    let x = model[0][0] * src[0] + model[0][1] * src[1] + model[0][2] - dst[0];
    let y = model[1][0] * src[0] + model[1][1] * src[1] + model[1][2] - dst[1];
    x * x + y * y
}

/// One image's keypoints, without its descriptors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageKeypoints {
    /// Keypoint centers in image pixels.
    pub positions: Vec<[f32; 2]>,
    /// The matching 2x2 affine shapes, one per position.
    pub affine_shapes: Vec<[[f32; 2]; 2]>,
}

impl ImageKeypoints {
    /// Read every keypoint of a `.sift` file, and none of its descriptors.
    pub fn read(sift_path: &Path) -> Result<Self, KdfError> {
        let (positions, affine_shapes) =
            sfmtool_sift_format::read_sift_keypoints(sift_path, usize::MAX)
                .map_err(|e| sift_error(sift_path, e))?;
        Ok(Self {
            positions,
            affine_shapes,
        })
    }

    /// Number of keypoints.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the image has no keypoints.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Rows within `radius` pixels of `center`, in ascending row order.
    pub fn within(&self, center: [f32; 2], radius: f32) -> Vec<u32> {
        let limit = (radius as f64) * (radius as f64);
        self.positions
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let dx = p[0] as f64 - center[0] as f64;
                let dy = p[1] as f64 - center[1] as f64;
                dx * dx + dy * dy <= limit
            })
            .map(|(row, _)| row as u32)
            .collect()
    }
}

/// The image a patch is taken from.
#[derive(Clone, Copy, Debug)]
pub struct QueryImage<'a> {
    /// Its `.sift` file. Read for the keypoints when `keypoints` is `None`, and
    /// for descriptors only when the corpus does not index this image.
    pub sift_path: &'a Path,
    /// Its keypoints, when the caller has already read them.
    pub keypoints: Option<&'a ImageKeypoints>,
    /// Its index in the corpus image table, when the corpus indexes it.
    pub image_index: Option<u32>,
}

/// A patch's constellation and what the index made of it.
#[derive(Clone, Debug)]
pub struct PatchConstellation {
    /// The query image's `.sift` rows that made up the constellation, in the
    /// order the correspondences' `query_index` refers to.
    pub feature_rows: Vec<u32>,
    /// Their corpus feature IDs, when the query image is indexed; empty when it
    /// is not.
    pub feature_ids: Vec<u32>,
    /// Candidate images, most inliers first.
    pub matches: Vec<ConstellationMatch>,
}

/// The radius that holds about `target` keypoints of one image.
///
/// `sqrt(target * A / (pi * K))` for image area `A` and `K` keypoints in the
/// image: a disc of that radius is `target / K` of the frame, so a uniform
/// scattering of `K` keypoints leaves `target` of them inside it.
///
/// Fifty is the size to ask for. Across five captures the share of found images
/// whose warp places the ground truth's own correspondences within 3 px is
/// 0.76 / 0.75 / 0.65 / 0.89 / 0.33 at fifty features against 0.54 / 0.33 /
/// 0.28 / 0.37 / 0.06 at two hundred and 0.23 / 0.06 / 0.04 / 0.07 / 0.01 at
/// eight hundred, while image recall climbs only 0.03 to 0.40 over that whole
/// range, because the affine is the first-order approximation of a homography
/// about the patch centre and the term it drops grows with the patch.
///
/// Keypoints cluster where there is texture and a patch is usually centred on
/// one, so the radius measured at fifty features ran 70 to 100% of what this
/// predicts; it is a starting point, not a count. An image with no keypoints
/// has no such radius, and the answer is then zero.
pub fn radius_for_feature_count(
    image_width: u32,
    image_height: u32,
    keypoint_count: usize,
    target: usize,
) -> f32 {
    if keypoint_count == 0 {
        return 0.0;
    }
    let area = image_width as f64 * image_height as f64;
    let radius_sq = (target as f64 * area) / (std::f64::consts::PI * keypoint_count as f64);
    radius_sq.sqrt() as f32
}

/// [`constellation_query`] for a pixel and a radius in one image.
///
/// The constellation is taken from the image's **`.sift` file**, not from the
/// corpus: a `.kdf` stores geometry in corpus storage order, so one image's
/// keypoints are scattered across every block and selecting a radius out of
/// them would touch most of the file, while the `.sift` file holds exactly that
/// image's keypoints in one entry. So this reads the keypoints there, selects
/// the few inside the radius, and only then fetches those few descriptors: from
/// the corpus by feature ID when the image is indexed, and from the `.sift`
/// file's descriptor entry when it is not. In the indexed case the descriptor
/// payload of the `.sift` file is never decompressed.
///
/// Features inside the radius that the corpus does not index are dropped from
/// the constellation, so an index built over a subset stays usable.
///
/// `center` selects the constellation and is then carried inside it, so the
/// reported warp is fitted towards the pixel that was asked about, as
/// [`ConstellationParams::refit`] says.
pub fn constellation_at_pixel<I, F>(
    index: &I,
    sources: &F,
    image: &QueryImage<'_>,
    center: [f32; 2],
    radius: f32,
    params: &ConstellationParams,
) -> Result<PatchConstellation, KdfError>
where
    I: NeighborIndex<u8> + ?Sized,
    F: FeatureSources + ?Sized,
{
    let read;
    let keypoints = match image.keypoints {
        Some(keypoints) => keypoints,
        None => {
            read = ImageKeypoints::read(image.sift_path)?;
            &read
        }
    };
    if let Some(indexed) = image.image_index {
        return constellation_from_keypoints(
            index, sources, keypoints, indexed, center, radius, params,
        );
    }

    let rows = keypoints.within(center, radius);
    let descriptors = read_sift_rows(image.sift_path, &rows)?;
    let positions: Vec<[f32; 2]> = rows
        .iter()
        .map(|&row| keypoints.positions[row as usize])
        .collect();
    let query = Constellation {
        positions: &positions,
        descriptors: ConstellationDescriptors::Vectors(&descriptors),
        image_index: None,
        center: Some(center),
    };
    let matches = constellation_query(index, sources, &query, params)?;
    Ok(PatchConstellation {
        feature_rows: rows,
        feature_ids: Vec::new(),
        matches,
    })
}

/// [`constellation_query`] for a pixel and a radius in an image the corpus
/// already indexes, whose keypoints the caller is holding.
///
/// The half of [`constellation_at_pixel`] that opens no `.sift` file at all: the
/// keypoints come from the caller -- a viewer's own feature cache, a corpus of
/// them read once -- and the descriptors come from the corpus by feature ID.
/// This is the path a window takes, where a `.sift` read per gesture would be a
/// second copy of what the window already has in memory.
///
/// `image_index` is the query image's row in the **corpus** image table, and
/// the rows of `keypoints` are that image's `.sift` rows: the two are joined
/// through the origin table, which is what
/// [`FeatureSources::image_feature_ids`] resolves. Features inside the radius
/// that the corpus does not index are dropped from the constellation, so an
/// index built over a subset stays usable, and candidates from `image_index`
/// itself are never reported. `center` is carried into the constellation as
/// well as used to select it, so the reported warp is fitted towards it.
pub fn constellation_from_keypoints<I, F>(
    index: &I,
    sources: &F,
    keypoints: &ImageKeypoints,
    image_index: u32,
    center: [f32; 2],
    radius: f32,
    params: &ConstellationParams,
) -> Result<PatchConstellation, KdfError>
where
    I: NeighborIndex<u8> + ?Sized,
    F: FeatureSources + ?Sized,
{
    let mut rows = keypoints.within(center, radius);
    let map = sources.image_feature_ids(image_index)?;
    rows.retain(|row| map.contains_key(row));
    let feature_ids: Vec<u32> = rows.iter().map(|row| map[row]).collect();
    let positions: Vec<[f32; 2]> = rows
        .iter()
        .map(|&row| keypoints.positions[row as usize])
        .collect();
    let query = Constellation {
        positions: &positions,
        descriptors: ConstellationDescriptors::FeatureIds(&feature_ids),
        image_index: Some(image_index),
        center: Some(center),
    };
    let matches = constellation_query(index, sources, &query, params)?;
    Ok(PatchConstellation {
        feature_rows: rows,
        feature_ids,
        matches,
    })
}

/// Gather the named descriptor rows from a `.sift` file.
///
/// The partial reader takes a prefix length rather than a row list, so this
/// costs the file's descriptors up to the highest selected row. It is the
/// fallback for an image the corpus does not index; an indexed image reads its
/// descriptors from the corpus instead and never lands here.
fn read_sift_rows(path: &Path, rows: &[u32]) -> Result<Vec<u8>, KdfError> {
    let Some(&highest) = rows.iter().max() else {
        return Ok(Vec::new());
    };
    let data = sfmtool_sift_format::read_sift_partial(path, highest as usize + 1)
        .map_err(|e| sift_error(path, e))?;
    let descriptors = data.descriptors;
    let dim = descriptors.ncols();
    let mut out = Vec::with_capacity(rows.len() * dim);
    for &row in rows {
        let row = row as usize;
        if row >= descriptors.nrows() {
            return Err(KdfError::InvalidQuery(format!(
                "{} has no feature {row}",
                path.display()
            )));
        }
        out.extend(descriptors.row(row).iter().copied());
    }
    Ok(out)
}

/// Carry a `.sift` failure in the error type the rest of this query uses.
///
/// One `Result` for a caller is worth more than the extra variant an error of
/// its own would add: an unreadable `.sift` file and an unreadable `.kdf` are
/// the same problem to the caller, and the I/O kind is preserved so a missing
/// file still surfaces as a missing file.
fn sift_error(path: &Path, err: SiftError) -> KdfError {
    match err {
        SiftError::Io(e) => KdfError::Io(e),
        SiftError::IoPath { source, .. } => KdfError::Io(source),
        other => KdfError::InvalidFormat(format!("{}: {other}", path.display())),
    }
}
