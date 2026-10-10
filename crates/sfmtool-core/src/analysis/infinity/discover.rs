// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Discover points at infinity (and near-infinite distant points) in an
//! existing reconstruction by clustering the world-space directions of
//! keypoints across all images and confirming clusters with SIFT descriptors.
//!
//! See `specs/cli/reconstruction/xform/find-points-at-infinity.md` for the design. This
//! complements [`SfmrReconstruction::classify_points_at_infinity`], which only
//! *reclassifies* points the reconstruction already has; here we *discover* new
//! tracks the solve's parallax filters threw away.
//!
//! The geometric insight: a point at infinity is seen along the *same*
//! world-space direction from every camera (its rays are parallel), so
//! un-projecting every keypoint to a world direction makes the keypoints
//! belonging to one infinite point land on the same spot on the unit sphere. A
//! single nearest-neighbour query on the unit sphere replaces the per-pair
//! epipolar search a finite point would need. Descriptor agreement then
//! confirms co-directional keypoints are the same physical feature.
//!
//! [`find_infinity_tracks`] assembles the candidate tracks, and
//! [`decide_candidate_tracks`] decides each one with the point-or-bearing test,
//! as reclassification decides a stored point. Only the bearings are appended:
//! a track whose rays ask for a depth is a finite point, not a point at
//! infinity, and is left out.

use std::collections::{HashMap, HashSet};

use nalgebra::{Point3, Vector3};

use super::convert::camera_extents;
use crate::analysis::point_or_bearing::{
    observation_ray, ObservationRay, PointOrBearingError, RayBatch,
};
use crate::analysis::reprojection_noise::ReprojectionNoise;
use crate::features::feature_match::descriptor::descriptor_distance_l2_squared;
use crate::reconstruction::data::observation_reprojection_error;
use crate::reconstruction::triangulation::{
    bearing_score_batch, depth_uncertainty_batch, is_finite, triangulate_batch,
    DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};
use crate::reconstruction::{
    ImageTable, ObservationSource, Point3D, ReconstructionError, SfmrReconstruction,
    TrackObservation,
};
use sfmtool_sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE};

/// Parameters governing the points-at-infinity search.
#[derive(Debug, Clone, Copy)]
pub struct InfinityParams {
    /// Angular clustering radius in degrees. Smaller values demand more nearly
    /// parallel rays — a tighter `eps_deg` raises the distance cutoff toward
    /// true infinity, a looser one sweeps in merely "distant" points.
    pub eps_deg: f64,
    /// Maximum L2 descriptor distance for a candidate match (compared in
    /// squared space against `desc_thresh^2`).
    pub desc_thresh: f64,
    /// Lowe ratio test against the second-best in-image match.
    pub ratio: f64,
    /// A surviving track must span at least this many distinct images.
    pub min_views: usize,
}

/// A candidate track: its member observations, one per distinct image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfinityTrack {
    /// Member observations as `(image_index, feature_index)` pairs, sorted,
    /// one per distinct image.
    pub members: Vec<(u32, u32)>,
}

/// What [`decide_candidate_tracks`] made of one candidate track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CandidateDecision {
    /// A bearing verdict whose bearing is in front of every observing camera:
    /// a point at infinity along this unit direction, the score's closed-form
    /// bearing.
    Bearing(Vector3<f64>),
    /// A finite verdict: the rays ask for a depth.
    Finite,
    /// A bearing verdict whose bearing is behind an observing camera, so it
    /// describes no sighting there.
    BearingBehindCamera,
    /// Fewer than two of the track's observations give a usable ray.
    Unscored,
}

/// What [`SfmrReconstruction::find_points_at_infinity`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InfinityDiscovery {
    /// The noise level the candidates were scored at: the caller's, or the
    /// measured one. `None` when none was given and none could be measured
    /// (no observation of a finite point), in which case nothing was searched
    /// and every count is 0.
    pub sigma_px: Option<f64>,
    /// The measurement behind `sigma_px` when the pass measured it; `None`
    /// when the caller gave one.
    pub noise: Option<ReprojectionNoise>,
    /// Candidate tracks the clustering assembled, each spanning at least
    /// `min_views` images.
    pub candidates: usize,
    /// Candidates appended as points at infinity.
    pub bearings: usize,
    /// Of the appended points at infinity, those whose observing cameras
    /// have a baseline too short to tell a point at the capture's own scale
    /// (the camera extents) from infinity: `resolvable_distance` under
    /// `camera_extents` at the noise level.
    pub short_baseline: usize,
    /// Candidates dropped with a finite verdict: their rays ask for a depth.
    pub finite: usize,
    /// Bearing verdicts dropped because the bearing is behind an observing
    /// camera.
    pub bearing_behind_camera: usize,
    /// Candidates dropped because fewer than `min_views` of their members
    /// (and fewer than two) give a usable ray.
    pub unscored: usize,
}

/// Assemble candidate tracks from un-projected keypoint directions. Pure and
/// file-IO-free so it is unit-testable without `.sift` files.
///
/// All slices have the same length `T`, one entry per keypoint. The result is
/// sorted by its members, so it does not depend on hashing order.
pub fn find_infinity_tracks(
    dirs: &[Vector3<f64>],
    descriptors: &[[u8; 128]],
    image_index: &[u32],
    feature_index: &[u32],
    params: &InfinityParams,
) -> Vec<InfinityTrack> {
    let t = dirs.len();
    assert_eq!(descriptors.len(), t, "descriptors length must equal dirs");
    assert_eq!(image_index.len(), t, "image_index length must equal dirs");
    assert_eq!(
        feature_index.len(),
        t,
        "feature_index length must equal dirs"
    );
    if t == 0 {
        return Vec::new();
    }

    // 1. Build the direction KD-tree and query each point's neighbours within
    //    the chord radius corresponding to the angular radius eps.
    let eps_rad = params.eps_deg.to_radians();
    let chord_radius = (2.0 * (1.0 - eps_rad.cos())).sqrt();
    let flat: Vec<f64> = dirs.iter().flat_map(|d| [d.x, d.y, d.z]).collect();
    let cloud = crate::spatial::PointCloud3::<f64>::new(&flat, t);
    let (offsets, indices) = cloud.within_radius(&flat, t, chord_radius);

    let thresh_sq = (params.desc_thresh * params.desc_thresh) as i64;
    let ratio_sq = params.ratio * params.ratio;

    // 2-3. Directed per-image best + Lowe ratio edges. For each source `a` and
    //    each distinct neighbour image `j`, keep the best (and check the
    //    second-best) neighbour in image `j`. The directed edge `a -> b`
    //    survives if best_dist < desc_thresh^2 and (no second-best, or
    //    best_dist < ratio^2 * second_best_dist).
    //
    // Directed edges are recorded in a HashSet for O(1) mutual lookup.
    let mut directed: HashSet<(u32, u32)> = HashSet::new();
    // Per-image best/second-best scratch, keyed by neighbour image index.
    let mut best_per_image: HashMap<u32, (i64, u32)> = HashMap::new();
    let mut second_per_image: HashMap<u32, i64> = HashMap::new();

    for a in 0..t {
        best_per_image.clear();
        second_per_image.clear();
        let img_a = image_index[a];
        let start = offsets[a] as usize;
        let end = offsets[a + 1] as usize;
        for &b_u in &indices[start..end] {
            let b = b_u as usize;
            if b == a {
                continue;
            }
            let img_b = image_index[b];
            if img_b == img_a {
                continue;
            }
            let dist = descriptor_distance_l2_squared(&descriptors[a], &descriptors[b]);
            match best_per_image.get_mut(&img_b) {
                None => {
                    best_per_image.insert(img_b, (dist, b_u));
                }
                Some(best) => {
                    if dist < best.0 {
                        // Old best becomes second-best.
                        let prev_best = best.0;
                        *best = (dist, b_u);
                        let s = second_per_image.entry(img_b).or_insert(i64::MAX);
                        *s = (*s).min(prev_best);
                    } else {
                        let s = second_per_image.entry(img_b).or_insert(i64::MAX);
                        *s = (*s).min(dist);
                    }
                }
            }
        }

        for (img_b, &(best_dist, b_u)) in &best_per_image {
            if best_dist >= thresh_sq {
                continue;
            }
            let second = second_per_image.get(img_b).copied().unwrap_or(i64::MAX);
            // Ratio test in squared space; accept when there is no second-best.
            let ratio_ok = second == i64::MAX || (best_dist as f64) < ratio_sq * (second as f64);
            if ratio_ok {
                directed.insert((a as u32, b_u));
            }
        }
    }

    // 4. Mutual edges: keep undirected {a, b} when both directions exist.
    let mut union = UnionFind::new(t);
    let mut touched = vec![false; t];
    for &(a, b) in &directed {
        if a < b && directed.contains(&(b, a)) {
            union.union(a as usize, b as usize);
            touched[a as usize] = true;
            touched[b as usize] = true;
        }
    }

    // 5. Connected components over mutual edges (only touched nodes).
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &is_touched) in touched.iter().enumerate() {
        if is_touched {
            components.entry(union.find(node)).or_default().push(node);
        }
    }

    // For the one-per-image step we need each member's summed descriptor
    // distance to its mutual neighbours within the same component. Build an
    // undirected mutual-neighbour adjacency over touched nodes.
    let mut mutual_neighbours: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &directed {
        if a < b && directed.contains(&(b, a)) {
            mutual_neighbours.entry(a).or_default().push(b);
            mutual_neighbours.entry(b).or_default().push(a);
        }
    }

    let mut tracks = Vec::new();
    for members in components.into_values() {
        if members.len() < 2 {
            continue;
        }

        // 6. One feature per image: when an image contributes more than one
        //    feature, keep only the single best-supported one (smallest sum of
        //    descriptor distances to its mutual neighbours in the component).
        //    SPLIT rather than drop the whole component.
        let in_component: HashSet<u32> = members.iter().map(|&m| m as u32).collect();
        let mut best_for_image: HashMap<u32, (i64, u32)> = HashMap::new();
        for &m in &members {
            let m_u = m as u32;
            let img = image_index[m];
            let support: i64 = mutual_neighbours
                .get(&m_u)
                .map(|nbrs| {
                    nbrs.iter()
                        .filter(|&&nb| in_component.contains(&nb))
                        .map(|&nb| {
                            descriptor_distance_l2_squared(
                                &descriptors[m],
                                &descriptors[nb as usize],
                            )
                        })
                        .sum()
                })
                .unwrap_or(0);
            best_for_image
                .entry(img)
                .and_modify(|entry| {
                    if support < entry.0 {
                        *entry = (support, m_u);
                    }
                })
                .or_insert((support, m_u));
        }

        // One (image_index, feature_index) per distinct image.
        let mut chosen: Vec<(u32, u32)> = best_for_image
            .values()
            .map(|&(_, m_u)| (image_index[m_u as usize], feature_index[m_u as usize]))
            .collect();
        chosen.sort_unstable();

        // 7. Drop tracks spanning fewer than `min_views` distinct images.
        if chosen.len() < params.min_views {
            continue;
        }
        tracks.push(InfinityTrack { members: chosen });
    }
    tracks.sort_unstable_by(|a, b| a.members.cmp(&b.members));
    tracks
}

/// Decide each candidate track of `rays` with the point-or-bearing test, the
/// one reclassification decides a stored point with.
///
/// Each track is scored with [`bearing_score_batch`], and [`is_finite`] at
/// [`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`] gives the verdict. A bearing
/// verdict is the score's closed-form bearing when that is in front of every
/// observing camera. One decision per track of `rays`, in order.
///
/// # Example
///
/// ```
/// use nalgebra::{Point3, Vector3};
/// use sfmtool_core::analysis::infinity::{decide_candidate_tracks, CandidateDecision};
/// use sfmtool_core::analysis::point_or_bearing::RayBatch;
/// use sfmtool_core::reconstruction::triangulation::isotropic_ray_weights;
///
/// // Three cameras 1 unit apart see one direction: parallel rays.
/// let dirs = vec![Vector3::z(); 3];
/// let rays = RayBatch {
///     weights: isotropic_ray_weights(&dirs, &[1e-3; 3]),
///     dirs,
///     centers: (0..3).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect(),
///     offsets: vec![0, 3],
/// };
/// let decisions = decide_candidate_tracks(&rays);
/// assert!(matches!(decisions[0], CandidateDecision::Bearing(_)));
/// ```
pub fn decide_candidate_tracks(rays: &RayBatch) -> Vec<CandidateDecision> {
    bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights)
        .iter()
        .map(|score| match score {
            None => CandidateDecision::Unscored,
            Some(s) if is_finite(s, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD) => {
                CandidateDecision::Finite
            }
            Some(s) if s.bearing_in_front_of_all_cameras => {
                CandidateDecision::Bearing(s.bearing.normalize())
            }
            Some(_) => CandidateDecision::BearingBehindCamera,
        })
        .collect()
}

/// The rays of a set of candidate tracks, and the members they came from.
struct CandidateRays {
    /// One track per candidate, in order; empty for a candidate left with
    /// fewer than `min_views` members that give a ray.
    rays: RayBatch,
    /// The image of each ray.
    ray_images: Vec<usize>,
    /// Each candidate's members that gave a ray (empty where its track is).
    members: Vec<Vec<(u32, u32)>>,
}

/// Each candidate's rays, one per member keypoint whose pixel (from `pixels`)
/// gives one at `sigma_px`. A member whose pixel gives no ray (outside the
/// camera model's domain, as past the wide-angle blend of a fisheye's
/// un-projection) is left out of the track, so what is appended is what was
/// decided; a candidate left with fewer than `min_views` members gets no rays,
/// and so no score.
fn candidate_rays(
    image_table: &ImageTable,
    candidates: &[InfinityTrack],
    pixels: &HashMap<(u32, u32), [f64; 2]>,
    sigma_px: f64,
    min_views: usize,
) -> CandidateRays {
    let mut out = CandidateRays {
        rays: RayBatch {
            offsets: vec![0],
            ..RayBatch::default()
        },
        ray_images: Vec::new(),
        members: Vec::with_capacity(candidates.len()),
    };
    for track in candidates {
        let observed: Vec<((u32, u32), ObservationRay)> = track
            .members
            .iter()
            .filter_map(|&(img, feat)| {
                let pixel = pixels[&(img, feat)];
                observation_ray(image_table, img as usize, pixel, sigma_px)
                    .map(|r| ((img, feat), r))
            })
            .collect();
        if observed.len() >= min_views {
            for ((img, _), r) in &observed {
                out.rays.dirs.push(r.dir);
                out.rays.centers.push(r.center);
                out.rays.weights.push(r.weight);
                out.ray_images.push(*img as usize);
            }
            out.members
                .push(observed.into_iter().map(|(m, _)| m).collect());
        } else {
            out.members.push(Vec::new());
        }
        out.rays.offsets.push(out.rays.dirs.len());
    }
    out
}

impl InfinityDiscovery {
    /// Add one count per decision: every candidate lands in exactly one of
    /// `bearings`, `finite`, `bearing_behind_camera` and `unscored`, and a
    /// bearing flagged in the `short_baseline` argument also in the field of
    /// the same name.
    fn count_decisions(&mut self, decisions: &[CandidateDecision], short_baseline: &[bool]) {
        for (decision, &short) in decisions.iter().zip(short_baseline) {
            match decision {
                CandidateDecision::Bearing(_) => {
                    self.bearings += 1;
                    if short {
                        self.short_baseline += 1;
                    }
                }
                CandidateDecision::Finite => self.finite += 1,
                CandidateDecision::BearingBehindCamera => self.bearing_behind_camera += 1,
                CandidateDecision::Unscored => self.unscored += 1,
            }
        }
    }
}

/// Minimal union-find over `0..n` for connected-component assembly.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Largest focal length (pixels) for each image's camera.
fn per_image_focal_max(recon: &SfmrReconstruction) -> Vec<f64> {
    recon
        .image_table
        .images
        .iter()
        .map(|im| {
            let (fx, fy) = recon.image_table.cameras[im.camera_index as usize].focal_lengths();
            fx.max(fy)
        })
        .collect()
}

impl SfmrReconstruction {
    /// Discover points at infinity by clustering world-space keypoint
    /// directions across all images, confirming clusters with SIFT
    /// descriptors, deciding each surviving track with the point-or-bearing
    /// test, and appending the bearings as new points and observations.
    /// Returns a new reconstruction and what was found.
    ///
    /// Loads each image's keypoints from its `.sift` file (capped to the largest
    /// `max_features`, or all when `None`), skipping any keypoint already
    /// assigned to an existing 3D point, un-projects the rest to world-space
    /// directions, and runs [`find_infinity_tracks`]. Each candidate's rays are
    /// weighted at the per-axis pixel noise `sigma_px` (the reconstruction's
    /// [`reprojection_noise`](Self::reprojection_noise) when `None`) and
    /// decided by [`decide_candidate_tracks`]. A bearing becomes a `w = 0`
    /// point at the score's closed-form bearing, and the other candidates are
    /// dropped and counted: a finite verdict is a finite point, which this
    /// pass does not add. Every appended track is built only from previously
    /// untracked features, so no feature observes two points.
    ///
    /// When `sigma_px` is `None` and the reconstruction has no observation of a
    /// finite point to measure the noise from, nothing is searched: the result
    /// is the reconstruction unchanged, with a summary whose `sigma_px` is
    /// `None`.
    ///
    /// # Errors
    ///
    /// [`PointOrBearingError::InvalidNoiseLevel`] for a `sigma_px` that is not
    /// finite and positive, and [`PointOrBearingError::Reconstruction`] for an
    /// `embedded_patches` reconstruction (which has no `.sift` files to search)
    /// or a `.sift` file that cannot be read.
    pub fn find_points_at_infinity(
        &self,
        eps_deg: f64,
        desc_thresh: f64,
        ratio: f64,
        min_views: usize,
        max_features: Option<usize>,
        sigma_px: Option<f64>,
    ) -> Result<(Self, InfinityDiscovery), PointOrBearingError> {
        if let Some(s) = sigma_px {
            if !(s.is_finite() && s > 0.0) {
                return Err(PointOrBearingError::InvalidNoiseLevel(s));
            }
        }
        // Discovery un-projects keypoints read from per-image `.sift` files and
        // appends new sift_files observations, so it only applies to a
        // sift_files reconstruction. Refuse embedded_patches up front rather
        // than failing obscurely at the first `.sift` read.
        if self.feature_indexes().is_none() {
            return Err(ReconstructionError::Unsupported(format!(
                "find_points_at_infinity is not supported for {} reconstructions",
                self.feature_source()
            ))
            .into());
        }

        let mut summary = InfinityDiscovery::default();
        let sigma = match sigma_px {
            Some(s) => s,
            None => {
                let noise = self.reprojection_noise()?;
                let measured = noise.sigma_px;
                summary.noise = Some(noise);
                match measured {
                    Some(s) => s,
                    None => return Ok((self.clone(), summary)),
                }
            }
        };
        summary.sigma_px = Some(sigma);

        // Un-project every keypoint in every image to a world-space direction.
        let read_count = max_features.unwrap_or(usize::MAX);
        let mut dirs: Vec<Vector3<f64>> = Vec::new();
        let mut descriptors: Vec<[u8; 128]> = Vec::new();
        let mut image_index: Vec<u32> = Vec::new();
        let mut feature_index: Vec<u32> = Vec::new();
        // Observed pixel position of each candidate keypoint, keyed by
        // (image, feature). Retained to build a candidate's rays for the test
        // and to measure an appended point's reprojection error.
        let mut obs_xy: HashMap<(u32, u32), [f64; 2]> = HashMap::new();

        for (img_idx, image) in self.image_table.images.iter().enumerate() {
            let camera = &self.image_table.cameras[image.camera_index as usize];
            let sift_path = self.sift_path_for_image(img_idx);
            let sift =
                sfmtool_sift_format::read_sift_partial(&sift_path, read_count).map_err(|e| {
                    ReconstructionError::SiftRead {
                        path: sift_path.clone(),
                        source: e.to_string(),
                    }
                })?;

            let n = sift.positions_xy.nrows();
            // World-rotation: ray_world = R^T * ray_cam, where R = world->cam
            // quaternion. quaternion_wxyz.inverse() is the camera->world
            // rotation.
            let cam_to_world = image.quaternion_wxyz.inverse();
            let tracked = &self.point_set.image_feature_to_point[img_idx];
            for f in 0..n {
                // Discovery only considers keypoints the solve left untracked. A
                // 2D feature already assigned to a 3D point cannot also belong to
                // a new track; reusing it would make one feature observe two
                // points, which the .sfmr list tolerates but COLMAP export (and
                // thus bundle adjustment) rejects.
                if tracked.contains_key(&(f as u32)) {
                    continue;
                }
                let u = sift.positions_xy[[f, 0]] as f64;
                let v = sift.positions_xy[[f, 1]] as f64;
                let ray_cam = camera.pixel_to_ray(u, v);
                let world = cam_to_world * Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
                let norm = world.norm();
                let unit = if norm > 0.0 { world / norm } else { world };
                dirs.push(unit);
                obs_xy.insert((img_idx as u32, f as u32), [u, v]);

                let mut desc = [0u8; 128];
                for (k, slot) in desc.iter_mut().enumerate() {
                    *slot = sift.descriptors[[f, k]];
                }
                descriptors.push(desc);
                image_index.push(img_idx as u32);
                feature_index.push(f as u32);
            }
        }

        let params = InfinityParams {
            eps_deg,
            desc_thresh,
            ratio,
            min_views,
        };
        let found =
            find_infinity_tracks(&dirs, &descriptors, &image_index, &feature_index, &params);
        summary.candidates = found.len();

        let CandidateRays {
            rays,
            ray_images,
            members: kept_members,
        } = candidate_rays(&self.image_table, &found, &obs_xy, sigma, min_views);
        let decisions = decide_candidate_tracks(&rays);

        // How far each track's cameras can tell a point from infinity, at the
        // same noise as an angle (σ_px over the focal length), against the
        // camera extents: the count of appended bearings the cameras could not
        // resolve at the capture's own scale.
        let camera_centers: Vec<Point3<f64>> = self
            .image_table
            .images
            .iter()
            .map(|im| im.camera_center())
            .collect();
        let capture_scale = camera_extents(&camera_centers);
        let focal_max = per_image_focal_max(self);
        let sigma_rad: Vec<f64> = ray_images.iter().map(|&i| sigma / focal_max[i]).collect();
        let tris = triangulate_batch(&rays.dirs, &rays.centers, &rays.offsets);
        let resolvable: Vec<f64> =
            depth_uncertainty_batch(&tris, &rays.dirs, &rays.centers, &rays.offsets, &sigma_rad)
                .iter()
                .map(|du| du.resolvable_distance)
                .collect();

        // Mean reprojection error (pixels) of a discovered bearing against the
        // features it was built from, via the shared single-observation helper.
        // A point with no in-front observation scores 0.0.
        let reprojection_error = |bearing: &Point3<f64>, members: &[(u32, u32)]| -> f32 {
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for &(img, feat) in members {
                let Some(&observed) = obs_xy.get(&(img, feat)) else {
                    continue;
                };
                let image = &self.image_table.images[img as usize];
                let camera = &self.image_table.cameras[image.camera_index as usize];
                if let Some(e) = observation_reprojection_error(
                    &image.quaternion_wxyz,
                    &image.translation_xyz,
                    camera,
                    bearing,
                    true,
                    observed,
                ) {
                    sum += e;
                    count += 1;
                }
            }
            if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            }
        };

        // Append the bearings; drop the rest.
        // Every member is a previously untracked feature, so no appended
        // observation collides with an existing point's observation.
        let mut recon = self.clone_for_edit();
        // A `sift_files` reconstruction may carry an inline copy of its
        // observation coordinates; the appended observations extend it in
        // lockstep with `feature_indexes`. Their pixels are the ones unprojected
        // above, collected here and spliced on once the loop is done. `None`
        // when there is no inline column to extend.
        let mut appended_keypoints: Option<Vec<[f32; 2]>> =
            recon.keypoints_xy().map(|_| Vec::new());
        let short_baseline: Vec<bool> = resolvable.iter().map(|&d| d < capture_scale).collect();
        summary.count_decisions(&decisions, &short_baseline);
        for (members, decision) in kept_members.iter().zip(&decisions) {
            let CandidateDecision::Bearing(dir) = *decision else {
                continue;
            };
            let position = Point3::from(dir);
            let error = reprojection_error(&position, members);
            let new_point_id = recon.point_set.points.len() as u32;
            recon.point_set.points.push(Point3D {
                position,
                w: 0.0,
                color: [200, 200, 200],
                error,
                normal: Vector3::zeros(),
            });
            for (img, _feat) in members {
                recon.point_set.tracks.push(TrackObservation {
                    image_index: *img,
                    point_index: new_point_id,
                });
            }
            // Infinity discovery runs on sift_files reconstructions; append the
            // new observations' feature indices to the parallel column.
            if let ObservationSource::SiftFiles {
                feature_indexes, ..
            } = &mut recon.point_set.observations
            {
                for (_img, feat) in members {
                    feature_indexes.push(*feat);
                }
            }
            // Every member is one of the candidate keypoints unprojected above,
            // so its pixel is in `obs_xy`.
            if let Some(rows) = appended_keypoints.as_mut() {
                for (img, feat) in members {
                    let xy = obs_xy[&(*img, *feat)];
                    rows.push([xy[0] as f32, xy[1] as f32]);
                }
            }
            // A newly discovered observation was never measured, so it gets the
            // "no data-derived support" code rather than inheriting anything.
            if let Some(confidence) = recon.point_set.observation_confidence.as_mut() {
                confidence.extend(std::iter::repeat_n(0u8, members.len()));
            }
            // Nor read on any render.
            if let Some(readings) = recon.point_set.observation_readings.as_mut() {
                readings.rows.extend(std::iter::repeat_n(
                    crate::reconstruction::ObservationReading::NOT_MEASURED,
                    members.len(),
                ));
            }
            // Nothing outside the solve owns a track this pass discovered, so
            // its constraint row is free -- the row the reconstruction would
            // hold for it if it carried no constraint columns at all.
            if let Some(constraints) = recon.point_set.point_constraints.as_mut() {
                constraints.point_constraints.push(POINT_CONSTRAINT_FREE);
                constraints.constraint_distances.push(f64::NAN);
                constraints
                    .constraint_reference_images
                    .push(NO_REFERENCE_IMAGE);
            }
            // A new bearing has no bitmap rendered from any of its observations.
            if let Some(references) = recon.point_set.reference_observations.as_mut() {
                references.push(sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION);
            }
            // Nor a pick the display render made, so its mark stays in step.
            if let Some(marks) = recon.point_set.display_only_references.as_mut() {
                marks.push(false);
            }
            recon
                .point_set
                .observation_counts
                .push(members.len() as u32);
        }

        if let (
            Some(rows),
            ObservationSource::SiftFiles {
                keypoints_xy: Some(keypoints_xy),
                ..
            },
        ) = (appended_keypoints, &mut recon.point_set.observations)
        {
            for xy in rows {
                keypoints_xy
                    .push_row(ndarray::ArrayView1::from(&xy[..]))
                    .expect("appended keypoint row is 2 wide");
            }
        }

        recon.rebuild_derived_fields();
        // The appended bearings change the count the metadata states as well.
        recon.metadata.infinity_point_count = recon.point_set.infinity_point_count as u32;
        Ok((recon, summary))
    }
}

#[cfg(test)]
mod tests;
