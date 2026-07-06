// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Discover points at infinity (and near-infinite distant points) in an
//! existing reconstruction by clustering the world-space directions of
//! keypoints across all images and confirming clusters with SIFT descriptors.
//!
//! See `specs/cli/xform-find-points-at-infinity.md` for the design. This
//! complements [`SfmrReconstruction::classify_points_at_infinity`], which only
//! *reclassifies* points the solve already triangulated; here we *discover* new
//! tracks the solve's parallax filters threw away.
//!
//! The geometric insight: a point at infinity is seen along the *same*
//! world-space direction from every camera (its rays are parallel), so
//! un-projecting every keypoint to a world direction makes the keypoints
//! belonging to one infinite point land on the same spot on the unit sphere. A
//! single nearest-neighbour query on the unit sphere replaces the per-pair
//! epipolar search a finite point would need. Descriptor agreement then
//! confirms co-directional keypoints are the same physical feature.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use super::convert::{
    camera_extents, classify_rays_at_infinity, Classification, RayClassification,
    DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
};
use crate::features::feature_match::descriptor::descriptor_distance_l2_squared;
use crate::reconstruction::data::observation_reprojection_error;
use crate::reconstruction::{
    ObservationSource, Point3D, ReconstructionError, SfmrReconstruction, TrackObservation,
};

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
    /// SIFT keypoint localisation noise floor (pixels). A track whose parallax
    /// signal falls below this floor is emitted as a `w = 0` point at infinity.
    pub noise_floor_px: f64,
}

/// A discovered track: its member observations and the classification of its
/// rays (finite / at infinity / indeterminate), with the diagnostics behind
/// the call for debug review of dropped tracks.
#[derive(Debug, Clone)]
pub struct InfinityTrack {
    /// Member observations as `(image_index, feature_index)` pairs, one per
    /// distinct image.
    pub members: Vec<(u32, u32)>,
    /// Classification plus observability diagnostics.
    pub classification: RayClassification,
}

/// Discover infinite/distant-point tracks from un-projected keypoint
/// directions. Pure and file-IO-free so it is unit-testable without `.sift`
/// files.
///
/// All keypoint-indexed slices (`dirs`, `descriptors`, `image_index`,
/// `feature_index`) have the same length `T`. `camera_centers` and `focal_max`
/// are indexed by image (length `N`).
#[allow(clippy::too_many_arguments)]
pub fn find_infinity_tracks(
    dirs: &[Vector3<f64>],
    descriptors: &[[u8; 128]],
    image_index: &[u32],
    feature_index: &[u32],
    camera_centers: &[Point3<f64>],
    focal_max: &[f64],
    params: &InfinityParams,
    finite_horizon: f64,
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
    let mut directed: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
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
        let in_component: std::collections::HashSet<u32> =
            members.iter().map(|&m| m as u32).collect();
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

        // 8. Classify on the triangulation's observability diagnostics into
        //    finite / at-infinity / indeterminate. See `classify_track`. The
        //    caller drops indeterminate tracks; the diagnostics ride along for
        //    debug review.
        let member_dirs: Vec<Vector3<f64>> = best_for_image
            .values()
            .map(|&(_, m_u)| dirs[m_u as usize])
            .collect();
        let member_images: Vec<usize> = best_for_image
            .values()
            .map(|&(_, m_u)| image_index[m_u as usize] as usize)
            .collect();

        let classification = classify_track(
            &member_dirs,
            &member_images,
            camera_centers,
            focal_max,
            params.noise_floor_px,
            finite_horizon,
        );

        tracks.push(InfinityTrack {
            members: chosen,
            classification,
        });
    }

    tracks
}

/// Classify a track from its member directions into finite / at-infinity /
/// indeterminate, decided on the triangulation's observability diagnostics —
/// see [`classify_rays_at_infinity`]. Discovered tracks carry no reprojection
/// error, so the per-ray angular noise is `noise_floor_px` divided by each
/// observing camera's focal length.
fn classify_track(
    member_dirs: &[Vector3<f64>],
    member_images: &[usize],
    camera_centers: &[Point3<f64>],
    focal_max: &[f64],
    noise_floor_px: f64,
    finite_horizon: f64,
) -> RayClassification {
    let member_centers: Vec<Point3<f64>> = member_images
        .iter()
        .map(|&img| camera_centers[img])
        .collect();
    let sigma_rad: Vec<f64> = member_images
        .iter()
        .map(|&img| noise_floor_px / focal_max[img])
        .collect();
    classify_rays_at_infinity(
        member_dirs,
        &member_centers,
        &sigma_rad,
        DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
        finite_horizon,
    )
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
        .images
        .iter()
        .map(|im| {
            let (fx, fy) = recon.cameras[im.camera_index as usize].focal_lengths();
            fx.max(fy)
        })
        .collect()
}

impl SfmrReconstruction {
    /// Discover points at infinity (and near-infinite distant points) by
    /// clustering world-space keypoint directions across all images, confirming
    /// clusters with SIFT descriptors, and appending the surviving tracks as new
    /// points and observations. Returns a new reconstruction.
    ///
    /// Loads each image's keypoints from its `.sift` file (capped to the largest
    /// `max_features`, or all when `None`), skipping any keypoint already
    /// assigned to an existing 3D point, un-projects the rest to world-space
    /// directions, and runs [`find_infinity_tracks`]. Every surviving track
    /// becomes a new point with a new track, built only from previously
    /// untracked features so no feature observes two points.
    pub fn find_points_at_infinity(
        &self,
        eps_deg: f64,
        desc_thresh: f64,
        ratio: f64,
        min_views: usize,
        max_features: Option<usize>,
        noise_floor_px: f64,
    ) -> Result<Self, ReconstructionError> {
        // Discovery un-projects keypoints read from per-image `.sift` files and
        // appends new sift_files observations, so it only applies to a
        // sift_files reconstruction. Refuse embedded_patches up front rather
        // than failing obscurely at the first `.sift` read.
        if self.feature_indexes().is_none() {
            return Err(ReconstructionError::Unsupported(format!(
                "find_points_at_infinity is not supported for {} reconstructions",
                self.feature_source()
            )));
        }

        // Un-project every keypoint in every image to a world-space direction.
        let read_count = max_features.unwrap_or(usize::MAX);
        let mut dirs: Vec<Vector3<f64>> = Vec::new();
        let mut descriptors: Vec<[u8; 128]> = Vec::new();
        let mut image_index: Vec<u32> = Vec::new();
        let mut feature_index: Vec<u32> = Vec::new();
        // Observed pixel position of each candidate keypoint, keyed by
        // (image, feature). Retained so a discovered point's reprojection error
        // can be measured inline against the features it was built from.
        let mut obs_xy: HashMap<(u32, u32), [f64; 2]> = HashMap::new();

        for (img_idx, image) in self.images.iter().enumerate() {
            let camera = &self.cameras[image.camera_index as usize];
            let sift_path = self.sift_path_for_image(img_idx);
            let sift = sift_format::read_sift_partial(&sift_path, read_count).map_err(|e| {
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
            let tracked = &self.image_feature_to_point[img_idx];
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

        let camera_centers: Vec<Point3<f64>> =
            self.images.iter().map(|im| im.camera_center()).collect();
        let focal_max = per_image_focal_max(self);

        // `finite_horizon` defaults to the camera extents — the scale of the
        // region the capture explored. A track whose observing baseline can't
        // resolve a point even at this distance is indeterminate and dropped.
        // (Initial value; worth sweeping other multiples of the extents later.)
        let finite_horizon = camera_extents(&camera_centers);

        let params = InfinityParams {
            eps_deg,
            desc_thresh,
            ratio,
            min_views,
            noise_floor_px,
        };
        let found = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &params,
            finite_horizon,
        );

        // Mean reprojection error (pixels) of a discovered point against the
        // features it was built from, via the shared single-observation helper
        // (handles the w = 0 vs finite projection). A point with no in-front
        // observation scores 0.0.
        let reprojection_error =
            |position: &Point3<f64>, at_infinity: bool, members: &[(u32, u32)]| -> f32 {
                let mut sum = 0.0f64;
                let mut count = 0u32;
                for &(img, feat) in members {
                    let Some(&observed) = obs_xy.get(&(img, feat)) else {
                        continue;
                    };
                    let image = &self.images[img as usize];
                    let camera = &self.cameras[image.camera_index as usize];
                    if let Some(e) = observation_reprojection_error(
                        &image.quaternion_wxyz,
                        &image.translation_xyz,
                        camera,
                        position,
                        at_infinity,
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

        // Append finite and at-infinity tracks; drop indeterminate ones (the
        // baseline couldn't adjudicate them) with a debug line for review. Every
        // member is a previously untracked feature, so no appended observation
        // collides with an existing point's observation.
        let mut recon = self.clone();
        let old_point_count = recon.points.len();
        let (mut n_finite, mut n_infinity, mut n_dropped) = (0usize, 0usize, 0usize);
        for track in found {
            let rc = &track.classification;
            let (position, w) = match rc.class {
                Classification::Finite(p) => {
                    n_finite += 1;
                    (p, 1.0)
                }
                Classification::Infinity(dir) => {
                    n_infinity += 1;
                    (dir, 0.0)
                }
                Classification::Indeterminate => {
                    n_dropped += 1;
                    let images: Vec<u32> = track.members.iter().map(|(i, _)| *i).collect();
                    eprintln!(
                        "[find-infinity] DROP indeterminate: views={} cond={:.1} \
                         resolvable={:.2} < finite_horizon={:.2} z={:.2} \
                         bearing=[{:.3}, {:.3}, {:.3}] images={:?}",
                        rc.num_views,
                        rc.condition_number,
                        rc.resolvable_distance,
                        finite_horizon,
                        rc.inverse_depth_z,
                        rc.bearing.x,
                        rc.bearing.y,
                        rc.bearing.z,
                        images,
                    );
                    continue;
                }
            };
            let error = reprojection_error(&position, w == 0.0, &track.members);
            let new_point_id = recon.points.len() as u32;
            recon.points.push(Point3D {
                position,
                w,
                color: [200, 200, 200],
                error,
                normal: Vector3::zeros(),
            });
            for (img, _feat) in &track.members {
                recon.tracks.push(TrackObservation {
                    image_index: *img,
                    point_index: new_point_id,
                });
            }
            // Infinity discovery runs on sift_files reconstructions; append the
            // new observations' feature indices to the parallel column.
            if let ObservationSource::SiftFiles {
                feature_indexes, ..
            } = &mut recon.observations
            {
                for (_img, feat) in &track.members {
                    feature_indexes.push(*feat);
                }
            }
            recon.observation_counts.push(track.members.len() as u32);
        }
        eprintln!(
            "[find-infinity] discovered {n_finite} new finite + {n_infinity} new \
             at-infinity points, dropped {n_dropped} indeterminate \
             (finite_horizon={finite_horizon:.2})"
        );

        debug_assert!(recon.points.len() >= old_point_count);
        recon.rebuild_derived_fields();
        Ok(recon)
    }
}

#[cfg(test)]
mod tests;
