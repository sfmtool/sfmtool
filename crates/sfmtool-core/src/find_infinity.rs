// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Discover points at infinity (and near-infinite distant points) in an
//! existing reconstruction by clustering the world-space directions of
//! keypoints across all images and confirming clusters with SIFT descriptors.
//!
//! See `specs/drafts/xform-find-points-at-infinity.md` for the design. This
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

use nalgebra::{Matrix3, Point3, Vector3};

use crate::feature_match::descriptor::descriptor_distance_l2_squared;
use crate::reconstruction::{Point3D, ReconstructionError, SfmrReconstruction, TrackObservation};
use crate::viewing_angle::max_viewing_angle;

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

/// A discovered track: its member observations and resulting 3D point.
#[derive(Debug, Clone)]
pub struct InfinityTrack {
    /// Member observations as `(image_index, feature_index)` pairs, one per
    /// distinct image.
    pub members: Vec<(u32, u32)>,
    /// Euclidean position (`w = 1`) or unit direction (`w = 0`).
    pub position: Point3<f64>,
    /// Homogeneous coordinate kind: `1.0` finite, `0.0` at infinity.
    pub w: f64,
}

/// Discover infinite/distant-point tracks from un-projected keypoint
/// directions. Pure and file-IO-free so it is unit-testable without `.sift`
/// files.
///
/// All keypoint-indexed slices (`dirs`, `descriptors`, `image_index`,
/// `feature_index`) have the same length `T`. `camera_centers` and `focal_max`
/// are indexed by image (length `N`).
pub fn find_infinity_tracks(
    dirs: &[Vector3<f64>],
    descriptors: &[[u8; 128]],
    image_index: &[u32],
    feature_index: &[u32],
    camera_centers: &[Point3<f64>],
    focal_max: &[f64],
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

        // 8. Classify: parallax_px = alpha_max * f_max. Below the floor → w = 0
        //    bearing mean; above → triangulate, fall back to bearing mean when
        //    the solve is ill-conditioned or the point is behind the cameras.
        let member_dirs: Vec<Vector3<f64>> = best_for_image
            .values()
            .map(|&(_, m_u)| dirs[m_u as usize])
            .collect();
        let member_images: Vec<usize> = best_for_image
            .values()
            .map(|&(_, m_u)| image_index[m_u as usize] as usize)
            .collect();

        let (position, w) = classify_track(
            &member_dirs,
            &member_images,
            camera_centers,
            focal_max,
            params.noise_floor_px,
        );

        tracks.push(InfinityTrack {
            members: chosen,
            position,
            w,
        });
    }

    tracks
}

/// Classify a track from its member directions: a `w = 0` bearing-mean point at
/// infinity, or a triangulated finite point when the parallax clears the noise
/// floor and the midpoint solve is well-conditioned and in front of the
/// cameras.
fn classify_track(
    member_dirs: &[Vector3<f64>],
    member_images: &[usize],
    camera_centers: &[Point3<f64>],
    focal_max: &[f64],
    noise_floor_px: f64,
) -> (Point3<f64>, f64) {
    let bearing_mean = || -> Point3<f64> {
        let mut sum = Vector3::zeros();
        for d in member_dirs {
            sum += d;
        }
        let norm = sum.norm();
        if norm > 0.0 {
            Point3::from(sum / norm)
        } else {
            // Rays cancel exactly — degenerate; fall back to the first member.
            Point3::from(member_dirs[0])
        }
    };

    // alpha_max over the member directions: the largest pairwise angle.
    let rays: Vec<(usize, Vector3<f64>)> = member_images
        .iter()
        .zip(member_dirs.iter())
        .map(|(&img, &d)| (img, d))
        .collect();
    let alpha_max = max_viewing_angle(&rays);
    let f_max = member_images
        .iter()
        .map(|&img| focal_max[img])
        .fold(0.0_f64, f64::max);
    let parallax_px = alpha_max * f_max;

    if parallax_px < noise_floor_px {
        return (bearing_mean(), 0.0);
    }

    // Triangulate by the midpoint / least-squares closest point to N lines
    // `cᵢ + t·dᵢ`: solve `A p = b` with `A = Σ(I - dᵢdᵢᵀ)`,
    // `b = Σ(I - dᵢdᵢᵀ)cᵢ`.
    let mut a_mat = Matrix3::<f64>::zeros();
    let mut b_vec = Vector3::<f64>::zeros();
    let identity = Matrix3::<f64>::identity();
    for (&img, &d) in member_images.iter().zip(member_dirs.iter()) {
        let proj = identity - d * d.transpose();
        a_mat += proj;
        b_vec += proj * camera_centers[img].coords;
    }

    // Well-conditioned check: the projection sum is rank-deficient when all
    // directions are parallel (the infinite-point case). Guard with the
    // determinant relative to the trace scale.
    let det = a_mat.determinant();
    let scale = a_mat.norm();
    if scale <= 0.0 || det.abs() < 1e-9 * scale.powi(3) {
        return (bearing_mean(), 0.0);
    }
    let Some(inv) = a_mat.try_inverse() else {
        return (bearing_mean(), 0.0);
    };
    let p = inv * b_vec;
    let point = Point3::from(p);

    // The triangulated point must lie in front of every observing camera (the
    // ray from camera to point agrees with the un-projected direction).
    let in_front = member_images
        .iter()
        .zip(member_dirs.iter())
        .all(|(&img, &d)| {
            let to_point = point.coords - camera_centers[img].coords;
            to_point.dot(&d) > 0.0
        });
    if !in_front || !p.iter().all(|c| c.is_finite()) {
        return (bearing_mean(), 0.0);
    }

    (point, 1.0)
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
        // Un-project every keypoint in every image to a world-space direction.
        let read_count = max_features.unwrap_or(usize::MAX);
        let mut dirs: Vec<Vector3<f64>> = Vec::new();
        let mut descriptors: Vec<[u8; 128]> = Vec::new();
        let mut image_index: Vec<u32> = Vec::new();
        let mut feature_index: Vec<u32> = Vec::new();

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
        );

        // Append surviving tracks. Every member is a previously untracked
        // feature (already-tracked keypoints were excluded above), so no
        // appended observation collides with an existing point's observation.
        let mut recon = self.clone();
        let old_point_count = recon.points.len();
        for track in found {
            let new_point_id = recon.points.len() as u32;
            recon.points.push(Point3D {
                position: track.position,
                w: track.w,
                color: [200, 200, 200],
                error: 0.0,
                estimated_normal: Vector3::zeros(),
            });
            for (img, feat) in &track.members {
                recon.tracks.push(TrackObservation {
                    image_index: *img,
                    feature_index: *feat,
                    point_index: new_point_id,
                });
            }
            recon.observation_counts.push(track.members.len() as u32);
        }

        debug_assert!(recon.points.len() >= old_point_count);
        recon.rebuild_derived_indexes();
        Ok(recon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> InfinityParams {
        InfinityParams {
            eps_deg: 0.5,
            desc_thresh: 300.0,
            ratio: 0.8,
            min_views: 2,
            noise_floor_px: 1.0,
        }
    }

    /// Build a descriptor that is `value` in every component.
    fn flat_desc(value: u8) -> [u8; 128] {
        [value; 128]
    }

    #[test]
    fn finds_single_infinite_point() {
        // 3 keypoints in 3 distinct images, near-identical world directions and
        // identical descriptors. Cameras spread out but the ray directions are
        // essentially parallel (an infinite point).
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let dirs = vec![dir, dir, dir];
        let descriptors = vec![flat_desc(50), flat_desc(50), flat_desc(50)];
        let image_index = vec![0, 1, 2];
        let feature_index = vec![0, 0, 0];
        let camera_centers = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let focal_max = vec![1000.0, 1000.0, 1000.0];

        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &default_params(),
        );

        assert_eq!(tracks.len(), 1, "exactly one track");
        let track = &tracks[0];
        assert_eq!(track.members.len(), 3, "three members, one per image");
        assert_eq!(track.w, 0.0, "parallel rays → point at infinity");
        assert!(
            (track.position.coords.norm() - 1.0).abs() < 1e-9,
            "position is a unit direction"
        );
    }

    #[test]
    fn distinct_descriptors_not_merged() {
        // Same direction, different descriptors → no match.
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let dirs = vec![dir, dir];
        let descriptors = vec![flat_desc(0), flat_desc(255)];
        let image_index = vec![0, 1];
        let feature_index = vec![0, 0];
        let camera_centers = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let focal_max = vec![1000.0, 1000.0];

        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &default_params(),
        );
        assert!(tracks.is_empty(), "far descriptors must not merge");
    }

    #[test]
    fn distinct_directions_not_merged() {
        // Identical descriptors but very different directions → no neighbour.
        let dirs = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 0.0)];
        let descriptors = vec![flat_desc(50), flat_desc(50)];
        let image_index = vec![0, 1];
        let feature_index = vec![0, 0];
        let camera_centers = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let focal_max = vec![1000.0, 1000.0];

        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &default_params(),
        );
        assert!(tracks.is_empty(), "distant directions must not merge");
    }

    #[test]
    fn one_feature_per_image_after_split() {
        // Image 1 contributes two co-directional, identical-descriptor features
        // alongside one each in images 0 and 2. The track must keep exactly one
        // feature per image.
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let dirs = vec![dir, dir, dir, dir];
        let descriptors = vec![
            flat_desc(50), // img 0 feat 0
            flat_desc(50), // img 1 feat 0
            flat_desc(50), // img 1 feat 1 (duplicate image)
            flat_desc(50), // img 2 feat 0
        ];
        let image_index = vec![0, 1, 1, 2];
        let feature_index = vec![0, 0, 1, 0];
        let camera_centers = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let focal_max = vec![1000.0, 1000.0, 1000.0];

        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &default_params(),
        );

        assert_eq!(tracks.len(), 1, "one track");
        let imgs: Vec<u32> = tracks[0].members.iter().map(|(i, _)| *i).collect();
        let mut distinct = imgs.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            imgs.len(),
            distinct.len(),
            "no image appears twice in a track"
        );
    }

    #[test]
    fn min_views_filter_drops_short_track() {
        // A 2-image track is dropped when min_views = 3.
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let dirs = vec![dir, dir];
        let descriptors = vec![flat_desc(50), flat_desc(50)];
        let image_index = vec![0, 1];
        let feature_index = vec![0, 0];
        let camera_centers = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let focal_max = vec![1000.0, 1000.0];

        let mut params = default_params();
        params.min_views = 3;
        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &params,
        );
        assert!(tracks.is_empty(), "2-image track dropped at min_views=3");

        params.min_views = 2;
        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            &params,
        );
        assert_eq!(tracks.len(), 1, "kept at min_views=2");
    }

    #[test]
    fn wide_parallax_track_triangulates_finite() {
        // Three cameras on the x-axis look at a point ~2 units in front; the
        // ray directions differ by tens of degrees (wide parallax, well above
        // the noise floor), so the track triangulates to a finite point.
        let point = Point3::new(0.0, 0.0, 2.0);
        let camera_centers = vec![
            Point3::new(-2.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let dirs: Vec<Vector3<f64>> = camera_centers
            .iter()
            .map(|c| (point.coords - c.coords).normalize())
            .collect();
        let descriptors = vec![flat_desc(50), flat_desc(50), flat_desc(50)];
        let image_index = vec![0, 1, 2];
        let feature_index = vec![0, 0, 0];
        let focal_max = vec![1000.0, 1000.0, 1000.0];

        let tracks = find_infinity_tracks(
            &dirs,
            &descriptors,
            &image_index,
            &feature_index,
            &camera_centers,
            &focal_max,
            // eps must be loose enough to cluster these spread-out directions.
            &InfinityParams {
                eps_deg: 90.0,
                ..default_params()
            },
        );

        assert_eq!(tracks.len(), 1, "one track");
        assert_eq!(tracks[0].w, 1.0, "wide parallax → finite point");
        let recovered = tracks[0].position;
        assert!(
            (recovered.coords - point.coords).norm() < 1e-6,
            "triangulated position {:?} near true point {:?}",
            recovered,
            point
        );
    }
}
