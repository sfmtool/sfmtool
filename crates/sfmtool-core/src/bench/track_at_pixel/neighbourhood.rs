// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the track-at-pixel members read about the surroundings of a pixel: the
//! reconstruction's observations near it, the `.matches` clusters with a member
//! near it, and each view's camera as plain arithmetic.
//!
//! The observation index is built from the [`EditedReconstruction`] the
//! operation is handed, over its live points only, so a point the version has
//! deleted is not found by any query here. That is what keeps a held-out point
//! out of reach: it is deleted from the version, and nothing reads the base's
//! point columns directly.

use std::cell::OnceCell;

use nalgebra::{Matrix3, Point3, Vector3};
use sfmtool_matches_format::{MatchesData, CLUSTER_REFERENCE_UNREFINABLE};

use crate::patch::normal_refine::ProjectedImage;
use crate::reconstruction::edited::EditedReconstruction;
use crate::spatial::PointCloud;

/// One image's camera, as the members do arithmetic with it: `cam_from_world`
/// as a matrix and a translation, the centre, the mean focal length and the
/// frame size. Cameras look down their own `-Z` axis, so a point's depth is
/// `-z` in the camera frame.
pub(super) struct ViewCamera<'a> {
    view: &'a ProjectedImage<'a>,
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    /// The camera centre in world coordinates.
    pub(super) center: Vector3<f64>,
    /// The mean of the two focal lengths, in px.
    pub(super) focal: f64,
    /// The frame width, in px.
    pub(super) width: u32,
    /// The frame height, in px.
    pub(super) height: u32,
}

impl<'a> ViewCamera<'a> {
    pub(super) fn new(view: &'a ProjectedImage<'a>) -> Self {
        let rotation = view.cam_from_world.to_rotation_matrix();
        let translation = view.cam_from_world.translation;
        let (fx, fy) = view.camera.focal_lengths();
        Self {
            view,
            rotation,
            translation,
            center: -(rotation.transpose() * translation),
            focal: 0.5 * (fx + fy),
            width: view.camera.width,
            height: view.camera.height,
        }
    }

    /// A world point in the camera frame.
    fn to_camera(&self, xyz: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * xyz + self.translation
    }

    /// Distance in front of the camera along its axis.
    pub(super) fn depth(&self, xyz: &Vector3<f64>) -> f64 {
        -self.to_camera(xyz).z
    }

    /// The pixel a world point projects to, or `None` when it is behind the
    /// camera or outside the lens model's domain.
    pub(super) fn project(&self, xyz: &Vector3<f64>) -> Option<[f64; 2]> {
        let p = self.to_camera(xyz);
        if -p.z <= 1e-12 {
            return None;
        }
        let ray = p / p.norm();
        let (u, v) = self.view.camera.ray_to_pixel([ray.x, ray.y, ray.z])?;
        Some([u, v])
    }

    /// The world-frame unit ray through a pixel.
    pub(super) fn ray(&self, pixel: [f64; 2]) -> Vector3<f64> {
        let d = self.view.camera.pixel_to_ray(pixel[0], pixel[1]);
        self.rotation.transpose() * Vector3::new(d[0], d[1], d[2])
    }

    /// Whether `pixel` lies at least `margin` px inside the frame.
    pub(super) fn in_frame(&self, pixel: [f64; 2], margin: f64) -> bool {
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        margin <= pixel[0] && pixel[0] < w - margin && margin <= pixel[1] && pixel[1] < h - margin
    }
}

/// One reconstruction observation near a pixel, with what a member reads off
/// its point.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyObservation {
    /// The point, as an index into the edited reconstruction.
    pub point: u32,
    /// The observation's keypoint in the queried image.
    pub keypoint: [f64; 2],
    /// How far the keypoint is from the pixel asked about, in px.
    pub distance_px: f64,
    /// The point's depth along the queried camera's axis, or `None` for a point
    /// at infinity.
    pub depth: Option<f64>,
    /// The point's unit outward normal, the cross product of its patch
    /// half-vectors; zero when the point carries no patch.
    pub normal: Vector3<f64>,
    /// The length of the patch's `u` half-vector, in world units.
    pub half_extent: f64,
    /// That half-extent's apparent size in the queried image, in px; `NaN` for
    /// a point at infinity or one behind the camera.
    pub half_px: f64,
    /// Whether the point is a bearing (`w == 0`).
    pub at_infinity: bool,
}

impl NearbyObservation {
    /// The depth when there is one and it is not zero: what the members mean by
    /// "a neighbour that states a depth".
    pub(super) fn stated_depth(&self) -> Option<f64> {
        self.depth.filter(|&d| d != 0.0)
    }

    /// The depth when the point is finite and in front of the camera.
    pub(super) fn positive_depth(&self) -> Option<f64> {
        self.stated_depth()
            .filter(|&d| !self.at_infinity && d > 0.0)
    }
}

/// One image's observations, with its 2D index built the first time it is
/// asked about.
struct ImageObservations {
    points: Vec<u32>,
    xy: Vec<f64>,
    cloud: OnceCell<Option<PointCloud<f64, 2>>>,
}

/// The live points' observations of an edited reconstruction, per image.
///
/// A point observed twice in one image is listed at its first observation
/// there, in track order.
pub(super) struct ObservationIndex<'a> {
    edited: &'a EditedReconstruction,
    per_image: Vec<ImageObservations>,
}

impl<'a> ObservationIndex<'a> {
    pub(super) fn new(edited: &'a EditedReconstruction) -> Self {
        let mut per_image: Vec<ImageObservations> = (0..edited.image_count())
            .map(|_| ImageObservations {
                points: Vec::new(),
                xy: Vec::new(),
                cloud: OnceCell::new(),
            })
            .collect();
        let mut seen: Vec<u32> = Vec::new();
        for point in edited.live_indexes() {
            let Some(view) = edited.point(point) else {
                continue;
            };
            seen.clear();
            for (k, obs) in view.observations().iter().enumerate() {
                let image = obs.image_index;
                if seen.contains(&image) || image as usize >= per_image.len() {
                    continue;
                }
                seen.push(image);
                let Some(kp) = view.keypoint_xy(k) else {
                    continue;
                };
                let entry = &mut per_image[image as usize];
                entry.points.push(point);
                entry.xy.push(f64::from(kp[0]));
                entry.xy.push(f64::from(kp[1]));
            }
        }
        Self { edited, per_image }
    }

    /// Every observation in `image` within `radius_px` of `pixel`, nearest
    /// first, read through `camera` (the queried image's).
    pub(super) fn near(
        &self,
        image: u32,
        pixel: [f64; 2],
        radius_px: f64,
        camera: &ViewCamera<'_>,
    ) -> Vec<NearbyObservation> {
        let Some(entry) = self.per_image.get(image as usize) else {
            return Vec::new();
        };
        let cloud = entry.cloud.get_or_init(|| {
            (!entry.points.is_empty()).then(|| PointCloud::new(&entry.xy, entry.points.len()))
        });
        let Some(cloud) = cloud else {
            return Vec::new();
        };
        let (_, rows) = cloud.within_radius(&pixel, 1, radius_px);
        let mut out: Vec<NearbyObservation> = rows
            .into_iter()
            .filter_map(|row| {
                let row = row as usize;
                let point = entry.points[row];
                let keypoint = [entry.xy[2 * row], entry.xy[2 * row + 1]];
                self.describe(point, keypoint, pixel, camera)
            })
            .collect();
        out.sort_by(|a, b| a.distance_px.total_cmp(&b.distance_px));
        out
    }

    fn describe(
        &self,
        point: u32,
        keypoint: [f64; 2],
        pixel: [f64; 2],
        camera: &ViewCamera<'_>,
    ) -> Option<NearbyObservation> {
        let view = self.edited.point(point)?;
        let p = view.point();
        let finite = p.w != 0.0;
        let halfvec = |h: Option<[f32; 3]>| {
            h.map_or_else(Vector3::zeros, |h| {
                Vector3::new(f64::from(h[0]), f64::from(h[1]), f64::from(h[2]))
            })
        };
        let u = halfvec(view.patch_u_halfvec());
        let v = halfvec(view.patch_v_halfvec());
        let cross = u.cross(&v);
        let norm = cross.norm();
        let normal = if norm > 0.0 {
            cross / norm
        } else {
            Vector3::zeros()
        };
        let half_extent = u.norm();
        let depth = finite.then(|| camera.depth(&p.position.coords));
        let half_px = match depth {
            Some(d) if d > 0.0 => half_extent * camera.focal / d,
            _ => f64::NAN,
        };
        let dx = keypoint[0] - pixel[0];
        let dy = keypoint[1] - pixel[1];
        Some(NearbyObservation {
            point,
            keypoint,
            distance_px: (dx * dx + dy * dy).sqrt(),
            depth,
            normal,
            half_extent,
            half_px,
            at_infinity: !finite,
        })
    }

    /// Every observation of `point`, as `(image, keypoint)` in track order.
    pub(super) fn point_observations(&self, point: u32) -> Vec<(u32, [f64; 2])> {
        let Some(view) = self.edited.point(point) else {
            return Vec::new();
        };
        view.observations()
            .iter()
            .enumerate()
            .filter_map(|(k, obs)| {
                let kp = view.keypoint_xy(k)?;
                Some((obs.image_index, [f64::from(kp[0]), f64::from(kp[1])]))
            })
            .collect()
    }

    /// A live point's position, or `None` for one the version does not hold.
    pub(super) fn position(&self, point: u32) -> Option<Point3<f64>> {
        Some(self.edited.point(point)?.point().position)
    }
}

/// The member-status legend of the `.matches` cluster-patches section: the
/// reference member.
const STATUS_REFERENCE: u8 = 0;
/// The member-status legend: a member the refinement kept.
const STATUS_KEPT: u8 = 1;
/// The member-status legend: a member nothing evaluated, which is what every
/// member of a file with no cluster-patches section is.
const STATUS_NOT_EVALUATED: u8 = 5;

/// Why a `.matches` file cannot serve as the clusters a track-at-pixel query
/// reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchesClustersError {
    /// The file stores the pairwise backbone, which has no clusters.
    NoClusters,
    /// The file's clusters carry no member positions and shapes, which only a
    /// value built in memory can lack.
    NoMemberGeometry,
}

impl std::fmt::Display for MatchesClustersError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoClusters => write!(
                f,
                "the .matches file stores image pairs, not clusters; run `sfm match --cluster` \
                 and `sfm cluster-patches` to make one that does"
            ),
            Self::NoMemberGeometry => write!(
                f,
                "the .matches clusters carry no member positions or shapes"
            ),
        }
    }
}

impl std::error::Error for MatchesClustersError {}

/// One member of a `.matches` cluster, as a track-at-pixel query reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterMember {
    /// The member's image in the reconstruction, or `None` when the file's image
    /// is not one the reconstruction holds.
    pub image: Option<u32>,
    /// The member's position in its image, in px: refined where the
    /// cluster-patches pass measured it, the detection otherwise.
    pub position: [f64; 2],
    /// The member's affine shape, the detector's unit frame mapped onto its
    /// image's pixels, as the file stores it (`shape[row][column]`).
    pub shape: [[f64; 2]; 2],
    /// The `member_status` discriminant; `not_evaluated` for every member of a
    /// file with no cluster-patches section.
    pub status: u8,
    /// The ZNCC the refinement achieved against the reference, `NaN` where it
    /// did not evaluate the member.
    pub zncc: f64,
}

impl ClusterMember {
    /// Whether the refinement kept this member or made it the reference.
    pub fn is_kept_or_reference(&self) -> bool {
        self.status == STATUS_REFERENCE || self.status == STATUS_KEPT
    }
}

/// A cluster with a member near a pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyCluster {
    /// The cluster's index in the file.
    pub cluster: u32,
    /// How far its nearest member in the queried image is from the pixel, in px.
    pub distance_px: f64,
    /// That member, by its global index in the file.
    pub member: usize,
    /// Every member of the cluster, by global index, in file order.
    pub members: std::ops::Range<usize>,
}

/// The clusters of a cluster-patches `.matches` file, re-indexed onto a
/// reconstruction's images by name, with a 2D index over each image's members.
///
/// Built once per file and reconstruction, and read by every query: the file
/// holds nothing of the reconstruction's points, so a query needs no version
/// of it.
pub struct MatchesClusters {
    cluster_starts: Vec<u32>,
    members: Vec<ClusterMember>,
    member_cluster: Vec<u32>,
    reference_members: Vec<Option<u32>>,
    per_image: Vec<(Vec<u32>, Option<PointCloud<f64, 2>>)>,
}

impl std::fmt::Debug for MatchesClusters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchesClusters")
            .field("clusters", &self.cluster_count())
            .field("members", &self.members.len())
            .finish()
    }
}

impl MatchesClusters {
    /// Index the clusters of `matches` onto the images named `image_names`, in
    /// the reconstruction's order.
    ///
    /// A member is matched to a reconstruction image by its name in the file's
    /// image table. A member in an image the reconstruction does not hold is
    /// kept, with no image, and is never found by a query.
    pub fn new(matches: &MatchesData, image_names: &[&str]) -> Result<Self, MatchesClustersError> {
        let clusters = matches
            .clusters
            .as_ref()
            .ok_or(MatchesClustersError::NoClusters)?;
        let (Some(positions), Some(shapes)) = (
            clusters.member_positions.as_ref(),
            clusters.member_affine_shapes.as_ref(),
        ) else {
            return Err(MatchesClustersError::NoMemberGeometry);
        };
        let to_recon: Vec<Option<u32>> = matches
            .image_names
            .iter()
            .map(|name| image_names.iter().position(|n| n == name).map(|i| i as u32))
            .collect();
        let patches = matches.cluster_patches.as_ref();
        let m = clusters.member_images.len();
        let members: Vec<ClusterMember> = (0..m)
            .map(|k| ClusterMember {
                image: to_recon
                    .get(clusters.member_images[k] as usize)
                    .copied()
                    .flatten(),
                position: [f64::from(positions[[k, 0]]), f64::from(positions[[k, 1]])],
                shape: [
                    [f64::from(shapes[[k, 0, 0]]), f64::from(shapes[[k, 0, 1]])],
                    [f64::from(shapes[[k, 1, 0]]), f64::from(shapes[[k, 1, 1]])],
                ],
                status: patches.map_or(STATUS_NOT_EVALUATED, |p| p.member_status[k]),
                zncc: patches.map_or(f64::NAN, |p| f64::from(p.member_zncc[k])),
            })
            .collect();
        let cluster_starts: Vec<u32> = clusters.cluster_starts.to_vec();
        let mut member_cluster = vec![0u32; m];
        for c in 0..cluster_starts.len().saturating_sub(1) {
            for slot in
                &mut member_cluster[cluster_starts[c] as usize..cluster_starts[c + 1] as usize]
            {
                *slot = c as u32;
            }
        }
        let reference_members = match patches {
            Some(p) => p
                .reference_members
                .iter()
                .map(|&r| (r != CLUSTER_REFERENCE_UNREFINABLE).then_some(r))
                .collect(),
            None => vec![None; cluster_starts.len().saturating_sub(1)],
        };
        let per_image = (0..image_names.len() as u32)
            .map(|image| {
                let rows: Vec<u32> = (0..m as u32)
                    .filter(|&k| members[k as usize].image == Some(image))
                    .collect();
                let xy: Vec<f64> = rows
                    .iter()
                    .flat_map(|&k| members[k as usize].position)
                    .collect();
                let cloud = (!rows.is_empty()).then(|| PointCloud::new(&xy, rows.len()));
                (rows, cloud)
            })
            .collect();
        Ok(Self {
            cluster_starts,
            members,
            member_cluster,
            reference_members,
            per_image,
        })
    }

    /// How many clusters the file holds.
    pub fn cluster_count(&self) -> usize {
        self.cluster_starts.len().saturating_sub(1)
    }

    /// How many images the clusters were indexed onto.
    pub fn image_count(&self) -> usize {
        self.per_image.len()
    }

    /// The member at global index `k`.
    pub fn member(&self, k: usize) -> &ClusterMember {
        &self.members[k]
    }

    /// The reference member of `cluster`, or `None` when the refinement named
    /// none.
    pub fn reference(&self, cluster: u32) -> Option<u32> {
        self.reference_members
            .get(cluster as usize)
            .copied()
            .flatten()
    }

    /// The clusters with a member in `image` within `radius_px` of `pixel`,
    /// nearest first, each with its member there that is nearest the pixel.
    pub fn near(&self, image: u32, pixel: [f64; 2], radius_px: f64) -> Vec<NearbyCluster> {
        let Some((rows, Some(cloud))) = self.per_image.get(image as usize) else {
            return Vec::new();
        };
        let (_, hits) = cloud.within_radius(&pixel, 1, radius_px);
        // The nearest member per cluster, in the order the clusters were first
        // met, so a tie in distance keeps that order through the sort.
        let mut nearest: Vec<(u32, f64, usize)> = Vec::new();
        for hit in hits {
            let k = rows[hit as usize] as usize;
            let p = self.members[k].position;
            let d = ((p[0] - pixel[0]).powi(2) + (p[1] - pixel[1]).powi(2)).sqrt();
            let c = self.member_cluster[k];
            match nearest.iter_mut().find(|(cluster, _, _)| *cluster == c) {
                Some(entry) => {
                    if d < entry.1 {
                        entry.1 = d;
                        entry.2 = k;
                    }
                }
                None => nearest.push((c, d, k)),
            }
        }
        nearest.sort_by(|a, b| a.1.total_cmp(&b.1));
        nearest
            .into_iter()
            .map(|(cluster, distance_px, member)| NearbyCluster {
                cluster,
                distance_px,
                member,
                members: self.cluster_starts[cluster as usize] as usize
                    ..self.cluster_starts[cluster as usize + 1] as usize,
            })
            .collect()
    }
}
