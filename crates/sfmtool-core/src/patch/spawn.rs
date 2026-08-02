// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate track spawning: congeal a new track at an in-plane offset from an
//! existing patched point.
//!
//! A **candidate** is a synthetic patch placed at `X_p + du·hu_p + dv·hv_p` —
//! the parent's frame translated along its own half-extent vectors, so a request
//! speaks the patch's own scale and stays in its plane. The candidate is then put
//! through exactly the machinery a real track is congealed with: discrete
//! cross-view [localization](super::keypoint_localize), [sub-pixel
//! refinement](super::keypoint_subpixel), triangulation of the refined keypoints'
//! rays, and the same acceptance gates. What comes back is a vetted
//! `(position, views, keypoints)` per request.
//!
//! Callers pick the offsets and assemble the survivors; the primitive does
//! neither. Surfel-normal expansion places candidates along directions its
//! adjacency graph left empty and feeds the surviving *positions* back in as
//! extra fit neighbours; densification places them over image regions no
//! observation claims and assembles the survivors into new reconstruction
//! tracks.
//!
//! A candidate that fails a gate is reported with the stage that killed it
//! rather than dropped, so a caller can budget and diagnose on the counts.
//!
//! See `specs/core/candidate-track-spawning.md`.

use nalgebra::{Point3, Vector3};

use super::cloud::{OrientedPatch, PatchCloud};
use super::keypoint_localize::{
    localize_patch_cloud_keypoints, project_unclipped, KeypointLocalizeParams,
};
use super::keypoint_subpixel::{refine_patch_cloud_keypoints, KeypointSubpixelParams};
use super::normal_refine::ProjectedImage;
use crate::reconstruction::triangulation::triangulate_batch;

/// Tunables for [`spawn_candidate_tracks`]. The defaults are the spec's.
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnParams {
    /// The `R×R` sampling grid every stage scores on, as in localization.
    pub resolution: u32,
    /// Localizer search half-width, in patch-grid px
    /// ([`KeypointLocalizeParams::search`]).
    pub search: f64,
    /// Localizer shift gate, in source-image px
    /// ([`KeypointLocalizeParams::max_shift_px`]). Wider than the localizer's own
    /// default: a candidate is a hypothesis, and the offset that produced it is
    /// exactly the displacement the search has to be free to walk back.
    pub max_shift_px: f64,
    /// Sub-pixel refinement outer sweeps
    /// ([`KeypointSubpixelParams::max_outer_sweeps`]). `0` skips refinement, so
    /// the discrete keypoints go straight to triangulation.
    pub subpixel_sweeps: u32,
    /// Surviving-view floor: a candidate with fewer surviving views is
    /// [`SpawnStatus::TooFewViews`].
    pub min_views: u32,
    /// Acceptance gate on the RMS reprojection error of the triangulated
    /// position against the refined keypoints, in source-image px.
    pub max_reproj_rms_px: f64,
}

impl Default for SpawnParams {
    fn default() -> Self {
        Self {
            resolution: 24,
            search: 6.0,
            max_shift_px: 8.0,
            subpixel_sweeps: 1,
            min_views: 3,
            max_reproj_rms_px: 2.0,
        }
    }
}

/// Why a candidate is or is not a track. The discriminants are the wire values
/// the binding reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpawnStatus {
    /// Passed every gate: the triangulated position and its observations stand.
    Spawned = 0,
    /// Fewer than [`SpawnParams::min_views`] views survived localization.
    TooFewViews = 1,
    /// Non-finite triangulation, a degenerate (unobservable-depth) solve, or a
    /// point that is not in front of every surviving camera.
    BadTriangulation = 2,
    /// The RMS reprojection error against the refined keypoints exceeds
    /// [`SpawnParams::max_reproj_rms_px`].
    HighReproj = 3,
}

/// The result of one [`spawn_candidate_tracks`] batch: per-candidate arrays plus
/// the surviving observations in CSR layout.
#[derive(Debug, Clone, Default)]
pub struct SpawnedTracks {
    /// Per candidate, the [`SpawnStatus`] discriminant.
    pub status: Vec<u8>,
    /// Per candidate, the triangulated position. `NaN` for a candidate that is
    /// not [`SpawnStatus::Spawned`].
    pub positions: Vec<[f64; 3]>,
    /// Per candidate, the requested centre `X_c` — always filled, so a caller can
    /// measure how far the congealed position moved from what it asked for.
    pub requested_centers: Vec<[f64; 3]>,
    /// Per candidate, the RMS reprojection error of [`positions`](Self::positions)
    /// against the refined keypoints. `NaN` for a candidate that died before the
    /// reprojection stage.
    pub reproj_rms_px: Vec<f64>,
    /// Per candidate, how many views survived to triangulation (`0` when none
    /// did).
    pub n_views: Vec<u32>,
    /// CSR row boundaries over the observation arrays, `n + 1` entries. A
    /// candidate that never reached triangulation owns an empty range.
    pub obs_offsets: Vec<u32>,
    /// Per surviving observation, the image index, ascending within a candidate.
    pub obs_view_indexes: Vec<u32>,
    /// Per surviving observation, the refined keypoint in source-image px,
    /// parallel to [`obs_view_indexes`](Self::obs_view_indexes).
    pub obs_keypoints_xy: Vec<[f64; 2]>,
}

impl SpawnedTracks {
    /// An all-empty result with `n + 1` CSR offsets, ready to be filled.
    fn with_capacity(n: usize) -> Self {
        Self {
            status: Vec::with_capacity(n),
            positions: Vec::with_capacity(n),
            requested_centers: Vec::with_capacity(n),
            reproj_rms_px: Vec::with_capacity(n),
            n_views: Vec::with_capacity(n),
            obs_offsets: Vec::with_capacity(n + 1),
            obs_view_indexes: Vec::new(),
            obs_keypoints_xy: Vec::new(),
        }
    }

    /// Number of candidates.
    pub fn len(&self) -> usize {
        self.status.len()
    }

    pub fn is_empty(&self) -> bool {
        self.status.is_empty()
    }
}

/// Spawn one candidate track per `(parent, offset)` request.
///
/// `views` is one [`ProjectedImage`] per image (indexed by image index);
/// `cloud` holds the parents, indexed by `parents`; `offsets_uv[i]` is
/// candidate `i`'s `(du, dv)` in units of its parent's half-extent vectors;
/// `view_sets[i]` lists the views candidate `i` is searched in (typically its
/// parent's). A repeated parent is fine — candidates are independent.
///
/// Every candidate goes through the batch kernels as **one** batch each (one
/// patch cloud, one localization call, one refinement call, one triangulation
/// call). The kernels parallelize internally, so this composition adds no
/// parallelism of its own, and its results are deterministic.
///
/// Candidates are built **finite** (`w = 1`): the offsets displace the frame in
/// world units and the output is a triangulated 3D position, neither of which a
/// point at infinity has. Callers filter their parents accordingly (the Python
/// binding rejects an infinity parent outright).
///
/// # Panics
///
/// Panics if `offsets_uv` or `view_sets` is not parallel to `parents`, or a
/// parent index is out of range for `cloud`. The binding validates these and
/// raises `ValueError` instead.
pub fn spawn_candidate_tracks(
    views: &[ProjectedImage<'_>],
    cloud: &PatchCloud,
    parents: &[u32],
    offsets_uv: &[[f64; 2]],
    view_sets: &[Vec<u32>],
    params: &SpawnParams,
) -> SpawnedTracks {
    let n = parents.len();
    assert_eq!(
        offsets_uv.len(),
        n,
        "offsets_uv must be parallel to parents"
    );
    assert_eq!(view_sets.len(), n, "view_sets must be parallel to parents");

    let mut out = SpawnedTracks::with_capacity(n);
    out.obs_offsets.push(0);
    if n == 0 {
        return out;
    }

    // ── The candidate cloud: each parent's frame translated to its own X_c ──
    let mut candidates = PatchCloud {
        patches: Vec::with_capacity(n),
        point_indexes: (0..n as u32).collect(),
    };
    for (i, &p) in parents.iter().enumerate() {
        let parent = cloud.patch(p as usize);
        let [du, dv] = offsets_uv[i];
        let center = parent.center
            + parent.u_axis * (du * parent.half_extent[0])
            + parent.v_axis * (dv * parent.half_extent[1]);
        candidates.patches.push(OrientedPatch::new(
            center,
            parent.u_axis,
            parent.v_axis,
            parent.half_extent,
        ));
    }

    // ── 1. Discrete localization, seeded at each view's projection of X_c ──
    let localize_params = KeypointLocalizeParams {
        resolution: params.resolution,
        search: params.search,
        max_shift_px: params.max_shift_px,
        // A candidate carries its parent's view set, which is small; every view
        // congeals against every other (the uncapped consensus).
        basis_max_views: 0,
        ..KeypointLocalizeParams::default()
    };
    let localized = localize_patch_cloud_keypoints(
        &candidates,
        views,
        view_sets,
        None,
        None,
        &localize_params,
        None,
    );

    // The view floor decides what the later stages spend anything on: a candidate
    // below it gets an empty refinement/triangulation set and never renders again.
    let alive: Vec<bool> = localized
        .iter()
        .map(|l| l.views.len() as u32 >= params.min_views)
        .collect();

    // ── 2. Sub-pixel refinement, seeded at the localized keypoints ──
    let refine_sets: Vec<Vec<u32>> = localized
        .iter()
        .zip(&alive)
        .map(|(l, &ok)| if ok { l.views.clone() } else { Vec::new() })
        .collect();
    let observations: Vec<(Vec<u32>, Vec<[f64; 2]>)> = if params.subpixel_sweeps > 0 {
        let seeds: Vec<Vec<Option<[f64; 2]>>> = localized
            .iter()
            .zip(&alive)
            .map(|(l, &ok)| {
                if ok {
                    l.keypoints.iter().map(|&k| Some(k)).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let refine_params = KeypointSubpixelParams {
            resolution: params.resolution,
            max_outer_sweeps: params.subpixel_sweeps,
            ..KeypointSubpixelParams::default()
        };
        refine_patch_cloud_keypoints(
            &candidates,
            views,
            &refine_sets,
            Some(&seeds),
            &refine_params,
        )
        .into_iter()
        .map(|r| (r.views, r.keypoints))
        .collect()
    } else {
        localized
            .iter()
            .zip(&alive)
            .map(|(l, &ok)| {
                if ok {
                    (l.views.clone(), l.keypoints.clone())
                } else {
                    (Vec::new(), Vec::new())
                }
            })
            .collect()
    };

    // Observations go out in view-index order regardless of the order the caller
    // listed the view set in, so the CSR layout is a stable contract.
    let observations: Vec<(Vec<u32>, Vec<[f64; 2]>)> = observations
        .into_iter()
        .map(|(v, k)| {
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_by_key(|&j| v[j]);
            (
                order.iter().map(|&j| v[j]).collect(),
                order.iter().map(|&j| k[j]).collect(),
            )
        })
        .collect();

    // ── 3. Triangulation from the refined keypoints' world rays ──
    let mut dirs: Vec<Vector3<f64>> = Vec::new();
    let mut centers: Vec<Point3<f64>> = Vec::new();
    let mut tri_offsets: Vec<usize> = Vec::with_capacity(n + 1);
    tri_offsets.push(0);
    for (obs_views, keypoints) in &observations {
        for (&image, &kp) in obs_views.iter().zip(keypoints) {
            let view = &views[image as usize];
            let ray = view.camera.pixel_to_ray(kp[0], kp[1]);
            // Camera-to-world rotation carries the canonical (-Z forward) ray into
            // the world frame the triangulator solves in.
            let rot = view.cam_from_world.to_rotation_matrix();
            dirs.push(rot.transpose() * Vector3::new(ray[0], ray[1], ray[2]));
            centers.push(view.cam_from_world.inverse_translation_origin());
        }
        tri_offsets.push(dirs.len());
    }
    let triangulated = triangulate_batch(&dirs, &centers, &tri_offsets);

    // ── 4. Gates, each recording its casualty ──
    const NAN3: [f64; 3] = [f64::NAN; 3];
    for i in 0..n {
        let c = candidates.patches[i].center;
        out.requested_centers.push([c.x, c.y, c.z]);
        let (obs_views, keypoints) = &observations[i];
        out.n_views.push(obs_views.len() as u32);

        if !alive[i] {
            out.status.push(SpawnStatus::TooFewViews as u8);
            out.positions.push(NAN3);
            out.reproj_rms_px.push(f64::NAN);
            out.obs_offsets.push(out.obs_view_indexes.len() as u32);
            continue;
        }

        // Everything below reached triangulation, so its observations are reported
        // whatever the verdict — that is what makes a casualty diagnosable.
        out.obs_view_indexes.extend_from_slice(obs_views);
        out.obs_keypoints_xy.extend_from_slice(keypoints);
        out.obs_offsets.push(out.obs_view_indexes.len() as u32);

        let tri = triangulated[i];
        let position = tri.point;
        // A degenerate solve (parallel rays, unobservable depth) reports an
        // infinite condition number; its "position" is the minimum-norm point in
        // the observable subspace, not a triangulation.
        if !position.coords.iter().all(|c| c.is_finite())
            || !tri.condition_number.is_finite()
            || !tri.in_front_of_all_cameras
        {
            out.status.push(SpawnStatus::BadTriangulation as u8);
            out.positions.push(NAN3);
            out.reproj_rms_px.push(f64::NAN);
            continue;
        }

        // RMS reprojection of the triangulated position against the keypoints that
        // produced it. A view the position no longer projects into at all (behind
        // the camera, or outside the model's domain) makes the RMS infinite, which
        // the gate below rejects.
        let mut sum_sq = 0.0_f64;
        for (&image, &kp) in obs_views.iter().zip(keypoints) {
            match project_unclipped(&views[image as usize], &position, 1.0) {
                Some((px, py)) => {
                    sum_sq += (px - kp[0]).powi(2) + (py - kp[1]).powi(2);
                }
                None => {
                    sum_sq = f64::INFINITY;
                    break;
                }
            }
        }
        let rms = (sum_sq / obs_views.len() as f64).sqrt();
        out.reproj_rms_px.push(rms);
        if rms > params.max_reproj_rms_px {
            out.status.push(SpawnStatus::HighReproj as u8);
            out.positions.push(NAN3);
            continue;
        }

        out.status.push(SpawnStatus::Spawned as u8);
        out.positions.push([position.x, position.y, position.z]);
    }

    out
}

#[cfg(test)]
mod tests;
