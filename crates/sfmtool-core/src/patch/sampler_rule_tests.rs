// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The sampler rule across the patch kernels: every kernel chooses a view's
//! sampler from the same observation, a view the rule leaves on
//! `bilinear_mip` renders the same tile as before, bit for bit, and a moved
//! view renders a different one.

use nalgebra::{Point3, Vector3};

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::sampler::{Sampler, SamplerChoice};
use crate::camera::warp_map::patch_grid_jacobian;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_subpixel::{fuse_patch_bitmap, KeypointSubpixelParams};
use crate::patch::normal_refine::{
    normalized_stack, view_samplers, LevelContext, ProjectedImage, ViewSamplers,
};
use crate::progress::Progress;

const R: u32 = 24;

/// A pinhole camera 640 × 480 with focal length `f`.
fn camera(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: f,
            focal_length_y: f,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// A camera at `x` along the world `x` axis, with the axes of the world.
fn pose(x: f64) -> RigidTransform {
    RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-x, 0.0, 0.0])
}

/// A deterministic textured RGB photograph and its pyramid.
fn pyramid() -> ImageU8Pyramid {
    let (w, h) = (640u32, 480u32);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 7 + y * 13) ^ (x * y);
            data.extend_from_slice(&[(v % 251) as u8, (v % 241) as u8, ((x + y) % 256) as u8]);
        }
    }
    ImageU8Pyramid::build(&ImageU8::new(w, h, 3, data), 6)
}

/// A square of half-extent 0.5 at depth 4, turned `tilt_deg` about `y`.
fn tilted(tilt_deg: f64) -> OrientedPatch {
    let t = tilt_deg.to_radians();
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(t.sin(), 0.0, t.cos()),
        Vector3::y(),
        [0.5, 0.5],
    )
}

/// Where `patch`'s centre projects in each view.
fn keypoints(patch: &OrientedPatch, views: &[ProjectedImage<'_>]) -> Vec<[f64; 2]> {
    views
        .iter()
        .map(|v| {
            v.camera
                .project_homogeneous(v.cam_from_world, patch.center.coords, patch.w)
                .expect("projects")
        })
        .collect()
}

/// The sampler the bench and Track View read for an observation: the rule on
/// the patch re-anchored at its keypoint, through `patch_grid_jacobian` at `R`.
fn bench_choice(patch: &OrientedPatch, view: &ProjectedImage<'_>, kp: [f64; 2]) -> Sampler {
    let frame = patch
        .anchored_at_keypoint(view.camera, view.cam_from_world, kp)
        .unwrap_or_else(|| patch.clone());
    SamplerChoice::per_view().for_jacobian(patch_grid_jacobian(
        &frame,
        view.camera,
        view.cam_from_world,
        R as usize,
    ))
}

/// Three views of a square turned 65° away: two at a 500 px focal length,
/// which compress it about five times and twice as much across the turn, and
/// one at 100 px, which compresses it under `√2` and so reads level 0. The
/// rule moves the first two and leaves the third, every kernel agrees, and in
/// one render of the three only the moved views' tiles change.
#[test]
fn unmoved_views_render_the_same_tiles_and_moved_views_do_not() {
    let cams = [camera(500.0), camera(100.0), camera(500.0)];
    let poses = [pose(-0.2), pose(0.0), pose(0.2)];
    let pyr = pyramid();
    let views: Vec<ProjectedImage<'_>> = cams
        .iter()
        .zip(&poses)
        .map(|(camera, cam_from_world)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid: &pyr,
        })
        .collect();
    let patch = tilted(65.0);
    let kps = keypoints(&patch, &views);
    let some_kps: Vec<Option<[f64; 2]>> = kps.iter().map(|&k| Some(k)).collect();

    let rule = SamplerChoice::per_view();
    let frozen = view_samplers(rule, &patch, &views, Some(&some_kps), R);
    assert_eq!(
        frozen,
        [
            Sampler::Anisotropic,
            Sampler::BilinearMip,
            Sampler::Anisotropic
        ]
    );
    for (i, view) in views.iter().enumerate() {
        assert_eq!(frozen[i], bench_choice(&patch, view, kps[i]), "view {i}");
    }

    let ctx = LevelContext {
        kept: vec![0, 1, 2],
        pixels: (0..(R * R) as usize).collect(),
        weights: vec![1.0; (R * R) as usize],
    };
    let render = |samplers| {
        normalized_stack(
            &patch,
            &ctx,
            &views,
            R,
            samplers,
            Some(&some_kps),
            &Progress::none(),
        )
        .expect("every view covers the tile")
    };
    let (by_rule, channels) = render(ViewSamplers::Each(rule));
    let (by_mip, _) = render(ViewSamplers::Frozen(&[Sampler::BilinearMip; 3]));
    let n = (R * R) as usize;
    let tile = |raw: &[f32], v: usize| raw[v * channels * n..(v + 1) * channels * n].to_vec();
    assert_eq!(
        tile(&by_rule, 1),
        tile(&by_mip, 1),
        "the unmoved view moved"
    );
    assert_ne!(
        tile(&by_rule, 0),
        tile(&by_mip, 0),
        "a moved view is unchanged"
    );
    assert_ne!(
        tile(&by_rule, 2),
        tile(&by_mip, 2),
        "a moved view is unchanged"
    );
}

/// Where the rule moves no view, the fused bitmap is bit for bit the one
/// `bilinear_mip` fuses; where it moves them, it is not.
#[test]
fn the_fuse_changes_only_where_a_view_moves() {
    let cams = [camera(500.0), camera(500.0), camera(500.0)];
    let poses = [pose(-0.2), pose(0.0), pose(0.2)];
    let pyr = pyramid();
    let views: Vec<ProjectedImage<'_>> = cams
        .iter()
        .zip(&poses)
        .map(|(camera, cam_from_world)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid: &pyr,
        })
        .collect();
    let fuse = |patch: &OrientedPatch, sampler: SamplerChoice| {
        let kps = keypoints(patch, &views);
        let params = KeypointSubpixelParams {
            resolution: R,
            sampler,
            ..KeypointSubpixelParams::default()
        };
        fuse_patch_bitmap(patch, &views, &[0, 1, 2], &kps, &params).expect("fuses")
    };
    let mip = SamplerChoice::Fixed(Sampler::BilinearMip);
    let facing = tilted(0.0);
    assert_eq!(
        fuse(&facing, SamplerChoice::per_view()),
        fuse(&facing, mip),
        "no view moved, so the bitmap must not change"
    );
    let turned = tilted(65.0);
    assert_ne!(fuse(&turned, SamplerChoice::per_view()), fuse(&turned, mip));
    assert_eq!(
        fuse(&turned, SamplerChoice::per_view()),
        fuse(&turned, SamplerChoice::Fixed(Sampler::Anisotropic)),
        "every view of the turned square moves"
    );
}

/// A stored frame whose axes are not perpendicular is read by the rule as
/// stored, re-anchored on the keypoint, whatever frame a kernel then renders
/// through. The kernels that rebuild an orthonormal frame around the stored
/// `v` axis (the sub-pixel refiner, the localizer, normal refinement) would
/// see this square facing the camera and compressed alike on both axes, and
/// would keep `bilinear_mip`; the stored frame is sheared 30°, which
/// `bilinear_mip` reads at level 3 and the sampler rule moves. The fuse
/// follows the stored frame, as the bench does.
#[test]
fn every_kernel_reads_the_rule_from_the_stored_frame() {
    let cams = [camera(500.0), camera(500.0)];
    let poses = [pose(-0.1), pose(0.1)];
    let pyr = pyramid();
    let views: Vec<ProjectedImage<'_>> = cams
        .iter()
        .zip(&poses)
        .map(|(camera, cam_from_world)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid: &pyr,
        })
        .collect();
    let s = 30f64.to_radians();
    let sheared = OrientedPatch {
        center: Point3::new(0.0, 0.0, -4.0),
        u_axis: Vector3::x(),
        v_axis: Vector3::new(s.sin(), s.cos(), 0.0),
        half_extent: [0.5, 0.5],
        w: 1.0,
    };
    let rebuilt = OrientedPatch::from_center_normal(
        sheared.center,
        sheared.normal(),
        sheared.v_axis,
        sheared.half_extent,
    );
    let kps = keypoints(&sheared, &views);
    let rule = SamplerChoice::per_view();
    for (view, &kp) in views.iter().zip(&kps) {
        assert_eq!(bench_choice(&sheared, view, kp), Sampler::Anisotropic);
        assert_eq!(bench_choice(&rebuilt, view, kp), Sampler::BilinearMip);
        assert_eq!(
            rule.for_observation(&sheared, view.camera, view.cam_from_world, Some(kp), R),
            Sampler::Anisotropic
        );
    }
    let fuse = |sampler: SamplerChoice| {
        let params = KeypointSubpixelParams {
            resolution: R,
            sampler,
            ..KeypointSubpixelParams::default()
        };
        fuse_patch_bitmap(&sheared, &views, &[0, 1], &kps, &params).expect("fuses")
    };
    assert_eq!(
        fuse(rule),
        fuse(SamplerChoice::Fixed(Sampler::Anisotropic)),
        "the fuse read the rule from a rebuilt frame"
    );
}
