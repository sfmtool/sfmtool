// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A bitmap the bench renders and a writer stores is the render of the patch
//! frame as the value stores it: rendering the stored value's bitmaps again
//! (`--drop-patch-bitmaps` then `--add-patch-bitmaps`) gives back the same
//! bytes after `refine_normals`, after `refit_points`, and after the viewer's
//! tilt, spin or resize followed by its live evaluation and a commit.
//!
//! A step builds its frame in `f64` and the value stores it as `f32`
//! half-vectors. Rendered through the unrounded frame, a sample moves by a
//! few `f32` steps of the frame, and over many samples a few land on the
//! other side of an 8-bit rounding or of a sampler step. The scene is sized
//! so that, without the rounding at render time, some do: a grid of points on
//! the textured plane, each with a 48 x 48 bitmap.

use super::grown_plane_normal::{camera, textured_plane, wide_centres};
use super::scene::{fixture_points, with_columns, Scene, PLANE_Z, WORLD};
use super::*;
use crate::patch::cloud::PatchCloud;
use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::stored_bitmap::{render_patch_cloud_bitmaps, UnreferencedPoints};

/// The bitmap side: large enough that a render through the unrounded frame
/// differs from one through the stored frame somewhere in the grid.
const BITMAP_R: usize = 48;

/// The textured plane seen from six views out to 50 degrees.
fn plane_scene() -> Scene {
    Scene::looking_at(&wide_centres(), WORLD, camera(), textured_plane)
}

/// A 21 x 21 grid of points on the plane around [`WORLD`], 0.04 apart.
fn grid() -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    for i in -10..=10 {
        for j in -10..=10 {
            points.push(Point3::new(
                WORLD.x + 0.04 * f64::from(i),
                WORLD.y + 0.04 * f64::from(j),
                PLANE_Z,
            ));
        }
    }
    points
}

/// `grid()` over `scene`, every point on a patch facing the cameras, with a
/// zero bitmap column of side [`BITMAP_R`] and a reference column naming
/// none, so each writer records the observation it rendered.
fn grid_value(scene: &Scene) -> SfmrReconstruction {
    let mut recon = with_columns(fixture_points(scene, &grid()), BITMAP_R);
    recon.point_set.reference_observations = Some(vec![-1; recon.point_count()]);
    recon.rebuild_derived_fields();
    recon
}

/// The same value with every patch turned `by` degrees off the plane, about
/// an axis that leaves no half-vector component an `f32` holds exactly.
fn tilted_grid_value(scene: &Scene, by: f64) -> SfmrReconstruction {
    let mut recon = grid_value(scene);
    let turn = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.5, 0.0)),
        by.to_radians(),
    );
    let set = &mut recon.point_set;
    for column in [&mut set.patch_u_halfvec_xyz, &mut set.patch_v_halfvec_xyz] {
        let column = column.as_mut().expect("a framed value");
        for mut row in column.rows_mut() {
            let v = turn * Vector3::new(f64::from(row[0]), f64::from(row[1]), f64::from(row[2]));
            row.assign(&ndarray::arr1(&[v.x as f32, v.y as f32, v.z as f32]));
        }
    }
    recon.rebuild_derived_fields();
    recon
}

/// How many bytes of `recon`'s bitmap column differ from a render of its
/// stored frames, keypoints and reference observations, as
/// `--add-patch-bitmaps` renders them, over the points that name a
/// reference; and how many points that is.
fn bytes_off_a_render_of_the_stored_value(
    scene: &Scene,
    recon: &SfmrReconstruction,
) -> (usize, usize) {
    let views: Vec<Option<ProjectedImage<'_>>> = scene.views().into_iter().map(Some).collect();
    let cloud = PatchCloud::from_stored_patch_frames(recon).expect("a framed value");
    let params = KeypointSubpixelParams {
        resolution: BITMAP_R as u32,
        ..FitOptions::default().refine
    };
    let again = render_patch_cloud_bitmaps(
        &cloud,
        recon,
        &views,
        &params,
        UnreferencedPoints::Skip,
        None,
        &Progress::none(),
    )
    .expect("not cancelled");
    let stored = recon
        .point_set
        .patch_bitmaps_y_x_rgba
        .as_deref()
        .expect("a bitmap column");
    let references = recon
        .point_set
        .reference_observations
        .as_deref()
        .expect("a reference column");
    let mut bytes = 0;
    let mut named = 0;
    for (p, &reference) in references.iter().enumerate() {
        if reference < 0 {
            continue;
        }
        named += 1;
        let row = stored.index_axis(ndarray::Axis(0), p);
        let render = again.bitmaps.index_axis(ndarray::Axis(0), p);
        bytes += row
            .iter()
            .zip(render.iter())
            .filter(|(a, b)| a != b)
            .count();
    }
    assert!(named > 0, "no point names a reference");
    (bytes, named)
}

#[test]
fn refine_normals_stores_the_render_of_the_frame_it_stores() {
    let scene = plane_scene();
    // Three starting tilts, so that over all of them some sample of a render
    // through the unrounded turned frame lands one level off.
    let (mut bytes, mut named) = (0, 0);
    for by in [15.0, 25.0, 35.0] {
        let recon = tilted_grid_value(&scene, by);
        let (out, report) = refine_normals(
            &recon,
            &scene.views(),
            &RefineNormalsOptions::default(),
            &Progress::none(),
        )
        .expect("the run refines");
        assert!(report.turned() > grid().len() / 2, "{by}: {report:?}");
        let (b, n) = bytes_off_a_render_of_the_stored_value(&scene, &out);
        bytes += b;
        named += n;
    }
    assert_eq!(bytes, 0, "{bytes} bytes off over {named} bitmaps");
}

#[test]
fn refit_points_stores_the_render_of_the_frame_it_stores() {
    let scene = plane_scene();
    // A frame longer along `v` than along `u`: the fit squares it, to the
    // geometric mean of the two half-lengths, which no `f32` holds.
    let mut recon = tilted_grid_value(&scene, 10.0);
    let v = recon
        .point_set
        .patch_v_halfvec_xyz
        .as_mut()
        .expect("a framed value");
    v.mapv_inplace(|x| x * 1.13);
    let (out, report) = refit_points(
        &recon,
        &scene.views(),
        &RefitPointsOptions {
            fit: test_fit_options(),
            ..RefitPointsOptions::default()
        },
        &Progress::none(),
    )
    .expect("the run refits");
    assert!(report.refit() > grid().len() / 2, "{report:?}");
    let (bytes, named) = bytes_off_a_render_of_the_stored_value(&scene, &out);
    assert_eq!(bytes, 0, "{bytes} bytes off over {named} bitmaps");
}

/// Put every point of `recon` on the bench, apply `step` to it, run the
/// viewer's live evaluation (which renders the bitmap the step dropped) and
/// commit it, one point after another, and return the value written.
fn committed_after(
    scene: &Scene,
    recon: &SfmrReconstruction,
    step: impl Fn(&EditableTrack, &EditedReconstruction) -> EditableTrack,
) -> SfmrReconstruction {
    let views = scene.views();
    let mut edited = EditedReconstruction::new(Arc::new(recon.clone()));
    for p in 0..recon.point_count() as u32 {
        let (bench, created) =
            create_track(&Bench::new(), &edited, p, &CreateTrackOptions::default())
                .expect("a live point");
        let track = bench.track(&created.label).expect("just put on");
        let stepped = step(track, &edited);
        let (evaluated, _) = evaluate_rendering_bitmap(
            &stepped,
            &edited,
            &views,
            &EvaluateOptions::default(),
            &FitOptions::default(),
            &Progress::none(),
        )
        .expect("the evaluation runs");
        if let Ok((next, _)) = commit(&edited, &evaluated) {
            edited = next;
        }
    }
    edited.materialize().0
}

#[test]
fn a_viewer_tilt_then_commit_stores_the_render_of_the_frame_it_stores() {
    let scene = plane_scene();
    let recon = grid_value(&scene);
    let out = committed_after(&scene, &recon, |track, edited| {
        let turn = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.5, 0.0)),
            12f64.to_radians(),
        );
        tilt_patch(track, edited, turn * Vector3::new(0.0, 0.0, -1.0))
            .expect("the tilt runs")
            .0
    });
    let (bytes, named) = bytes_off_a_render_of_the_stored_value(&scene, &out);
    assert!(named > grid().len() / 2, "{named} committed bitmaps");
    assert_eq!(bytes, 0, "{bytes} bytes off over {named} bitmaps");
}

#[test]
fn a_viewer_spin_and_resize_then_commit_store_the_render_of_the_frame_they_store() {
    let scene = plane_scene();
    let recon = grid_value(&scene);
    let out = committed_after(&scene, &recon, |track, edited| {
        let (spun, _) = spin_patch(track, edited, 0.3).expect("the spin runs");
        let half = spun
            .track()
            .and_then(|t| t.patch_frame.as_ref())
            .expect("a patch frame")
            .half_extent[0];
        resize_patch(&spun, edited, half * 1.0371, None)
            .expect("the resize runs")
            .0
    });
    let (bytes, named) = bytes_off_a_render_of_the_stored_value(&scene, &out);
    assert!(named > grid().len() / 2, "{named} committed bitmaps");
    assert_eq!(bytes, 0, "{bytes} bytes off over {named} bitmaps");
}
