use super::*;

fn classify(dirs: &[Vector3<f64>], centers: &[Point3<f64>], finite_horizon: f64) -> Classification {
    let sigma = vec![1.0e-3; dirs.len()];
    classify_rays_at_infinity(
        dirs,
        centers,
        &sigma,
        DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
        finite_horizon,
    )
    .class
}

#[test]
fn classify_rays_finite_when_well_conditioned() {
    // Rays converging on a near point with wide parallax → finite via the
    // condition pre-filter, regardless of finite_horizon.
    let target = Point3::new(0.0, 0.0, 3.0);
    let centers = vec![
        Point3::new(-2.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| (target.coords - c.coords).normalize())
        .collect();
    assert!(matches!(
        classify(&dirs, &centers, 1e6),
        Classification::Finite(_)
    ));
}

#[test]
fn classify_rays_confident_infinity_with_wide_baseline() {
    // Parallel rays seen across a wide perpendicular baseline: resolvable far
    // past finite_horizon, so the parallel solve is a *confident* infinity.
    let dirs = vec![Vector3::z(), Vector3::z(), Vector3::z()];
    let centers = vec![
        Point3::new(-50.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(50.0, 0.0, 0.0),
    ];
    match classify(&dirs, &centers, 100.0) {
        Classification::Infinity(dir) => {
            assert!((dir.coords - Vector3::z()).norm() < 1e-9)
        }
        other => panic!("expected Infinity, got {other:?}"),
    }
}

#[test]
fn classify_rays_indeterminate_without_baseline() {
    // Near-parallel rays from nearly-coincident cameras: the perpendicular
    // baseline can't reach finite_horizon, so neither finite nor infinity is
    // earned — indeterminate. (This is the 97221 / 102031 single-stop case.)
    let target = Point3::new(0.0, 0.0, 1000.0);
    let centers = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.01, 0.0, 0.0)];
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| (target.coords - c.coords).normalize())
        .collect();
    assert!(matches!(
        classify(&dirs, &centers, 100.0),
        Classification::Indeterminate
    ));
}

#[test]
fn camera_extents_is_bbox_diagonal() {
    let centers = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(3.0, 4.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ];
    assert!((camera_extents(&centers) - 5.0).abs() < 1e-9);
    assert_eq!(camera_extents(&[]), 0.0);
}

#[test]
fn classify_leaves_well_conditioned_points_finite() {
    // demo() points sit ~1 unit from cameras ~5 units away: wide parallax.
    let recon = SfmrReconstruction::demo(100);
    let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    let n_inf = classified.points.iter().filter(|p| p.w == 0.0).count();
    assert_eq!(n_inf, 0, "well-conditioned demo points must stay finite");
    // The cached count tracks the actual w=0 count after classify.
    assert_eq!(classified.infinity_point_count, n_inf);
}

#[test]
fn classify_detects_far_point_as_infinity() {
    // demo(1): point 0 observed by images 0 and 1. Pushing it far away
    // collapses the parallax below the noise floor.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::new(0.0, 0.0, 10_000.0);
    recon.points[0].error = 0.5;

    let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(classified.points[0].is_at_infinity());
    // The cached count reflects the freshly-classified point.
    assert_eq!(classified.infinity_point_count, 1);
    // The stored direction is a unit vector.
    assert!((classified.points[0].position.coords.norm() - 1.0).abs() < 1e-9);
    // normal is zeroed for a point at infinity.
    assert_eq!(classified.points[0].normal, Vector3::zeros());
}

#[test]
fn classify_uses_per_point_reprojection_error() {
    // A distant, ill-conditioned point: its inverse-depth z-score sits
    // above the cutoff at the 1 px noise floor (depth still resolvable), but
    // a large per-point reprojection error inflates the angular noise enough
    // to push it below the cutoff. The per-point error folds into σ.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::new(0.0, 0.0, 300.0);

    recon.points[0].error = 0.5;
    let with_floor = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(
        !with_floor.points[0].is_at_infinity(),
        "z-score above the cutoff at the 1 px floor — point stays finite"
    );
    assert_eq!(with_floor.infinity_point_count, 0);

    recon.points[0].error = 20.0;
    let with_error = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(
        with_error.points[0].is_at_infinity(),
        "the point's own 20 px error inflates σ, dropping the z-score below the cutoff"
    );
    assert_eq!(with_error.infinity_point_count, 1);
}

#[test]
fn classify_idempotent_on_infinity_points() {
    let mut recon = SfmrReconstruction::demo(20);
    for pt in &mut recon.points {
        pt.position = Point3::from(pt.position.coords.normalize());
        pt.w = 0.0;
    }
    let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(classified.points.iter().all(|p| p.is_at_infinity()));
    assert_eq!(classified.infinity_point_count, classified.points.len());
    for pt in &classified.points {
        assert!((pt.position.coords.norm() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn materialize_makes_every_point_finite() {
    let mut recon = SfmrReconstruction::demo(50);
    for pt in &mut recon.points {
        pt.position = Point3::from(pt.position.coords.normalize());
        pt.w = 0.0;
        pt.normal = Vector3::zeros();
    }
    let materialised = recon.materialize_points_at_infinity();
    assert!(materialised.points.iter().all(|p| p.w == 1.0));
    assert_eq!(materialised.infinity_point_count, 0);
}

#[test]
fn materialize_places_point_along_its_direction() {
    let mut recon = SfmrReconstruction::demo(1);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    recon.points[0].position = Point3::from(dir);
    recon.points[0].w = 0.0;

    let materialised = recon.materialize_points_at_infinity();
    let pt = &materialised.points[0];
    assert_eq!(pt.w, 1.0);
    // The point lies on the ray origin + t·d, so (pt - origin) ∥ d.
    let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
    let mut origin = Vector3::zeros();
    for c in &centers {
        origin += c.coords;
    }
    origin /= centers.len() as f64;
    let offset = pt.position.coords - origin;
    let perp = offset - offset.dot(&dir) * dir;
    assert!(perp.norm() < 1e-6, "materialised point must lie along d");
    assert!(offset.dot(&dir) > 0.0, "placed in the +d half-line");
}

#[test]
fn materialize_renormalises_non_unit_directions() {
    // A w = 0 point whose stored direction drifted off the unit sphere
    // must materialise to the same finite point as the unit direction:
    // the placement geometry depends only on the normalised direction.
    let mut unit = SfmrReconstruction::demo(1);
    unit.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
    unit.points[0].w = 0.0;

    let mut scaled = SfmrReconstruction::demo(1);
    scaled.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 7.5));
    scaled.points[0].w = 0.0;

    let from_unit = unit.materialize_points_at_infinity();
    let from_scaled = scaled.materialize_points_at_infinity();
    let delta =
        (from_unit.points[0].position.coords - from_scaled.points[0].position.coords).norm();
    assert!(
        delta < 1e-9,
        "non-unit direction must materialise identically"
    );
}

#[test]
fn materialize_leaves_zero_direction_point_untouched() {
    // A malformed w = 0 point with a zero-length direction cannot be
    // placed — it is left at w = 0 rather than producing a bogus finite
    // coordinate.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::from(Vector3::zeros());
    recon.points[0].w = 0.0;

    let materialised = recon.materialize_points_at_infinity();
    assert!(materialised.points[0].is_at_infinity());
}

#[test]
fn materialize_is_inverse_free_but_round_trips_classification() {
    // A genuine infinity point, materialised then reclassified, returns to
    // infinity: its parallax is still nil.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
    recon.points[0].w = 0.0;
    recon.points[0].error = 0.5;

    let materialised = recon.materialize_points_at_infinity();
    assert!(!materialised.points[0].is_at_infinity());
    assert_eq!(materialised.infinity_point_count, 0);

    let reclassified = materialised.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(reclassified.points[0].is_at_infinity());
    assert_eq!(reclassified.infinity_point_count, 1);
}

/// Attach a patch frame (and a bitmap) to point 0 of a demo reconstruction.
fn with_patch(recon: &mut SfmrReconstruction, half_u: f32, half_v: f32) {
    let n = recon.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    u[[0, 0]] = half_u;
    v[[0, 1]] = half_v;
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);
    let mut b = Array4::<u8>::zeros((n, 2, 2, 4));
    b.index_axis_mut(Axis(0), 0).fill(200);
    recon.patch_bitmaps_y_x_rgba = Some(b);
}

#[test]
fn classify_rescales_patch_frame_to_angular_extents() {
    // Demoting a far finite point to infinity turns its anchor into a unit
    // direction; the world-unit patch half-vectors must become angular
    // extents (divided by the demotion-time distance from the camera-cloud
    // centroid, preserving apparent size), tangent to the direction sphere,
    // with u x v along -d — the format's infinity-patch convention.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::new(0.0, 0.0, 10_000.0);
    recon.points[0].error = 0.5;
    with_patch(&mut recon, 50.0, 25.0);

    let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
    let origin = camera_cloud_centroid(&centers);
    let dist = (recon.points[0].position.coords - origin.coords).norm();

    let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(classified.points[0].is_at_infinity());
    let d = classified.points[0].position.coords.normalize();
    let get = |a: &Option<Array2<f32>>| {
        let a = a.as_ref().unwrap();
        Vector3::new(
            f64::from(a[[0, 0]]),
            f64::from(a[[0, 1]]),
            f64::from(a[[0, 2]]),
        )
    };
    let u = get(&classified.patch_u_halfvec_xyz);
    let v = get(&classified.patch_v_halfvec_xyz);
    // Tangent to the direction sphere.
    assert!(u.dot(&d).abs() < 1e-6 * u.norm());
    assert!(v.dot(&d).abs() < 1e-6 * v.norm());
    // Right-handed with the outward normal along -d.
    assert!(u.cross(&v).dot(&d) < 0.0);
    // Apparent half-sizes preserved (the swap may exchange the axes).
    let mut sizes = [u.norm(), v.norm()];
    sizes.sort_by(f64::total_cmp);
    assert!((sizes[0] - 25.0 / dist).abs() / (25.0 / dist) < 0.01);
    assert!((sizes[1] - 50.0 / dist).abs() / (50.0 / dist) < 0.01);
    // The bitmap is untouched by a demotion that keeps the patch.
    assert_eq!(
        classified.patch_bitmaps_y_x_rgba.as_ref().unwrap()[[0, 0, 0, 0]],
        200
    );
}

#[test]
fn classify_leaves_finite_point_patches_untouched() {
    let mut recon = SfmrReconstruction::demo(1);
    with_patch(&mut recon, 5.0, 5.0);
    let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
    assert!(!classified.points[0].is_at_infinity());
    assert_eq!(
        classified.patch_u_halfvec_xyz.as_ref().unwrap()[[0, 0]],
        5.0
    );
}

#[test]
fn materialize_rescales_patch_frame_to_world_extents() {
    // The inverse boundary crossing: angular extents on the direction sphere
    // become world-unit extents at the placement depth.
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
    recon.points[0].w = 0.0;
    recon.points[0].normal = Vector3::zeros();
    with_patch(&mut recon, 0.01, 0.005);

    let materialised = recon.materialize_points_at_infinity();
    let pt = &materialised.points[0];
    assert_eq!(pt.w, 1.0);
    let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
    let origin = camera_cloud_centroid(&centers);
    let t = (pt.position.coords - origin.coords).norm();
    let u = materialised.patch_u_halfvec_xyz.as_ref().unwrap();
    assert!((f64::from(u[[0, 0]]) - 0.01 * t).abs() / (0.01 * t) < 1e-6);
}

#[test]
fn classify_then_materialize_preserves_apparent_patch_size() {
    // Round trip: world extent / distance is invariant across the demotion
    // and the re-materialisation (the depths differ, the ratio must not).
    let mut recon = SfmrReconstruction::demo(1);
    recon.points[0].position = Point3::new(0.0, 0.0, 10_000.0);
    recon.points[0].error = 0.5;
    with_patch(&mut recon, 50.0, 25.0);

    let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
    let origin = camera_cloud_centroid(&centers);
    let dist0 = (recon.points[0].position.coords - origin.coords).norm();
    let apparent0 = 50.0 / dist0;

    let round = recon
        .classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX)
        .materialize_points_at_infinity();
    assert!(!round.points[0].is_at_infinity());
    let dist1 = (round.points[0].position.coords - origin.coords).norm();
    let ua = round.patch_u_halfvec_xyz.as_ref().unwrap();
    let va = round.patch_v_halfvec_xyz.as_ref().unwrap();
    let norm_row = |a: &Array2<f32>| {
        (f64::from(a[[0, 0]]).powi(2) + f64::from(a[[0, 1]]).powi(2) + f64::from(a[[0, 2]]).powi(2))
            .sqrt()
    };
    // The demotion may swap the axes (handedness) — compare the larger one.
    let apparent1 = norm_row(ua).max(norm_row(va)) / dist1;
    assert!(
        (apparent0 - apparent1).abs() / apparent0 < 0.01,
        "apparent size must survive the round trip: {apparent0} vs {apparent1}"
    );
}
