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

/// `find_infinity_tracks` with `finite_horizon` defaulted to the camera
/// extents, as the production path does.
#[allow(clippy::too_many_arguments)]
fn find(
    dirs: &[Vector3<f64>],
    descriptors: &[[u8; 128]],
    image_index: &[u32],
    feature_index: &[u32],
    camera_centers: &[Point3<f64>],
    focal_max: &[f64],
    params: &InfinityParams,
) -> Vec<InfinityTrack> {
    let finite_horizon = camera_extents(camera_centers);
    find_infinity_tracks(
        dirs,
        descriptors,
        image_index,
        feature_index,
        camera_centers,
        focal_max,
        params,
        finite_horizon,
    )
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

    let tracks = find(
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
    match track.classification.class {
        Classification::Infinity(dir) => assert!(
            (dir.coords.norm() - 1.0).abs() < 1e-9,
            "position is a unit direction"
        ),
        other => panic!("parallel rays → point at infinity, got {other:?}"),
    }
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

    let tracks = find(
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

    let tracks = find(
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

    let tracks = find(
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
    let tracks = find(
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
    let tracks = find(
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

    let tracks = find(
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
    match tracks[0].classification.class {
        Classification::Finite(recovered) => assert!(
            (recovered.coords - point.coords).norm() < 1e-6,
            "triangulated position {:?} near true point {:?}",
            recovered,
            point
        ),
        other => panic!("wide parallax → finite point, got {other:?}"),
    }
}
