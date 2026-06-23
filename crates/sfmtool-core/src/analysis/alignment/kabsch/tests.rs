use super::*;
use approx::assert_relative_eq;

#[test]
fn test_zero_points_is_error() {
    let result = kabsch_algorithm(&[], &[], 0);
    assert!(result.is_err());
}

#[test]
fn test_single_point_gives_translation() {
    // With a single point, rotation is identity, scale is 1, and
    // translation bridges the two points.
    let source = [1.0, 2.0, 3.0];
    let target = [4.0, 5.0, 6.0];
    let t = kabsch_algorithm(&source, &target, 1).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
    assert_relative_eq!(t.translation, Vector3::new(3.0, 3.0, 3.0), epsilon = 1e-10);
}

#[test]
fn test_coincident_source_points() {
    // All identical source points — rotation/scale undetermined,
    // defaults to identity rotation, scale 1, translation to target centroid.
    let source = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let target = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let t = kabsch_algorithm(&source, &target, 2).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
    // Target centroid is (5.5, 6.5, 7.5), source centroid is (1, 2, 3)
    assert_relative_eq!(t.translation, Vector3::new(4.5, 4.5, 4.5), epsilon = 1e-10);
}

#[test]
fn test_collinear_points_identity() {
    // Collinear points mapped to themselves — should recover identity transform.
    let points = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let t = kabsch_algorithm(&points, &points, 3).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(t.translation, Vector3::zeros(), epsilon = 1e-10);
    assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
}

#[test]
fn test_collinear_points_with_transform() {
    // Collinear points with a known translation + scale (no rotation component
    // along the line).  The cross-axis rotation is unconstrained but the
    // transform should still map source to target correctly.
    let source = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let target: Vec<f64> = source.iter().map(|&x| x * 2.0 + 5.0).collect();

    let t = kabsch_algorithm(&source, &target, 3).unwrap();

    // Verify the transform maps source to target
    for i in 0..3 {
        let s = nalgebra::Point3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
        let expected = Vector3::new(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
        let transformed = t.apply_to_point(&s);
        assert_relative_eq!(transformed.coords, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_identity_alignment() {
    // Same non-collinear points should give identity transform
    let points = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let t = kabsch_algorithm(&points, &points, 4).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(t.translation, Vector3::zeros(), epsilon = 1e-10);
    assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
}

#[test]
fn test_known_rotation() {
    // 90-degree rotation around Z axis: (x, y, z) -> (-y, x, z)
    let source = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let target = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let t = kabsch_algorithm(&source, &target, 3).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
    assert_relative_eq!(t.translation.norm(), 0.0, epsilon = 1e-10);

    // Check that rotation maps source to target
    for i in 0..3 {
        let s = Vector3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
        let expected = Vector3::new(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
        let transformed = rot * s;
        assert_relative_eq!(transformed, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_known_scale() {
    // Scale by factor of 2 with translation
    let source = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let target: Vec<f64> = source
        .chunks(3)
        .flat_map(|p| [p[0] * 2.0 + 1.0, p[1] * 2.0 + 2.0, p[2] * 2.0 + 3.0])
        .collect();

    let t = kabsch_algorithm(&source, &target, 4).unwrap();
    let rot = t.rotation.to_rotation_matrix();

    assert_relative_eq!(t.scale, 2.0, epsilon = 1e-10);
    assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(t.translation, Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-10);
}

#[test]
fn test_random_similarity_round_trip() {
    use rand::rngs::StdRng;
    use rand::RngExt;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(12345);
    let n_points = 20;

    for _ in 0..10 {
        // Random rotation via Rodrigues formula on a random axis/angle
        let axis = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        )
        .normalize();
        let angle: f64 = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        let k = Matrix3::new(
            0.0, -axis[2], axis[1], axis[2], 0.0, -axis[0], -axis[1], axis[0], 0.0,
        );
        let expected_rot = Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * k * k;

        let expected_scale: f64 = rng.random_range(0.1..10.0);
        let expected_trans = Vector3::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        // Generate random 3D source points
        let mut source = vec![0.0f64; n_points * 3];
        for v in source.iter_mut() {
            *v = rng.random_range(-5.0..5.0);
        }

        // Apply similarity transform: target = scale * R * source + translation
        let mut target = vec![0.0f64; n_points * 3];
        for i in 0..n_points {
            let s = Vector3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
            let t = expected_scale * (expected_rot * s) + expected_trans;
            target[i * 3] = t[0];
            target[i * 3 + 1] = t[1];
            target[i * 3 + 2] = t[2];
        }

        let result = kabsch_algorithm(&source, &target, n_points).unwrap();
        let rot = result.rotation.to_rotation_matrix();

        assert_relative_eq!(result.scale, expected_scale, epsilon = 1e-8);
        assert_relative_eq!(rot, expected_rot, epsilon = 1e-8);
        assert_relative_eq!(result.translation, expected_trans, epsilon = 1e-8);
    }
}
