use super::*;
use nalgebra::{Matrix3, Vector3};

fn test_intrinsics() -> Matrix3<f64> {
    Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
}

/// Smoke test: match_image_pair works with per-image dimensions and
/// forward motion. The actual both-epipoles safety logic is tested in
/// rectification::tests::test_rectification_safe_requires_both_epipoles_outside.
#[test]
fn test_match_image_pair_forward_motion_asymmetric_dimensions() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 1.0);

    // 4 features at the image corners, each with a unique descriptor.
    // Avoid the epipole (320, 240) where polar matching is degenerate.
    let n = 4;
    let positions = [
        100.0, 80.0, // top-left
        500.0, 80.0, // top-right
        100.0, 400.0, // bottom-left
        500.0, 400.0, // bottom-right
    ];
    let mut descriptors = vec![0u8; n * 128];
    for i in 0..n {
        descriptors[i * 128] = (i * 50) as u8;
    }

    let matches = match_image_pair(
        &k,
        &k,
        &r,
        &t1,
        &r,
        &t2,
        640,
        480, // image 1: normal size
        10,
        10, // image 2: different size
        &positions,
        &descriptors,
        n,
        &positions,
        &descriptors,
        n,
        128,
        5,
        None,
        50,
        None,
        None,
        None,
    );

    assert_eq!(
        matches.len(),
        n,
        "All {n} features should match (got {})",
        matches.len()
    );
    for (idx1, idx2, dist) in &matches {
        assert_eq!(idx1, idx2, "Feature {idx1} should match itself");
        assert_eq!(*dist, 0.0);
    }
}
