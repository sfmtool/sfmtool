use super::*;

#[test]
fn test_argsort_by_y() {
    // 3 points: (0, 10), (0, 5), (0, 15)
    let kpts = [0.0, 10.0, 0.0, 5.0, 0.0, 15.0];
    let indices = argsort_by_y(&kpts, 3);
    assert_eq!(indices, vec![1, 0, 2]); // sorted by Y: 5, 10, 15
}

#[test]
fn test_one_way_sweep_basic() {
    // 3 features in image 1, 3 in image 2
    // Sorted by Y already
    let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

    // Descriptors: make feature 0 in img1 closest to feature 0 in img2, etc.
    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    // Feature 0: descs1[0] = [1, 0, ...], descs2[0] = [1, 0, ...]
    descs1[0] = 1;
    descs2[0] = 1;
    // Feature 1: descs1[128] = [0, 2, ...], descs2[128] = [0, 2, ...]
    descs1[129] = 2;
    descs2[129] = 2;
    // Feature 2: descs1[256] = [0, 0, 3, ...], descs2[256] = [0, 0, 3, ...]
    descs1[258] = 3;
    descs2[258] = 3;

    let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, None);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[&0].0, 0);
    assert_eq!(matches[&1].0, 1);
    assert_eq!(matches[&2].0, 2);
}

#[test]
fn test_mutual_match_basic() {
    // Two sets of identical features
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 20;
    descs1[258] = 30;
    descs2[258] = 30;

    let mutual = mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, None);

    assert_eq!(mutual.len(), 3);
    // All should be identity matches
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_empty_inputs() {
    let matches = match_one_way_sweep(&[], &[], 0, &[0.0, 1.0], &[0u8; 128], 1, 30, None);
    assert!(matches.is_empty());

    let matches = match_one_way_sweep(&[0.0, 1.0], &[0u8; 128], 1, &[], &[], 0, 30, None);
    assert!(matches.is_empty());
}

#[test]
fn test_mutual_match_geometric_basic() {
    use nalgebra::{Matrix3, Vector3};

    // Same setup as test_mutual_match_basic but with identity affines and identity cameras
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 20;
    descs1[258] = 30;
    descs2[258] = 30;

    // Identity affines for all features
    let affines1 = [
        1.0, 0.0, 0.0, 1.0, // feature 0
        1.0, 0.0, 0.0, 1.0, // feature 1
        1.0, 0.0, 0.0, 1.0, // feature 2
    ];
    let affines2 = [
        1.0, 0.0, 0.0, 1.0, // feature 0
        1.0, 0.0, 0.0, 1.0, // feature 1
        1.0, 0.0, 0.0, 1.0, // feature 2
    ];

    // Identity cameras (same K, identity R, zero t)
    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t = Vector3::zeros();
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);
    let config = GeometricFilterConfig::default();

    let mutual = mutual_best_match_sweep_geometric(
        &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 128, 3, None, &geom, &config,
    );

    assert_eq!(mutual.len(), 3);
    // All should be identity matches
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_one_way_sweep_with_threshold_rejects() {
    // 1 feature each, with descriptors far apart
    let kpts1 = [0.0, 1.0];
    let kpts2 = [0.0, 1.0];

    let descs1 = vec![0u8; 128];
    let descs2 = vec![100u8; 128];
    // Distance = sqrt(100^2 * 128) ≈ 1131.4

    // Tight threshold: should reject
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(10.0));
    assert!(
        matches.is_empty(),
        "Tight threshold should reject distant descriptors"
    );

    // Generous threshold: should accept
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(2000.0));
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_one_way_sweep_with_threshold_accepts() {
    // With a generous threshold, all matches should pass
    let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 1;
    descs2[0] = 1;
    descs1[129] = 2;
    descs2[129] = 2;
    descs1[258] = 3;
    descs2[258] = 3;

    let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, Some(2000.0));
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_one_way_sweep_multiple_thresholds() {
    let kpts1 = [0.0, 1.0];
    let kpts2 = [0.0, 1.0];

    let descs1 = vec![0u8; 128];
    let mut descs2 = vec![0u8; 128];
    descs2[0] = 3;
    descs2[1] = 4;
    // Distance = sqrt(9 + 16) = 5.0

    // Threshold below distance: no match
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(4.0));
    assert!(matches.is_empty());

    // Threshold at distance: accepted (<=)
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(5.0));
    assert_eq!(matches.len(), 1);

    // Threshold above distance: accepted
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(100.0));
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_asymmetric_feature_counts() {
    // 5 features in img1, 3 in img2
    let n1 = 5;
    let n2 = 3;
    let mut kpts1 = vec![0.0; n1 * 2];
    let mut kpts2 = vec![0.0; n2 * 2];
    for i in 0..n1 {
        kpts1[i * 2 + 1] = i as f64;
    }
    for i in 0..n2 {
        kpts2[i * 2 + 1] = i as f64;
    }

    let mut descs1 = vec![0u8; n1 * 128];
    let mut descs2 = vec![0u8; n2 * 128];
    // Make each descriptor unique
    for i in 0..n1 {
        descs1[i * 128] = (i * 10) as u8;
    }
    for i in 0..n2 {
        descs2[i * 128] = (i * 10) as u8;
    }

    let matches = match_one_way_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 5, None);
    // First 3 features in img1 should match perfectly to img2 features
    assert!(matches.len() >= 3);
    for i in 0..3 {
        assert_eq!(matches[&i].0, i);
        assert_eq!(matches[&i].1, 0.0);
    }
}

#[test]
fn test_mutual_match_asymmetric() {
    // 4 features in img1, 6 in img2
    let n1 = 4;
    let n2 = 6;
    let mut kpts1 = vec![0.0; n1 * 2];
    let mut kpts2 = vec![0.0; n2 * 2];
    for i in 0..n1 {
        kpts1[i * 2 + 1] = i as f64;
    }
    for i in 0..n2 {
        kpts2[i * 2 + 1] = i as f64;
    }

    let mut descs1 = vec![0u8; n1 * 128];
    let mut descs2 = vec![0u8; n2 * 128];
    // First n1 features match between the two sets
    for i in 0..n1 {
        descs1[i * 128] = (i * 20 + 10) as u8;
        descs2[i * 128] = (i * 20 + 10) as u8;
    }
    // Extra features in img2 are distinct
    for i in n1..n2 {
        descs2[i * 128] = 200;
        descs2[i * 128 + 1] = (i * 30) as u8;
    }

    let mutual = mutual_best_match_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 128, 10, None);

    assert_eq!(mutual.len(), n1);
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_mutual_match_with_threshold() {
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    // Feature 0: identical (dist=0), Feature 1: close (dist=5), Feature 2: far (dist≈1131)
    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 23; // diff = 3
    descs2[130] = 4; // diff = 4, distance = 5
    descs1[258] = 30;
    descs2[256..384].fill(100); // very far

    // Threshold that accepts feature 0 and 1, rejects feature 2
    let mutual =
        mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, Some(10.0));
    assert_eq!(mutual.len(), 2);
}

#[test]
fn test_larger_window_sizes() {
    // 10 features, test window sizes 5, 10, 20
    let n = 10;
    let mut kpts = vec![0.0; n * 2];
    for i in 0..n {
        kpts[i * 2 + 1] = (i * 10) as f64;
    }
    let mut descs = vec![0u8; n * 128];
    for i in 0..n {
        descs[i * 128] = (i * 15) as u8;
    }

    for window in [5, 10, 20] {
        let mutual = mutual_best_match_sweep(&kpts, &descs, n, &kpts, &descs, n, 128, window, None);
        // All features should match themselves regardless of window size
        assert_eq!(
            mutual.len(),
            n,
            "Window size {window}: expected {n} matches, got {}",
            mutual.len()
        );
        for (idx1, idx2, dist) in &mutual {
            assert_eq!(idx1, idx2);
            assert_eq!(*dist, 0.0);
        }
    }
}

#[test]
fn test_match_one_way_sweep_geometric_rejects_bad_orientation() {
    use nalgebra::{Matrix3, Vector3};

    // 3 features in image 1, 3 in image 2, sorted by Y
    let kpts1 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];
    let kpts2 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];

    // Make all descriptors identical so descriptor matching always succeeds
    let descs1 = vec![1u8; 3 * 128];
    let descs2 = vec![1u8; 3 * 128];

    // Query affines: identity orientation
    let affines1 = [
        5.0, 0.0, 0.0, 5.0, // feature 0
        5.0, 0.0, 0.0, 5.0, // feature 1
        5.0, 0.0, 0.0, 5.0, // feature 2
    ];
    // Target affines: perpendicular orientation (first col rotated 90 degrees)
    let affines2 = [
        0.0, 5.0, 5.0, 0.0, // feature 0: first col = (0, 5), perpendicular
        0.0, 5.0, 5.0, 0.0, // feature 1
        0.0, 5.0, 5.0, 0.0, // feature 2
    ];

    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    let matches = match_one_way_sweep_geometric(
        &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 3, None, &geom, &config,
    );

    // All candidates should be rejected due to bad orientation
    assert!(
        matches.is_empty(),
        "All matches should be rejected due to perpendicular orientation"
    );
}

/// A permissive filter admits the whole window, so the geometric path must
/// return exactly what the plain one does: the filter is the *only* difference
/// between them, which is the premise of sharing one body.
///
/// This does not check the index remap — with everything admitted, the passing
/// offsets are the identity `[0, 1, 2, …]` and a remap bug would be invisible.
/// The test below covers that.
#[test]
fn geometric_path_matches_plain_when_the_filter_admits_everything() {
    use nalgebra::{Matrix3, Vector3};

    let n = 12;
    let mut kpts = Vec::new();
    let mut descs1 = Vec::new();
    let mut descs2 = Vec::new();
    let mut affines = Vec::new();
    for i in 0..n {
        kpts.push(300.0 + (i % 3) as f64);
        kpts.push(i as f64 * 4.0);
        for d in 0..128 {
            descs1.push(((i * 7 + d * 3) % 251) as u8);
            descs2.push(((i * 7 + d * 3 + 1) % 251) as u8);
        }
        // One shared shape, so orientation and size agree for every candidate.
        affines.extend_from_slice(&[3.0, 0.0, 0.0, 3.0]);
    }

    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.7, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let permissive = GeometricFilterConfig {
        max_angle_difference: 180.0,
        // 90° => cos == 0, so every candidate short-circuits stage 2 as "rays
        // too parallel to judge". This is the permissive extreme despite
        // reading like a strict one, and it is load-bearing: the ratio bounds
        // below cannot cover a failed triangulation or a non-positive depth,
        // both of which reject before any ratio is compared.
        min_triangulation_angle: 90.0,
        geometric_size_ratio_min: 0.0,
        geometric_size_ratio_max: f64::INFINITY,
    };

    let plain = match_one_way_sweep(&kpts, &descs1, n, &kpts, &descs2, n, 5, None);
    let filtered = match_one_way_sweep_geometric(
        &kpts,
        &descs1,
        n,
        &kpts,
        &descs2,
        n,
        &affines,
        &affines,
        5,
        None,
        &geom,
        &permissive,
    );

    assert!(!plain.is_empty(), "the plain sweep should find matches");
    assert_eq!(plain, filtered);
}

/// When the filter rejects part of a window, the returned index must name the
/// candidate's place in the *sorted* array, not its offset within the surviving
/// subset — the remap the merged one-way sweep performs through its passing
/// offsets.
#[test]
fn filtered_windows_report_sorted_indices_not_offsets_within_the_survivors() {
    use nalgebra::{Matrix3, Vector3};

    // One query, four candidates all at the same Y so the whole window is live.
    let kpts1 = [320.0, 10.0];
    let kpts2 = [320.0, 10.0, 320.0, 10.0, 320.0, 10.0, 320.0, 10.0];

    let descs1 = vec![0u8; 128];
    let mut descs2 = vec![0u8; 4 * 128];
    // Candidate 0 is the nearest descriptor, candidate 3 the next nearest;
    // candidates 0-2 will be filtered out, leaving 3 as the only survivor.
    descs2[128] = 90; // candidate 1
    descs2[256] = 60; // candidate 2
    descs2[3 * 128] = 30; // candidate 3

    let query_affine = [4.0, 0.0, 0.0, 4.0];
    let perpendicular = [0.0, 4.0, 4.0, 0.0];
    let mut affines2 = Vec::new();
    for _ in 0..3 {
        affines2.extend_from_slice(&perpendicular);
    }
    affines2.extend_from_slice(&query_affine);

    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    let matches = match_one_way_sweep_geometric(
        &kpts1,
        &descs1,
        1,
        &kpts2,
        &descs2,
        4,
        &query_affine,
        &affines2,
        4,
        None,
        &geom,
        &config,
    );

    let (target, _) = matches[&0];
    // 3, the survivor's index in the candidate array. Reporting its offset
    // within the survivors would give 0 — which is also a real candidate here,
    // and the one the unfiltered sweep would have picked.
    assert_eq!(target, 3);

    // The unfiltered sweep does pick candidate 0, so the assertion above is
    // discriminating rather than vacuous.
    let unfiltered = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 4, 4, None);
    assert_eq!(unfiltered[&0].0, 0);
}

/// The in-plane rotation, in radians, between the two views of
/// [`asymmetric_geometry`].
const RELATIVE_ROLL: f64 = 0.6;

/// Build the two-view setup the backward-sweep test needs: different
/// intrinsics, a non-zero baseline, and — the part that matters — a relative
/// rotation about the **optical axis**, so `geom.swapped()` differs from `geom`
/// in the one quantity stage 1 of the filter reads.
///
/// A rotation about X or Y would not do: `r_2d` is the upper-left 2×2 of
/// `R2 · R1ᵀ`, and swapping transposes it, so an out-of-plane rotation gives a
/// near-symmetric `r_2d` whose transpose is nearly itself. The swap would then
/// be undetectable no matter what else the fixture did.
fn asymmetric_geometry() -> StereoPairGeometry {
    use nalgebra::{Matrix3, Vector3};

    let mut k1 = Matrix3::identity();
    k1[(0, 0)] = 500.0;
    k1[(1, 1)] = 500.0;
    k1[(0, 2)] = 320.0;
    k1[(1, 2)] = 240.0;

    let mut k2 = Matrix3::identity();
    k2[(0, 0)] = 800.0;
    k2[(1, 1)] = 780.0;
    k2[(0, 2)] = 400.0;
    k2[(1, 2)] = 300.0;

    let r1 = Matrix3::identity();
    let (s, c) = RELATIVE_ROLL.sin_cos();
    let r2 = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);

    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.8, 0.1, 0.05);

    StereoPairGeometry::new(&k1, &k2, &r1, &r2, &t1, &t2)
}

/// The bidirectional geometric sweep must run its backward pass with the
/// geometry *and* the affine arrays swapped, since that pass reverses the roles
/// of query and target. Nothing else pins this: the older bidirectional test is
/// symmetric in every input (equal affines, equal cameras, `n1 == n2`), so both
/// swaps are no-ops there and dropping either one leaves it green.
///
/// The expectation is composed from the one-way entry point, called twice with
/// the roles exchanged by hand — a different code path from the one under test,
/// so this is a cross-check rather than a restatement.
#[test]
fn bidirectional_geometric_sweep_reverses_geometry_and_affines_for_the_backward_pass() {
    let n1 = 5;
    let n2 = 3;
    let desc_len = 4;

    // Y-sorted and distinct, so the internal argsort is the identity and the
    // one-way calls below see exactly the arrays the bidirectional pass builds.
    let kpts1 = [
        300.0, 10.0, 310.0, 20.0, 295.0, 30.0, 330.0, 40.0, 305.0, 50.0,
    ];
    let kpts2 = [420.0, 12.0, 380.0, 26.0, 410.0, 44.0];

    let descs1: Vec<u8> = vec![
        10, 0, 0, 0, //
        0, 20, 0, 0, //
        0, 0, 30, 0, //
        0, 0, 0, 40, //
        11, 0, 0, 0, //
    ];
    let descs2: Vec<u8> = vec![
        12, 0, 0, 0, //
        0, 22, 0, 0, //
        0, 0, 0, 41, //
    ];

    // Image 1's shapes are axis-aligned; image 2's are the same shapes seen
    // through the relative roll. Stage 1 compares the query shape *rotated by
    // `r_2d`* against the candidate's, so this pair agrees in the forward
    // direction and — only with `r_2d` transposed — in the backward one too.
    // Run backward with the un-swapped geometry and the shapes come out 2 ×
    // RELATIVE_ROLL apart, well outside the tolerance below.
    let (s, c) = RELATIVE_ROLL.sin_cos();
    let mut affines1 = [0.0; 5 * 4];
    for f in 0..5 {
        affines1[f * 4] = 4.0;
        affines1[f * 4 + 3] = 4.0;
    }
    let mut affines2 = [0.0; 3 * 4];
    for f in 0..3 {
        affines2[f * 4] = 4.0 * c;
        affines2[f * 4 + 1] = -4.0 * s;
        affines2[f * 4 + 2] = 4.0 * s;
        affines2[f * 4 + 3] = 4.0 * c;
    }

    let geom = asymmetric_geometry();
    let config = GeometricFilterConfig {
        // Tight enough that 2 × RELATIVE_ROLL (≈ 68.8°) is rejected while the
        // correctly-oriented pairing (0°) passes.
        max_angle_difference: 20.0,
        // 90° => every candidate short-circuits stage 2 as "rays too parallel
        // to judge", leaving stage 1's orientation check as the only filter.
        // That is what makes this test about the geometry swap and nothing else.
        min_triangulation_angle: 90.0,
        geometric_size_ratio_min: 0.1,
        geometric_size_ratio_max: 10.0,
    };
    let window = 4;

    let mutual = mutual_best_match_sweep_geometric(
        &kpts1, &descs1, n1, &kpts2, &descs2, n2, &affines1, &affines2, desc_len, window, None,
        &geom, &config,
    );

    // Forward: image 1 queries image 2, with image 1's shapes as the query side.
    let forward = match_one_way_sweep_geometric(
        &kpts1, &descs1, n1, &kpts2, &descs2, n2, &affines1, &affines2, window, None, &geom,
        &config,
    );
    // Backward: the roles reverse, so both the geometry and the affine arrays do.
    let backward = match_one_way_sweep_geometric(
        &kpts2,
        &descs2,
        n2,
        &kpts1,
        &descs1,
        n1,
        &affines2,
        &affines1,
        window,
        None,
        &geom.swapped(),
        &config,
    );

    let mut expected: Vec<(usize, usize, f64)> = forward
        .iter()
        .filter(|(idx1, (idx2, _))| backward.get(idx2).map(|&(back, _)| back) == Some(**idx1))
        .map(|(&idx1, &(idx2, dist))| (idx1, idx2, dist))
        .collect();
    expected.sort_by_key(|a| (a.0, a.1));

    let mut got = mutual.clone();
    got.sort_by_key(|a| (a.0, a.1));

    assert!(
        !expected.is_empty(),
        "the fixture must produce mutual matches, or the comparison below is vacuous"
    );
    assert_eq!(got, expected);
}
