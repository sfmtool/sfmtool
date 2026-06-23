use super::*;

#[test]
fn u8_kernel_matches_scalar() {
    // Exercise several lengths, including non-multiples of 16 and > 16.
    for &len in &[0usize, 1, 7, 15, 16, 17, 31, 128, 130] {
        let a: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i * 13 + 200) as u8).collect();
        assert_eq!(
            u8_dist_sq(&a, &b),
            u8_dist_sq_scalar(&a, &b),
            "mismatch at len {len}"
        );
    }
}

/// `u8_dist_sq` only exercises the *auto-selected* path (AVX2 on most CI
/// hosts), so test each SIMD kernel directly against the scalar reference —
/// otherwise a bug in the SSE2 path would never be compared on an AVX2 box.
#[test]
#[cfg(target_arch = "x86_64")]
fn simd_kernels_match_scalar() {
    for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 128, 130] {
        let a: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i * 13 + 200) as u8).collect();
        let scalar = u8_dist_sq_scalar(&a, &b);
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 confirmed available.
            assert_eq!(unsafe { u8_dist_sq_sse2(&a, &b) }, scalar, "sse2 len {len}");
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed available.
            assert_eq!(unsafe { u8_dist_sq_avx2(&a, &b) }, scalar, "avx2 len {len}");
        }
    }
}

#[test]
fn u8_kernel_extremes() {
    let a = [0u8; 128];
    let b = [255u8; 128];
    assert_eq!(u8_dist_sq(&a, &b), 255i64 * 255 * 128);
    assert_eq!(u8_dist_sq(&a, &a), 0);
}

#[test]
fn f32_cutoff_and_order() {
    assert_eq!(f32::cutoff_sq(2.0), OrdF32(4.0));
    assert!(OrdF32(1.0) < OrdF32(2.0));
    assert!(OrdF32(0.0) < OrdF32(f32::INFINITY));
    assert_eq!(OrdF32(1.0) + OrdF32(2.0), OrdF32(3.0));
}

#[test]
fn u8_cutoff_rounds_up() {
    // sqrt(5) ≈ 2.236; squared back must not drop below 5.
    let c = u8::cutoff_sq(5.0f32.sqrt());
    assert!(c >= 5, "cutoff {c} under-rounded");
}
