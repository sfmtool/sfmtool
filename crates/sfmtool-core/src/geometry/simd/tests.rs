// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::Vector3;

/// The `AoS → SoA` transposes are pure lane shuffles, so an exact element
/// comparison is the whole contract: any mis-encoded `permute`/`blend` immediate
/// shows up as a permuted lane here rather than as a fp discrepancy a hundred
/// thousand residuals later.
#[test]
fn vec3x4_transpose_places_every_lane() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let v: Vec<Vector3<f64>> = (0..4)
        .map(|i| Vector3::new(i as f64, 10.0 + i as f64, 100.0 + i as f64))
        .collect();
    let mut x = [0.0f64; 4];
    let mut y = [0.0f64; 4];
    let mut z = [0.0f64; 4];
    // SAFETY: avx2 confirmed above; `v` holds four consecutive 3-vectors.
    unsafe {
        let (vx, vy, vz) = load_vec3x4(v.as_ptr() as *const f64);
        std::arch::x86_64::_mm256_storeu_pd(x.as_mut_ptr(), vx);
        std::arch::x86_64::_mm256_storeu_pd(y.as_mut_ptr(), vy);
        std::arch::x86_64::_mm256_storeu_pd(z.as_mut_ptr(), vz);
    }
    assert_eq!(x, [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(y, [10.0, 11.0, 12.0, 13.0]);
    assert_eq!(z, [100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn vec2x4_transpose_places_every_lane() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let p: Vec<[f64; 2]> = (0..4).map(|i| [i as f64, 10.0 + i as f64]).collect();
    let mut x = [0.0f64; 4];
    let mut y = [0.0f64; 4];
    // SAFETY: avx2 confirmed above; `p` holds four consecutive 2-vectors.
    unsafe {
        let (vx, vy) = load_vec2x4(p.as_ptr() as *const f64);
        std::arch::x86_64::_mm256_storeu_pd(x.as_mut_ptr(), vx);
        std::arch::x86_64::_mm256_storeu_pd(y.as_mut_ptr(), vy);
    }
    assert_eq!(x, [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(y, [10.0, 11.0, 12.0, 13.0]);
}
