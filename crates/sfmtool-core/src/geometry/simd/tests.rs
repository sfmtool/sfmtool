// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::geometry::numeric::{acos_poly_scalar, splitmix64};
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

/// The rotation residual's vector body and its scalar tail (and fallback) have
/// to be the *same* arithmetic, not merely close: a `to_bits` comparison is the
/// contract, because anything weaker would let `SFMTOOL_FOCAL_VOTE_NO_SIMD`
/// change a consensus count. NaN and the branch boundary are included — a NaN
/// argument has to come back NaN through both forms.
#[test]
fn acos_vector_and_scalar_agree_bit_for_bit() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut inputs = vec![
        -1.0f64,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        -1.0 + f64::EPSILON,
        1.0 - f64::EPSILON,
        0.5 + f64::EPSILON,
        -0.5 - f64::EPSILON,
        f64::NAN,
        -f64::NAN,
    ];
    let mut state = 0x9e37_79b9u64;
    for _ in 0..4_000 {
        let u = (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        inputs.push(2.0 * u - 1.0);
        // Crowd the ends, where the big branch and the `√z` live.
        let mag = 1.0 - u * u * u;
        inputs.push(mag);
        inputs.push(-mag);
    }
    while inputs.len() % 4 != 0 {
        inputs.push(0.25);
    }
    let mut out = vec![0.0f64; inputs.len()];
    for b in 0..inputs.len() / 4 {
        // SAFETY: avx2 confirmed above; four readable/writable `f64` at `b * 4`.
        unsafe {
            let v = std::arch::x86_64::_mm256_loadu_pd(inputs.as_ptr().add(b * 4));
            std::arch::x86_64::_mm256_storeu_pd(out.as_mut_ptr().add(b * 4), acos_pd(v));
        }
    }
    for (d, got) in inputs.iter().zip(out.iter()) {
        let want = acos_poly_scalar(*d);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "acos({d:?}): simd {got:?} vs scalar {want:?}"
        );
    }
}
