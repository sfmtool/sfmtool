// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// `atan2_approx` agrees with `libm` to < 2e-3 rad over a dense angle/radius
/// grid, including the axes and origin.
#[test]
fn test_atan2_approx_accuracy() {
    if !*HAS_AVX2_FMA {
        return;
    }
    let mut max_err = 0.0f32;
    // A grid of (x, y) including both signs, the axes, and (0, 0).
    let coords: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    for &yv in &coords {
        for chunk in coords.chunks(8) {
            let mut xs = [0.0f32; 8];
            xs[..chunk.len()].copy_from_slice(chunk);
            let ys = [yv; 8];
            let mut got = [0.0f32; 8];
            // SAFETY: guarded by HAS_AVX2_FMA above.
            unsafe {
                let x = _mm256_loadu_ps(xs.as_ptr());
                let y = _mm256_loadu_ps(ys.as_ptr());
                _mm256_storeu_ps(got.as_mut_ptr(), atan2_approx(y, x));
            }
            for k in 0..chunk.len() {
                let want = ys[k].atan2(xs[k]);
                // Compare on the circle to ignore the ±π wrap equivalence.
                let d = (got[k] - want).abs();
                let d = d.min((2.0 * std::f32::consts::PI - d).abs());
                max_err = max_err.max(d);
            }
        }
    }
    assert!(max_err < 2.0e-3, "atan2 max error {max_err}");
}

/// `exp_approx` matches `libm` to small relative error over `[-87, 5]` — well
/// past the caller's Gaussian-weight range, exercising the large-`|n|`
/// exponent-construction branch down to the `-87` clamp boundary.
#[test]
fn test_exp_approx_accuracy() {
    if !*HAS_AVX2_FMA {
        return;
    }
    let mut max_rel = 0.0f32;
    let xs: Vec<f32> = (-870..=50).map(|i| i as f32 * 0.1).collect();
    for chunk in xs.chunks(8) {
        let mut buf = [0.0f32; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let mut got = [0.0f32; 8];
        // SAFETY: guarded by HAS_AVX2_FMA above.
        unsafe {
            let v = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(got.as_mut_ptr(), exp_approx(v));
        }
        for k in 0..chunk.len() {
            let want = buf[k].exp();
            let rel = (got[k] - want).abs() / want;
            max_rel = max_rel.max(rel);
        }
    }
    assert!(max_rel < 1.0e-5, "exp max relative error {max_rel}");
}
