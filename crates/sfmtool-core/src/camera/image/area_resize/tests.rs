// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn area_taps_are_the_exact_overlaps() {
    // 3 to 2: the middle pixel is split half and half.
    assert_eq!(
        area_taps(3, 2),
        vec![
            vec![(0, 2.0f32 / 3.0), (1, 1.0 / 3.0)],
            vec![(1, 1.0 / 3.0), (2, 2.0 / 3.0)],
        ]
    );
    // An integer factor is equal weights; the same size is the identity.
    assert_eq!(
        area_taps(4, 2),
        vec![vec![(0, 0.5), (1, 0.5)], vec![(2, 0.5), (3, 0.5)]]
    );
    assert_eq!(
        area_taps(3, 3),
        vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]
    );
    // Enlarging 2 to 3: the middle output straddles the two sources.
    assert_eq!(
        area_taps(2, 3),
        vec![vec![(0, 1.0)], vec![(0, 0.5), (1, 0.5)], vec![(1, 1.0)]]
    );
    // Each weight is the f32 nearest the exact ratio, every output's weights
    // sum to one, and every source pixel is used in full.
    for (n, m) in [
        (1920, 480),
        (2040, 480),
        (641, 97),
        (1000, 7),
        (5, 3),
        (7, 300),
    ] {
        let taps = area_taps(n, m);
        let mut used = vec![0f64; n];
        for row in &taps {
            let total: f64 = row.iter().map(|t| f64::from(t.1)).sum();
            assert!((total - 1.0).abs() < 1e-6, "{n} -> {m}: {total}");
            for &(s, w) in row {
                used[s] += f64::from(w) * n as f64 / m as f64;
            }
        }
        assert!(used.iter().all(|&u| (u - 1.0).abs() < 1e-5), "{n} -> {m}");
    }
    // 3000 to 7: output 1 covers [3000, 6000) in sevenths of a pixel, and
    // source 428 covers [2996, 3003), so its overlap is 3 and its weight
    // 3 / 3000, the f32 nearest 0.001.
    let (s, w) = area_taps(3000, 7)[1][0];
    assert_eq!((s, w), (428, 0.001f32));
}

#[test]
fn a_resize_is_the_area_integral() {
    // Each output pixel against a direct two-dimensional integral of the
    // piecewise-constant source over its footprint, in f64.
    let (w, h, ow, oh) = (11usize, 7usize, 4usize, 3usize);
    let value = |x: usize, y: usize, c: usize| ((x * 37 + y * 101 + c * 59) % 256) as f32;
    let out = area_resize(
        (w, h),
        2,
        (ow, oh),
        || (),
        |y, row, _| {
            for x in 0..w {
                for c in 0..2 {
                    row[x * 2 + c] = value(x, y, c);
                }
            }
        },
    );
    assert_eq!(out.len(), ow * oh * 2);
    for j in 0..oh {
        for i in 0..ow {
            let (x0, x1) = (
                i as f64 * w as f64 / ow as f64,
                (i + 1) as f64 * w as f64 / ow as f64,
            );
            let (y0, y1) = (
                j as f64 * h as f64 / oh as f64,
                (j + 1) as f64 * h as f64 / oh as f64,
            );
            for c in 0..2 {
                let mut integral = 0.0;
                for y in 0..h {
                    for x in 0..w {
                        let ox = (x1.min(x as f64 + 1.0) - x0.max(x as f64)).max(0.0);
                        let oy = (y1.min(y as f64 + 1.0) - y0.max(y as f64)).max(0.0);
                        integral += ox * oy * f64::from(value(x, y, c));
                    }
                }
                let expected = integral / ((x1 - x0) * (y1 - y0));
                let got = f64::from(out[(j * ow + i) * 2 + c]);
                assert!(
                    (got - expected).abs() < 2e-6 * expected.max(1.0),
                    "({i}, {j}, {c}): {got} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn the_rows_are_filled_once_each_and_the_result_does_not_depend_on_the_thread_count() {
    let (w, h) = (301usize, 157usize);
    let fill = |y: usize, row: &mut [f32], calls: &mut usize| {
        *calls += 1;
        for (x, v) in row.iter_mut().enumerate() {
            *v = ((x * 7919 + y * 104_729) % 255) as f32 / 255.0;
        }
    };
    let filled = std::sync::atomic::AtomicUsize::new(0);
    let count = |y: usize, row: &mut [f32], calls: &mut usize| {
        filled.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        fill(y, row, calls)
    };
    let many = area_resize((w, h), 1, (97, 50), || 0usize, count);
    assert_eq!(filled.into_inner(), h);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| area_resize((w, h), 1, (97, 50), || 0usize, fill));
    assert_eq!(
        many.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        one.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn an_empty_size_gives_an_empty_result() {
    let never = |_: usize, _: &mut [f32], _: &mut ()| panic!("no row to fill");
    assert!(area_resize((0, 5), 1, (3, 3), || (), never).is_empty());
    assert!(area_resize((5, 5), 1, (0, 3), || (), never).is_empty());
}
