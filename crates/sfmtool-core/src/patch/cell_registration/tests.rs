// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests of the cell registration kernel through its own interface: a
//! reference given as planes of samples and a working patch given as
//! interleaved samples, with no cluster or render behind them. The piecewise
//! refinement's tests cover the same code as it runs inside that stage.

use super::*;

/// A band-limited texture with fine terms, so every cell carries texture.
fn texture(x: f64, y: f64) -> f64 {
    127.0
        + 40.0 * (0.11 * x + 0.06 * y + 1.3).sin()
        + 28.0 * (0.05 * x - 0.12 * y + 0.7).sin()
        + 20.0 * (0.17 * x + 0.13 * y + 2.9).sin()
        + 10.0 * (0.29 * x - 0.23 * y + 0.4).cos()
        + 15.0 * (0.83 * x + 0.47 * y + 0.2).sin()
        + 12.0 * (-0.52 * x + 0.88 * y + 1.1).sin()
}

const R: u32 = 24;
const BOUND: f32 = 2.0;

fn gates() -> CellGateOptions {
    CellGateOptions {
        min_cell_zncc: 0.8,
        min_cell_curvature: 0.02,
    }
}

/// The reference square: one plane of `R×R` samples of `f`.
fn reference(f: impl Fn(f64, f64) -> f64) -> Vec<f32> {
    let r = R as usize;
    (0..r * r)
        .map(|p| f((p % r) as f64, (p / r) as f64) as f32)
        .collect()
}

/// A one-channel working patch whose content at grid position `g` is the
/// reference's at `g − d`, so every cell's content lies at its placement plus
/// `d`.
fn working_patch(layout: &CellLayout, d: [f64; 2]) -> WorkingPatch {
    let side = layout.patch_side();
    let m = layout.margin as f64;
    let samples = (0..side * side)
        .map(|p| {
            let (x, y) = ((p % side) as f64 - m, (p / side) as f64 - m);
            texture(x - d[0], y - d[1]) as f32
        })
        .collect();
    WorkingPatch::new(side, 1, samples)
}

#[test]
fn a_shifted_render_registers_every_cell_at_the_shift() {
    let layout = CellLayout::new(R, BOUND);
    let templates = CellTemplates::from_planes(&reference(texture), 1, &layout);
    let d = [0.3, -0.4];
    let patch = working_patch(&layout, d);
    let readings = read_cells(&templates, &[0], &patch, &layout, &gates());
    for (row, line) in readings.iter().enumerate() {
        for (col, rd) in line.iter().enumerate() {
            assert_eq!(rd.status, CellStatus::Fitted, "cell ({row}, {col})");
            assert!(rd.zncc > 0.95, "cell ({row}, {col}): {}", rd.zncc);
            let err = (rd.shift[0] - d[0]).hypot(rd.shift[1] - d[1]);
            assert!(err < 0.15, "cell ({row}, {col}): {:?}", rd.shift);
        }
    }
    let pass = fit_pass(readings, &layout).expect("a fit");
    assert_eq!(pass.model, UpdateModel::Affine);
    assert!(pass.weights.iter().flatten().all(|&w| w > 0.0));
    let at_centre = pass.update.moves([0.0, 0.0]);
    assert!((at_centre[0] - d[0]).abs() < 0.1 && (at_centre[1] - d[1]).abs() < 0.1);
}

#[test]
fn the_bars_and_the_bound_refuse_cells() {
    let layout = CellLayout::new(R, BOUND);
    let templates = CellTemplates::from_planes(&reference(texture), 1, &layout);

    // A ZNCC bar no score reaches refuses every cell for ZNCC, with its
    // shift still measured.
    let patch = working_patch(&layout, [0.0, 0.0]);
    let strict = CellGateOptions {
        min_cell_zncc: 1.5,
        ..gates()
    };
    for rd in read_cells(&templates, &[0], &patch, &layout, &strict)
        .iter()
        .flatten()
    {
        assert_eq!(rd.status, CellStatus::RefusedZncc);
        assert!(rd.shift.iter().all(|v| v.is_finite()));
    }

    // Content moved by the whole search bound peaks on the bound.
    let patch = working_patch(&layout, [2.0, 0.0]);
    for rd in read_cells(&templates, &[0], &patch, &layout, &gates())
        .iter()
        .flatten()
    {
        assert_eq!(rd.status, CellStatus::RefusedBound);
        assert!(rd.shift.iter().all(|v| v.is_nan()));
    }

    // A flat reference refuses every cell for curvature before any search.
    let flat = CellTemplates::from_planes(&reference(|_, _| 100.0), 1, &layout);
    let patch = working_patch(&layout, [0.0, 0.0]);
    for rd in read_cells(&flat, &[0], &patch, &layout, &gates())
        .iter()
        .flatten()
    {
        assert_eq!(rd.status, CellStatus::RefusedCurvature);
    }
    assert!(fit_pass(read_cells(&flat, &[0], &patch, &layout, &gates()), &layout).is_none());
}
