// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pair fit on flows the pair model itself writes: the exact flow of a
//! known plane recovers it, a disc with no information has no fit, outliers
//! are down-weighted, and the residual σ is honest about noisy flow.

use nalgebra::Vector3;

use super::super::model::{tangent_basis, CommonPlane, PairModel};
use super::super::texture::{Disc, Gaussian};
use super::super::PairwiseFlowNormalOptions;
use super::fit_pair;

fn angle_deg(a: [f64; 3], b: Vector3<f64>) -> f64 {
    let a = Vector3::from(a);
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

/// Two cameras 4 units from the point, 30° apart, both near the round's
/// plane normal.
fn model() -> PairModel {
    let normal = Vector3::new(0.0, 0.0, 1.0);
    let (x_axis, y_axis) = tangent_basis(normal, Vector3::x());
    PairModel {
        plane: CommonPlane {
            point: Vector3::zeros(),
            normal,
            x_axis,
            y_axis,
            spacing: 0.02,
        },
        from: Vector3::new(-1.0, 0.2, 3.85),
        to: Vector3::new(1.05, -0.1, 3.84),
        level_scale: 1.0,
    }
}

/// The plane the tests' flows are made from: 25° off the round's plane.
fn truth() -> Vector3<f64> {
    let t = 25f64.to_radians();
    Vector3::new(t.sin() * 0.6, t.sin() * 0.8, t.cos())
}

/// The samples within `radius` grid samples of the centre, each with the
/// flow `truth` predicts plus `offset` and `noise` samples of white noise
/// per component, and the same information `w` at every sample.
fn disc(radius: f64, offset: [f64; 2], noise: f64, w: [f64; 3], l2: f64, seed: u64) -> Disc {
    let m = model();
    let mut normal = Gaussian::new(seed);
    let r = radius.ceil() as i32;
    let mut positions = Vec::new();
    let mut flows = Vec::new();
    for y in -r..=r {
        for x in -r..=r {
            let q = [f64::from(x) + 0.5, f64::from(y) + 0.5];
            if q[0].hypot(q[1]) > radius {
                continue;
            }
            let f = m.flow(q, truth()).unwrap();
            positions.push(q);
            flows.push([
                f[0] + offset[0] + noise * normal.next(),
                f[1] + offset[1] + noise * normal.next(),
            ]);
        }
    }
    let information = vec![w; positions.len()];
    Disc {
        positions,
        flows,
        information,
        residual_variance: 1e-4,
        l2,
    }
}

#[test]
fn the_exact_flow_of_a_plane_recovers_it() {
    let options = PairwiseFlowNormalOptions::default();
    let fit = fit_pair(
        &model(),
        &disc(12.0, [0.4, -0.3], 0.0, [1.0, 0.0, 1.0], 1.0, 1),
        &options,
    )
    .unwrap();
    let error = angle_deg(fit.normal, truth());
    assert!(error < 1e-4, "{error}°");
    assert!((fit.offset[0] - 0.4).abs() < 1e-6 && (fit.offset[1] + 0.3).abs() < 1e-6);
    assert!(fit.rms_residual < 1e-6, "{}", fit.rms_residual);
    assert!(fit.sigma_residual_deg < 1e-3, "{}", fit.sigma_residual_deg);
    assert_eq!(
        fit.sigma_deg,
        fit.sigma_texture_deg.max(fit.sigma_residual_deg)
    );
}

#[test]
fn a_disc_with_no_information_has_no_fit() {
    let options = PairwiseFlowNormalOptions::default();
    let flat = disc(12.0, [0.0; 2], 0.0, [0.0; 3], 64.0, 1);
    assert!(fit_pair(&model(), &flat, &options).is_none());
}

/// Texture along one image axis at every sample, such as a crop of parallel
/// stripes, leaves the offset across them free: the Hessian is singular,
/// and the pair has no fit rather than a confident one.
#[test]
fn stripes_in_every_window_give_no_fit() {
    let options = PairwiseFlowNormalOptions::default();
    let one_axis = disc(12.0, [0.0; 2], 0.0, [1.0, 0.0, 0.0], 1.0, 1);
    assert!(fit_pair(&model(), &one_axis, &options).is_none());
}

/// A tenth of the disc carries flow from another layer, 3 samples off; the
/// Tukey weights leave the plane where the rest of the disc puts it.
#[test]
fn flow_from_another_layer_is_down_weighted() {
    let options = PairwiseFlowNormalOptions::default();
    let mut d = disc(12.0, [0.0; 2], 0.02, [1.0, 0.0, 1.0], 1.0, 5);
    for (k, f) in d.flows.iter_mut().enumerate() {
        if k % 10 == 0 {
            f[0] += 3.0;
        }
    }
    let fit = fit_pair(&model(), &d, &options).unwrap();
    let error = angle_deg(fit.normal, truth());
    assert!(error < 0.5, "{error}°");
}

/// White noise of 0.1 sample on every flow component, each sample its own
/// correlated square: over 12 draws the residual σ is within a factor of
/// two of the normals' spread about the truth, and the texture σ, which
/// reads an information of 1 as a noise of 1 sample, is ten times larger.
#[test]
fn the_residual_sigma_matches_the_spread_of_noisy_fits() {
    let options = PairwiseFlowNormalOptions::default();
    let draws = 12;
    let (mut squared_error, mut sigma) = (0.0, 0.0);
    for seed in 0..draws {
        let d = disc(10.0, [0.0; 2], 0.1, [1.0, 0.0, 1.0], 1.0, 100 + seed);
        let fit = fit_pair(&model(), &d, &options).unwrap();
        squared_error += angle_deg(fit.normal, truth()).powi(2);
        sigma += fit.sigma_residual_deg;
        let ratio = fit.sigma_texture_deg / fit.sigma_residual_deg;
        assert!((7.0..14.0).contains(&ratio), "{ratio}");
    }
    let rms_error = (squared_error / draws as f64).sqrt();
    let mean_sigma = sigma / draws as f64;
    // The angle combines the two axes, so its rms is about √2 times one
    // axis's σ; the largest-axis σ is at least that axis's.
    assert!(
        rms_error < 2.0 * std::f64::consts::SQRT_2 * mean_sigma && rms_error > 0.25 * mean_sigma,
        "rms error {rms_error}°, mean σ {mean_sigma}°"
    );
}
