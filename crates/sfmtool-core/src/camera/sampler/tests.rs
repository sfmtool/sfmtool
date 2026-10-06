// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::SQRT_2;
use std::sync::Mutex;

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear, remap_bilinear_mip};
use crate::camera::CameraModel;
use crate::progress::Event;

const A: f64 = DEFAULT_ANISOTROPIC_THRESHOLD;

/// `σ_major` just under `√2` keeps `BilinearMip` however compressed the other
/// axis is, because `BilinearMip` reads level 0 for both axes there; just over
/// it the level is 1 and a minor axis at or under one photograph px per grid
/// px is read twice too coarse.
#[test]
fn the_rule_switches_at_sigma_major_root_two() {
    let below = SQRT_2 * (1.0 - 1e-9);
    let above = SQRT_2 * (1.0 + 1e-9);
    for minor in [0.1, 0.5, 1.0] {
        assert_eq!(rule_sampler([below, minor], A), Sampler::BilinearMip);
        assert_eq!(rule_sampler([above, minor], A), Sampler::Anisotropic);
    }
    assert_eq!(bilinear_mip_level(below), 0);
    assert_eq!(bilinear_mip_level(above), 1);
    // At exactly √2 the level rounds up, as `remap_bilinear_mip` does.
    assert_eq!(bilinear_mip_level(SQRT_2), 1);
    assert_eq!(rule_sampler([SQRT_2, 1.0], A), Sampler::Anisotropic);
}

/// A view that compresses both axes alike reads `L` between `1/√2` and `√2`
/// from the level rounding alone. Swept across six octaves, including either
/// side of every level boundary, it stays on `BilinearMip` at the default
/// threshold.
#[test]
fn isotropic_views_stay_on_bilinear_mip_at_every_level_boundary() {
    let mut worst: f64 = 0.0;
    let mut sigma = 1.0;
    while sigma < 64.0 {
        let loss = minor_axis_loss([sigma, sigma]);
        worst = worst.max(loss);
        assert!(loss <= SQRT_2 * (1.0 + 1e-12), "σ {sigma}: L {loss}");
        assert_eq!(
            rule_sampler([sigma, sigma], A),
            Sampler::BilinearMip,
            "σ {sigma}"
        );
        sigma *= 1.0007;
    }
    for level in 1..6 {
        let boundary = 2f64.powf(level as f64 - 0.5);
        for sigma in [boundary * (1.0 - 1e-9), boundary * (1.0 + 1e-9)] {
            assert_eq!(rule_sampler([sigma, sigma], A), Sampler::BilinearMip);
        }
    }
    // The sweep reaches the √2 the threshold is set above.
    assert!(worst > 1.41, "{worst}");
}

/// A minor axis that magnifies the photograph (`σ_minor < 1`) loses detail
/// only down to one photograph pixel, so its loss is `2^l` whatever its
/// magnification.
#[test]
fn a_magnifying_minor_axis_counts_as_one_photograph_pixel() {
    for minor in [0.05, 0.3, 0.99, 1.0] {
        assert_eq!(minor_axis_loss([3.0, minor]), 4.0);
        assert_eq!(rule_sampler([3.0, minor], A), Sampler::Anisotropic);
    }
    // Magnified along both axes: level 0, nothing to lose.
    assert_eq!(rule_sampler([1.2, 0.3], A), Sampler::BilinearMip);
    // Compressed 3× and 2.2×: level 2, so the minor axis is read 4 / 2.2 too
    // coarse, above the threshold; at 2.9× it is not.
    assert_eq!(rule_sampler([3.0, 2.2], A), Sampler::Anisotropic);
    assert_eq!(rule_sampler([3.0, 2.9], A), Sampler::BilinearMip);
}

/// An infinite `σ_major` reads an infinite loss rather than the `0.5` an
/// overflowing level would give, and moves the view; a huge finite one gives
/// a huge loss.
#[test]
fn an_infinite_major_axis_reads_an_infinite_loss() {
    assert_eq!(bilinear_mip_level(f64::INFINITY), u32::MAX);
    assert_eq!(minor_axis_loss([f64::INFINITY, 2.0]), f64::INFINITY);
    assert_eq!(rule_sampler([f64::INFINITY, 2.0], A), Sampler::Anisotropic);
    assert_eq!(minor_axis_loss([1e300, 1.0]), 2f64.powi(997));
    assert!(minor_axis_loss([f64::INFINITY, f64::INFINITY]).is_nan());
}

/// The threshold is the bar `L` must reach, inclusive.
#[test]
fn the_threshold_is_inclusive() {
    // l = 2, L = 4 / 2.5 = 1.6.
    assert_eq!(rule_sampler([4.0, 2.5], 1.6), Sampler::Anisotropic);
    assert_eq!(rule_sampler([4.0, 2.5], 1.6 + 1e-9), Sampler::BilinearMip);
}

/// A view with no Jacobian, or a degenerate one, keeps `BilinearMip`; a fixed
/// choice is the sampler it names whatever the Jacobian.
#[test]
fn no_jacobian_keeps_bilinear_mip_and_a_fixed_choice_ignores_it() {
    let rule = SamplerChoice::default();
    assert_eq!(rule, SamplerChoice::per_view());
    assert_eq!(rule.anisotropic_threshold(), Some(A));
    assert_eq!(rule.for_singular_values(None), Sampler::BilinearMip);
    assert_eq!(rule.for_jacobian(None), Sampler::BilinearMip);
    assert_eq!(
        rule.for_singular_values(Some([f64::NAN, 1.0])),
        Sampler::BilinearMip
    );
    for sampler in [
        Sampler::Bilinear,
        Sampler::BilinearMip,
        Sampler::Anisotropic,
    ] {
        let fixed = SamplerChoice::from(sampler);
        assert_eq!(fixed.anisotropic_threshold(), None);
        assert_eq!(fixed.name(), sampler.name());
        assert_eq!(fixed.for_singular_values(None), sampler);
        assert_eq!(fixed.for_singular_values(Some([12.0, 1.0])), sampler);
        assert_eq!(fixed.for_singular_values(Some([1.0, 1.0])), sampler);
    }
    assert_eq!(rule.name(), "per_view");
}

/// A pinhole camera 640 × 480 with a 500 px focal length, at the origin
/// looking down `−z`.
fn camera() -> (CameraIntrinsics, RigidTransform) {
    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    (camera, pose)
}

/// A square of half-extent 0.5 at depth 4, turned `tilt_deg` about the `y`
/// axis: at `R = 24` one grid px spans 125 / 24 ≈ 5.2 image px facing the
/// camera, and `cos(tilt)` of that along `u`.
fn tilted_square(tilt_deg: f64) -> OrientedPatch {
    let t = tilt_deg.to_radians();
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(t.sin(), 0.0, t.cos()),
        Vector3::y(),
        [0.5, 0.5],
    )
}

/// The placement form reads the Jacobian `patch_grid_jacobian` reads: a
/// facing square stays on `BilinearMip`, the same square at 70° moves.
#[test]
fn the_placement_form_applies_the_rule_to_the_patch_grid_jacobian() {
    let (camera, pose) = camera();
    let rule = SamplerChoice::default();
    for tilt in [0.0, 30.0, 50.0, 70.0] {
        let patch = tilted_square(tilt);
        let jacobian = patch_grid_jacobian(&patch, &camera, &pose, 24);
        assert_eq!(
            rule.for_placement(&patch, &camera, &pose, 24),
            rule.for_jacobian(jacobian),
            "tilt {tilt}"
        );
    }
    assert_eq!(
        rule.for_placement(&tilted_square(0.0), &camera, &pose, 24),
        Sampler::BilinearMip
    );
    assert_eq!(
        rule.for_placement(&tilted_square(70.0), &camera, &pose, 24),
        Sampler::Anisotropic
    );
    // A context tile twice as wide at twice the resolution has the same texel,
    // so it chooses as the core at its centre does.
    for tilt in [0.0, 30.0, 50.0, 70.0] {
        let core = tilted_square(tilt);
        let mut wide = core.clone();
        wide.half_extent = [1.0, 1.0];
        assert_eq!(
            rule.for_placement(&wide, &camera, &pose, 48),
            rule.for_placement(&core, &camera, &pose, 24),
            "tilt {tilt}"
        );
    }
}

/// A deterministic textured 640 × 480 RGB photograph and its pyramid.
fn pyramid() -> ImageU8Pyramid {
    let (w, h) = (640u32, 480u32);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 7 + y * 13) ^ (x * y);
            data.extend_from_slice(&[(v % 251) as u8, (v % 241) as u8, ((x + y) % 256) as u8]);
        }
    }
    ImageU8Pyramid::build(&ImageU8::new(w, h, 3, data), 6)
}

/// `render_tile` is the remap the sampler names, bit for bit, and computes
/// the SVD it needs itself.
#[test]
fn render_tile_is_the_named_remap() {
    let (camera, pose) = camera();
    let pyramid = pyramid();
    let patch = tilted_square(60.0);
    for sampler in [
        Sampler::Bilinear,
        Sampler::BilinearMip,
        Sampler::Anisotropic,
    ] {
        let mut map = WarpMap::from_patch(&patch, &camera, &pose, 24);
        let tile = render_tile(&pyramid, &mut map, sampler);
        let mut reference = WarpMap::from_patch(&patch, &camera, &pose, 24);
        reference.compute_svd();
        let want = match sampler {
            Sampler::Bilinear => remap_bilinear(pyramid.level(0), &reference),
            Sampler::BilinearMip => remap_bilinear_mip(&pyramid, &reference),
            Sampler::Anisotropic => remap_aniso_with_pyramid(&pyramid, &reference, MAX_ANISOTROPY),
        };
        assert_eq!(tile.data(), want.data(), "{sampler:?}");
    }
}

/// On the 60° square the anisotropic render keeps detail the mip render
/// averages away across the tilt: the two tiles differ.
#[test]
fn a_moved_view_renders_a_different_tile() {
    let (camera, pose) = camera();
    let pyramid = pyramid();
    let patch = tilted_square(60.0);
    let mut a = WarpMap::from_patch(&patch, &camera, &pose, 24);
    let mut b = WarpMap::from_patch(&patch, &camera, &pose, 24);
    let mip = render_tile(&pyramid, &mut a, Sampler::BilinearMip);
    let aniso = render_tile(&pyramid, &mut b, Sampler::Anisotropic);
    assert_ne!(mip.data(), aniso.data());
}

/// `render_tiles` returns the tiles in the order of the maps, each rendered
/// with its own sampler, and a detailed run records one phase per sampler with
/// the number of views it rendered. A run that is not detailed records nothing.
#[test]
fn render_tiles_groups_the_renders_by_sampler() {
    let (camera, pose) = camera();
    let pyramid = pyramid();
    let tilts = [0.0, 60.0, 30.0, 70.0];
    let samplers = [
        Sampler::BilinearMip,
        Sampler::Anisotropic,
        Sampler::BilinearMip,
        Sampler::Anisotropic,
    ];
    let maps = || -> Vec<WarpMap> {
        tilts
            .iter()
            .map(|&t| WarpMap::from_patch(&tilted_square(t), &camera, &pose, 24))
            .collect()
    };
    let pyramids = vec![&pyramid; tilts.len()];

    let seen: Mutex<Vec<(&'static str, Option<String>)>> = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave { phase, note, .. } = event {
            seen.lock().unwrap().push((phase, note.map(str::to_string)));
        }
    };
    let mut detailed_maps = maps();
    let tiles = render_tiles(
        &pyramids,
        &mut detailed_maps,
        &samplers,
        &Progress::to(&sink).detailed(true),
    );
    for (i, tile) in tiles.iter().enumerate() {
        let mut map = WarpMap::from_patch(&tilted_square(tilts[i]), &camera, &pose, 24);
        let want = render_tile(&pyramid, &mut map, samplers[i]);
        assert_eq!(tile.data(), want.data(), "tile {i}");
    }
    assert_eq!(
        *seen.lock().unwrap(),
        [
            ("render bilinear_mip", Some("2 views".to_string())),
            ("render anisotropic", Some("2 views".to_string())),
        ]
    );

    seen.lock().unwrap().clear();
    let mut quiet_maps = maps();
    render_tiles(&pyramids, &mut quiet_maps, &samplers, &Progress::to(&sink));
    assert!(seen.lock().unwrap().is_empty());
}
