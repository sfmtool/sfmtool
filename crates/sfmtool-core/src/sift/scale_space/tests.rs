use super::*;

fn params() -> SiftParams {
    SiftParams::default()
}

#[test]
fn test_gaussian_kernel_normalized() {
    let k = gaussian_kernel(1.6, 3.0);
    let sum: f32 = k.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "kernel sum {sum}");
    // Symmetric.
    let n = k.len();
    for i in 0..n / 2 {
        assert!((k[i] - k[n - 1 - i]).abs() < 1e-6);
    }
}

#[test]
fn test_blur_constant_image_stays_constant() {
    let img = GrayImage::new_constant(40, 30, 0.42);
    let blurred = gaussian_blur(&img, 2.0, 3.0, &mut Vec::new());
    for &v in blurred.data() {
        assert!((v - 0.42).abs() < 1e-4, "got {v}");
    }
}

#[test]
fn test_dog_of_constant_is_zero() {
    let img = GrayImage::new_constant(48, 48, 0.6);
    let ss = ScaleSpace::build(&img, &params());
    for level in 0..ss.dogs_per_octave() {
        let dog = ss.dog(0, level);
        for &v in dog.data() {
            assert!(
                v.abs() < 1e-4,
                "DoG should be ~0 on a constant image, got {v}"
            );
        }
    }
}

#[test]
fn test_octave_dims_halve() {
    // No doubling so octave 0 is the original size, easy to reason about.
    let p = SiftParams {
        double_image: false,
        ..SiftParams::default()
    };
    let img = GrayImage::new_constant(256, 128, 0.5);
    let ss = ScaleSpace::build(&img, &p);
    assert!(ss.num_octaves() >= 2);
    let (w0, h0) = ss.octave_dims(0);
    assert_eq!((w0, h0), (256, 128));
    let (w1, h1) = ss.octave_dims(1);
    assert_eq!((w1, h1), (128, 64));
}

#[test]
fn test_doubling_octave0_dims() {
    let p = SiftParams {
        double_image: true,
        ..SiftParams::default()
    };
    let img = GrayImage::new_constant(64, 48, 0.5);
    let ss = ScaleSpace::build(&img, &p);
    let (w0, h0) = ss.octave_dims(0);
    assert_eq!((w0, h0), (128, 96));
}

#[test]
fn test_levels_per_octave() {
    let img = GrayImage::new_constant(64, 64, 0.5);
    let ss = ScaleSpace::build(&img, &params());
    // s + 3 Gaussian, s + 2 DoG.
    assert_eq!(ss.gaussians_per_octave(), (3 + 3) as usize);
    assert_eq!(ss.dogs_per_octave(), (3 + 2) as usize);
}

#[test]
fn test_gradient_of_linear_ramp() {
    // I(x, y) = 2*x (column index scaled), so dx = 2*1 differenced over 2 px
    // central difference = (2*(x+1) - 2*(x-1)) = 4; dy = 0.
    let w = 32u32;
    let h = 16u32;
    let data: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(|c| 2.0 * c as f32))
        .collect();
    let img = GrayImage::new(w, h, data);
    // Interior pixel.
    let idx = 8 * w as usize + 10;
    let (mag, theta) = pixel_gradient(img.data(), w as usize, idx);
    assert!((mag - 4.0).abs() < 1e-4, "mag {}", mag);
    // Gradient points in +x, so theta ~ 0.
    assert!(theta.abs() < 1e-4, "theta {}", theta);
}

#[test]
fn test_abs_sigma() {
    let img = GrayImage::new_constant(64, 64, 0.5);
    let ss = ScaleSpace::build(&img, &params());
    // Level 0 has absolute blur σ.
    assert!((ss.abs_sigma(0.0) - 1.6).abs() < 1e-9);
    // Level s has absolute blur 2σ (k^s = 2).
    assert!((ss.abs_sigma(ss.octave_layers() as f64) - 3.2).abs() < 1e-9);
    // A fractional layer.
    let k = 2.0f64.powf(1.0 / 3.0);
    assert!((ss.abs_sigma(1.5) - 1.6 * k.powf(1.5)).abs() < 1e-9);
}

#[test]
fn test_octave_pixel_step_and_mapping() {
    let img = GrayImage::new_constant(64, 64, 0.5);
    // Doubling on: octave 0 step is 0.5, octave 1 step is 1.0.
    let ss = ScaleSpace::build(&img, &params());
    assert!((ss.octave_pixel_step(0) - 0.5).abs() < 1e-12);
    assert!((ss.octave_pixel_step(1) - 1.0).abs() < 1e-12);
    // Octave-0 pixel (0,0) center maps to full-res 0*0.5 + 0.25 = 0.25
    // (constant offset c = base_step/2 = 0.25 with doubling on).
    let (x, y) = ss.octave_to_image(0, 0.0, 0.0);
    assert!((x - 0.25).abs() < 1e-12);
    assert!((y - 0.25).abs() < 1e-12);
    // The offset is a constant, not scaled per octave: octave-1 pixel 0 also
    // maps to 0.25 (xo=0), and a nonzero xo scales only by step.
    let (x1, _) = ss.octave_to_image(1, 0.0, 0.0);
    assert!((x1 - 0.25).abs() < 1e-12);
    let (x2, _) = ss.octave_to_image(1, 3.0, 0.0);
    assert!((x2 - (3.0 * 1.0 + 0.25)).abs() < 1e-12);
    // octave_to_image and image_to_octave round-trip.
    let (rx, ry) = ss.image_to_octave(1, x2, 0.25);
    assert!((rx - 3.0).abs() < 1e-12 && ry.abs() < 1e-12);
    // abs_sigma_full at octave 0, layer 0 = σ * 0.5.
    assert!((ss.abs_sigma_full(0, 0.0) - 1.6 * 0.5).abs() < 1e-9);
}

#[test]
fn test_gaussian_pyramid_retained() {
    let img = GrayImage::new_constant(64, 64, 0.5);
    let ss = ScaleSpace::build(&img, &params());
    // DoG is virtual (computed on the fly); the Gaussian pyramid is retained
    // for detection, orientation, and description.
    assert_eq!(ss.dogs_per_octave(), 5); // s + 2
    assert_eq!(ss.gaussians_per_octave(), 6); // s + 3
    let (gw, _) = ss.octave_dims(0);
    assert_eq!(gw, 64 * 2); // doubled octave 0
    assert_eq!(ss.gaussian(0, 0).data().len(), 64 * 64 * 4);
}

#[test]
fn test_upsample_2x_constant() {
    let img = GrayImage::new_constant(16, 16, 0.3);
    let up = upsample_2x(&img);
    assert_eq!(up.width(), 32);
    assert_eq!(up.height(), 32);
    for &v in up.data() {
        assert!((v - 0.3).abs() < 1e-5);
    }
}

#[test]
fn test_blur_matches_scalar_reference() {
    // Cross-check the SSE2/border path against a naive scalar blur on a ramp.
    let w = 40usize;
    let h = 8usize;
    let data: Vec<f32> = (0..h)
        .flat_map(|r| (0..w).map(move |c| (c as f32 * 0.1 + r as f32 * 0.05).sin()))
        .collect();
    let img = GrayImage::new(w as u32, h as u32, data.clone());
    let sigma = 1.6;
    let got = gaussian_blur(&img, sigma, 3.0, &mut Vec::new());

    let kernel = gaussian_kernel(sigma, 3.0);
    let radius = (kernel.len() / 2) as i32;
    // Naive separable reference.
    let mut horiz = vec![0.0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0f32;
            for (kidx, &kw) in kernel.iter().enumerate() {
                let c = (col as i32 + kidx as i32 - radius).clamp(0, w as i32 - 1) as usize;
                acc += kw * data[row * w + c];
            }
            horiz[row * w + col] = acc;
        }
    }
    let mut reference = vec![0.0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0f32;
            for (kidx, &kw) in kernel.iter().enumerate() {
                let rr = (row as i32 + kidx as i32 - radius).clamp(0, h as i32 - 1) as usize;
                acc += kw * horiz[rr * w + col];
            }
            reference[row * w + col] = acc;
        }
    }
    for (a, b) in got.data().iter().zip(reference.iter()) {
        assert!((a - b).abs() < 1e-5, "blur mismatch {a} vs {b}");
    }
}
