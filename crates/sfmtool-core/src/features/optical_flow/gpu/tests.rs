use super::super::{FlowField, GrayImage};
use super::*;

/// Derive the image subdirectory from the filename (e.g. "seoul_bull_sculpture_08.jpg" -> "seoul_bull_sculpture").
fn image_subdir(filename: &str) -> &str {
    let stem = filename.rsplit_once('.').map_or(filename, |(s, _)| s);
    stem.rsplit_once('_').map_or(stem, |(prefix, _)| prefix)
}

/// Load a grayscale image from the test data directory.
fn load_test_image(filename: &str) -> GrayImage {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = std::path::PathBuf::from(manifest)
        .join("../../test-data/images")
        .join(image_subdir(filename))
        .join(filename);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e))
        .into_luma8();
    let (w, h) = img.dimensions();
    GrayImage::from_u8(w, h, img.as_raw())
}

#[test]
fn test_gpu_variational_identical_images() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Use the combined DIS+variational path on identical images
    let img = GrayImage::checkerboard(32, 32);
    let params = super::super::DisFlowParams {
        variational_refinement: true,
        gpu_min_pixels: 0, // force GPU even for small images
        ..super::super::DisFlowParams::default_quality()
    };

    let flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

    // For identical images, flow should remain near zero
    let mut max_flow = 0.0f32;
    for row in 0..32 {
        for col in 0..32 {
            let (dx, dy) = flow.get(col, row);
            max_flow = max_flow.max(dx.abs()).max(dy.abs());
        }
    }
    assert!(
        max_flow < 0.5,
        "Expected near-zero flow for identical images, got max_flow={}",
        max_flow
    );
}

#[test]
fn test_gpu_vs_cpu_variational() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Test GPU vs CPU with variational refinement enabled
    let ref_img = GrayImage::checkerboard(64, 64);
    let tgt_img = GrayImage::shifted(&ref_img, 2.0, 1.0);

    let params = super::super::DisFlowParams {
        variational_refinement: true,
        gpu_min_pixels: 0,
        finest_scale: Some(0),
        coarsest_scale: Some(3),
        ..super::super::DisFlowParams::default_quality()
    };

    let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.001,
        "GPU vs CPU RMSE too large: {} (max_diff={})",
        rmse,
        max_diff
    );
}

/// Helper to compute RMSE between two flow fields.
fn flow_rmse(a: &FlowField, b: &FlowField) -> (f64, f32) {
    let w = a.width();
    let h = a.height();
    let n = (w * h) as usize;
    let mut max_diff = 0.0f32;
    let mut sum_diff_sq = 0.0f64;
    for row in 0..h {
        for col in 0..w {
            let (au, av) = a.get(col, row);
            let (bu, bv) = b.get(col, row);
            let du = (au - bu).abs();
            let dv = (av - bv).abs();
            max_diff = max_diff.max(du).max(dv);
            sum_diff_sq += (du as f64).powi(2) + (dv as f64).powi(2);
        }
    }
    let rmse = (sum_diff_sq / (2 * n) as f64).sqrt();
    (rmse, max_diff)
}

#[test]
fn test_gpu_vs_cpu_full_pipeline_shifted() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    let ref_img = GrayImage::checkerboard(128, 128);
    let tgt_img = GrayImage::shifted(&ref_img, 3.0, 1.5);
    let params = super::super::DisFlowParams {
        finest_scale: Some(0),
        coarsest_scale: Some(3),
        ..super::super::DisFlowParams::default_quality()
    };

    let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.001,
        "Full pipeline GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_vs_cpu_full_pipeline_identical() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    let img = GrayImage::checkerboard(64, 64);
    let params = super::super::DisFlowParams::default_quality();

    let cpu_flow = super::super::compute_optical_flow(&img, &img, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.001,
        "Identical images GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

// --- Real image tests using dataset images checked into the repo ---

#[test]
fn test_gpu_vs_cpu_seoul_bull_consecutive() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Seoul bull: 270x480, horizontal orbit — consecutive pair has modest motion
    let img_a = load_test_image("seoul_bull_sculpture_08.jpg");
    let img_b = load_test_image("seoul_bull_sculpture_09.jpg");
    let params = super::super::DisFlowParams::default_quality();

    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.01,
        "Seoul bull consecutive GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_vs_cpu_seoul_bull_wider_baseline() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Skip 2 frames for a wider baseline with more motion
    let img_a = load_test_image("seoul_bull_sculpture_05.jpg");
    let img_b = load_test_image("seoul_bull_sculpture_08.jpg");
    let params = super::super::DisFlowParams::default_quality();

    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.01,
        "Seoul bull wide baseline GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_vs_cpu_seattle_backyard_consecutive() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Seattle backyard: 360x640, forward push — consecutive pair
    let img_a = load_test_image("seattle_backyard_10.jpg");
    let img_b = load_test_image("seattle_backyard_11.jpg");
    let params = super::super::DisFlowParams::default_quality();

    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
            rmse < 0.01,
            "Seattle backyard consecutive GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
}

#[test]
#[ignore] // slow diagnostic test — run manually with: cargo test -p sfmtool-core --lib gpu::tests::diagnostic -- --ignored --nocapture
fn test_gpu_vs_cpu_seattle_backyard_panning_diagnostic() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    let img_a = load_test_image("seattle_backyard_20.jpg");
    let img_b = load_test_image("seattle_backyard_22.jpg");

    let w = img_a.width();
    let h = img_a.height();
    eprintln!("Image size: {w}x{h}");

    let base_params = super::super::DisFlowParams::default_quality();
    // Replicate compute_coarsest_scale logic
    let coarsest = (2.0 * w as f64 / (5.0 * base_params.patch_size as f64))
        .log2()
        .floor() as u32;
    let finest = coarsest.saturating_sub(2);
    eprintln!("Pyramid: coarsest={coarsest}, finest={finest}");

    // Test each scale independently to isolate divergence
    for test_scale in (finest..=coarsest).rev() {
        let params = super::super::DisFlowParams {
            finest_scale: Some(test_scale),
            coarsest_scale: Some(test_scale),
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        eprintln!(
            "  Scale {test_scale} only: RMSE={rmse:.6}, max_diff={max_diff:.4}, \
                 flow_size={}x{}",
            cpu_flow.width(),
            cpu_flow.height()
        );
    }

    // Test cumulative: coarsest down to each finer level
    for stop_scale in (finest..coarsest).rev() {
        let params = super::super::DisFlowParams {
            finest_scale: Some(stop_scale),
            coarsest_scale: Some(coarsest),
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        eprintln!("  Scales {coarsest}→{stop_scale}: RMSE={rmse:.6}, max_diff={max_diff:.4}");
    }

    // Full pipeline — find where the worst pixels are
    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &base_params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &base_params, Some(&gpu));

    // Find top-5 worst pixels
    let total = cpu_flow.width() as usize * cpu_flow.height() as usize;
    let mut diffs: Vec<(u32, u32, f32)> = Vec::new();
    for row in 0..cpu_flow.height() {
        for col in 0..cpu_flow.width() {
            let (cu, cv) = cpu_flow.get(col, row);
            let (gu, gv) = gpu_flow.get(col, row);
            let d = (cu - gu).abs().max((cv - gv).abs());
            diffs.push((col, row, d));
        }
    }
    diffs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    eprintln!("  Top-5 worst pixels:");
    for &(col, row, d) in diffs.iter().take(5) {
        let (cu, cv) = cpu_flow.get(col, row);
        let (gu, gv) = gpu_flow.get(col, row);
        eprintln!("    ({col}, {row}): diff={d:.4}, cpu=({cu:.3},{cv:.3}), gpu=({gu:.3},{gv:.3})");
    }

    // RMSE excluding pixels whose flow points outside the image (occluded regions)
    let fw = cpu_flow.width() as f32;
    let fh = cpu_flow.height() as f32;
    let mut inbounds_sum_sq = 0.0f64;
    let mut inbounds_count = 0usize;
    for row in 0..cpu_flow.height() {
        for col in 0..cpu_flow.width() {
            let (cu, cv) = cpu_flow.get(col, row);
            let tx = col as f32 + cu;
            let ty = row as f32 + cv;
            if tx >= 0.0 && tx < fw && ty >= 0.0 && ty < fh {
                let (gu, gv) = gpu_flow.get(col, row);
                let du = (cu - gu) as f64;
                let dv = (cv - gv) as f64;
                inbounds_sum_sq += du * du + dv * dv;
                inbounds_count += 1;
            }
        }
    }
    if inbounds_count > 0 {
        let inbounds_rmse = (inbounds_sum_sq / (2 * inbounds_count) as f64).sqrt();
        eprintln!(
            "  In-bounds RMSE={inbounds_rmse:.6} ({inbounds_count}/{total} pixels, \
                 {:.1}% excluded)",
            100.0 * (1.0 - inbounds_count as f64 / total as f64)
        );
    }

    // Count pixels above various thresholds
    let thresholds = [0.01, 0.1, 1.0, 5.0];
    for &t in &thresholds {
        let count = diffs.iter().filter(|d| d.2 > t).count();
        eprintln!(
            "    pixels with diff > {t}: {count} ({:.2}%)",
            100.0 * count as f64 / total as f64
        );
    }

    // Spatial distribution: which image quadrant are the diff>1 pixels in?
    let hw = cpu_flow.width() / 2;
    let hh = cpu_flow.height() / 2;
    let mut quadrants = [0u32; 4]; // TL, TR, BL, BR
                                   // Also check how many are within 8 pixels of any edge
    let mut edge_count = 0u32;
    for &(col, row, d) in &diffs {
        if d > 1.0 {
            let qi = if row < hh { 0 } else { 2 } + if col < hw { 0 } else { 1 };
            quadrants[qi as usize] += 1;
            if col < 8 || col >= cpu_flow.width() - 8 || row < 8 || row >= cpu_flow.height() - 8 {
                edge_count += 1;
            }
        }
    }
    eprintln!(
        "  Pixels with diff>1 by quadrant: TL={} TR={} BL={} BR={}",
        quadrants[0], quadrants[1], quadrants[2], quadrants[3]
    );
    eprintln!(
        "  Of those, within 8px of edge: {edge_count}/{}",
        quadrants.iter().sum::<u32>()
    );

    // Check if divergence is purely in the DIS patch phase by running
    // the full pipeline WITHOUT variational refinement
    // Check if divergence is purely in the variational refinement
    let no_var_params = super::super::DisFlowParams {
        variational_refinement: false,
        ..super::super::DisFlowParams::default_quality()
    };
    let cpu_novar = super::super::compute_optical_flow(&img_a, &img_b, &no_var_params, None);
    let gpu_novar = super::super::compute_optical_flow(&img_a, &img_b, &no_var_params, Some(&gpu));
    let (rmse_nv, max_nv) = flow_rmse(&cpu_novar, &gpu_novar);
    eprintln!("  Without variational: RMSE={rmse_nv:.6}, max_diff={max_nv:.4}");

    // Test DIS+variational via the combined path with more outer iterations,
    // starting from the identical DIS-only flow. This isolates whether
    // variational refinement diverges on this image pair.
    let var_3_params = super::super::DisFlowParams {
        variational_outer_iterations_base: 3,
        gpu_min_pixels: 0,
        ..super::super::DisFlowParams::default_quality()
    };
    let cpu_var_flow = super::super::compute_optical_flow(&img_a, &img_b, &var_3_params, None);
    let gpu_var_flow =
        super::super::compute_optical_flow(&img_a, &img_b, &var_3_params, Some(&gpu));
    let (rmse_v, max_v) = flow_rmse(&cpu_var_flow, &gpu_var_flow);
    eprintln!("  DIS+Variational (3 outer): RMSE={rmse_v:.6}, max_diff={max_v:.4}");
}

#[test]
fn test_gpu_vs_cpu_seattle_backyard_panning() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Seattle backyard: frames in the panning section (camera stops and pans right).
    // This pair has higher GPU/CPU divergence (~0.45 RMSE) than others (<0.001).
    // Investigation shows this is NOT a GPU variational bug:
    //   - DIS patch matching without variational: bit-exact (RMSE=0.0)
    //   - Variational from same init flow: near-exact (RMSE=0.00006)
    //   - Per-scale independently: near-exact (<0.00002)
    // The divergence comes from the non-convex DIS inverse search at scale 2:
    // tiny variational differences at coarser scales (RMSE=0.0001) change the
    // initial flow enough to push some patches into different local minima.
    // All affected pixels are in the top-left quadrant where flow vectors exceed
    // 40px (pointing off-frame), a poorly-conditioned occluded region.
    let img_a = load_test_image("seattle_backyard_20.jpg");
    let img_b = load_test_image("seattle_backyard_22.jpg");
    let params = super::super::DisFlowParams::default_quality();

    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.5,
        "Seattle backyard panning GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

// --- DIS inverse search + densification tests ---

#[test]
fn test_gpu_dis_identical_images() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Without variational refinement, DIS on identical images should give ~0 flow
    let img = GrayImage::checkerboard(64, 64);
    let params = super::super::DisFlowParams {
        variational_refinement: false,
        ..super::super::DisFlowParams::default_quality()
    };

    let cpu_flow = super::super::compute_optical_flow(&img, &img, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.01,
        "DIS-only identical GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_dis_shifted_image() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // DIS without variational on a known shift
    let ref_img = GrayImage::checkerboard(128, 128);
    let tgt_img = GrayImage::shifted(&ref_img, 3.0, 1.5);
    let params = super::super::DisFlowParams {
        variational_refinement: false,
        finest_scale: Some(0),
        coarsest_scale: Some(3),
        ..super::super::DisFlowParams::default_quality()
    };

    let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.1,
        "DIS-only shifted GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_dis_real_images() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    // Real images without variational — isolates DIS inverse search + densification
    let img_a = load_test_image("seoul_bull_sculpture_08.jpg");
    let img_b = load_test_image("seoul_bull_sculpture_09.jpg");
    let params = super::super::DisFlowParams {
        variational_refinement: false,
        ..super::super::DisFlowParams::default_quality()
    };

    let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
    let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

    let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
    assert!(
        rmse < 0.1,
        "DIS-only real images GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
    );
}

#[test]
fn test_gpu_pyramid_vs_cpu_pyramid() {
    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU test: no GPU available");
        return;
    };

    let img = load_test_image("seoul_bull_sculpture_08.jpg");
    let num_levels = 4;

    // Build CPU pyramid
    let cpu_pyr = super::super::pyramid::ImagePyramid::build(&img, num_levels);

    // Build GPU pyramid
    let device = &gpu.ctx.device;
    let queue = &gpu.ctx.queue;

    let pyr_pool =
        gpu.pyramid_pipeline
            .create_pool(device, img.width(), img.height(), num_levels as usize);

    // Use a dummy image for tgt (we only validate ref)
    let dummy = GrayImage::new_constant(img.width(), img.height(), 0.0);
    gpu.pyramid_pipeline.upload_and_build(
        &pyr_pool,
        device,
        queue,
        &img,
        &dummy,
        num_levels as usize,
    );

    // Read back each GPU level and compare to CPU
    for level in 0..num_levels as usize {
        let cpu_level = cpu_pyr.level(level);
        let n = (cpu_level.width() as usize) * (cpu_level.height() as usize);
        let gpu_data = read_buffer(device, queue, &pyr_pool.ref_level_bufs[level], n);

        let mut max_diff = 0.0f32;
        let mut sum_diff_sq = 0.0f64;
        for (cpu, gpu) in cpu_level.data().iter().zip(gpu_data.iter()) {
            let d = (cpu - gpu).abs();
            max_diff = max_diff.max(d);
            sum_diff_sq += (d as f64).powi(2);
        }
        let rmse = (sum_diff_sq / n as f64).sqrt();

        assert!(
            rmse < 1e-5,
            "Pyramid level {level} RMSE too large: {rmse:.8} (max_diff={max_diff:.8})"
        );
    }
}

/// Benchmark: GPU vs CPU optical flow pipeline.
///
/// Run with: cargo test -p sfmtool-core --release --lib gpu::tests::bench_gpu_vs_cpu -- --ignored --nocapture
#[test]
#[ignore]
fn bench_gpu_vs_cpu() {
    use std::time::Instant;

    let Some(gpu) = GpuFlowContext::new() else {
        eprintln!("Skipping GPU benchmark: no GPU available");
        return;
    };

    let pairs: &[(&str, &str, &str)] = &[
        (
            "seoul_bull",
            "seoul_bull_sculpture_08.jpg",
            "seoul_bull_sculpture_09.jpg",
        ),
        (
            "seattle_backyard",
            "seattle_backyard_10.jpg",
            "seattle_backyard_11.jpg",
        ),
        ("dino_dog_toy", "dino_dog_toy_42.jpg", "dino_dog_toy_43.jpg"),
    ];

    let presets: &[(&str, super::super::DisFlowParams)] = &[
        ("default", super::super::DisFlowParams::default_quality()),
        ("high_quality", super::super::DisFlowParams::high_quality()),
    ];

    for &(pair_name, file_a, file_b) in pairs {
        let img_a = load_test_image(file_a);
        let img_b = load_test_image(file_b);
        eprintln!("\n{pair_name} ({}x{}):", img_a.width(), img_a.height());

        for (preset_name, params) in presets {
            // Warmup
            let _ = super::super::compute_optical_flow(&img_a, &img_b, params, Some(&gpu));
            let _ = super::super::compute_optical_flow(&img_a, &img_b, params, None);

            let n = 5;
            let mut gpu_times = Vec::new();
            let mut cpu_times = Vec::new();

            for _ in 0..n {
                let t = Instant::now();
                let _ = super::super::compute_optical_flow(&img_a, &img_b, params, Some(&gpu));
                gpu_times.push(t.elapsed().as_secs_f64());
            }
            for _ in 0..n {
                let t = Instant::now();
                let _ = super::super::compute_optical_flow(&img_a, &img_b, params, None);
                cpu_times.push(t.elapsed().as_secs_f64());
            }

            gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let gpu_med = gpu_times[n / 2];
            let cpu_med = cpu_times[n / 2];

            eprintln!(
                "  {preset_name}: GPU={:.0}ms CPU={:.0}ms speedup={:.2}x",
                gpu_med * 1000.0,
                cpu_med * 1000.0,
                cpu_med / gpu_med,
            );
        }
    }
}
