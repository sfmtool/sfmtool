// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::*;

#[test]
fn x86_64_levels_follow_avx2_and_fma() {
    assert_eq!(SimdLevel::x86_64(true, true), SimdLevel::Avx2Fma);
    assert_eq!(SimdLevel::x86_64(true, false), SimdLevel::Avx2);
    assert_eq!(SimdLevel::x86_64(false, true), SimdLevel::Sse2);
    assert_eq!(SimdLevel::x86_64(false, false), SimdLevel::Sse2);
    assert_eq!(SimdLevel::Avx2Fma.label(), "AVX2+FMA");
    assert_eq!(SimdLevel::Scalar.label(), "none (scalar)");
}

#[test]
fn simd_level_matches_the_kernels_checks() {
    #[cfg(target_arch = "x86_64")]
    let expected = match (
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
    ) {
        (true, true) => SimdLevel::Avx2Fma,
        (true, false) => SimdLevel::Avx2,
        (false, _) => SimdLevel::Sse2,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let expected = SimdLevel::Scalar;
    assert_eq!(simd_level(), expected);
}

#[test]
fn model_name_is_trimmed_at_the_first_nul() {
    let mut bytes = b"       Intel(R) Core(TM)  i9-14900HX".to_vec();
    bytes.extend_from_slice(&[0, 0, b'x']);
    assert_eq!(
        clean_model_name(&bytes).as_deref(),
        Some("Intel(R) Core(TM) i9-14900HX")
    );
    assert_eq!(clean_model_name(&[0; 48]), None);
    assert_eq!(clean_model_name(b"   "), None);
}

#[test]
fn rayon_threads_reads_the_pool_it_runs_in() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    assert_eq!(pool.install(rayon_threads), 3);
}

fn adapter() -> GpuAdapter {
    GpuAdapter {
        name: "Test GPU".into(),
        backend: "vulkan".into(),
        device_type: GpuDeviceType::Cpu,
    }
}

#[test]
fn wait_for_returns_the_request_answer() {
    let found = wait_for(|| Ok(adapter()), Duration::from_secs(10), &Progress::none());
    assert_eq!(found, Ok(GpuProbe::Found(adapter())));
    let none = wait_for(
        || Err("no adapter".into()),
        Duration::from_secs(10),
        &Progress::none(),
    );
    assert_eq!(none, Ok(GpuProbe::NoAdapter("no adapter".into())));
}

#[test]
fn wait_for_reports_a_panicking_request_as_failed() {
    let failed = wait_for(
        || panic!("driver crashed"),
        Duration::from_secs(10),
        &Progress::none(),
    );
    assert_eq!(failed, Ok(GpuProbe::Failed));
}

#[test]
fn wait_for_gives_up_at_the_timeout() {
    let timeout = Duration::from_millis(100);
    let start = Instant::now();
    let hung = wait_for(
        || {
            std::thread::sleep(Duration::from_secs(5));
            Ok(adapter())
        },
        timeout,
        &Progress::none(),
    );
    assert_eq!(hung, Ok(GpuProbe::TimedOut(timeout)));
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn wait_for_stops_when_cancelled() {
    let cancel = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&cancel);
    let result = wait_for(
        || {
            std::thread::sleep(Duration::from_secs(5));
            Ok(adapter())
        },
        Duration::from_secs(10),
        &progress,
    );
    assert_eq!(result, Err(Cancelled));
}

#[test]
fn wait_for_takes_a_timeout_too_long_for_a_deadline() {
    let found = wait_for(|| Ok(adapter()), Duration::MAX, &Progress::none());
    assert_eq!(found, Ok(GpuProbe::Found(adapter())));
}
