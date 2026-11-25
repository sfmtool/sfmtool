// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlay ribbon store's regions, against a headless `wgpu` device.
//!
//! The device is wgpu's `noop` backend, which validates every write and keeps
//! no buffer contents. So the tests read what the store did from its own
//! record: the writes it issued, each as a byte range of the buffer, and the
//! buffer's bytes as those writes left them.

use super::{IssuedWrite, OverlayRibbons, OverlayTarget, ProducerId, STRIDE};
use crate::scene_renderer::gpu_types::OverlayInstance;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("noop device")
}

/// `n` instances, each told apart from every other by `tag` and its index.
fn instances(n: u32, tag: f32) -> Vec<OverlayInstance> {
    (0..n)
        .map(|i| {
            OverlayInstance::segment(
                [tag, i as f32, 0.0, 1.0],
                [tag, i as f32, 1.0, 1.0],
                [tag as u8, 128, 64, 255],
                1.5,
                OverlayInstance::HOLD,
            )
        })
        .collect()
}

/// The write of `n` instances at instance `offset` of the buffer.
fn issued(producer: ProducerId, offset: u32, n: u32) -> IssuedWrite {
    let start = u64::from(offset) * STRIDE;
    IssuedWrite {
        producer,
        bytes: start..start + u64::from(n) * STRIDE,
    }
}

/// Whether the buffer holds `expected` at `producer`'s region, as many as the
/// region counts.
fn holds(store: &OverlayRibbons, producer: ProducerId, expected: &[OverlayInstance]) -> bool {
    let (offset, count, _) = store.region(producer).expect("the region");
    let start = (u64::from(offset) * STRIDE) as usize;
    let bytes: &[u8] = bytemuck::cast_slice(expected);
    count as usize == expected.len() && store.contents()[start..start + bytes.len()] == *bytes
}

#[test]
fn a_write_within_capacity_writes_its_own_region_and_nothing_else() {
    let (device, queue) = device();
    let mut store = OverlayRibbons::default();
    let figure = instances(10, 1.0);
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &figure,
    );
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &instances(10, 2.0),
    );
    let (measurement_offset, _, capacity) = store.region(ProducerId::Measurements).unwrap();
    assert_eq!(capacity, 64);
    assert_eq!(measurement_offset, 64, "after the figure's region");
    store.issued.clear();

    let measurements = instances(40, 3.0);
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &measurements,
    );
    assert_eq!(
        store.issued,
        [issued(ProducerId::Measurements, measurement_offset, 40)]
    );
    assert_eq!(store.region(ProducerId::Measurements), Some((64, 40, 64)));
    assert!(holds(&store, ProducerId::Measurements, &measurements));
    assert!(holds(&store, ProducerId::BenchFigure, &figure), "untouched");

    store.issued.clear();
    let figure = instances(5, 4.0);
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &figure,
    );
    assert_eq!(store.issued, [issued(ProducerId::BenchFigure, 0, 5)]);
    assert!(holds(&store, ProducerId::BenchFigure, &figure));
    assert!(holds(&store, ProducerId::Measurements, &measurements));
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn growing_past_capacity_repacks_and_every_region_keeps_its_instances() {
    let (device, queue) = device();
    let mut store = OverlayRibbons::default();
    let measurements = instances(10, 2.0);
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &measurements,
    );
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &instances(10, 1.0),
    );
    assert_eq!(store.buffer_capacity(), 256);
    let first = store.buffer().expect("a buffer").clone();

    // The figure's region doubles to 128 and the measurements' moves after
    // it, within the buffer it has.
    store.issued.clear();
    let figure = instances(100, 5.0);
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &figure,
    );
    assert_eq!(store.region(ProducerId::BenchFigure), Some((0, 100, 128)));
    assert_eq!(store.region(ProducerId::Measurements), Some((128, 10, 64)));
    assert_eq!(
        store.issued,
        [
            issued(ProducerId::BenchFigure, 0, 100),
            issued(ProducerId::Measurements, 128, 10),
        ]
    );
    assert_eq!(store.buffer(), Some(&first), "it still fits");
    assert!(holds(&store, ProducerId::BenchFigure, &figure));
    assert!(holds(&store, ProducerId::Measurements, &measurements));

    // The last region grows, so nothing before it moves.
    store.issued.clear();
    let measurements = instances(100, 6.0);
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &measurements,
    );
    assert_eq!(
        store.region(ProducerId::Measurements),
        Some((128, 100, 128))
    );
    assert_eq!(store.issued, [issued(ProducerId::Measurements, 128, 100)]);
    assert_eq!(store.buffer(), Some(&first), "256 instances still fit");

    // Past the buffer: a new one, by doubling, with every region written into
    // it.
    store.issued.clear();
    let figure = instances(1000, 7.0);
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &figure,
    );
    assert_eq!(store.region(ProducerId::BenchFigure), Some((0, 1000, 1024)));
    assert_eq!(
        store.region(ProducerId::Measurements),
        Some((1024, 100, 128))
    );
    assert_eq!(store.buffer_capacity(), 2048);
    let grown = store.buffer().expect("a buffer");
    assert_ne!(grown, &first);
    assert_eq!(grown.size(), 2048 * STRIDE);
    assert_eq!(
        store.issued,
        [
            issued(ProducerId::BenchFigure, 0, 1000),
            issued(ProducerId::Measurements, 1024, 100),
        ]
    );
    assert!(holds(&store, ProducerId::BenchFigure, &figure));
    assert!(holds(&store, ProducerId::Measurements, &measurements));
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn an_empty_write_draws_nothing_and_keeps_the_capacity() {
    let (device, queue) = device();
    let mut store = OverlayRibbons::default();
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &instances(100, 1.0),
    );
    assert_eq!(store.region(ProducerId::Measurements), Some((0, 100, 128)));
    store.issued.clear();

    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &[],
    );
    assert_eq!(store.region(ProducerId::Measurements), Some((0, 0, 128)));
    assert!(store.issued.is_empty(), "{:?}", store.issued);
    assert_eq!(store.draws(OverlayTarget::MeasurementLayer).count(), 0);
    assert_eq!(store.drawn_count(OverlayTarget::MeasurementLayer), 0);

    // The room kept takes the next write without growing.
    let buffer = store.buffer().expect("a buffer").clone();
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &instances(100, 2.0),
    );
    assert_eq!(store.issued, [issued(ProducerId::Measurements, 0, 100)]);
    assert_eq!(store.buffer(), Some(&buffer));

    // An empty write from a producer that never wrote makes no buffer.
    let mut empty = OverlayRibbons::default();
    empty.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &[],
    );
    assert!(empty.buffer().is_none());
    assert_eq!(empty.draws(OverlayTarget::Scene).count(), 0);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn regions_are_drawn_in_the_fixed_order_whatever_order_they_were_written_in() {
    let (device, queue) = device();
    let mut store = OverlayRibbons::default();
    // Written last-first, and all on one target, so the order the draws come
    // in is the store's and not the writes'.
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::Scene,
        &instances(3, 2.0),
    );
    store.write(
        &device,
        &queue,
        ProducerId::BenchFigure,
        OverlayTarget::Scene,
        &instances(7, 1.0),
    );
    store.write(
        &device,
        &queue,
        ProducerId::TrackRays,
        OverlayTarget::Scene,
        &instances(5, 3.0),
    );
    let draws: Vec<_> = store.draws(OverlayTarget::Scene).collect();
    assert_eq!(
        draws,
        [
            (ProducerId::TrackRays, 0..5),
            (ProducerId::BenchFigure, 64..71),
            (ProducerId::Measurements, 128..131),
        ]
    );
    assert_eq!(store.drawn_count(OverlayTarget::Scene), 15);
    assert_eq!(store.draws(OverlayTarget::MeasurementLayer).count(), 0);

    // Each target draws only its own regions.
    store.write(
        &device,
        &queue,
        ProducerId::Measurements,
        OverlayTarget::MeasurementLayer,
        &instances(3, 2.0),
    );
    let scene: Vec<_> = store.draws(OverlayTarget::Scene).collect();
    let layer: Vec<_> = store.draws(OverlayTarget::MeasurementLayer).collect();
    assert_eq!(
        scene,
        [
            (ProducerId::TrackRays, 0..5),
            (ProducerId::BenchFigure, 64..71)
        ]
    );
    assert_eq!(layer, [(ProducerId::Measurements, 128..131)]);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}
