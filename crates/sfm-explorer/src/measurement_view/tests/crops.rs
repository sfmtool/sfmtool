// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests of Measurement View's endpoint crops: they are cut from
//! the photograph tiles they need, read off the GUI thread, and equal the
//! crops the whole pyramid gives.

use std::collections::HashMap;
use std::sync::Arc;

use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::{PhotographCache, PhotographCacheOptions};

use super::super::*;
use super::{input, painted};
use crate::patch_pictures::tests::whole_pyramid_pixels;
use crate::state::{AppState, PhotographRequests, TaskQueue};

/// The pictures the frames uploaded, by texture.
type Uploaded = HashMap<egui::TextureId, egui::ColorImage>;

/// The fixture's measurement whose endpoints are in two photographs: A on a
/// point in image 0, B moved to a pixel of image 1 no square holds.
const TWO_PHOTOGRAPHS: usize = 1;

/// Where B is moved to: its crop lies in the first 512 px column of
/// photograph tiles, and its hover view, three times as wide, reaches
/// into the second, so the two pictures need different tiles.
const B_PIXEL: [f32; 2] = [440.0, 240.0];

/// One frame of the panel, adding every whole picture it uploaded to
/// `uploaded`.
fn uploading_frame(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    uploaded: &mut Uploaded,
) {
    let mut output = ctx.run_ui(input(Vec::new()), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    });
    for (id, deltas) in &output.textures_delta.set {
        for delta in deltas {
            if let (None, egui::ImageData::Color(image)) = (delta.pos, &delta.image) {
                uploaded.insert(*id, (**image).clone());
            }
        }
    }
    output.textures_delta.clear();
}

/// The measurement fixture with its photographs of images 0 and 1 written as
/// PNGs of hashed pixels in `dir`, its workspace, and B of
/// [`TWO_PHOTOGRAPHS`] at [`B_PIXEL`], in a state whose photograph cache
/// has `budget_bytes` and whose reads wait in the queue answered beside it,
/// with that measurement selected.
fn on_disk(dir: &std::path::Path, budget_bytes: u64) -> (AppState, ReconId, TaskQueue) {
    let mut recon = crate::measurements::fixture::measured_reconstruction(Some("m"));
    recon.workspace_dir = dir.to_path_buf();
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("image_{i:03}.png");
    }
    let camera = recon.image_set.cameras[0].clone();
    for i in 0..2u32 {
        let seed = i.wrapping_mul(0x9e37_79b9);
        image::RgbImage::from_fn(camera.width, camera.height, |x, y| {
            let p = (y * camera.width + x).wrapping_add(seed);
            let h = |c: u32| {
                (p.wrapping_mul(3)
                    .wrapping_add(c)
                    .wrapping_mul(2_654_435_761)
                    >> 13) as u8
            };
            image::Rgb([h(0), h(1), h(2)])
        })
        .save(dir.join(format!("image_{i:03}.png")))
        .unwrap();
    }
    let measurements = recon.image_set.measurements.distance.as_mut().unwrap();
    measurements.endpoints_xy[[TWO_PHOTOGRAPHS, 1, 0]] = B_PIXEL[0];
    measurements.endpoints_xy[[TWO_PHOTOGRAPHS, 1, 1]] = B_PIXEL[1];
    let mut state = AppState::new();
    state.photographs = Arc::new(PhotographCache::new(PhotographCacheOptions {
        budget_bytes,
        ..Default::default()
    }));
    let (requests, queue) = PhotographRequests::queued();
    state.photograph_requests = requests;
    let id = state.append_node(crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    state.select_measurement(MeasurementRef::distance(id, TWO_PHOTOGRAPHS));
    (state, id, queue)
}

/// The two endpoints' crops and hover views, read from the photograph tiles
/// they need with a cache budget and with one that keeps nothing, equal the
/// crops the whole pyramid gives. Until they are drawn each crop says
/// *loading…*, each photograph's read is asked for once however many frames
/// look, and nothing is decoded on the GUI thread; a crop and its hover view
/// need different tiles and are asked for together, so the frames need no
/// more reads than one render does.
#[test]
fn endpoint_crops_drawn_from_photograph_tiles_equal_the_whole_pyramid_crops() {
    // A header read and a decode that keeps the photograph; with a budget of
    // 0, a header read, the first tile and the tiles both pictures need.
    for (budget, reads) in [(1u64 << 30, 2), (0, 3)] {
        let dir = tempfile::tempdir().unwrap();
        let (state, id, queue) = on_disk(dir.path(), budget);
        let mut panel = MeasurementView::new();
        let ctx = egui::Context::default();
        let mut uploaded = Uploaded::new();
        let mut rounds = 0;
        loop {
            uploading_frame(&mut panel, &ctx, &state, &mut uploaded);
            if panel.pictures.len() == 2 {
                break;
            }
            assert!(rounds < 6, "the crops were never drawn, budget {budget}");
            let decodes = state.photographs.stats().decodes;
            for _ in 0..3 {
                let texts = painted(&mut panel, &ctx, &state);
                let loading = texts.iter().filter(|t| *t == "loading\u{2026}").count();
                assert_eq!(loading, 2, "{texts:?}");
            }
            assert_eq!(queue.len(), 2, "one read a photograph, budget {budget}");
            assert_eq!(state.photographs.stats().decodes, decodes, "no decode here");
            assert_eq!(queue.run_off_this_thread(), 2);
            rounds += 1;
        }
        assert_eq!(rounds, reads, "budget {budget}");
        assert_eq!(queue.len(), 0);

        let node = state.node(id).expect("loaded");
        let recon = node.recon();
        let readings = crate::measurements::MeasurementReadings::of(node.edited());
        let resolved = &readings.distance[TWO_PHOTOGRAPHS];
        let row =
            &crate::measurements::distance_rows(&recon.image_set.measurements)[TWO_PHOTOGRAPHS];
        for side in EndpointSide::BOTH {
            let image = row.image(side);
            let pixel = row.pixel(side);
            let key = PictureKey::new(
                id,
                node.history.current_version().document_serial,
                side,
                image,
                pixel,
            );
            let Crop::Ready {
                cell,
                context,
                caption,
            } = &panel.pictures[&key].crop
            else {
                panic!("endpoint {side:?} has no crop");
            };
            let square = crate::measurements::endpoint(resolved, side)
                .as_ref()
                .ok()
                .map(|resolution| (resolution.chosen().point, resolution.chosen().observation));
            assert_eq!(square.is_some(), side == EndpointSide::A, "{caption}");
            let (outline, _) = endpoint_outline(node, image as usize, pixel, square);
            let region = crop::region(&outline).expect("a crop region");
            let path = crate::state::photograph_path(recon, image as usize).unwrap();
            let decoded = ImageU8::read_rgb(&path).unwrap();
            let levels = ImageU8Pyramid::full_levels(decoded.width(), decoded.height());
            let whole = ImageU8Pyramid::from_image(decoded, levels);
            let drawn = &uploaded[&cell.texture_id()];
            assert!(
                *drawn == whole_pyramid_pixels(&whole, region, drawn.size, false),
                "the crop of {side:?}, budget {budget}"
            );
            let context = context.as_ref().expect("a hover view");
            let drawn = &uploaded[&context.texture_id()];
            assert!(
                *drawn == whole_pyramid_pixels(&whole, region, drawn.size, true),
                "the hover view of {side:?}, budget {budget}"
            );
        }
    }
}

/// An endpoint whose photograph cannot be read, found once its header read
/// has run off the GUI thread, says *no crop*, and why on hover.
#[test]
fn an_endpoint_whose_photograph_cannot_be_read_says_so() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    let (requests, queue) = PhotographRequests::queued();
    state.photograph_requests = requests;
    state.select_measurement(MeasurementRef::distance(id, TWO_PHOTOGRAPHS));
    let mut panel = MeasurementView::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(texts.iter().filter(|t| *t == "loading\u{2026}").count(), 2);
    assert_eq!(queue.run_off_this_thread(), 2, "a header read a photograph");
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(
        texts.iter().filter(|t| *t == "no crop").count(),
        2,
        "{texts:?}"
    );
    assert_eq!(panel.pictures.len(), 2);
    for pictures in panel.pictures.values() {
        assert!(matches!(
            pictures.crop,
            Crop::Absent("The photograph could not be read")
        ));
    }
    assert_eq!(queue.len(), 0, "nothing more asked for");
}

/// An endpoint's crop drawn after the read of its photograph's header failed
/// says *no crop*, and that answer is asked again rather than kept, so the
/// crop is drawn once the photograph arrives. A frame meets this when the
/// header read ends while it draws the panel.
#[test]
fn an_unreadable_crop_is_drawn_once_its_photograph_arrives() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    let (requests, queue) = PhotographRequests::queued();
    state.photograph_requests = requests;
    state.select_measurement(MeasurementRef::distance(id, TWO_PHOTOGRAPHS));
    let mut panel = MeasurementView::new();
    let ctx = egui::Context::default();
    painted(&mut panel, &ctx, &state);
    assert_eq!(queue.run_off_this_thread(), 2, "a header read a photograph");
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(texts.iter().filter(|t| *t == "no crop").count(), 2);

    let recon = state.node(id).expect("loaded").recon();
    let camera = recon.image_set.cameras[0].clone();
    for image in 0..2 {
        state.insert_photograph(
            id,
            image,
            ImageU8::new(
                camera.width,
                camera.height,
                3,
                vec![90; (camera.width * camera.height * 3) as usize],
            ),
        );
    }
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(
        texts.iter().filter(|t| *t == "no crop").count(),
        0,
        "{texts:?}"
    );
    assert!(panel
        .pictures
        .values()
        .all(|pictures| matches!(pictures.crop, Crop::Ready { .. })));
    assert_eq!(queue.len(), 0, "nothing more asked for");
}
