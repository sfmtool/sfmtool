// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Track Table's Crop column: when it is offered, what
//! each cell draws and why, the wait for the scroll to stop before anything is
//! asked for, and the reference picks for the points that store none.

use std::sync::{mpsc, Arc};

use ndarray::Array4;
use sfmtool_core::analysis::track_table::{TrackTableColumn, TrackTableQueryOptions};
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

use super::super::pictures::{Absent, CellPicture, PictureColumn};
use super::super::{TrackTablePanel, SCROLL_SETTLE_S};
use crate::background::{Finished, Operation};
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;
use crate::test_support::{painted_text_rects, painted_texts, run_frame_headless};

/// How many points the fixture holds: enough that a short window shows a few
/// rows and a scroll passes through many.
const POINTS: usize = 60;

/// Points below this store reference observation 0, whose photograph is image
/// 0's. Above it, a point seen once stores 0, a point seen three times stores
/// none, and a point seen twice stores 1, in image 1.
const FIRST_FAR: u32 = 12;

/// A point with zero patch half-vectors: no patch frame.
const NO_FRAME: u32 = 3;
/// A point moved behind image 0's camera, its patch facing away from it: no
/// sample of its outline projects into the photograph.
const BEHIND: u32 = 4;
/// A point at infinity, a bearing that projects onto its keypoints.
const BEARING: u32 = 7;

/// The reference observation the fixture stores for point `p`.
fn stored_reference(p: usize) -> i32 {
    let seen = p % 3 + 1;
    if (p as u32) < FIRST_FAR || seen == 1 {
        0
    } else if seen == 3 {
        -1
    } else {
        1
    }
}

/// An `embedded_patches` reconstruction of [`POINTS`] points with stored 8 × 8
/// bitmaps and the references of [`stored_reference`], with [`NO_FRAME`],
/// [`BEHIND`] and [`BEARING`] as their docs say.
fn crop_recon() -> SfmrReconstruction {
    let mut recon = projected_embedded_demo(POINTS);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::from_shape_fn(
        (POINTS, 8, 8, 4),
        |(p, y, x, c)| (20 + p + 9 * y + 5 * x + 30 * c) as u8,
    )));
    recon.point_set.reference_observations = Some((0..POINTS).map(stored_reference).collect());
    let mut u = recon
        .point_set
        .patch_u_halfvec_xyz
        .take()
        .expect("patch frames");
    let mut v = recon
        .point_set
        .patch_v_halfvec_xyz
        .take()
        .expect("patch frames");
    for c in 0..3 {
        u[[NO_FRAME as usize, c]] = 0.0;
        v[[NO_FRAME as usize, c]] = 0.0;
    }

    // Behind image 0's camera, which looks along -Z, the patch parallel to
    // its sensor, so the keypoint's ray never meets the patch's plane in
    // front of it.
    let image = recon.image_set.images[0].clone();
    let rotation = image.quaternion_wxyz.to_rotation_matrix();
    let to_world = |cam: nalgebra::Vector3<f64>| rotation.inverse() * (cam - image.translation_xyz);
    let behind = to_world(nalgebra::Vector3::new(0.0, 0.0, 1.0));
    recon.point_set.points[BEHIND as usize].position = nalgebra::Point3::from(behind);
    for (axis, half) in [(0, &mut u), (1, &mut v)] {
        let direction = rotation.inverse() * nalgebra::Vector3::ith(axis, 1.0);
        for c in 0..3 {
            half[[BEHIND as usize, c]] = (direction[c] * 0.02) as f32;
        }
    }

    // A bearing: the direction from image 0's centre to where the point was,
    // its keypoints its projections at infinity, its patch tangent to the
    // direction sphere.
    let point = &mut recon.point_set.points[BEARING as usize];
    let direction = (point.position - image.camera_center()).normalize();
    point.position = nalgebra::Point3::from(direction);
    point.w = 0.0;
    let across = direction.cross(&nalgebra::Vector3::z()).normalize();
    let up = direction.cross(&across).normalize();
    for c in 0..3 {
        u[[BEARING as usize, c]] = (across[c] * 0.01) as f32;
        v[[BEARING as usize, c]] = (up[c] * 0.01) as f32;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.rebuild_derived_fields();
    let start = recon.point_set.observation_offsets[BEARING as usize];
    let end = recon.point_set.observation_offsets[BEARING as usize + 1];
    for row in start..end {
        let seen = &recon.image_set.images[recon.point_set.tracks[row].image_index as usize];
        let cam = seen.quaternion_wxyz.to_rotation_matrix() * direction;
        let camera = &recon.image_set.cameras[seen.camera_index as usize];
        let (x, y) = camera
            .ray_to_pixel([cam.x, cam.y, cam.z])
            .expect("the bearing is in front of the demo cameras");
        if let sfmtool_core::ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
            &mut recon.point_set.observations
        {
            keypoints_xy[[row, 0]] = x as f32;
            keypoints_xy[[row, 1]] = y as f32;
        }
    }
    recon.rebuild_derived_fields();
    recon
}

/// A textured photograph for image `image` of `id`, the size its camera says.
fn cache_photograph(state: &AppState, id: ReconId, image: usize) {
    let node = state.node(id).expect("loaded");
    let camera = &node.recon().image_set.cameras[0];
    let (w, h) = (camera.width, camera.height);
    let data: Vec<u8> = (0..(w * h * 3))
        .map(|i| {
            let p = i / 3;
            ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
        })
        .collect();
    state.insert_photograph(id, image, ImageU8::new(w, h, 3, data));
}

/// A state holding the [`crop_recon`] node, selected, with the photographs of
/// `cached` in the photograph cache.
fn state_with(recon: SfmrReconstruction, cached: &[usize]) -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    state.select_recon(id);
    for &image in cached {
        cache_photograph(&state, id, image);
    }
    (state, id)
}

/// The fixture with every photograph its tracks see cached.
fn crop_state() -> (AppState, ReconId) {
    state_with(crop_recon(), &[0, 1, 2])
}

/// A window `height` points high at egui time `time`.
fn at(height: f32, time: f64) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, height),
        )),
        time: Some(time),
        ..Default::default()
    }
}

/// A clock for the frames of one test, a frame each 1/60 s.
struct Clock(f64);

impl Clock {
    fn tick(&mut self) -> f64 {
        self.0 += 1.0 / 60.0;
        self.0
    }

    fn wait(&mut self, seconds: f64) -> f64 {
        self.0 += seconds;
        self.0
    }
}

/// A panel drawn in a window `height` high, its table job run, and drawn again.
fn shown(state: &mut AppState, height: f32, clock: &mut Clock) -> (egui::Context, TrackTablePanel) {
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    run_frame_headless(&ctx, at(height, clock.tick()), |ui| panel.show(ui, state));
    state.settle_track_table_columns();
    run_frame_headless(&ctx, at(height, clock.tick()), |ui| panel.show(ui, state));
    (ctx, panel)
}

/// One frame at the clock's next tick.
fn frame(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    height: f32,
    time: f64,
) {
    run_frame_headless(ctx, at(height, time), |ui| panel.show(ui, state));
}

/// What `point`'s cell in `column` drew on the last frame.
fn cell(panel: &TrackTablePanel, point: u32, column: PictureColumn) -> Option<CellPicture> {
    panel
        .drawn_pictures
        .iter()
        .find(|&&(p, c, _)| p == point && c == column)
        .map(|&(_, _, picture)| picture)
}

/// The Crop column is offered exactly where the points carry a patch frame and
/// the observations a keypoint. It is shown by default, between Point and
/// Patch.
#[test]
fn the_crop_column_shows_only_with_patch_frames_and_keypoints() {
    let mut clock = Clock(0.0);
    // No patch frames: the plain demo, with bitmaps so the Patch column shows.
    let mut plain = SfmrReconstruction::demo(20);
    plain.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::from_elem((20, 8, 8, 4), 90u8)));
    let (mut state, _) = state_with(plain, &[]);
    let (ctx, mut panel) = shown(&mut state, 1200.0, &mut clock);
    let texts = painted_texts(&ctx, at(1200.0, clock.tick()), |ui| {
        panel.show(ui, &mut state)
    });
    assert!(texts.iter().any(|t| t == "Patch"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Crop"), "{texts:?}");

    // Patch frames and no keypoints.
    let mut framed = SfmrReconstruction::demo(20);
    let frames = projected_embedded_demo(20).point_set;
    framed.point_set.patch_u_halfvec_xyz = frames.patch_u_halfvec_xyz;
    framed.point_set.patch_v_halfvec_xyz = frames.patch_v_halfvec_xyz;
    framed.rebuild_derived_fields();
    let (mut state, _) = state_with(framed, &[]);
    let (ctx, mut panel) = shown(&mut state, 1200.0, &mut clock);
    let texts = painted_texts(&ctx, at(1200.0, clock.tick()), |ui| {
        panel.show(ui, &mut state)
    });
    assert!(!texts.iter().any(|t| t == "Crop"), "{texts:?}");

    // Both: Crop between Point and Patch, and shown by default.
    let (mut state, _) = crop_state();
    let (ctx, mut panel) = shown(&mut state, 1200.0, &mut clock);
    let texts = painted_text_rects(&ctx, at(1200.0, clock.tick()), |ui| {
        panel.show(ui, &mut state)
    });
    let left_of = |text: &str| {
        texts
            .iter()
            .find(|t| t.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .left()
    };
    assert!(left_of("Point") < left_of("Crop"));
    assert!(left_of("Crop") < left_of("Patch"));
    assert!(left_of("Patch") < left_of("Track"));
    assert!(panel.query.pictures.contains(&PictureColumn::Crop));
    assert!(matches!(
        cell(&panel, 0, PictureColumn::Crop),
        Some(CellPicture::Texture(_))
    ));
}

/// A cell says why it has no picture: *no patch* for a point with no patch
/// frame, *off image* for one whose outline lands nowhere in the photograph,
/// *unreadable* for a photograph that cannot be read. A point at infinity
/// gets its crop like any other.
#[test]
fn each_crop_cell_draws_the_picture_or_says_why_not() {
    let mut clock = Clock(0.0);
    // Image 1's photograph is not cached and its file does not exist.
    let (mut state, _) = state_with(crop_recon(), &[0, 2]);
    let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);
    assert!(matches!(
        cell(&panel, 0, PictureColumn::Crop),
        Some(CellPicture::Texture(_))
    ));
    assert!(matches!(
        cell(&panel, BEARING, PictureColumn::Crop),
        Some(CellPicture::Texture(_))
    ));
    assert_eq!(
        cell(&panel, NO_FRAME, PictureColumn::Crop),
        Some(CellPicture::Absent(Absent::NoPatch))
    );
    assert_eq!(
        cell(&panel, BEHIND, PictureColumn::Crop),
        Some(CellPicture::Absent(Absent::OffImage))
    );
    // Image 1 is asked for and is decoding: the cells wait in an empty frame.
    let in_image_1 = (FIRST_FAR..POINTS as u32)
        .find(|&p| stored_reference(p as usize) == 1)
        .unwrap();
    assert_eq!(
        cell(&panel, in_image_1, PictureColumn::Crop),
        Some(CellPicture::Waiting)
    );
    assert!(panel.requested_photographs.contains(&1));
    // The decode fails, which the cache remembers, and the cell says so.
    for _ in 0..500 {
        frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
        if cell(&panel, in_image_1, PictureColumn::Crop) != Some(CellPicture::Waiting) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(
        cell(&panel, in_image_1, PictureColumn::Crop),
        Some(CellPicture::Absent(Absent::Unreadable))
    );
    let texts = painted_texts(&ctx, at(4800.0, clock.tick()), |ui| {
        panel.show(ui, &mut state)
    });
    for said in ["no patch", "off image", "unreadable"] {
        assert!(texts.iter().any(|t| t == said), "{said}: {texts:?}");
    }
}

/// Zero the bitmap rows of every point [`stored_reference`] leaves at -1, as
/// a bitmap writer does for a point whose reference-view rule found no
/// eligible view.
fn rejected_by_the_rule(recon: &mut SfmrReconstruction) {
    let mut bitmaps = (**recon.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap()).clone();
    for p in 0..POINTS {
        if stored_reference(p) == -1 {
            bitmaps.index_axis_mut(ndarray::Axis(0), p).fill(0);
        }
    }
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
}

/// A point a bitmap writer left at -1 with a zero bitmap row has no reference
/// to find: the writer ran the rule on the same keypoints and photographs and
/// found no eligible view. That holds for the open's display render and for a
/// file whose bitmaps are stored, as Convert to Embedded Patches writes them,
/// so such a point is not picked again. Its cell says *no reference* at once,
/// and nothing is queued for it.
#[test]
fn a_point_the_rule_already_rejected_says_so_without_a_pick() {
    for for_display in [true, false] {
        let mut clock = Clock(0.0);
        let mut recon = crop_recon();
        recon.point_set.patch_bitmaps_for_display = for_display;
        rejected_by_the_rule(&mut recon);
        let (mut state, _) = state_with(recon, &[0, 1, 2]);
        let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);
        let unreferenced = (FIRST_FAR..POINTS as u32)
            .find(|&p| stored_reference(p as usize) == -1)
            .unwrap();
        assert_eq!(
            cell(&panel, unreferenced, PictureColumn::Crop),
            Some(CellPicture::Absent(Absent::NoReference)),
            "for display: {for_display}"
        );
        assert_eq!(state.reference_picks.wanted_points(), Some(&[][..]));
        state.drive_track_table_picks();
        assert_eq!(state.reference_picks.queued_task(), None);
        assert_eq!(state.reference_picks.tasks_queued, 0);
        let texts = painted_texts(&ctx, at(4800.0, clock.tick()), |ui| {
            panel.show(ui, &mut state)
        });
        assert!(texts.iter().any(|t| t == "no reference"), "{texts:?}");
    }
}

/// Deleting an image drops the reference of every point whose reference
/// observation was in it, and those points keep the bitmap rendered from it.
/// On a reconstruction the open rendered display bitmaps for, they are not
/// read as rejected by the rule: they are queued for a pick, which finds a
/// reference among the views left, while a point the rule did reject still
/// says *no reference* without one.
#[test]
fn a_point_that_lost_its_reference_to_an_edit_is_picked_again() {
    let mut clock = Clock(0.0);
    let mut recon = crop_recon();
    recon.point_set.patch_bitmaps_for_display = true;
    // Above FIRST_FAR, points seen twice store reference 1, in image 1, and
    // so do the points seen three times, except the first, which the rule
    // rejected.
    let rejected = (FIRST_FAR..POINTS as u32)
        .find(|&p| stored_reference(p as usize) == -1)
        .unwrap();
    let references = recon.point_set.reference_observations.as_mut().unwrap();
    for (p, reference) in references.iter_mut().enumerate().skip(FIRST_FAR as usize) {
        if stored_reference(p) == -1 && p as u32 != rejected {
            *reference = 1;
        }
    }
    let in_image_1 = references.iter().filter(|&&r| r == 1).count();
    let mut bitmaps = (**recon.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap()).clone();
    bitmaps
        .index_axis_mut(ndarray::Axis(0), rejected as usize)
        .fill(0);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    let (mut state, id) = state_with(recon, &[0, 1, 2]);
    let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);

    state
        .delete_image(crate::scene::ImageRef::new(id, 1))
        .expect("deletable");
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    frame(
        &ctx,
        &mut panel,
        &mut state,
        4800.0,
        clock.wait(SCROLL_SETTLE_S + 0.01),
    );

    let node = state.node(id).expect("loaded");
    let edited = node.edited();
    let lost: Vec<u32> = panel
        .drawn()
        .iter()
        .map(|&(p, _)| p)
        .filter(|&p| {
            let view = edited.point(p).unwrap();
            view.reference_observation() == Some(-1)
                && view.patch_bitmap().unwrap().iter().any(|&b| b != 0)
        })
        .collect();
    assert_eq!(
        lost.len(),
        in_image_1,
        "the points whose reference was in image 1"
    );
    // The ones seen three times keep two views to pick from.
    let two_left: Vec<u32> = lost
        .iter()
        .copied()
        .filter(|&p| edited.point(p).unwrap().observations().len() >= 2)
        .collect();
    assert!(!two_left.is_empty());
    let wanted = state.reference_picks.wanted_points().expect("picks wanted");
    assert_eq!(wanted, &lost[..], "they are queued for a pick");
    for &p in &lost {
        assert_eq!(
            cell(&panel, p, PictureColumn::Crop),
            Some(CellPicture::Waiting)
        );
    }
    // The one the rule rejected, wherever the edit moved it, says so at once.
    let said: Vec<u32> = panel
        .drawn_pictures
        .iter()
        .filter(|&&(_, c, picture)| {
            c == PictureColumn::Crop && picture == CellPicture::Absent(Absent::NoReference)
        })
        .map(|&(p, _, _)| p)
        .collect();
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(edited
        .point(said[0])
        .unwrap()
        .patch_bitmap()
        .unwrap()
        .iter()
        .all(|&b| b == 0));

    state.settle_reference_picks();
    let document = state.scene[0].history.current_version().document_serial;
    for &p in &two_left {
        assert!(
            matches!(state.reference_pick(id, p, document), Some(Some(_))),
            "#{p} was picked from the views left"
        );
    }
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    for &p in &two_left {
        assert!(matches!(
            cell(&panel, p, PictureColumn::Crop),
            Some(CellPicture::Texture(_))
        ));
    }
}

/// A point that stores no reference waits in an empty frame while its pick is
/// made, then draws the crop of the photograph the rule picked; the pick is
/// held by the panel's state, not written into the version.
#[test]
fn a_point_with_no_stored_reference_is_cropped_from_the_rule_s_pick() {
    let mut clock = Clock(0.0);
    let (mut state, id) = crop_state();
    let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);
    let unreferenced: Vec<u32> = (0..POINTS as u32)
        .filter(|&p| stored_reference(p as usize) == -1)
        .collect();
    assert!(!unreferenced.is_empty());
    for &p in &unreferenced {
        assert_eq!(
            cell(&panel, p, PictureColumn::Crop),
            Some(CellPicture::Waiting)
        );
    }
    assert_eq!(
        state.reference_picks.wanted_points(),
        Some(&unreferenced[..])
    );
    let versions = state.scene[0].history.current_version().serial;
    state.settle_reference_picks();
    let document = state.scene[0].history.current_version().document_serial;
    for &p in &unreferenced {
        let pick = state
            .reference_pick(id, p, document)
            .unwrap_or_else(|| panic!("#{p} was picked"));
        let seen = p as usize % 3 + 1;
        assert!(pick.is_none_or(|r| (r as usize) < seen), "#{p}: {pick:?}");
    }
    assert_eq!(
        state.scene[0].history.current_version().serial,
        versions,
        "no version was pushed"
    );
    assert_eq!(
        state
            .node(id)
            .unwrap()
            .edited()
            .point(unreferenced[0])
            .unwrap()
            .reference_observation(),
        Some(-1),
        "the version still stores no reference"
    );
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    for &p in &unreferenced {
        let drawn = cell(&panel, p, PictureColumn::Crop).unwrap();
        let pick = state.reference_pick(id, p, document).unwrap();
        match pick {
            Some(_) => assert!(matches!(drawn, CellPicture::Texture(_)), "#{p}: {drawn:?}"),
            None => assert_eq!(drawn, CellPicture::Absent(Absent::NoReference), "#{p}"),
        }
    }
}

/// Scroll the short window down by `delta` points over the rows, at `time`.
fn wheel(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    time: f64,
    delta: f32,
) {
    let mut raw = at(400.0, time);
    raw.events = vec![
        egui::Event::PointerMoved(egui::pos2(400.0, 300.0)),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -delta),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    run_frame_headless(ctx, raw, |ui| panel.show(ui, state));
}

/// While the rows on screen change from frame to frame, the crops ask for no
/// photograph and no pick and draw only what is held. Once the rows have
/// stayed put for the settle time, the visible rows' photographs are asked
/// for and their picks queued.
#[test]
fn nothing_is_asked_for_until_the_scroll_stops() {
    let mut clock = Clock(0.0);
    // Image 1 is not cached, so a row whose crop is cut from it asks for it.
    let (mut state, _) = state_with(crop_recon(), &[0, 2]);
    let (ctx, mut panel) = shown(&mut state, 400.0, &mut clock);
    let top: Vec<u32> = panel.drawn().iter().map(|&(p, _)| p).collect();
    assert!(top.iter().all(|&p| p < FIRST_FAR), "{top:?}");
    assert!(
        panel.requested_photographs.is_empty(),
        "the top rows are in image 0"
    );
    assert_eq!(state.reference_picks.wanted_points(), Some(&[][..]));

    state.hide_track_table();
    let mut passed = Vec::new();
    let mut moving_frames = 0;
    for _ in 0..40 {
        let now = clock.tick();
        wheel(&ctx, &mut panel, &mut state, now, 30.0);
        state.drive_track_table_picks();
        passed.extend(panel.drawn().iter().map(|&(p, _)| p));
        // The wheel's first frame can draw the rows where they were, before
        // the scroll has moved them; every frame after it moves them or
        // follows a move within the settle time.
        let moving = panel
            .motion
            .changed_at
            .is_some_and(|t| now - t < SCROLL_SETTLE_S);
        if moving {
            moving_frames += 1;
            assert!(
                panel.requested_photographs.is_empty(),
                "a photograph was asked for while the rows moved"
            );
            assert_eq!(state.reference_picks.wanted_points(), None);
            assert_eq!(state.reference_picks.tasks_queued, 0);
        }
        state.hide_track_table();
    }
    assert!(
        moving_frames >= 35,
        "the rows moved on {moving_frames} frames"
    );
    assert!(
        passed.iter().any(|&p| stored_reference(p as usize) == 1),
        "the scroll passed rows cut from image 1"
    );
    // Any scroll animation runs out; then the rows stay put.
    let mut settled_at = None;
    for _ in 0..120 {
        let time = clock.tick();
        frame(&ctx, &mut panel, &mut state, 400.0, time);
        if !panel.requested_photographs.is_empty()
            || state.reference_picks.wanted_points().is_some()
        {
            settled_at = Some(time);
            break;
        }
        state.hide_track_table();
    }
    let settled_at = settled_at.expect("the rows were asked for once the scroll stopped");
    let range = panel.motion.changed_at.expect("the rows moved");
    assert!(
        settled_at - range >= SCROLL_SETTLE_S - 1e-9,
        "asked for {} s after the rows last moved",
        settled_at - range
    );
    let drawn: Vec<u32> = panel.drawn().iter().map(|&(p, _)| p).collect();
    let wanted = state.reference_picks.wanted_points().expect("picks wanted");
    let unreferenced: Vec<u32> = drawn
        .iter()
        .copied()
        .filter(|&p| stored_reference(p as usize) == -1)
        .collect();
    assert_eq!(
        wanted,
        &unreferenced[..],
        "the visible rows' picks, and no others"
    );
    if drawn.iter().any(|&p| stored_reference(p as usize) == 1) {
        assert!(panel.requested_photographs.contains(&1));
    }
    state.drive_track_table_picks();
    if !unreferenced.is_empty() {
        assert!(state.reference_picks.queued_task().is_some());
    }
    state.settle_reference_picks();
}

/// An edit while the picks run asks them to stop and keeps no pick made for
/// the version the node left. While the rows are of that version nothing is
/// queued; once the new table lands, the picks are queued for the new one.
#[test]
fn an_edit_cancels_the_picks_and_stale_rows_queue_none() {
    let mut clock = Clock(0.0);
    let (mut state, id) = crop_state();
    let (open, hold) = mpsc::channel();
    state.reference_picks.hold_next = Some(hold);
    let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);
    let before = state.scene[0].history.current_version().document_serial;
    state.drive_track_table_picks();
    let task = state.reference_picks.queued_task().expect("queued");
    assert!(
        state.quiet_task_running(task),
        "an idle fast queue starts it"
    );

    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    state.drive_track_table_picks();
    assert!(state.quiet_task_cancelled(task), "the edit asks it to stop");
    open.send(()).expect("the task waits at its gate");
    state.wait_for_reference_picks();
    let unreferenced = (FIRST_FAR..POINTS as u32)
        .find(|&p| stored_reference(p as usize) == -1)
        .unwrap();
    assert_eq!(state.reference_pick(id, unreferenced, before), None);

    // The rows are of the version before the edit until its table lands.
    state.hide_track_table();
    let queued = state.reference_picks.tasks_queued;
    for _ in 0..3 {
        frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
        state.drive_track_table_picks();
        assert_eq!(state.reference_picks.wanted_points(), None);
        assert_eq!(
            state.reference_picks.tasks_queued, queued,
            "nothing queued for stale rows"
        );
        state.hide_track_table();
    }
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    state.settle_track_table_columns();
    // The new rows are one fewer, which moves the rows on screen, so they
    // are asked for once they have stayed put for the settle time.
    frame(&ctx, &mut panel, &mut state, 4800.0, clock.tick());
    frame(
        &ctx,
        &mut panel,
        &mut state,
        4800.0,
        clock.wait(SCROLL_SETTLE_S + 0.01),
    );
    let after = state.scene[0].history.current_version().document_serial;
    assert_ne!(before, after);
    state.drive_track_table_picks();
    assert_eq!(state.reference_picks.tasks_queued, queued + 1);
    state.settle_reference_picks();
    assert!(state.reference_pick(id, unreferenced, after).is_some());
}

/// The picks give way to a task someone asks for in the fast queue: it asks
/// them to stop, starts once they have, and the picks are queued again behind
/// it. A slow task holding the node leaves them running.
#[test]
fn the_picks_give_way_in_their_queue_and_run_beside_a_slow_task() {
    let mut clock = Clock(0.0);
    let (mut state, id) = crop_state();
    let (open, hold) = mpsc::channel();
    state.reference_picks.hold_next = Some(hold);
    let (_ctx, _panel) = shown(&mut state, 4800.0, &mut clock);
    state.drive_track_table_picks();
    let task = state.reference_picks.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));

    let (open_slow, hold_slow) = mpsc::channel::<()>();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress: &Progress<'_>| {
                let _ = hold_slow.recv();
                Finished::Failed("held for the test".to_string())
            }),
        )
        .expect("the slow queue is idle");
    assert!(state.busy_refusal(id).is_some(), "the node is locked");
    assert!(
        !state.quiet_task_cancelled(task),
        "a locking task in the slow queue asked the picks to stop"
    );

    let find = state
        .start_find_points(
            id,
            vec![TrackTableColumn::MaxRayAngleDeg],
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    assert!(
        state.quiet_task_cancelled(task),
        "an asked-for task in the fast queue asks the picks to stop"
    );
    assert!(
        state.waiting_task(find).is_some(),
        "Find points waits for them"
    );
    open.send(()).expect("the task waits at its gate");
    state.wait_for_reference_picks();
    assert!(
        state.running_task(find).is_some(),
        "Find points runs once the picks have stopped"
    );
    state.drive_track_table_picks();
    let again = state.reference_picks.queued_task().expect("queued again");
    assert_ne!(again, task);
    assert!(
        !state.quiet_task_running(again),
        "the picks wait behind Find points"
    );

    // Once Find points lands, the picks run on the node the slow task holds.
    while state.running_task(find).is_some() {
        state.step_queue(crate::background::Queue::Fast);
    }
    state.settle_reference_picks();
    let document = state.scene[0].history.current_version().document_serial;
    let unreferenced = (FIRST_FAR..POINTS as u32)
        .find(|&p| stored_reference(p as usize) == -1)
        .unwrap();
    assert!(
        state.reference_pick(id, unreferenced, document).is_some(),
        "the picks ran beside the slow task"
    );
    open_slow.send(()).expect("the slow task waits at its gate");
    state.finish_background_tasks();
}

/// Hovering a crop shows Track View's crop hover view under the photograph's
/// name; hovering the patch of a point that stores no reference says its
/// bitmap is not the render of the crop's photograph.
#[test]
fn hovering_a_crop_shows_it_in_context_and_an_unreferenced_patch_says_so() {
    let mut clock = Clock(0.0);
    let (mut state, _) = crop_state();
    let (ctx, mut panel) = shown(&mut state, 4800.0, &mut clock);
    state.settle_reference_picks();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let hover_texts = |panel: &mut TrackTablePanel,
                       state: &mut AppState,
                       clock: &mut Clock,
                       column: &str,
                       point: u32| {
        // The pointer leaves the window first, closing any tooltip open.
        let mut gone = at(4800.0, clock.tick());
        gone.events = vec![egui::Event::PointerGone];
        run_frame_headless(&ctx, gone, |ui| panel.show(ui, state));
        let heading =
            painted_text_rects(&ctx, at(4800.0, clock.tick()), |ui| panel.show(ui, state))
                .into_iter()
                .find(|t| t.text == column)
                .unwrap()
                .rect
                .center();
        let &(_, row) = panel.drawn().iter().find(|(p, _)| *p == point).unwrap();
        let at_cell = egui::pos2(heading.x, row.center().y);
        let mut texts = Vec::new();
        for _ in 0..3 {
            let mut raw = at(4800.0, clock.tick());
            raw.events = vec![egui::Event::PointerMoved(at_cell)];
            texts = painted_texts(&ctx, raw, |ui| panel.show(ui, state));
        }
        texts
    };
    let texts = hover_texts(&mut panel, &mut state, &mut clock, "Crop", 1);
    let name = state.scene[0].recon().image_set.images[0].name.clone();
    assert!(texts.contains(&name), "the photograph's name: {texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Outline: the patch as it lands in the photograph")),
        "{texts:?}"
    );
    // Said of the point, not of a bench track's row and its tile.
    assert!(
        texts.iter().any(
            |t| t.contains("Dot: the keypoint of the point's reference observation")
                && t.contains("Ring: where the point projects")
                && t.contains("along the patch's first axis")
        ),
        "{texts:?}"
    );
    assert!(
        !texts
            .iter()
            .any(|t| t.contains("the track's point") || t.contains("across the tile")),
        "{texts:?}"
    );

    let unreferenced = (FIRST_FAR..POINTS as u32)
        .find(|&p| stored_reference(p as usize) == -1)
        .unwrap();
    let texts = hover_texts(&mut panel, &mut state, &mut clock, "Patch", unreferenced);
    let said = format!("{}.", crate::patch_pictures::UNREFERENCED_PATCH_HOVER);
    assert!(texts.contains(&said), "{texts:?}");
    let texts = hover_texts(&mut panel, &mut state, &mut clock, "Patch", 1);
    let said = format!("{}.", crate::patch_pictures::STORED_PATCH_HOVER);
    assert!(texts.contains(&said), "{texts:?}");
}

/// The Crop heading takes no click: a picture column cannot be sorted.
#[test]
fn the_crop_heading_does_not_sort() {
    let mut clock = Clock(0.0);
    let (mut state, _) = crop_state();
    let (ctx, mut panel) = shown(&mut state, 1200.0, &mut clock);
    let heading = painted_text_rects(&ctx, at(1200.0, clock.tick()), |ui| {
        panel.show(ui, &mut state)
    })
    .into_iter()
    .find(|t| t.text == "Crop")
    .unwrap()
    .rect
    .center();
    for pressed in [true, false] {
        let mut raw = at(1200.0, clock.tick());
        raw.events = vec![
            egui::Event::PointerMoved(heading),
            egui::Event::PointerButton {
                pos: heading,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            },
        ];
        run_frame_headless(&ctx, raw, |ui| panel.show(ui, &mut state));
    }
    assert_eq!(panel.query.sort, None);
}

/// The running task works through the points the panel still wants: a row
/// that leaves the view leaves its list, and with none wanted it picks nothing
/// more.
#[test]
fn rows_that_leave_the_view_leave_the_picks_list() {
    let mut clock = Clock(0.0);
    let (mut state, id) = crop_state();
    let (open, hold) = mpsc::channel();
    state.reference_picks.hold_next = Some(hold);
    let (_ctx, _panel) = shown(&mut state, 4800.0, &mut clock);
    let unreferenced: Vec<u32> = (0..POINTS as u32)
        .filter(|&p| stored_reference(p as usize) == -1)
        .collect();
    state.drive_track_table_picks();
    let task = state.reference_picks.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));
    assert_eq!(
        state.reference_picks.queued_points(),
        Some((unreferenced.clone(), None))
    );

    // Scrolled so only the first of them is on screen.
    let document = state.scene[0].history.current_version().document_serial;
    state.want_reference_picks(id, document, vec![unreferenced[0]]);
    state.drive_track_table_picks();
    assert_eq!(
        state.reference_picks.queued_points(),
        Some((vec![unreferenced[0]], None))
    );
    // Hidden, or scrolling: nothing is wanted.
    state.hide_track_table();
    state.drive_track_table_picks();
    assert_eq!(state.reference_picks.queued_points(), Some((vec![], None)));
    open.send(()).expect("the task waits at its gate");
    state.wait_for_reference_picks();
    assert!(unreferenced
        .iter()
        .all(|&p| state.reference_pick(id, p, document).is_none()));
}
