// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── The Image Detail panel's view ───────────────────────────────────────
//
// The panel's own geometry is the frame's, and these run without one, so the
// reading a drawn frame would have published is seeded directly -- which is
// exactly what the dock does with `ImageDetailResponse::view`. What is under
// test here is the boundary: that each target resolves to the right place in
// the right photograph, that the reply reports where the view landed rather
// than what the call asked for, and that a refusal names the fix. Where the
// arithmetic itself lands is `image_detail::view`'s, and `image_detail::tests`
// drives it through real frames.

/// The panel body these tests measure against, in points. Its aspect ratio
/// (4:3) differs from the demo photograph's (16:9), so the fit is settled by
/// the width and the letterboxed axis is a real one.
const VIEW_PANEL: [f32; 2] = [800.0, 600.0];

/// The demo photograph's size, which is its lens's.
const VIEW_IMAGE: [f32; 2] = [1920.0, 1080.0];

/// `alpha` with image 0 selected and the Image Detail panel standing at
/// `zoom`, centred, as a frame that drew it would have left things.
fn looking_at(zoom: f32) -> (AppState, Viewer3D) {
    let (mut state, viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    state.image_detail_view = Some(crate::image_detail::ViewGeometry {
        image,
        image_size: VIEW_IMAGE,
        panel_size: VIEW_PANEL,
        pan: [0.0, 0.0],
        zoom,
    });
    (state, viewer)
}

/// The image pixel at the centre of the panel, which is what every target
/// aims: the centre of the reported visible rectangle.
#[track_caller]
fn view_centre(reply: &Value) -> [f64; 2] {
    let view = &reply["image_detail_view"];
    let rect = view["visible_rect_px"]
        .as_array()
        .unwrap_or_else(|| panic!("a visible rectangle: {reply}"));
    let at = |i: usize| rect[i].as_f64().expect("a number");
    [(at(0) + at(2)) / 2.0, (at(1) + at(3)) / 2.0]
}

/// The reported visible rectangle's extent in image pixels.
#[track_caller]
fn view_extent(reply: &Value) -> [f64; 2] {
    let rect = reply["image_detail_view"]["visible_rect_px"]
        .as_array()
        .unwrap_or_else(|| panic!("a visible rectangle: {reply}"));
    let at = |i: usize| rect[i].as_f64().expect("a number");
    [at(2) - at(0), at(3) - at(1)]
}

#[track_caller]
fn assert_about(actual: [f64; 2], expected: [f64; 2], what: &str) {
    assert!(
        (actual[0] - expected[0]).abs() < 0.05 && (actual[1] - expected[1]).abs() < 0.05,
        "{what}: {actual:?} != {expected:?}"
    );
}

/// A pixel, a point's observation and a feature all mean "put this place at the
/// centre of the panel", and the reply says so in the one field that can be
/// checked against the request.
#[test]
fn a_pixel_target_centres_the_pixel_and_reports_the_zoom_it_took() {
    let (mut state, mut viewer) = looking_at(1.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [300.0, 200.0], "zoom": 4.0 }),
    );
    assert_about(
        view_centre(&reply),
        [300.0, 200.0],
        "the pixel is off centre",
    );
    assert_eq!(reply["image_detail_view"]["zoom"], json!(4.0), "{reply}");
    assert_eq!(
        reply["image_detail_view"]["camera_image"],
        json!(0),
        "{reply}"
    );
    assert_eq!(
        reply["image_detail_view"]["image_size_px"],
        json!([1920.0, 1080.0]),
        "{reply}"
    );
    assert_eq!(
        reply["image_detail_view"]["panel_size_points"],
        json!([800.0, 600.0]),
        "{reply}"
    );
}

/// A rectangle settles its own zoom: it ends exactly as wide as the panel on
/// the axis that binds, and no narrower on the other.
#[test]
fn a_rect_target_fits_the_rectangle_to_the_panel() {
    let (mut state, mut viewer) = looking_at(1.0);
    // 400 x 150, wider in proportion than the 4:3 panel, so the width binds.
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [100.0, 100.0, 500.0, 250.0] }),
    );
    assert_about(
        view_centre(&reply),
        [300.0, 175.0],
        "the rect is off centre",
    );
    let [width, height] = view_extent(&reply);
    assert!(
        (width - 400.0).abs() < 0.05,
        "the binding axis does not fill the panel: {reply}"
    );
    assert!(
        height >= 150.0 - 0.05,
        "the rectangle does not fit: {reply}"
    );
    // The zoom the fit implies: the panel spans 400 px where fitted it spans
    // the whole 1920.
    let fit = (VIEW_PANEL[0] / VIEW_IMAGE[0]).min(VIEW_PANEL[1] / VIEW_IMAGE[1]);
    let expected = f64::from((VIEW_PANEL[0] / 400.0) / fit);
    let zoom = reply["image_detail_view"]["zoom"].as_f64().expect("a zoom");
    assert!((zoom - expected).abs() < 1e-3, "{zoom} != {expected}");
}

/// A corner-to-corner rectangle given the other way round is the rectangle the
/// caller drew, not an empty one.
#[test]
fn a_rect_named_from_its_far_corner_frames_the_same_region() {
    let (mut state, mut viewer) = looking_at(1.0);
    let forwards = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [100.0, 100.0, 500.0, 250.0] }),
    );
    let backwards = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [500.0, 250.0, 100.0, 100.0] }),
    );
    assert_eq!(forwards, backwards, "the corners' order changed the view");
}

/// `fit` is what `Z` does: the whole photograph, centred.
#[test]
fn a_fit_target_shows_the_whole_photograph() {
    let (mut state, mut viewer) = looking_at(8.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true }),
    );
    assert_eq!(reply["image_detail_view"]["zoom"], json!(1.0), "{reply}");
    assert_about(view_centre(&reply), [960.0, 540.0], "the fit is off centre");
}

/// A point target is looked for in the photograph being looked at, and lands on
/// the pixel `get_point` reports for that row of its track.
#[test]
fn a_point_target_centres_that_point_s_observation() {
    let (mut state, mut viewer) = looking_at(1.0);
    let point = call(&mut state, &mut viewer, "get_point", json!({ "point": 0 }));
    let row = point["track"]
        .as_array()
        .expect("a track")
        .iter()
        .find(|row| row["camera_image_index"] == json!(0))
        .expect("the demo tracks observe every image")
        .clone();
    let xy = row["xy"].as_array().expect("a pixel");
    let expected = [
        xy[0].as_f64().expect("a number"),
        xy[1].as_f64().expect("a number"),
    ];

    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "point": 0, "zoom": 6.0 }),
    );
    assert_about(
        view_centre(&reply),
        expected,
        "the observation is off centre",
    );
}

/// A feature target reads the same cache the panel's own overlay draws its
/// ellipses from, so what an agent centres is the mark it is looking at.
#[test]
fn a_feature_target_centres_that_feature() {
    let (mut state, mut viewer) = looking_at(1.0);
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.sift_cache.insert(
        image,
        crate::state::CachedSiftFeatures {
            positions_xy: vec![[10.0, 20.0], [640.0, 480.0]],
            affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]; 2],
            read_count: 2,
        },
    );
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "feature": 1, "zoom": 3.0 }),
    );
    assert_about(
        view_centre(&reply),
        [640.0, 480.0],
        "the feature is off centre",
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "feature": 7 }),
    );
    // Out of range, one way or the other: this fixture has no `.sift` file on
    // disk, so the index past the end of the cache is refused by the read that
    // would have gone looking for it. Either way the refusal names the
    // photograph, which is what an agent needs to act on it.
    assert!(
        past_the_end.0.contains("images/A_000.jpg"),
        "{past_the_end}"
    );
}

/// A zoom past the panel's own range is clamped, and the reply says where it
/// landed rather than echoing what was asked for.
#[test]
fn a_zoom_past_the_panel_s_range_comes_back_clamped() {
    let (mut state, mut viewer) = looking_at(1.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [100.0, 100.0], "zoom": 500.0 }),
    );
    assert_eq!(
        reply["image_detail_view"]["zoom"],
        json!(f64::from(crate::image_detail::MAX_ZOOM)),
        "{reply}"
    );
    let floored = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [100.0, 100.0], "zoom": 0.25 }),
    );
    assert_eq!(
        floored["image_detail_view"]["zoom"],
        json!(1.0),
        "{floored}"
    );
}

/// The tool's own description carries the closest the panel goes, and the panel
/// owns that number.
#[test]
fn the_view_tool_advertises_the_panel_s_own_zoom_limit() {
    let catalog = tools::catalog();
    let spec = catalog
        .iter()
        .find(|spec| spec.name == "set_image_detail_view")
        .expect("the tool is in the catalog");
    let limit = format!("{:.1}", crate::image_detail::MAX_ZOOM);
    let zoom = spec.schema["properties"]["zoom"]["description"]
        .as_str()
        .expect("the zoom argument is described");
    assert!(
        zoom.contains(&limit),
        "the zoom argument does not name the panel's own limit {limit}: {zoom}"
    );
}

/// Every way of asking for nothing, or for two things at once, or for
/// something that is not there.
#[test]
fn the_view_tool_refuses_in_its_own_words() {
    let (mut state, mut viewer) = looking_at(1.0);

    let nothing = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "zoom": 2.0 }),
    );
    assert!(nothing.0.contains("no place to look"), "{nothing}");

    let both = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [10.0, 10.0], "fit": true }),
    );
    assert!(both.0.contains("2 places to look"), "{both}");

    // A rect settles its own zoom, so one beside it is a contradiction.
    let zoomed_rect = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [0.0, 0.0, 10.0, 10.0], "zoom": 2.0 }),
    );
    assert!(zoomed_rect.0.contains("settles its own"), "{zoomed_rect}");

    let flat = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [10.0, 10.0, 10.0, 90.0] }),
    );
    assert!(flat.0.contains("frames nothing"), "{flat}");

    // A track named with no bench observation to find in it.
    let stray = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true, "track": "bull-nose" }),
    );
    assert!(stray.0.contains("bench_observation"), "{stray}");

    // Nothing selected, and nothing named.
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let unselected = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true }),
    );
    assert!(unselected.0.contains("select_camera_image"), "{unselected}");
}

/// A point with no observation in the photograph being looked at is refused by
/// name, rather than the panel quietly moving to some other image.
#[test]
fn a_point_not_in_this_photograph_is_refused_rather_than_followed() {
    let (mut state, mut viewer) = looking_at(1.0);
    // A point of the demo whose track does not reach image 0. Found rather
    // than assumed: the demo's tracks are subsets of its camera ring, and
    // which point misses which image is the fixture's business.
    let missing = (0..40)
        .map(|index| {
            (
                index,
                call(
                    &mut state,
                    &mut viewer,
                    "get_point",
                    json!({ "point": index }),
                ),
            )
        })
        .find(|(_, point)| {
            !point["track"]
                .as_array()
                .expect("a track")
                .iter()
                .any(|row| row["camera_image_index"] == json!(0))
        })
        .map(|(index, _)| index)
        .expect("some demo point misses image 0");

    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "point": missing }),
    );
    assert!(
        refusal
            .0
            .contains(&format!("no observation of point {missing}")),
        "{refusal}"
    );
    assert!(refusal.0.contains("get_point"), "{refusal}");
    // And the photograph was not changed for it.
    assert_eq!(state.selected_image.map(|image| image.index()), Some(0));
}

/// One frame of the Image Detail panel, as the dock runs one: the standing
/// request taken and applied in a panel of [`VIEW_PANEL`] points, and the
/// reading published on `AppState`. That publication is the whole of what a
/// deferred view call waits for.
fn panel_draws(state: &mut AppState, image: crate::scene::ImageRef) {
    panel_draws_at(state, image, VIEW_PANEL);
}

/// The same, in a panel body of a named size: what a frame after a layout change
/// publishes, which is a different panel from the one before it.
fn panel_draws_at(state: &mut AppState, image: crate::scene::ImageRef, panel: [f32; 2]) {
    let mut geometry = crate::image_detail::ViewGeometry {
        image,
        image_size: VIEW_IMAGE,
        panel_size: panel,
        pan: [0.0, 0.0],
        zoom: 1.0,
    };
    if let Some(look) = state.take_look(image) {
        geometry = crate::image_detail::look_at(geometry, &look);
    }
    // Through the publication, not the field: the serial and the layout it was
    // drawn in are what a waiting call and a view tool read it by.
    state.publish_image_detail_view(geometry);
}

/// The wait a view call left behind, or a panic saying it did not wait.
#[track_caller]
fn deferred_view(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    arguments: Value,
) -> super::super::PendingView {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    let command = tools::parse("set_image_detail_view", Some(&map))
        .unwrap_or_else(|e| panic!("set_image_detail_view: {e}"));
    match agent(state, viewer, command) {
        Outcome::Deferred(super::super::Deferred::ImageDetailView(pending)) => pending,
        Outcome::Deferred(_) => panic!("expected the view's deferral, got another tool's"),
        Outcome::Done(Ok(_)) => panic!("expected the call to wait for a drawn frame"),
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
    }
}

/// What a waiting call is answered with, once a frame has drawn.
#[track_caller]
fn view_reply(state: &AppState, pending: &super::super::PendingView) -> Value {
    match super::super::display::pending_view_reply(state, pending).expect("the frame answered it")
    {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// Before the panel has drawn anything there is no frame to do the arithmetic
/// in, so the read reports nulls and the write **waits** rather than refusing:
/// the look is standing, and the reply is the reading of the frame that applies
/// it.
#[test]
fn a_panel_that_has_never_drawn_reports_nothing_and_waits_for_a_frame() {
    let (mut state, mut viewer) = two_reconstructions();
    let view = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(
        view["image_detail_view"]["camera_image"],
        Value::Null,
        "{view}"
    );
    assert_eq!(view["image_detail_view"]["zoom"], Value::Null, "{view}");

    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "pixel": [300.0, 200.0], "zoom": 4.0 }),
    );
    assert_eq!(pending.image, image);
    assert!(
        super::super::display::pending_view_reply(&state, &pending).is_none(),
        "answered before any frame had drawn"
    );

    panel_draws(&mut state, image);
    let landed = view_reply(&state, &pending);
    assert_eq!(landed["image_detail_view"]["zoom"], json!(4.0), "{landed}");
    assert_about(
        view_centre(&landed),
        [300.0, 200.0],
        "the pixel is off centre",
    );
}

/// A frame that never comes is not waited on for ever: a photograph the
/// workspace no longer holds draws nothing however long it is given, so the
/// wait has a deadline and the answer past it says what did not happen.
#[test]
fn a_view_the_panel_never_draws_is_given_up_on() {
    let (mut state, mut viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    let mut pending = deferred_view(&mut state, &mut viewer, json!({ "fit": true }));
    pending.started = std::time::Instant::now() - std::time::Duration::from_secs(5);
    let refusal = match super::super::display::pending_view_reply(&state, &pending)
        .expect("the deadline answers it")
    {
        Err(refusal) => refusal,
        Ok(_) => panic!("a frame that never drew has no view to report"),
    };
    assert!(refusal.0.contains("has not drawn"), "{refusal}");
    assert!(
        refusal.0.contains(&state.image_name(image)),
        "the refusal does not name the photograph: {refusal}"
    );
}

/// A sighting *is* a place in a particular photograph, so a
/// `bench_observation` needs no selection and no open panel: the tool selects
/// the image, opens the panel if it was closed, and the look lands on the first
/// frame that draws it.
#[test]
fn a_bench_observation_target_selects_its_photograph_with_none_selected() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    state.hide_panel(Tab::ImageDetail);
    assert_eq!(state.selected_image, None, "the fixture selected an image");

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let row = &track["observations"][1];
    let pixel = row["pixel"].as_array().expect("where it sits").clone();
    let at = |i: usize| pixel[i].as_f64().expect("a number");
    let image = row["camera_image"].as_u64().expect("its photograph") as usize;

    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a", "bench_observation": 1, "zoom": 6.0 }),
    );
    assert_eq!(
        state.selected_image.map(|image| image.index()),
        Some(image),
        "the call did not select the photograph its target named"
    );
    assert!(
        state.panel_is_in_front(Tab::ImageDetail),
        "the call left the panel where nothing would draw it"
    );

    panel_draws(&mut state, pending.image);
    let landed = view_reply(&state, &pending);
    assert_eq!(landed["image_detail_view"]["zoom"], json!(6.0), "{landed}");
    assert_eq!(
        landed["image_detail_view"]["camera_image"],
        json!(image),
        "{landed}"
    );
    assert_about(
        view_centre(&landed),
        [at(0), at(1)],
        "the observation is off centre",
    );
}

/// `camera_image` names the photograph just as directly, so a `pixel` target
/// beside one also needs nothing selected.
#[test]
fn a_pixel_target_naming_a_camera_image_needs_no_selection() {
    let (mut state, mut viewer) = two_reconstructions();
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "camera_image": 3, "pixel": [640.0, 480.0], "zoom": 8.0 }),
    );
    assert_eq!(state.selected_image.map(|image| image.index()), Some(3));

    panel_draws(&mut state, pending.image);
    let landed = view_reply(&state, &pending);
    assert_eq!(
        landed["image_detail_view"]["camera_image"],
        json!(3),
        "{landed}"
    );
    assert_about(
        view_centre(&landed),
        [640.0, 480.0],
        "the pixel is off centre",
    );
}

/// What is left to refuse: a target that names no photograph, with none
/// selected. The sentence says both ways out of it.
#[test]
fn a_pixel_target_with_no_photograph_named_or_selected_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [10.0, 20.0], "zoom": 4.0 }),
    );
    assert!(refusal.0.contains("camera_image"), "{refusal}");
    assert!(refusal.0.contains("select_camera_image"), "{refusal}");
    assert_eq!(state.look, None, "a refused call left a request standing");
}

/// `get_image_detail_view` answers from the panel's own last frame, which is
/// what a screenshot of it would show -- not from the selection, which a tool
/// may have moved since.
#[test]
fn get_image_detail_view_reports_the_panel_s_last_frame() {
    let (mut state, mut viewer) = looking_at(4.0);
    let before = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(before["image_detail_view"]["zoom"], json!(4.0), "{before}");
    assert_eq!(
        before["image_detail_view"]["camera_image_name"],
        json!("images/A_000.jpg"),
        "{before}"
    );

    // A selection the panel has not drawn yet does not move the reading.
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 5 }),
    );
    let after = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(after, before, "the reading followed the selection");
}

/// A bench observation names its own photograph, so walking a track's sightings
/// takes one argument.
#[test]
fn a_bench_observation_target_selects_its_own_camera_image() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let image = track["observations"][1]["camera_image"]
        .as_u64()
        .expect("an image index");
    // The panel is standing on some *other* photograph, so the target has to
    // move the selection for the view to mean anything.
    let first = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(first));
    let size = {
        let camera = &state.scene[0].recon().image_table.cameras[0];
        [camera.width as f32, camera.height as f32]
    };
    state.image_detail_view = Some(crate::image_detail::ViewGeometry {
        image: first,
        image_size: size,
        panel_size: VIEW_PANEL,
        pan: [0.0, 0.0],
        zoom: 1.0,
    });

    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "reconstruction_label": "run_a", "bench_observation": 1, "track": item }),
    );
    assert_eq!(
        reply["image_detail_view"]["camera_image"],
        json!(image),
        "{reply}"
    );
    assert_eq!(
        state.selected_image.map(|image| image.index() as u64),
        Some(image),
        "the target did not select its own photograph"
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "reconstruction_label": "run_a", "bench_observation": 99 }),
    );
    assert!(
        past_the_end.0.contains("no observation 99"),
        "{past_the_end}"
    );
}

// ── What the two photometric steps refuse before they start ─────────────

/// Turn every observation of the active track out but the first.
#[track_caller]
fn leave_one_in(state: &mut AppState, viewer: &mut Viewer3D) {
    let track = call(
        state,
        viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the rows").len();
    for observation in 1..rows {
        call(
            state,
            viewer,
            "set_bench_track_verdict",
            json!({
                "reconstruction_label": "run_a",
                "observation": observation,
                "verdict": "out",
            }),
        );
    }
}

/// A stage change the track alone rules out is refused **inline**, before a
/// photograph is read: no deferral, no operation, no version.
///
/// The failure this pins down is a refusal that cost a decode: the step used to
/// answer `running: true`, hand the task a dozen images to read, and fail it a
/// second later with a sentence that was knowable before anything started.
#[test]
fn a_stage_change_the_track_rules_out_is_refused_before_the_worker() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    leave_one_in(&mut state, &mut viewer);
    let before = version_count(&state);

    // `refused_call` panics on a deferral, so this is also the assertion that
    // nothing went to a worker.
    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert_eq!(
        refusal.0,
        format!(
            "Cannot set the stage of {item}: 1 observations are in, and the track stage needs \
             two or more"
        )
    );
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");
}

/// The same for a fit of a track-stage track with nothing to register against
/// -- and, beside it, the reading of that very track, which is **not** refused:
/// one sighting is something to report, and only moving it needs a consensus.
#[test]
fn a_fit_the_track_rules_out_is_refused_before_the_worker() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    leave_one_in(&mut state, &mut viewer);
    let before = version_count(&state);

    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        refusal.0,
        format!("Cannot fit {item}: 1 observations are in, and the track stage needs two or more")
    );
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");

    let read = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        read["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Evaluated {item}")),
        "a reading of one sighting is a measurement, not a refusal: {read}"
    );
}

// ── What a commit names ─────────────────────────────────────────────────

/// A commit names the point it wrote, by index and by the id a later call can
/// address it with, so an agent can `get_point` it without hunting through the
/// counts for whichever row is new.
#[test]
fn a_commit_names_the_point_it_wrote() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let reply = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let index = reply["point"]["index"]
        .as_u64()
        .unwrap_or_else(|| panic!("the commit named no point: {reply}"));
    let id = reply["point"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("the commit minted no id: {reply}"))
        .to_string();
    // This track came off a point, so the commit replaced that point rather
    // than creating one, and the reply says which.
    assert_eq!(reply["point"]["replaced"], json!(BENCH_POINT), "{reply}");

    // Both handles resolve, which is the whole of what naming it is for.
    let by_index = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": index }),
    );
    assert_eq!(by_index["id"], json!(id), "{by_index}");
    let by_id = call(&mut state, &mut viewer, "get_point", json!({ "point": id }));
    assert_eq!(by_id["index"], json!(index), "{by_id}");
}

// ── The hash a version's point ids are minted from ──────────────────────

/// `get_scene`'s `content_hash` is the hash the version's point ids carry, so
/// an agent holding the field and an agent holding an id read off a point are
/// holding the same digits -- and an edit that mints a new one moves it.
///
/// The failure this pins down is a field that never moved: it reported the
/// *base*'s hash, which a point edit leaves exactly where it was, while the
/// point that edit created was already being named by the edit's own.
#[test]
fn the_scene_s_content_hash_is_the_hash_its_point_ids_carry() {
    let (mut state, mut viewer) = benchable();
    let hash_of = |state: &mut AppState, viewer: &mut Viewer3D| -> String {
        call(state, viewer, "get_scene", json!({}))["scene"][0]["content_hash"]
            .as_str()
            .expect("a content hash")
            .to_string()
    };
    let before = hash_of(&mut state, &mut viewer);
    assert_eq!(before.len(), 8, "a content hash is the id's eight digits");

    // A point the reconstruction had no row for: it is named by the hash of
    // the edit that made it, and that is what the node now reports. Pushed
    // directly, because what is under test is the field rather than any one
    // step, and every step that creates a point pushes this shape.
    let record = state.scene[0]
        .edited()
        .point(BENCH_POINT)
        .expect("a live point")
        .to_record();
    let node = &mut state.scene[0];
    let mut next = node.history.current().clone();
    let hash = next
        .point_edit_hash(std::slice::from_ref(&record))
        .expect("a hashable base");
    let created = next.add_point(record).expect("a well-formed record");
    node.history.push_creating(
        next,
        crate::document::PointMap::Removed(Vec::new()),
        "Created a point",
        Some(crate::document::CreatedPoints {
            hash,
            indexes: vec![created],
        }),
    );
    let id = state.scene[0].id;
    state.select_point(crate::scene::PointRef::new(id, created as usize));
    let after = hash_of(&mut state, &mut viewer);
    assert_ne!(
        after, before,
        "an edit that minted a new hash did not move it"
    );

    let selected = call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["point"]
        ["id"]
        .as_str()
        .expect("the created point is selected")
        .to_string();
    assert!(
        selected.starts_with(&format!("pt3d_{after}_")),
        "{selected} is not named by the hash get_scene reports ({after})"
    );

    // And an undo takes the node back to the hash it had, so the field tracks
    // the version rather than accumulating.
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(hash_of(&mut state, &mut viewer), before);
}

// ── What a cursor move does to the selection ────────────────────────────

/// Undo, redo and a jump are steps through a node's history, not statements
/// about what the person is looking at: the photograph on screen stays on
/// screen, and so does the point.
///
/// The failure this pins down is Image Detail going to "No image selected" on
/// every step of a run of bench steps, none of which touches the image table.
#[test]
fn a_history_move_keeps_the_photograph_and_the_point_selected() {
    let (mut state, mut viewer) = benchable();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 3 }),
    );
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": BENCH_POINT }),
    );
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 1, "verdict": "out" }),
    );

    let selection = |state: &mut AppState, viewer: &mut Viewer3D| -> Value {
        call(state, viewer, "get_scene", json!({}))["selection"].clone()
    };
    let standing = selection(&mut state, &mut viewer);
    assert_eq!(standing["camera_image"]["index"], json!(3), "{standing}");

    for step in ["undo", "undo", "redo", "redo"] {
        call(
            &mut state,
            &mut viewer,
            step,
            json!({ "reconstruction_label": "run_a" }),
        );
        let now = selection(&mut state, &mut viewer);
        assert_eq!(
            now["camera_image"], standing["camera_image"],
            "{step} dropped the photograph"
        );
        assert_eq!(
            now["point"]["index"], standing["point"]["index"],
            "{step} dropped the point"
        );
    }

    // And a jump, which walks several steps at once.
    let history = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    );
    let first = history["versions"][0]["serial"]
        .as_str()
        .expect("a serial")
        .to_string();
    call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    let jumped = selection(&mut state, &mut viewer);
    assert_eq!(
        jumped["camera_image"], standing["camera_image"],
        "a jump dropped the photograph"
    );
    let _ = item;
}

/// The one move that *can* take the photograph away takes it away: undoing a
/// `delete_camera_image` puts the image back under a different index, and
/// redoing the delete leaves nothing to show.
#[test]
fn a_move_across_a_deleted_camera_image_follows_it_by_name() {
    let (mut state, mut viewer) = two_reconstructions();
    let name = call(
        &mut state,
        &mut viewer,
        "get_camera_image",
        json!({ "camera_image": 5 }),
    )["name"]
        .as_str()
        .expect("a name")
        .to_string();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 5 }),
    );
    // Deleting an earlier image renumbers the one being looked at.
    call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 1 }),
    );
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "alpha" }),
    );
    let back =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(
        back["name"],
        json!(name),
        "the undo did not put the selection back on the same photograph"
    );
    assert_eq!(back["index"], json!(5), "{back}");

    // Deleting the selected photograph itself leaves nothing to show, so the
    // selection clears -- and an undo does not resurrect it, because a
    // selection is not part of a version.
    call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 5 }),
    );
    let gone =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(gone, Value::Null, "the deleted photograph stayed selected");
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "alpha" }),
    );
    let after =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(after, Value::Null, "{after}");
}

// ── The index files and the search through them ────────────────────────

/// `get_bench` reports the index files a search would read, the index-files
/// tools give a node them, and the search itself is a background task an agent
/// polls for.
///
/// Over the workspace fixture rather than [`benchable`]: a search needs real
/// `.sift` files and a real `.kdf`, which is what
/// [`crate::sift_index::tests::searchable`] builds.
#[test]
fn the_index_files_and_the_search_are_on_the_wire() {
    use crate::sift_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, item) = fixture::searchable(dir.path());
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    let index = &bench["index_files"]["sift_index"];
    assert_eq!(index["state"], json!("current"), "{bench}");
    assert!(
        index["path"]
            .as_str()
            .expect("a path")
            .ends_with("demo-sift-index.kdf"),
        "{bench}"
    );
    assert!(
        index["descriptors"].as_u64().expect("a count") > 0,
        "{bench}"
    );
    let patches = &bench["index_files"]["cluster_patches"];
    assert_eq!(patches["state"], json!("current"), "{bench}");
    assert!(
        patches["path"]
            .as_str()
            .expect("a path")
            .ends_with("demo-cluster-patches.matches"),
        "{bench}"
    );
    assert!(
        patches["clusters"].as_u64().expect("a count") > 0,
        "{bench}"
    );
    assert_eq!(patches["stale_reason"], Value::Null, "{bench}");
    // The scene description carries the same object.
    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(
        scene["scene"][0]["index_files"], bench["index_files"],
        "{scene}"
    );

    let searched = worked(
        &mut state,
        &mut viewer,
        "search_bench_track_descriptors",
        json!({ "reconstruction_label": label, "track": item, "observation": 0 }),
    );
    let report = searched["report"].as_str().expect("a report");
    assert!(
        report.contains("Searched from observation 0 of 3") && report.contains("images matched"),
        "{searched}"
    );

    // The operation an agent polls for, under the name the panel shows.
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Search descriptors"), "{task}");

    // And the candidate it added is on the wire under the search's provenance.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": label, "track": item }),
    );
    let observations = track["observations"].as_array().expect("a list");
    assert_eq!(observations.len(), 4, "{track}");
    let added = observations.last().expect("the search added one");
    assert_eq!(
        added["camera_image"],
        json!(fixture::FOUND_IMAGE),
        "{track}"
    );
    assert_eq!(added["provenance"]["kind"], json!("search"), "{track}");
    assert!(added["provenance"]["inliers"].is_number(), "{track}");

    // Opening the same file again by path is the same index, and pushes no
    // version: the index files sit beside the workspace, they are not a value.
    let path = index["path"].as_str().expect("a path").to_string();
    let before = state.node(id).expect("loaded").history.versions().len();
    let opened = call(
        &mut state,
        &mut viewer,
        "open_index_files",
        json!({ "reconstruction_label": label, "sift_index_path": path }),
    );
    let files = &opened["index_files"];
    assert_eq!(files["sift_index"]["state"], json!("current"), "{opened}");
    assert_eq!(
        files["cluster_patches"]["state"],
        json!("current"),
        "{opened}"
    );
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        before,
        "the index files are not a version"
    );

    // Closing lets go of both, and a second close is refused.
    let closed = call(
        &mut state,
        &mut viewer,
        "close_index_files",
        json!({ "reconstruction_label": label }),
    );
    let files = &closed["index_files"];
    assert_eq!(files["sift_index"]["state"], json!("none"), "{closed}");
    assert_eq!(files["cluster_patches"]["state"], json!("none"), "{closed}");
    let command = tools::parse(
        "close_index_files",
        Some(
            &json!({ "reconstruction_label": label })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(
        error.to_string().contains("No index file is open"),
        "{error}"
    );

    // Opening with no paths opens the node's own files again.
    let reopened = call(
        &mut state,
        &mut viewer,
        "open_index_files",
        json!({ "reconstruction_label": label }),
    );
    let files = &reopened["index_files"];
    assert_eq!(files["sift_index"]["state"], json!("current"), "{reopened}");
    assert_eq!(
        files["cluster_patches"]["state"],
        json!("current"),
        "{reopened}"
    );
}

/// `build_index_files` over the wire: a node with an index and no cluster
/// patches gets its cluster patches from the index that is open, and the reply
/// is the handle an agent polls; once it lands both files read current.
#[test]
fn build_index_files_makes_the_missing_cluster_patches_on_the_wire() {
    use crate::sift_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = fixture::searchable(dir.path());
    let label = state.node(id).expect("loaded").label.clone();
    let patches = state
        .cluster_patches(id)
        .expect("the fixture built them")
        .path
        .clone();
    let index_before = std::fs::read(fixture::index_of(dir.path())).expect("built");
    std::fs::remove_file(&patches).unwrap();
    state.close_index_files(id).expect("both are open");
    state
        .open_index_files(id, None, None)
        .expect("the index is still there");
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    assert_eq!(
        bench["index_files"]["cluster_patches"]["state"],
        json!("none"),
        "{bench}"
    );

    worked(
        &mut state,
        &mut viewer,
        "build_index_files",
        json!({ "reconstruction_label": label }),
    );
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Build index files"), "{task}");
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    let files = &bench["index_files"];
    assert_eq!(files["sift_index"]["state"], json!("current"), "{bench}");
    assert_eq!(
        files["cluster_patches"]["state"],
        json!("current"),
        "{bench}"
    );
    assert!(patches.is_file(), "the build wrote the cluster patches");
    assert_eq!(
        std::fs::read(fixture::index_of(dir.path())).expect("still there"),
        index_before,
        "a current index was rebuilt when only the cluster patches were missing"
    );
}

/// A step reports **its own** sentence, even when it sets something else off.
///
/// Putting the first item on a bench looks for the node's default descriptor
/// index, and finding one writes an Action Log row of its own -- after the
/// step's. The reply reads the log back for what the call did, so that row is
/// the viewer's rather than the caller's, and the reply skips it.
#[test]
fn a_create_that_opens_the_default_index_still_reports_the_create() {
    use crate::sift_index::tests as fixture;

    // One session builds the index; a second opens the same workspace with
    // nothing open yet, which is the state the first create meets.
    let dir = tempfile::tempdir().unwrap();
    let (_built, _, _) = fixture::searchable(dir.path());
    let (mut state, id) = fixture::state_in(dir.path());
    assert!(
        state.sift_index(id).is_none(),
        "the fresh session has opened nothing yet"
    );
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": label, "point": fixture::POINT }),
    );
    let item = made["item"].as_str().expect("a create names its item");
    let report = made["report"].as_str().expect("a create reports itself");
    assert!(
        report.starts_with("Put point") && report.contains(item),
        "the reply carries the step's own sentence: {made}"
    );
    // The index really did open in the middle of it, and said so in a row of
    // the viewer's own.
    assert!(state.sift_index(id).is_some());
    let opened = state
        .action_log
        .entries()
        .find(|entry| entry.text.starts_with("Opened the SIFT index"))
        .expect("the open wrote a row");
    assert_eq!(opened.actor, crate::action_log::Actor::Viewer);
}

/// The geometry search over the wire: it needs no SIFT index, reports under an
/// operation name of its own, pushes one version for its own sentence, and is
/// refused at the cluster stage in front of the worker rather than from inside
/// it. That a view clearing the gates arrives as a `sweep` candidate at the
/// patch's projection is held by core, over a scene with real texture
/// (`sfmtool-core/src/bench/tests.rs`); this fixture's cameras admit nothing,
/// which is what makes it the honest test of the wire rather than of the
/// kernel.
#[test]
fn search_bench_track_geometry_needs_no_index_and_reports_as_itself() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    )["observations"]
        .as_array()
        .expect("a list")
        .len();

    // No index is open on this node, and the search answers all the same --
    // which is the difference between it and search_bench_track_descriptors.
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        bench["index_files"]["sift_index"]["state"],
        json!("none"),
        "{bench}"
    );

    let versions = version_count(&state);
    let searched = worked(
        &mut state,
        &mut viewer,
        "search_bench_track_geometry",
        json!({ "reconstruction_label": "run_a", "track": item, "observation": 0 }),
    );
    let report = searched["report"].as_str().expect("a report");
    assert!(
        report.contains(&format!("Geometry search from observation 0 of {before}")),
        "the reply carries some other step's sentence: {searched}"
    );
    assert_eq!(version_count(&state), versions + 1, "{searched}");
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Geometry search"), "{task}");

    // Nothing this fixture offers clears the gates, and a search that admits
    // nothing moves no row it did not add.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let observations = track["observations"].as_array().expect("a list");
    assert_eq!(observations.len(), before, "{track}");
    for row in observations {
        assert_eq!(row["provenance"]["kind"], json!("origin"), "{track}");
    }

    // A cluster carries no geometry to project, and the refusal says so in
    // front of the worker rather than from inside it.
    worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "track": item, "stage": "cluster" }),
    );
    let command = tools::parse(
        "search_bench_track_geometry",
        Some(
            &json!({ "reconstruction_label": "run_a", "track": item, "observation": 0 })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(error.to_string().contains("track stage"), "{error}");
}

/// With no index, `get_bench` says so and still names where a build would put
/// one, and the search is refused in the step's own sentence rather than
/// deferring to a worker that could never answer.
#[test]
fn a_search_with_no_index_is_refused_and_get_bench_names_the_default_path() {
    use crate::sift_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = fixture::state_in(dir.path());
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    let item = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": label, "point": fixture::POINT }),
    )["item"]
        .as_str()
        .expect("a create names the item it made")
        .to_string();

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    let files = &bench["index_files"];
    assert_eq!(files["sift_index"]["state"], json!("none"), "{bench}");
    assert!(
        files["sift_index"]["path"]
            .as_str()
            .expect("the default path is named even when nothing is open")
            .ends_with("demo-sift-index.kdf"),
        "{bench}"
    );
    assert_eq!(files["sift_index"]["descriptors"], Value::Null);
    assert_eq!(files["cluster_patches"]["state"], json!("none"), "{bench}");
    assert!(
        files["cluster_patches"]["path"]
            .as_str()
            .expect("the default path is named even when nothing is open")
            .ends_with("demo-cluster-patches.matches"),
        "{bench}"
    );
    assert_eq!(files["cluster_patches"]["clusters"], Value::Null);

    let command = tools::parse(
        "search_bench_track_descriptors",
        Some(
            &json!({ "reconstruction_label": label, "track": item, "observation": 0 })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(
        error.to_string().contains("No SIFT index is open"),
        "{error}"
    );

    // And a build over a node whose images have no `.sift` companion is refused
    // the same way, in front of the worker.
    let command = tools::parse(
        "build_index_files",
        Some(
            &json!({ "reconstruction_label": label })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(error.to_string().contains("No .sift file"), "{error}");
}

/// [`crate::state::edits::tests::projected_embedded_demo`] with [`BENCH_POINT`]
/// stored as a bearing: `w = 0`, a unit direction and a zero normal, which is
/// the row the format states for a point at infinity.
pub(super) fn bearing_demo() -> sfmtool_core::SfmrReconstruction {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    let point = &mut recon.point_set.points[BENCH_POINT as usize];
    point.position = nalgebra::Point3::from(point.position.coords.normalize());
    point.w = 0.0;
    point.normal = nalgebra::Vector3::zeros();
    recon.rebuild_derived_fields();
    recon
}

/// A bearing and a position are the same three numbers under different rules,
/// so the wire publishes the coordinate under the name of whichever it is, with
/// the flag beside it. An agent that read `position` off a `w = 0` track would
/// be holding a place one unit from the world origin.
#[test]
fn get_bench_track_publishes_a_bearing_as_a_direction() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["at_infinity"], json!(true), "{track}");
    let direction = stage["direction"].as_array().expect("a bearing");
    assert_eq!(direction.len(), 3, "{track}");
    let norm: f64 = direction
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-9, "a unit direction: {track}");
    assert!(stage["position"].is_null(), "{track}");

    // And the finite point of the same fixture comes back the other way round.
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["at_infinity"], json!(false), "{track}");
    assert!(stage["position"].is_array(), "{track}");
    assert!(stage["direction"].is_null(), "{track}");
}

/// A fit says which representation the rays earned and why, so an agent driving
/// the bench reads the decision rather than inferring it from a coordinate.
///
/// The demo's cameras sit on an arc forty-five degrees apart, so the three
/// sightings of a track on it resolve a depth easily: the stored bearing is
/// promoted, which is the answer, and the wire carries it.
#[test]
fn a_fit_of_a_bearing_reports_the_classification_and_can_promote_it() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    let item = on_the_bench(&mut state, &mut viewer);
    let fitted = worked(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let report = fitted["report"].as_str().expect("a report");
    assert!(
        report.contains("finite at (") && report.contains("rays up to"),
        "the report should name the call and the evidence: {report}"
    );

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(
        stage["at_infinity"],
        json!(false),
        "the promotion is on the wire: {track}"
    );
    assert!(stage["position"].is_array(), "{track}");
}

// ── The no-effect contract, the clamp, and a frameless bearing ─────────────

/// A step told to do what has already been done answers successfully and says
/// so: no version, `changed: false`, the cursor where it was, and the step's
/// **own** sentence under `report` -- not the previous step's label, which is
/// what a reply assembled from the cursor alone echoes.
#[test]
fn a_bench_step_with_no_effect_pushes_nothing_and_says_so() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let pixel = track["observations"][0]["pixel"]
        .as_array()
        .expect("a sighting has a pixel")
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .collect::<Vec<f64>>();

    let before = version_count(&state);
    let cursor = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    )["cursor"]
        .as_str()
        .expect("a cursor")
        .to_string();

    // The patch slid to the pixel its centre already sits under.
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "pixel": [pixel[0], pixel[1]],
        }),
    );
    assert_eq!(moved["changed"], json!(false), "{moved}");
    assert_eq!(moved["serial"], json!(cursor), "{moved}");
    assert_eq!(moved["cursor"], json!(cursor), "{moved}");
    let report = moved["report"].as_str().expect("a sentence");
    assert!(report.contains("no effect"), "{report}");
    assert!(report.starts_with("Moved"), "{report}");
    assert!(!report.contains("by 0.000 units"), "{report}");

    // A turn of nothing, and a verdict the observation already carries.
    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "track": item, "degrees": 0.0 }),
    );
    assert_eq!(turned["changed"], json!(false), "{turned}");
    assert!(
        turned["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{turned}"
    );

    let verdict = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "verdict": "in",
        }),
    );
    assert_eq!(verdict["changed"], json!(false), "{verdict}");
    assert!(
        verdict["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{verdict}"
    );

    // A painting that proposes the verdicts the track already carries.
    let painted = call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(painted["changed"], json!(false), "{painted}");
    assert!(
        painted["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{painted}"
    );

    // A resize repeated: the second drag names the size the first left.
    let target = [pixel[0] + 3.0, pixel[1] + 1.0];
    let first = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "edge": "+u", "pixel": target,
        }),
    );
    assert_eq!(first["changed"], json!(true), "{first}");
    let again = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "edge": "+u", "pixel": target,
        }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert!(
        again["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{again}"
    );

    // One version for the one drag that did something, and none for the rest.
    assert_eq!(
        version_count(&state),
        before + 1,
        "a no-effect step pushed a version"
    );
}

/// The commit is in the contract too: a second call on a track the point it
/// wrote already holds answers `changed: false`, names that point as any commit
/// names the one it wrote, and pushes no version.
#[test]
fn a_second_commit_of_the_same_track_answers_changed_false() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let first = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(first["changed"], json!(true), "{first}");
    let point = first["point"]["index"].clone();
    let cursor = first["cursor"].as_str().expect("a cursor").to_string();
    let before = version_count(&state);
    let points = state.scene[0].point_count();

    let again = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(again["serial"], json!(cursor), "{again}");
    assert_eq!(again["cursor"], json!(cursor), "{again}");
    assert_eq!(again["item"], json!(item), "{again}");
    assert_eq!(again["point"]["index"], point, "{again}");
    assert_eq!(again["point"]["replaced"], json!(null), "{again}");
    assert_eq!(again["point"]["id"], first["point"]["id"], "{again}");
    let report = again["report"].as_str().expect("a sentence");
    assert!(report.contains("no effect"), "{report}");
    assert!(report.starts_with("Committed "), "{report}");
    assert_eq!(
        version_count(&state),
        before,
        "a second commit is a version"
    );
    assert_eq!(state.scene[0].point_count(), points, "{again}");
}

/// A pixel off the photograph names no place on it, and the patch whose centre
/// was slid to meet that pixel's ray was flung across the reconstruction. The
/// step takes the nearest place the photograph does name, and the reply says so.
#[test]
fn a_bench_pixel_off_the_photograph_is_clamped_and_the_reply_says_so() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "pixel": [-500.0, -500.0],
        }),
    );
    assert_eq!(moved["clamped"], json!(true), "{moved}");
    assert_eq!(moved["clamped_from"], json!([-500.0, -500.0]), "{moved}");
    // `pixel` is where the moved centre now **projects**, not the clamp's own
    // output: the clamped place is unprojected onto the patch's plane, the patch
    // is carried there and the centre is projected back, and that round trip is
    // good to the arithmetic's last bits rather than bit for bit. So a corner
    // pixel comes back a rounding either side of the corner.
    let used = moved["pixel"].as_array().expect("the pixel it used");
    let camera = &state.scene[0].recon().image_table.cameras[0];
    for (value, extent) in used.iter().zip([camera.width, camera.height]) {
        let value = value.as_f64().expect("a number");
        assert!(
            value >= -1e-6 && value < f64::from(extent),
            "the centre landed at {value}, off a {extent} px axis: {moved}"
        );
    }
    let report = moved["report"].as_str().expect("a sentence");
    assert!(
        report.contains("clamped to the photograph from"),
        "{report}"
    );

    // A pixel on the photograph is left where it was named. Two px in on both
    // axes, and not along the edge the clamp above put the sighting on: the
    // number read back is a reprojection, so it sits a rounding either side of
    // that edge and a pixel named along it would be clamped for the rounding
    // rather than for anything the call said.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let pixel = track["observations"][0]["pixel"].clone();
    let at = pixel.as_array().expect("a pixel");
    let inside = call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({
            "reconstruction_label": "run_a", "track": item, "observation": 0,
            "pixel": [
                at[0].as_f64().expect("x") + 2.0,
                at[1].as_f64().expect("y") + 2.0,
            ],
        }),
    );
    assert_eq!(inside["clamped"], json!(false), "{inside}");
    assert_eq!(inside["clamped_from"], Value::Null, "{inside}");
}

/// [`bearing_demo`] with the patch-frame columns stripped: the bearing a node
/// with no `patch_u_halfvec` holds, which is every row of a `sift_files` value.
fn frameless_bearing_demo() -> sfmtool_core::SfmrReconstruction {
    let mut recon = bearing_demo();
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    recon.rebuild_derived_fields();
    recon
}

/// The flag is the **track's** and not its patch's `w`, so a bearing put on the
/// bench from a node with no patch frames publishes the bearing it is. Reading
/// the frame answered `at_infinity: false` and put a unit direction under
/// `position`, which is a place one unit from the world origin.
#[test]
fn a_frameless_bearing_is_published_as_a_bearing_on_the_wire() {
    let (mut state, mut viewer) = benchable_with(frameless_bearing_demo());

    // The point itself says so, through `get_point`.
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": BENCH_POINT }),
    );
    assert_eq!(point["at_infinity"], json!(true), "{point}");

    // And so does the track put on the bench from it.
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["frame_fitted"], json!(false), "{track}");
    assert_eq!(stage["at_infinity"], json!(true), "{track}");
    assert!(stage["position"].is_null(), "{track}");
    let direction = stage["direction"].as_array().expect("a bearing: {track}");
    let norm: f64 = direction
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-9, "a unit direction: {track}");
}

/// A view request is answered by the frame that draws it whenever the standing
/// reading is about a panel the dock no longer has. Answering from the stale
/// reading reported a `panel_size_points` the very next `get_image_detail_view`
/// contradicted, with the arithmetic done in a panel body that had gone.
#[test]
fn a_view_set_after_a_layout_change_reports_the_panel_as_it_now_is() {
    let (mut state, mut viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));

    // A frame draws the panel and publishes what it measured.
    panel_draws(&mut state, image);
    let view = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(
        view["image_detail_view"]["panel_size_points"],
        json!([VIEW_PANEL[0] as f64, VIEW_PANEL[1] as f64]),
        "{view}"
    );

    // The dock is re-laid out under it: the panel the reading describes is gone.
    lay_out(
        &mut state,
        Tab::ImageDetail,
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1366.0, 720.0)),
    );
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "pixel": [300.0, 200.0], "zoom": 2.0 }),
    );
    assert_eq!(pending.image, image);
    assert!(
        super::super::display::pending_view_reply(&state, &pending).is_none(),
        "answered from the reading the layout change invalidated"
    );

    // The frame that draws in the new layout is the one that answers.
    let after = [1366.0f32, 680.0f32];
    panel_draws_at(&mut state, image, after);
    let landed = view_reply(&state, &pending);
    assert_eq!(
        landed["image_detail_view"]["panel_size_points"],
        json!([after[0] as f64, after[1] as f64]),
        "{landed}"
    );
    assert_eq!(landed["image_detail_view"]["zoom"], json!(2.0), "{landed}");
    assert_about(
        view_centre(&landed),
        [300.0, 200.0],
        "the pixel is off centre",
    );
}

// ── The display transform ───────────────────────────────────────────────
//
// The three tools that set, frame from a patch, and bake a node's display
// transform. What each does to the picture is asserted in
// `display_transform::tests`; these assert the boundary: the reply, the
// `transform` block `get_scene` reports back, the refusals' words and that
// undo walks the framing.

/// `run_a`'s `transform` block, as `get_scene` reports it.
fn transform_block(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    let scene = call(state, viewer, "get_scene", json!({}));
    scene["scene"][0]["transform"].clone()
}

/// The identity, as the wire spells it.
fn identity_block() -> Value {
    json!({
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
        "scale": 1.0,
    })
}

/// A similarity with a turn, a shift and a scale, as the wire spells it.
fn similarity_block() -> Value {
    let half = 0.3_f64;
    json!({
        "rotation_wxyz": [half.cos(), 0.0, half.sin(), 0.0],
        "translation": [1.5, -2.0, 0.25],
        "scale": 1.25,
    })
}

#[test]
fn get_scene_reports_the_transform_block_and_a_set_lands_in_it() {
    let (mut state, mut viewer) = benchable();
    assert_eq!(transform_block(&mut state, &mut viewer), identity_block());

    let reply = call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform",
        json!({ "reconstruction_label": "run_a", "transform": similarity_block() }),
    );

    // A version of the framing and not a change to the data.
    assert_eq!(reply["changed"], true);
    assert_eq!(reply["dirty"], false);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Set transform of run_a: "), "{report}");
    let block = transform_block(&mut state, &mut viewer);
    let want = similarity_block();
    for field in ["rotation_wxyz", "translation"] {
        let (got, want) = (
            block[field].as_array().unwrap(),
            want[field].as_array().unwrap(),
        );
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g.as_f64().unwrap() - w.as_f64().unwrap()).abs() < 1e-12,
                "{block}"
            );
        }
    }
    assert_eq!(block["scale"], 1.25);
    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(scene["scene"][0]["transformed"], true);

    // Undo walks the framing back.
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(transform_block(&mut state, &mut viewer), identity_block());
}

#[test]
fn setting_the_identity_is_the_reset_and_is_refused_where_there_is_nothing_to_reset() {
    let (mut state, mut viewer) = benchable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform",
        json!({ "reconstruction_label": "run_a", "transform": identity_block() }),
    );
    assert!(error.0.contains("already in its own frame"), "{error}");
    assert_eq!(version_count(&state), 1);

    call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform",
        json!({ "reconstruction_label": "run_a", "transform": similarity_block() }),
    );
    let reply = call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform",
        json!({ "reconstruction_label": "run_a", "transform": identity_block() }),
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Reset transform of run_a"), "{report}");
}

#[test]
fn a_transform_that_is_not_a_similarity_is_refused_at_the_parse() {
    let (mut state, mut viewer) = benchable();
    for transform in [
        json!({ "rotation_wxyz": [0.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0], "scale": 1.0 }),
        json!({ "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0], "scale": 0.0 }),
    ] {
        refused_call(
            &mut state,
            &mut viewer,
            "set_reconstruction_transform",
            json!({ "reconstruction_label": "run_a", "transform": transform }),
        );
    }
    assert_eq!(version_count(&state), 1);
}

#[test]
fn set_reconstruction_transform_from_patch_frames_the_scene_on_the_active_patch() {
    let (mut state, mut viewer) = benchable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform_from_patch",
        json!({ "reconstruction_label": "run_a", "mode": "set_to_origin" }),
    );
    assert!(
        error.0.contains("nothing on its bench is active"),
        "{error}"
    );

    let item = on_the_bench(&mut state, &mut viewer);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform_from_patch",
        json!({ "reconstruction_label": "run_a", "mode": "set_to_origin" }),
    );
    assert_eq!(reply["dirty"], false);
    assert_eq!(
        reply["label"],
        format!("Set run_a to the frame of patch {item}")
    );
    assert_ne!(transform_block(&mut state, &mut viewer), identity_block());

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform_from_patch",
        json!({ "reconstruction_label": "run_a", "mode": "frame" }),
    );
    assert!(error.0.contains("set_to_origin"), "{error}");
}

#[test]
fn bake_reconstruction_transform_is_one_edit_and_undo_puts_the_framing_back() {
    let (mut state, mut viewer) = benchable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "bake_reconstruction_transform",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(error.0.contains("already in its own frame"), "{error}");
    assert_eq!(failures(&state).len(), 1, "{:?}", rows(&state));

    on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "set_reconstruction_transform_from_patch",
        json!({ "reconstruction_label": "run_a", "mode": "align_normal_to_z" }),
    );
    let framed = transform_block(&mut state, &mut viewer);

    let reply = call(
        &mut state,
        &mut viewer,
        "bake_reconstruction_transform",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(reply["changed"], true);
    assert_eq!(reply["dirty"], true);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Baked transform of run_a: "), "{report}");
    let last = rows(&state).pop().expect("a row");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
    assert_eq!(transform_block(&mut state, &mut viewer), identity_block());

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(transform_block(&mut state, &mut viewer), framed);
    assert!(!state.scene[0].is_dirty());
}
