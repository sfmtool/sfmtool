// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless frames of Image Detail drawing the fixture's measurements
//! ([`crate::measurements::fixture`]) over image 0: marks at the endpoint
//! pixels, the warning colour of a refused endpoint, the hover, a click and a
//! double-click, a press on a mark owning it (G7), the focused item's handle
//! drag, and the off-screen indicator.
//!
//! The photograph is a flat stand-in of the fixture's size; the panel is the
//! whole frame, so at zoom 1 the image is fitted and centred in it.

use crate::test_support::HeldPhotograph;
use eframe::egui;
use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};

use crate::measurement_marks::{DrawnId, MeasurementGesture};
use crate::measurements::{EndpointSide, MeasurementRef};
use crate::state::{AppState, FeatureDisplaySettings, IntrinsicsDisplaySettings};

use super::super::{BenchMenu, ImageDetail, ImageDetailResponse};
use super::Placement;

const PANEL: egui::Vec2 = egui::vec2(900.0, 700.0);
const IMAGE: egui::Vec2 = egui::vec2(640.0, 480.0);

/// What one frame left behind.
struct Frame {
    response: ImageDetailResponse,
    texts: Vec<String>,
    cursor: egui::CursorIcon,
}

fn pyramid() -> HeldPhotograph {
    let pixels = ImageU8::new(640, 480, 3, vec![90u8; 640 * 480 * 3]);
    HeldPhotograph::new(ImageU8Pyramid::build(&pixels, 1))
}

/// One frame of the panel on image 0 of the fixture's node, its layer built
/// as the dock builds it.
fn frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    state: &AppState,
    photograph: &HeldPhotograph,
    events: Vec<egui::Event>,
) -> Frame {
    detail.measurements = state.measurement_layer(state.feature_display.show_all_measurements);
    let node = &state.scene[0];
    let mut intrinsics = IntrinsicsDisplaySettings::default();
    let features = FeatureDisplaySettings::default();
    let view_only = state.bench_view_only_refusal(node.id);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
        events,
        // Half a second a frame, so clicks on separate frames are never a
        // double-click and the two of one frame are.
        time: Some(ctx.input(|i| i.time) + 0.5),
        ..Default::default()
    };
    let mut response = None;
    let mut output = ctx.run_ui(input, |ui| {
        response = Some(detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(0),
            None,
            None,
            None,
            BenchMenu {
                view_only: view_only.as_deref(),
                ..BenchMenu::default()
            },
            &[],
            &crate::platform::ScrollInput::default(),
            None,
            Some(photograph.source()),
            &features,
            &mut intrinsics,
            None,
        ));
    });
    output.textures_delta.clear();
    Frame {
        response: response.expect("the panel ran"),
        texts: crate::test_support::output_texts(&output),
        cursor: output.platform_output.cursor_icon,
    }
}

/// The placement of image 0 at `detail`'s view: fitted at zoom 1, and moved
/// by the pan and the zoom.
fn placement(state: &AppState, detail: &ImageDetail) -> Placement {
    let panel_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL);
    let fit = (PANEL.x / IMAGE.x).min(PANEL.y / IMAGE.y);
    let scale = fit * detail.zoom;
    Placement {
        image: crate::scene::ImageRef::new(state.scene[0].id, 0),
        image_size: IMAGE,
        image_rect: egui::Rect::from_center_size(panel_rect.center() + detail.pan, IMAGE * scale),
        scale,
        panel_rect,
    }
}

/// Where `side`'s pixel of stored measurement `index` lands in the panel.
fn at_pixel(
    state: &AppState,
    detail: &ImageDetail,
    index: usize,
    side: EndpointSide,
) -> egui::Pos2 {
    let m = MeasurementRef::distance(state.scene[0].id, index);
    let [x, y] = state.distance_row(m).unwrap().pixel(side);
    let at = placement(state, detail);
    at.image_rect.min + egui::vec2(x as f32, y as f32) * at.scale
}

fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Click `times` times at `pos`, all in one frame as a double-click arrives,
/// returning both frames' responses.
fn click(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    state: &AppState,
    photograph: &HeldPhotograph,
    pos: egui::Pos2,
    times: usize,
) -> Vec<ImageDetailResponse> {
    let events = (0..times)
        .flat_map(|_| [press(pos, true), press(pos, false)])
        .collect();
    vec![
        frame(
            detail,
            ctx,
            state,
            photograph,
            vec![egui::Event::PointerMoved(pos)],
        )
        .response,
        frame(detail, ctx, state, photograph, events).response,
    ]
}

/// A panel and context that have laid out twice.
fn settled(state: &AppState, photograph: &HeldPhotograph) -> (ImageDetail, egui::Context) {
    let mut detail = ImageDetail::new();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        frame(&mut detail, &ctx, state, photograph, Vec::new());
    }
    (detail, ctx)
}

fn gestures(responses: Vec<ImageDetailResponse>) -> Vec<MeasurementGesture> {
    responses
        .into_iter()
        .filter_map(|r| r.measurement_gesture)
        .collect()
}

/// The screen the panel builds for its current view.
fn screen(
    detail: &ImageDetail,
    ctx: &egui::Context,
    state: &AppState,
) -> crate::measurement_marks::Screen {
    let at = placement(state, detail);
    let mut out = None;
    crate::test_support::run_frame_headless(ctx, egui::RawInput::default(), |ui| {
        out = Some(detail.measurement_screen(ui.painter(), &at));
    });
    out.unwrap()
}

/// A mark at each endpoint pixel in the image shown, a line where both are
/// in it, and a refused endpoint's mark in the warning colour; an endpoint
/// in another image draws no mark there.
#[test]
fn marks_are_at_the_endpoint_pixels_in_the_image_shown() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    state.feature_display.show_all_measurements = true;
    let photograph = pyramid();
    let (detail, ctx) = settled(&state, &photograph);
    let screen = screen(&detail, &ctx, &state);
    let layer = &detail.measurements;
    let of = |index: usize| {
        screen
            .of(layer, &DrawnId::Stored(MeasurementRef::distance(id, index)))
            .unwrap_or_else(|| panic!("measurement {index} is not drawn"))
    };
    let both = of(0);
    assert_eq!(both.marks.len(), 2);
    assert!(
        both.marks[0]
            .pos
            .distance(at_pixel(&state, &detail, 0, EndpointSide::A))
            < 1e-3
    );
    assert!(
        both.marks[1]
            .pos
            .distance(at_pixel(&state, &detail, 0, EndpointSide::B))
            < 1e-3
    );
    assert_eq!(both.line, Some([both.marks[0].pos, both.marks[1].pos]));
    // Measurement 1's B is in image 1: its A alone, no line.
    assert_eq!(of(1).marks.len(), 1);
    assert_eq!(of(1).line, None);
    // Measurement 2's B is refused: drawn, in the warning colour.
    let refused = of(2);
    assert_eq!(refused.marks.len(), 2);
    assert!(!refused.marks[0].warning && refused.marks[1].warning);
    // Measurement 3's B is on a point at infinity: its pixel is drawn.
    assert_eq!(of(3).marks.len(), 2);
    assert!(of(3).marks.iter().all(|m| !m.warning));
}

/// Hovering a mark shows the row text and the pointing hand; a click on it
/// asks to select that endpoint and selects no feature's point under it
/// (G7); a double-click asks to look through it; a click on the line asks to
/// select the measurement.
#[test]
fn a_mark_owns_its_press_and_answers_a_click_and_a_double_click() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    let m = MeasurementRef::distance(id, 0);
    state.select_measurement(m);
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let b = at_pixel(&state, &detail, 0, EndpointSide::B);

    frame(
        &mut detail,
        &ctx,
        &state,
        &photograph,
        vec![egui::Event::PointerMoved(a)],
    );
    let hovered = frame(&mut detail, &ctx, &state, &photograph, Vec::new());
    let hover = detail.measurements.drawn[0].hover.clone();
    assert!(
        hovered.texts.contains(&hover),
        "{hover:?} not in {:?}",
        hovered.texts
    );
    assert_eq!(hovered.cursor, egui::CursorIcon::PointingHand);

    let responses = click(&mut detail, &ctx, &state, &photograph, a, 1);
    assert!(
        responses.iter().all(|r| r.select_point.is_none()),
        "a feature lies under the mark"
    );
    assert_eq!(
        gestures(responses),
        vec![MeasurementGesture::SelectEndpoint(
            DrawnId::Stored(m),
            EndpointSide::A
        )]
    );
    let responses = click(&mut detail, &ctx, &state, &photograph, b, 2);
    assert_eq!(
        gestures(responses).last(),
        Some(&MeasurementGesture::LookThrough(
            DrawnId::Stored(m),
            EndpointSide::B
        ))
    );
    assert_eq!(
        gestures(click(
            &mut detail,
            &ctx,
            &state,
            &photograph,
            a.lerp(b, 0.5),
            1
        )),
        vec![MeasurementGesture::Select(DrawnId::Stored(m))]
    );
    // The same click on the feature with no measurement drawn selects its
    // point.
    state.selected_measurement = None;
    let responses = click(&mut detail, &ctx, &state, &photograph, a, 1);
    assert!(
        responses.iter().any(|r| r.select_point.is_some()),
        "the feature is under the mark"
    );
}

/// Drag `from` by `by` in `steps` frames and release, returning every
/// frame's response.
fn drag(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    state: &AppState,
    photograph: &HeldPhotograph,
    from: egui::Pos2,
    by: egui::Vec2,
) -> Vec<ImageDetailResponse> {
    let mut out = vec![
        frame(
            detail,
            ctx,
            state,
            photograph,
            vec![egui::Event::PointerMoved(from)],
        )
        .response,
        frame(detail, ctx, state, photograph, vec![press(from, true)]).response,
    ];
    for step in 1..=4 {
        let at = from + by * (step as f32 / 4.0);
        out.push(
            frame(
                detail,
                ctx,
                state,
                photograph,
                vec![egui::Event::PointerMoved(at)],
            )
            .response,
        );
    }
    out.push(
        frame(
            detail,
            ctx,
            state,
            photograph,
            vec![press(from + by, false)],
        )
        .response,
    );
    out
}

/// A drag of the focused item's handle asks, on release, to set that
/// endpoint to the pixel under the release in the image shown, and the step
/// it asks for is one version; the photograph does not pan under it.
#[test]
fn a_handle_drag_sets_the_endpoint_in_one_version() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    let m = MeasurementRef::distance(id, 0);
    state.put_measurement_on_bench(m).unwrap();
    let label = state.focused_measurement_label(id).unwrap().to_string();
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let pan = detail.pan;
    let responses = drag(
        &mut detail,
        &ctx,
        &state,
        &photograph,
        a,
        egui::vec2(42.0, 21.0),
    );
    assert_eq!(detail.pan, pan, "the photograph panned under a handle");
    let at = placement(&state, &detail);
    let release = a + egui::vec2(42.0, 21.0);
    let want = [
        f64::from((release.x - at.image_rect.min.x) / at.scale),
        f64::from((release.y - at.image_rect.min.y) / at.scale),
    ];
    let sets: Vec<_> = gestures(responses);
    let [MeasurementGesture::SetEndpoint(set)] = sets.as_slice() else {
        panic!("one set asked for: {sets:?}");
    };
    assert_eq!(
        (set.node, set.item.as_str(), set.side, set.image),
        (id, label.as_str(), EndpointSide::A, 0)
    );
    assert!((set.pixel[0] - want[0]).abs() < 1e-3 && (set.pixel[1] - want[1]).abs() < 1e-3);
    let versions = state.scene[0].history.all_serials().len();
    state.set_endpoint(set);
    assert_eq!(
        state.scene[0].history.all_serials().len(),
        versions + 1,
        "one version"
    );
    let (_, pixel) = state.focused_item_endpoint(id, EndpointSide::A).unwrap();
    assert!((pixel[0] - want[0]).abs() < 1e-3, "{pixel:?}");
}

/// On a view-only bench the item's marks are not handles: a drag of one asks
/// for nothing, and still pans nothing, the press being the mark's.
#[test]
fn a_view_only_bench_draws_no_handles() {
    let mut recon = crate::measurements::fixture::measured_reconstruction(Some("m"));
    let count = recon.point_set.tracks.len();
    recon.point_set.observations = sfmtool_core::ObservationSource::SiftFiles {
        feature_indexes: vec![0; count],
        feature_tool_hashes: vec![[0u8; 16]; 3],
        sift_content_hashes: vec![[0u8; 16]; 3],
    };
    let mut state = AppState::new();
    let id = state.append_node(crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    assert!(state.bench_view_only_refusal(id).is_some());
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    assert!(detail.measurements.drawn.iter().all(|d| !d.handles));
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let pan = detail.pan;
    frame(
        &mut detail,
        &ctx,
        &state,
        &photograph,
        vec![egui::Event::PointerMoved(a)],
    );
    let hovered = frame(&mut detail, &ctx, &state, &photograph, Vec::new());
    assert_eq!(hovered.cursor, egui::CursorIcon::PointingHand, "not a grab");
    let responses = drag(
        &mut detail,
        &ctx,
        &state,
        &photograph,
        a,
        egui::vec2(42.0, 21.0),
    );
    assert_eq!(gestures(responses), Vec::new());
    assert_eq!(detail.pan, pan);
}

/// Zoomed in away from the selected measurement, an indicator points to its
/// endpoints, and a click on it centres the panel on them at the same zoom;
/// in view there is none.
#[test]
fn the_indicator_points_to_an_endpoint_outside_the_view_and_centres_it() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    assert!(screen(&detail, &ctx, &state).indicator.is_none(), "in view");
    // Zoomed in on the image's top-left corner, the endpoints near its middle
    // are off the panel's bottom right.
    detail.zoom = 8.0;
    detail.pan = egui::vec2(2000.0, 1500.0);
    frame(&mut detail, &ctx, &state, &photograph, Vec::new());
    let (zoom, pan) = (detail.zoom, detail.pan);
    let shown = screen(&detail, &ctx, &state);
    let indicator = shown.indicator.clone().expect("off the panel");
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let b = at_pixel(&state, &detail, 0, EndpointSide::B);
    let toward = (a.lerp(b, 0.5) - egui::pos2(450.0, 350.0)).normalized();
    assert!(
        indicator.direction.dot(toward) > 0.999,
        "{:?}",
        indicator.direction
    );
    let painted = frame(&mut detail, &ctx, &state, &photograph, Vec::new());
    assert!(painted.texts.contains(&"Distance #0".to_string()));
    click(
        &mut detail,
        &ctx,
        &state,
        &photograph,
        indicator.tip - indicator.direction * 6.0,
        1,
    );
    assert_eq!(detail.zoom, zoom, "the zoom is kept");
    assert_ne!(detail.pan, pan);
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let b = at_pixel(&state, &detail, 0, EndpointSide::B);
    assert!(
        a.lerp(b, 0.5).distance(egui::pos2(450.0, 350.0)) < 1.0,
        "the midpoint is centred: {:?}",
        a.lerp(b, 0.5)
    );
    assert!(screen(&detail, &ctx, &state).indicator.is_none());
}

/// Escape while a handle is held abandons the drag: the release asks for no
/// step, and the photograph does not pan.
#[test]
fn escape_abandons_a_handle_drag() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    let a = at_pixel(&state, &detail, 0, EndpointSide::A);
    let pan = detail.pan;
    let by = egui::vec2(42.0, 21.0);
    let escape = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let mut responses = vec![
        frame(
            &mut detail,
            &ctx,
            &state,
            &photograph,
            vec![egui::Event::PointerMoved(a)],
        )
        .response,
        frame(&mut detail, &ctx, &state, &photograph, vec![press(a, true)]).response,
        frame(
            &mut detail,
            &ctx,
            &state,
            &photograph,
            vec![egui::Event::PointerMoved(a + by * 0.5)],
        )
        .response,
        frame(
            &mut detail,
            &ctx,
            &state,
            &photograph,
            vec![egui::Event::PointerMoved(a + by), escape],
        )
        .response,
    ];
    responses.push(
        frame(
            &mut detail,
            &ctx,
            &state,
            &photograph,
            vec![press(a + by, false)],
        )
        .response,
    );
    assert_eq!(gestures(responses), Vec::new(), "Escape abandoned the drag");
    assert_eq!(detail.pan, pan);
    // Without the Escape, the same drag asks for the step.
    let responses = drag(&mut detail, &ctx, &state, &photograph, a, by);
    assert_eq!(gestures(responses).len(), 1);
}

/// A stored measurement whose columns disagree in length, one of whose
/// pixels is not a number, is drawn by a whole frame without a panic: that
/// pixel draws no mark and no line, and every mark drawn is at a finite
/// position.
#[test]
fn a_measurement_with_mismatched_columns_and_a_nan_pixel_draws() {
    let mut recon = crate::measurements::fixture::measured_reconstruction(Some("m"));
    let m = recon.image_set.measurements.distance.as_mut().unwrap();
    m.image_indexes = m.image_indexes.slice(ndarray::s![..4, ..]).to_owned();
    m.units = m.units.slice(ndarray::s![..4]).to_owned();
    m.endpoints_xy[[0, 0, 0]] = f32::NAN;
    let mut state = AppState::new();
    let id = state.append_node(crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    state.feature_display.show_all_measurements = true;
    state.select_measurement(MeasurementRef::distance(id, 0));
    let photograph = pyramid();
    let (mut detail, ctx) = settled(&state, &photograph);
    let shown = frame(&mut detail, &ctx, &state, &photograph, Vec::new());
    assert!(!shown.texts.is_empty(), "the frame drew");
    let screen = screen(&detail, &ctx, &state);
    assert!(
        !screen.measurements.is_empty(),
        "the other measurements are drawn"
    );
    let nan_row = DrawnId::Stored(MeasurementRef::distance(id, 0));
    let nan = screen
        .of(&detail.measurements, &nan_row)
        .expect("its other endpoint is in the image");
    let sides: Vec<EndpointSide> = nan.marks.iter().map(|m| m.side).collect();
    assert_eq!(sides, vec![EndpointSide::B], "A's pixel is not a number");
    assert!(nan.line.is_none());
    for m in &screen.measurements {
        for mark in &m.marks {
            assert!(mark.pos.x.is_finite() && mark.pos.y.is_finite(), "{mark:?}");
        }
    }
}
