// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The panel controls an agent reads and writes as documents, which are the
//! ones the Action Log files under [`crate::action_log::Kind::Display`]: what
//! the Image Detail panel draws over its photograph, and whether the Action
//! Log records detailed timing.
//!
//! ## The Image Detail panel's overlays
//!
//! Everything the panel's toolbar decides — the seven feature overlay modes,
//! the three filters on the features, and the intrinsics layer with its own
//! sub-toggles — is scene-level state on `AppState` (`feature_display`,
//! `intrinsics_display`) rather than a property of any image or
//! reconstruction. That is why it is one document and one pair of tools rather
//! than fields on `set_reconstruction_display`: it describes how the panel
//! looks at *whatever* is selected, and a `screenshot` of `image_detail` shows
//! whichever mode was last picked.
//!
//! Three things about the shape here are worth knowing before reading it:
//!
//! - **Every refusal is at the parse.** The vocabularies are static — seven
//!   modes, two ladders, the bounds on a size filter — so [`parse_change`]
//!   validates the whole call before a `Command` exists and [`set`] cannot
//!   fail. A call naming a good field and a bad one changes nothing.
//! - **`feature_size_px` is one thing here because it is one checkbox in the
//!   toolbar.** See [`FeatureSize`].
//! - **The Action Log entries are not written here.** They come from
//!   [`crate::state::record_image_detail_changes`], the one differ the panel's
//!   own frame goes through as well, which is what keeps the human's row and
//!   the agent's row for one control identical.
//!
//! ## Detailed timing
//!
//! The Action Log toolbar's **Detailed timing** checkbox, by the same rule:
//! [`set_timing_detail`] calls
//! [`ActionLog::set_detailed_timing`](crate::action_log::ActionLog::set_detailed_timing),
//! which is the call the checkbox itself makes, so the entry an agent leaves
//! is the entry a human leaves and neither can record one the other would not.

use serde_json::{json, Value};

use super::tools::Args;
use super::{
    resolve_camera_image, resolve_reconstruction, Deferred, ImageDetailDisplayChange,
    ImageDetailTarget, ImageDetailViewRequest, IntrinsicsChange, JsonReply, Outcome, PendingView,
    Reply, ToolError, ToolOutput,
};
use crate::dock::Tab;
use crate::image_detail::{Look, ViewGeometry};
use crate::scene::{ImageRef, ReconId};
use crate::state::{
    record_image_detail_changes, AppState, FeatureDisplaySettings, ImageDetailDisplay,
    IntrinsicsDisplaySettings, OverlayMode,
};

/// A `feature_size_px` a call carried: the two bounds, or the filter off.
///
/// One value rather than two optional bounds, because the toolbar re-derives
/// `min_feature_size` and `max_feature_size` from its single `Min/max size:`
/// checkbox **every frame** — ticked, both are written from the persisted drag
/// values; unticked, both are cleared. So the two are never independently
/// `Some` while the panel is open, and a tool that let an agent set one
/// without the other would have its half undone by the next frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FeatureSize {
    /// Both bounds, in pixels. Writing this writes **all four** fields — the
    /// two options and the two drag values — so the toolbar's next frame
    /// re-derives exactly what the agent asked for.
    Between { min: f32, max: f32 },
    /// The filter off, which is what unticking the checkbox does: the two
    /// options are cleared and the drag values stay where they were.
    Off,
}

/// `get_image_detail_display`: the whole document, no arguments.
pub(super) fn get(state: &AppState) -> JsonReply {
    Ok(document(state))
}

/// `set_image_detail_display`: write the fields the call named, record what
/// that changed, and answer with the whole document.
///
/// The reply is exactly what [`get`] would return, so the agent reads back the
/// state rather than the fields it happened to set — the same rule as
/// `set_reconstruction_display`.
pub(super) fn set(state: &mut AppState, change: &ImageDetailDisplayChange) -> JsonReply {
    let before = ImageDetailDisplay::snapshot(&state.feature_display, &state.intrinsics_display);
    apply_feature(&mut state.feature_display, change);
    apply_intrinsics(&mut state.intrinsics_display, &change.intrinsics);
    let after = ImageDetailDisplay::snapshot(&state.feature_display, &state.intrinsics_display);
    record_image_detail_changes(&mut state.action_log, &before, &after);
    Ok(document(state))
}

fn apply_feature(feature: &mut FeatureDisplaySettings, change: &ImageDetailDisplayChange) {
    if let Some(mode) = change.overlay_mode {
        feature.overlay_mode = mode;
    }
    if let Some(max_features) = change.max_features {
        feature.max_features = max_features;
    }
    match change.feature_size_px {
        None => {}
        Some(FeatureSize::Between { min, max }) => {
            feature.min_feature_size = Some(min);
            feature.max_feature_size = Some(max);
            feature.min_feature_size_value = min;
            feature.max_feature_size_value = max;
        }
        Some(FeatureSize::Off) => {
            feature.min_feature_size = None;
            feature.max_feature_size = None;
        }
    }
    if let Some(tracked_only) = change.tracked_only {
        feature.tracked_only = tracked_only;
    }
}

fn apply_intrinsics(intrinsics: &mut IntrinsicsDisplaySettings, change: &IntrinsicsChange) {
    if let Some(enabled) = change.enabled {
        intrinsics.enabled = enabled;
    }
    if let Some(axes) = change.axes {
        intrinsics.axes = axes;
    }
    if let Some(rings) = change.rings {
        intrinsics.rings = rings;
    }
    if let Some(distortion) = change.distortion {
        intrinsics.distortion = distortion;
    }
    if let Some(distortion_scale) = change.distortion_scale {
        intrinsics.distortion_scale = distortion_scale;
    }
    if let Some(grid_cols) = change.grid_cols {
        intrinsics.grid_cols = grid_cols;
    }
}

/// The document both tools answer with.
///
/// The feature overlay and its filters at the top level and the intrinsics
/// layer as an `intrinsics` sub-block, which is how the toolbar draws them: the
/// feature controls in a row, the layer behind one checkbox and a gear.
fn document(state: &AppState) -> Value {
    let feature = &state.feature_display;
    let intrinsics = &state.intrinsics_display;
    json!({
        "image_detail_display": {
            "overlay_mode": feature.overlay_mode.wire_name(),
            "max_features": feature.max_features,
            "feature_size_px": feature_size_px(feature),
            "tracked_only": feature.tracked_only,
            "intrinsics": {
                "enabled": intrinsics.enabled,
                "axes": intrinsics.axes,
                "rings": intrinsics.rings,
                "distortion": intrinsics.distortion,
                "distortion_scale": intrinsics.distortion_scale.map(f64::from),
                "grid_cols": intrinsics.grid_cols,
            },
        }
    })
}

/// The size filter as one object, or `null` for no size filter.
///
/// A bound somehow set on its own falls back to the drag value beside it,
/// which is the number the toolbar would show for it — the object always
/// carries both, because both is what setting it takes.
fn feature_size_px(feature: &FeatureDisplaySettings) -> Value {
    match (feature.min_feature_size, feature.max_feature_size) {
        (None, None) => Value::Null,
        (min, max) => json!({
            "min": f64::from(min.unwrap_or(feature.min_feature_size_value)),
            "max": f64::from(max.unwrap_or(feature.max_feature_size_value)),
        }),
    }
}

/// Every argument `set_image_detail_display` takes, and the whole of what it
/// refuses.
///
/// Called from [`super::tools::parse`], so a bad mode name, an off-ladder
/// value or an inverted size filter is turned away before a `Command` exists —
/// which is what makes a refusal atomic without [`set`] needing a rollback.
pub(super) fn parse_change(args: &Args) -> Result<ImageDetailDisplayChange, ToolError> {
    let change = ImageDetailDisplayChange {
        overlay_mode: parse_overlay_mode(args)?,
        // Doubly optional: absent leaves the cap alone, an explicit null lifts
        // it. Refused at 0, since "show no features" is what
        // `overlay_mode: "none"` says, and the `Max:` dropdown cannot show a
        // second spelling of it.
        max_features: match args.get("max_features") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(_) => {
                let max = args.required_usize("max_features")?;
                if max == 0 {
                    return Err(args.error(
                        "wants max_features to be 1 or more — \"show no features\" is \
                         overlay_mode \"none\".",
                    ));
                }
                Some(Some(max))
            }
        },
        feature_size_px: parse_feature_size(args)?,
        tracked_only: args.optional_bool("tracked_only")?,
        intrinsics: parse_intrinsics(args)?,
    };
    if change == ImageDetailDisplayChange::default() {
        return Err(args.error("was given nothing to change."));
    }
    Ok(change)
}

fn parse_overlay_mode(args: &Args) -> Result<Option<OverlayMode>, ToolError> {
    let Some(name) = args.optional_string("overlay_mode")? else {
        return Ok(None);
    };
    OverlayMode::from_wire_name(&name).map(Some).ok_or_else(|| {
        args.error(format!(
            "does not know the overlay mode {name:?} — the modes are {}.",
            OverlayMode::all_wire_names()
        ))
    })
}

/// The size filter, as an object of two bounds or an explicit `null`.
fn parse_feature_size(args: &Args) -> Result<Option<FeatureSize>, ToolError> {
    let value = match args.get("feature_size_px") {
        None => return Ok(None),
        Some(Value::Null) => return Ok(Some(FeatureSize::Off)),
        Some(value) => value,
    };
    let map = value.as_object().ok_or_else(|| {
        args.error("wants feature_size_px to be an object with min and max, or null.")
    })?;
    let inner = Args::new("set_image_detail_display.feature_size_px", map);
    inner.reject_unknown(&["min", "max"])?;
    let min = inner.required_f64("min")?;
    let max = inner.required_f64("max")?;
    for (name, bound) in [("min", min), ("max", max)] {
        if !bound.is_finite() || bound < 0.0 {
            return Err(inner.error(format!(
                "wants {name} to be a size in pixels, zero or more — got {bound}."
            )));
        }
    }
    if min > max {
        return Err(inner.error(format!(
            "was given min {min} above max {max}, which selects no feature at all."
        )));
    }
    Ok(Some(FeatureSize::Between {
        min: min as f32,
        max: max as f32,
    }))
}

fn parse_intrinsics(args: &Args) -> Result<IntrinsicsChange, ToolError> {
    let value = match args.get("intrinsics") {
        None | Some(Value::Null) => return Ok(IntrinsicsChange::default()),
        Some(value) => value,
    };
    let map = value
        .as_object()
        .ok_or_else(|| args.error("wants intrinsics to be an object of the layer's controls."))?;
    let inner = Args::new("set_image_detail_display.intrinsics", map);
    inner.reject_unknown(&[
        "enabled",
        "axes",
        "rings",
        "distortion",
        "distortion_scale",
        "grid_cols",
    ])?;
    Ok(IntrinsicsChange {
        enabled: inner.optional_bool("enabled")?,
        axes: inner.optional_bool("axes")?,
        rings: inner.optional_bool("rings")?,
        distortion: inner.optional_bool("distortion")?,
        // Doubly optional, and off its ladder it is refused: those are the
        // exaggerations the gear popup offers, and a value the popup cannot
        // show is a value the human cannot see they are looking at.
        distortion_scale: match inner.get("distortion_scale") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(_) => {
                let scale = inner.required_f64("distortion_scale")? as f32;
                if !IntrinsicsDisplaySettings::SCALE_LADDER.contains(&scale) {
                    return Err(inner.error(format!(
                        "does not offer the distortion scale {scale} — the ladder is {}, or null \
                         for auto.",
                        ladder(&IntrinsicsDisplaySettings::SCALE_LADDER)
                    )));
                }
                Some(Some(scale))
            }
        },
        grid_cols: match inner.optional_usize("grid_cols")? {
            None => None,
            Some(cols) => {
                if !IntrinsicsDisplaySettings::GRID_LADDER.contains(&cols) {
                    return Err(inner.error(format!(
                        "does not offer the grid density {cols} — the ladder is {}.",
                        ladder(&IntrinsicsDisplaySettings::GRID_LADDER)
                    )));
                }
                Some(cols)
            }
        },
    })
}

/// A ladder as an error message lists it: `1, 2, 3, 5, 10, 20, 50`.
fn ladder<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// ── The Image Detail panel's view ───────────────────────────────────────

/// `get_image_detail_view`: where the panel is looking, no arguments.
///
/// The **panel's** reading and not the selection's: it is what the last frame
/// that drew a photograph settled on, so a `select_camera_image` sent a moment
/// ago is not in it yet. That is the honest answer to "what is on screen", and
/// it is the reading a `screenshot` of the panel would show.
///
/// Every field is `null` before the panel has drawn an image at all -- a fresh
/// session, a panel that was never opened, no camera image selected -- because
/// what fit means, and therefore what the zoom and the visible rectangle mean,
/// is settled by a panel size that does not exist yet.
pub(super) fn get_view(state: &AppState) -> JsonReply {
    Ok(json!({ "image_detail_view": view_document(state, state.image_detail_view) }))
}

/// `set_image_detail_view`: point the panel at one place in one photograph.
///
/// Three steps, in this order, because each can refuse and a refusal should
/// change nothing: the call's node, photograph and target are resolved into a
/// pixel or a rectangle; the panel is brought to where the call can be seen --
/// the photograph selected, the panel surfaced if it was closed or behind
/// another tab; and the request is left for the panel to apply on the frame it
/// draws that photograph. The
/// reply is where the view landed in the same shape [`get_view`] answers in, so
/// the caller reads the applied view -- including a zoom the panel's range
/// clamped -- rather than the numbers it sent.
///
/// **The call selects the photograph it is about.** A `bench_observation` and a
/// `point` name their own, a `camera_image` argument names one outright, and
/// [`AppState::look_at_in_image`] selects whichever it is as the row click in
/// Track View does. So the only thing left to refuse is a call that names no
/// photograph with none selected.
///
/// **The panel's size is the panel's to say**, so a call arriving before it has
/// drawn one has no frame to compute a view in and cannot answer in its own
/// frame. It defers instead of refusing ([`super::PendingView`]): the look is
/// standing, the panel applies it on the frame it draws the photograph, and the
/// reply is the reading that frame publishes. Where a reading already stands,
/// the answer is in this frame, through
/// [`crate::image_detail::look_at`] -- the same function the panel itself runs
/// a frame later, so what the reply says and what the panel does are one
/// computation rather than two that could disagree.
pub(super) fn set_view(state: &mut AppState, request: &ImageDetailViewRequest) -> Outcome {
    match apply_view(state, request) {
        Ok(outcome) => outcome,
        Err(refusal) => super::done(Err(refusal)),
    }
}

/// [`set_view`]'s body, with its refusals as a `Result` so every one of them is
/// a `?` and none of them can be reached after the state has moved.
fn apply_view(
    state: &mut AppState,
    request: &ImageDetailViewRequest,
) -> Result<Outcome, ToolError> {
    let id = resolve_reconstruction(state, request.reconstruction_label.as_deref())?;
    let (image, look) = resolve_target(state, id, request)?;
    let standing = geometry_for(state, image)?;
    // Whether the standing reading still describes the panel the dock now has,
    // and whether the panel is docked at all -- both read before the surfacing
    // below, which gives the panel a new leaf when it opens one rather than
    // raising a tab that was already there.
    let current = state.image_detail_view_is_current();
    let docked = state.dock.find_tab(&Tab::ImageDetail).is_some();
    let after = state.image_detail_view_serial;
    state.look_at_in_image(image, look);
    // A view of a panel nobody can see is a view nobody asked for -- and a
    // panel docked behind another tab draws nothing, so there would also be no
    // frame to answer from. Surfaced only when it is not already in front,
    // because `show_panel` records a row of its own and an agent walking a
    // track's observations should leave one `Raised` line and not one per step.
    if !state.panel_is_in_front(Tab::ImageDetail) {
        state.show_panel(Tab::ImageDetail);
    }
    // Two reasons to let the frame answer instead: the panel has never drawn,
    // and the reading it did draw is about a layout that has moved. The second
    // is what a reply reporting a stale `panel_size_points` was -- the
    // arithmetic was done in a panel body that no longer existed -- and the
    // deferral costs nothing, because the drain runs before the egui pass and
    // the frame that applies the look is the frame that answers.
    let Some(geometry) = standing.filter(|_| current && docked) else {
        return Ok(Outcome::Deferred(Deferred::ImageDetailView(PendingView {
            image,
            started: std::time::Instant::now(),
            after,
        })));
    };
    let landed = crate::image_detail::look_at(geometry, &look);
    Ok(super::done(Ok(
        json!({ "image_detail_view": view_document(state, Some(landed)) }),
    )))
}

/// Whether the frame just drawn answered a waiting `set_image_detail_view`, and
/// with what.
///
/// `None` while the panel has still not drawn that photograph, which puts the
/// call back in the queue for the next frame. The deadline is what keeps a
/// caller from waiting on a frame that is never going to come: a photograph the
/// workspace no longer holds draws nothing however long it is given, and an
/// `embedded_patches` node has no photographs at all.
pub(super) fn pending_view_reply(state: &AppState, pending: &super::PendingView) -> Option<Reply> {
    if let Some(view) = state
        .image_detail_view
        .filter(|_| state.image_detail_view_serial > pending.after)
        .filter(|view| view.image == pending.image)
    {
        return Some(Ok(ToolOutput::Json(
            json!({ "image_detail_view": view_document(state, Some(view)) }),
        )));
    }
    if pending.started.elapsed() < DRAWS_WITHIN {
        return None;
    }
    Some(Err(ToolError::new(format!(
        "The Image Detail panel has not drawn {} since the call, so there is no view to report. \
         The look is standing and the panel will apply it on the frame it does draw it; a \
         photograph the workspace no longer holds never will. get_image_detail_view says what \
         the panel is looking at now.",
        state.image_name(pending.image),
    ))))
}

/// How long a view request waits for the panel to draw the photograph.
///
/// The panel draws in the very frame the call is applied in -- the drain runs
/// before the egui pass, and the decode is on this thread -- so this is not a
/// budget for the work but a bound on a frame that is never coming: a closed
/// window, or a photograph that cannot be read. Half a second is a dozen frames
/// of slack and still short of a wait anyone would sit through.
const DRAWS_WITHIN: std::time::Duration = std::time::Duration::from_millis(500);

/// The photograph a call is about, and the look it is asking for in it.
///
/// Where the photograph comes from is the order the request's own doc states:
/// the one the call named, else the one the target names, else the one already
/// selected. Only `bench_observation` names one -- a `point` is *looked for* in
/// the photograph being looked at, and says so when it is not there.
fn resolve_target(
    state: &mut AppState,
    id: ReconId,
    request: &ImageDetailViewRequest,
) -> Result<(ImageRef, Look), ToolError> {
    let named = match &request.camera_image {
        Some(selector) => Some(resolve_camera_image(state, id, selector)?),
        None => None,
    };
    let zoom = request.zoom;
    match &request.target {
        ImageDetailTarget::Fit => Ok((selected_image(state, id, named)?, Look::Fit)),
        ImageDetailTarget::Pixel(pixel) => Ok((
            selected_image(state, id, named)?,
            Look::Pixel {
                pixel: *pixel,
                zoom,
            },
        )),
        ImageDetailTarget::Rect(rect) => Ok((selected_image(state, id, named)?, Look::Rect(*rect))),
        ImageDetailTarget::Point(query) => {
            let image = selected_image(state, id, named)?;
            let point = super::resolve_point_in(state, id, query)?;
            let pixel =
                super::read::point_observation_xy(state, point, image).ok_or_else(|| {
                    ToolError::new(format!(
                        "{} has no observation of point {} -- there is nothing of it to look at in \
                     that photograph. get_point lists the camera images its track holds.",
                        state.image_name(image),
                        point.index(),
                    ))
                })?;
            Ok((image, Look::Pixel { pixel, zoom }))
        }
        ImageDetailTarget::Feature(feature) => {
            let image = selected_image(state, id, named)?;
            // Through the cache the panel's own overlay draws from, so the mark
            // an agent asked to be centred is the mark it is looking at.
            let (pixel, _) = state
                .sift_feature(image, *feature)
                .map_err(ToolError::new)?;
            Ok((
                image,
                Look::Pixel {
                    pixel: [pixel[0] as f32, pixel[1] as f32],
                    zoom,
                },
            ))
        }
        ImageDetailTarget::BenchObservation { track, observation } => {
            let (image, pixel) =
                super::bench::observation_place(state, id, track.as_deref(), *observation)?;
            // The photograph the observation names, unless the call named one
            // itself -- which is how an agent asks "where would this sighting
            // be in *that* image".
            Ok((named.unwrap_or(image), Look::Pixel { pixel, zoom }))
        }
    }
}

/// The photograph a call that named none is about: the one selected in `id`.
fn selected_image(
    state: &AppState,
    id: ReconId,
    named: Option<ImageRef>,
) -> Result<ImageRef, ToolError> {
    if let Some(image) = named {
        return Ok(image);
    }
    state
        .selected_image
        .filter(|image| image.recon == id)
        .ok_or_else(|| {
            ToolError::new(
                "No camera image is selected, so there is no photograph to look at. Name one \
                 with camera_image, or send select_camera_image first.",
            )
        })
}

/// The frame a look is computed in: the panel's size as it last drew, and
/// `image`'s own size.
///
/// The panel size can only come from the panel -- nothing else in the process
/// knows how big its body is -- so `None` is "there is no frame to do the
/// arithmetic in yet", which is what a call before the panel's first drawn
/// photograph gets, and what makes that call wait for a frame rather than be
/// answered against a guess. The image size is the lens's, which is what the
/// panel's texture is, and is known without decoding anything; where the panel
/// is already showing this photograph its own measurement is used instead,
/// since that is the number the frame will actually use.
fn geometry_for(state: &AppState, image: ImageRef) -> Result<Option<ViewGeometry>, ToolError> {
    let image_size = image_size_px(state, image).ok_or_else(|| {
        ToolError::new("That camera image is no longer in this version of the reconstruction.")
    })?;
    let Some(standing) = state.image_detail_view else {
        return Ok(None);
    };
    if standing.image == image {
        return Ok(Some(standing));
    }
    Ok(Some(ViewGeometry {
        image,
        image_size,
        ..standing
    }))
}

/// `image`'s size in its own pixels, from the lens it was shot through.
fn image_size_px(state: &AppState, image: ImageRef) -> Option<[f32; 2]> {
    let node = state.node(image.recon)?;
    let table = &node.recon().image_table;
    let camera = table
        .cameras
        .get(table.images.get(image.index())?.camera_index as usize)?;
    Some([camera.width as f32, camera.height as f32])
}

/// The document both view tools answer with.
///
/// `visible_rect_px` is **not** clipped to the photograph: at fit zoom the
/// letterboxed axis runs past both edges, and the useful invariant is that the
/// rectangle's centre is the image pixel at the centre of the panel, which is
/// what every target aims. `panel_size_points` is in egui points rather than
/// pixels, which is what the panel lays out in; `screenshot` is where physical
/// pixels are.
fn view_document(state: &AppState, view: Option<ViewGeometry>) -> Value {
    let Some(view) = view else {
        return json!({
            "reconstruction_label": Value::Null,
            "camera_image": Value::Null,
            "camera_image_name": Value::Null,
            "zoom": Value::Null,
            "visible_rect_px": Value::Null,
            "panel_size_points": Value::Null,
            "image_size_px": Value::Null,
        });
    };
    json!({
        "reconstruction_label": super::render::label_of(state, view.image.recon),
        "camera_image": view.image.index(),
        "camera_image_name": state.image_name(view.image),
        "zoom": f64::from(view.zoom),
        "visible_rect_px": view.visible_rect().map(f64::from),
        "panel_size_points": view.panel_size.map(f64::from),
        "image_size_px": view.image_size.map(f64::from),
    })
}

/// Everything `set_image_detail_view` takes, and the whole of what it refuses
/// before a `Command` exists.
///
/// The one-target rule is here rather than in [`set_view`] for the reason the
/// display parse holds its vocabularies: a call that named two places has asked
/// two questions, and answering half of it would be worse than turning it away.
pub(super) fn parse_view(args: &Args) -> Result<ImageDetailViewRequest, ToolError> {
    let target = parse_target(args)?;
    let zoom: Option<f32> = match args.optional_f64("zoom")? {
        None => None,
        Some(zoom) => {
            if !matches!(
                target,
                ImageDetailTarget::Pixel(_)
                    | ImageDetailTarget::Point(_)
                    | ImageDetailTarget::Feature(_)
                    | ImageDetailTarget::BenchObservation { .. }
            ) {
                return Err(args.error(
                    "was given a zoom with a target that settles its own: rect fits what it was \
                     given, and fit is the whole photograph.",
                ));
            }
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(args.error(format!(
                    "wants zoom to be a magnification above zero, 1.0 being the fit -- got {zoom}."
                )));
            }
            Some(zoom as f32)
        }
    };
    if args.get("track").is_some() && !matches!(target, ImageDetailTarget::BenchObservation { .. })
    {
        return Err(args.error("names a track without a bench_observation to find in it."));
    }
    let camera_image = match args.get("camera_image") {
        None | Some(Value::Null) => None,
        Some(_) => Some(args.camera_image("camera_image")?),
    };
    Ok(ImageDetailViewRequest {
        reconstruction_label: args.optional_string("reconstruction_label")?,
        camera_image,
        target,
        zoom,
    })
}

/// The one target a call named, or the refusal for none and for more than one.
fn parse_target(args: &Args) -> Result<ImageDetailTarget, ToolError> {
    let mut found: Vec<(&str, ImageDetailTarget)> = Vec::new();
    if args.get("pixel").is_some() {
        found.push(("pixel", ImageDetailTarget::Pixel(args.pixel("pixel")?)));
    }
    if args.get("rect").is_some() {
        found.push(("rect", ImageDetailTarget::Rect(parse_rect(args)?)));
    }
    if args.get("point").is_some() {
        found.push(("point", ImageDetailTarget::Point(args.point("point")?)));
    }
    if args.get("feature").is_some() {
        let feature = args.required_usize("feature")?;
        let feature = u32::try_from(feature)
            .map_err(|_| args.error(format!("was given a feature index of {feature}.")))?;
        found.push(("feature", ImageDetailTarget::Feature(feature)));
    }
    if args.get("bench_observation").is_some() {
        found.push((
            "bench_observation",
            ImageDetailTarget::BenchObservation {
                track: args.optional_string("track")?,
                observation: args.required_usize("bench_observation")?,
            },
        ));
    }
    // `fit: false` names no target: it is the absence of the request, not a
    // request for something else.
    if args.optional_bool("fit")? == Some(true) {
        found.push(("fit", ImageDetailTarget::Fit));
    }
    match found.len() {
        1 => Ok(found.pop().expect("one").1),
        0 => Err(args.error(
            "was given no place to look -- pass one of pixel, rect, point, feature, \
             bench_observation or fit.",
        )),
        _ => {
            let names: Vec<&str> = found.iter().map(|(name, _)| *name).collect();
            Err(args.error(format!(
                "was given {} places to look at once ({}) -- a call names one.",
                names.len(),
                names.join(", ")
            )))
        }
    }
}

/// A `rect` argument: four finite numbers in image pixels, spanning something.
///
/// A rectangle of no area is refused rather than fitted: it names a line or a
/// point, and the magnification that "fills the panel" with one is unbounded.
/// `pixel` with a `zoom` is what a caller asking for that means.
fn parse_rect(args: &Args) -> Result<[f32; 4], ToolError> {
    let [x0, y0, x1, y1] = args
        .optional_numbers::<4>("rect")?
        .ok_or_else(|| args.error("needs rect: [x0, y0, x1, y1] in image pixels."))?;
    if (x1 - x0).abs() <= 0.0 || (y1 - y0).abs() <= 0.0 {
        return Err(args.error(format!(
            "was given a rect of no area ({x0}, {y0})-({x1}, {y1}), which frames nothing."
        )));
    }
    Ok([x0 as f32, y0 as f32, x1 as f32, y1 as f32])
}

// ── Detailed timing ─────────────────────────────────────────────────────

/// `get_timing_detail`: the level the next operation will be recorded at, no
/// arguments.
pub(super) fn get_timing_detail(state: &AppState) -> JsonReply {
    Ok(timing_detail(state))
}

/// `set_timing_detail`: raise or lower the level for the operations to come,
/// and answer with it.
///
/// Through `ActionLog::set_detailed_timing` rather than by assigning the
/// field, because that call is where the checkbox's own rule lives: one
/// `Display` entry when the value changes and none when it is handed the value
/// it already has. An agent raising the level therefore leaves the row a human
/// raising it leaves, which is how the human at the window finds out.
///
/// It takes effect on the next operation. Nothing already recorded is
/// re-timed, and an entry keeps the detail it was recorded with.
pub(super) fn set_timing_detail(state: &mut AppState, enabled: bool) -> JsonReply {
    state.action_log.set_detailed_timing(enabled);
    Ok(timing_detail(state))
}

/// The document both tools answer with, read back off the log rather than
/// echoed from the call.
fn timing_detail(state: &AppState) -> Value {
    json!({
        "timing_detail": {
            "enabled": state.action_log.detailed_timing(),
        }
    })
}

// ── Schema fragments ────────────────────────────────────────────────────
//
// Here rather than in `tools`, beside the parse that has to accept every one
// of them: the two are one statement, and the catalog walk in `mcp::tests`
// checks that they stay one.

/// The `intrinsics` sub-block of `set_image_detail_display`.
pub(super) fn intrinsics_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "description":
            "The intrinsics overlay layer, which composes with whichever feature overlay_mode is \
             active — including \"none\", for the camera model alone. Every key is optional.",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description":
                    "Draw the layer at all: the principal point always, and whatever of axes, \
                     rings and distortion is on. On by default.",
            },
            "axes": {
                "type": "boolean",
                "description": "The angular axes through the principal point, with their ticks.",
            },
            "rings": {
                "type": "boolean",
                "description": "Iso-angle rings, at the same angular ladder as the axis ticks.",
            },
            "distortion": {
                "type": "boolean",
                "description":
                    "The displacement field. Ignored where the camera model has no distortion.",
            },
            "distortion_scale": {
                "type": ["number", "null"],
                "enum": scale_ladder_with_null(),
                "description":
                    "How far the displacement arrows are exaggerated, from the ladder the gear \
                     popup offers, or null to fit the scale to the lens automatically.",
            },
            "grid_cols": {
                "type": "integer",
                "enum": IntrinsicsDisplaySettings::GRID_LADDER,
                "description": "Arrows across the image width, from the popup's own ladder.",
            },
        },
    })
}

/// The `feature_size_px` argument: both bounds, or `null` for no filter.
pub(super) fn feature_size_schema() -> Value {
    json!({
        "type": ["object", "null"],
        "additionalProperties": false,
        "required": ["min", "max"],
        "description":
            "Show only features whose size falls between these two, in pixels, or null to turn \
             the filter off. Both bounds together, because the toolbar's one checkbox derives \
             both from its two persisted values every frame.",
        "properties": {
            "min": { "type": "number", "minimum": 0 },
            "max": { "type": "number", "minimum": 0 },
        },
    })
}

/// The ladder plus `null`, as `distortion_scale`'s enum spells it.
fn scale_ladder_with_null() -> Vec<Value> {
    IntrinsicsDisplaySettings::SCALE_LADDER
        .iter()
        .map(|scale| json!(f64::from(*scale)))
        .chain(std::iter::once(Value::Null))
        .collect()
}
