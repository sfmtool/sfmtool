// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An opt-in [Model Context Protocol][mcp] control surface for the running
//! viewer, so an agent can drive the window the human is looking at.
//!
//! See `specs/gui/mcp-server.md`. Started with `sfm-explorer --mcp`, the viewer
//! hosts a small HTTP server on loopback; a connected agent can enumerate the
//! scene graph, open and close `.sfmr` files, move the selection and the 3D
//! camera, and take a screenshot of the viewport.
//!
//! ## The shape of this module, and why
//!
//! Application state stays single-threaded on the GUI thread. `AppState`,
//! [`Viewer3D`] and the renderer are threaded through the frame as `&mut`, and
//! the panels rely on that — [`crate::dock::TabContext`] hands out seven
//! simultaneous `&mut` borrows, and the SIFT and full-res caches are
//! split-borrowed against the scene on purpose. So the server never touches app
//! state: it builds a [`Command`], hands it to the GUI thread over a channel,
//! wakes the event loop, and waits for the answer.
//!
//! That gives the module three layers, in dependency order:
//!
//! - [`tools`] — the tool table and the wire parse. Names, descriptions,
//!   `inputSchema`, and JSON arguments to [`Command`].
//! - [`apply_with_window`] and [`render`] — the whole command vocabulary, applied to
//!   `(&mut AppState, &mut Viewer3D)` and a [`window::WindowHost`]. **No
//!   `App`, no GPU handle**, which is what keeps twenty-one of the twenty-two
//!   tools under headless test.
//! - [`server`] — the `rmcp` handler and the `axum`/`tokio` plumbing that
//!   carries a [`Request`] to the GUI thread and its [`Reply`] back.
//!
//! [mcp]: https://modelcontextprotocol.io/

use std::path::PathBuf;

use serde_json::{json, Value};

use crate::action_log::Kind;
use crate::scene::{CameraRef, ImageRef, PointRef, ReconId};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

mod frame;
mod layout;
mod read;
pub(crate) mod server;
pub(crate) mod tools;
mod view;
pub(crate) mod window;
mod write;

pub(crate) mod render;

#[cfg(test)]
mod tests;

pub(crate) use server::serve;

/// Everything the MCP surface can ask the viewer to do. One variant per tool.
///
/// A reconstruction is named by its **label**, so these carry a `String` that
/// [`apply_with_window`] resolves against `AppState::scene`. `Option<String>` means "the
/// selected reconstruction if omitted". The `ReconId` never crosses the wire:
/// a label is unique across the scene and survives `Reload from Disk`, which
/// mints a fresh id (see "Addressing" in `specs/gui/mcp-server.md`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Command {
    GetScene,
    ListCameraImages {
        reconstruction_label: Option<String>,
        offset: usize,
        limit: usize,
    },
    GetCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    GetCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    GetPoint {
        point: crate::goto_point::PointQuery,
    },
    OpenReconstruction {
        path: PathBuf,
    },
    CloseReconstruction {
        target: CloseTarget,
    },
    SelectReconstruction {
        reconstruction_label: String,
    },
    SelectCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    SelectCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    SelectPoint {
        point: crate::goto_point::PointQuery,
    },
    ClearSelection {
        scope: SelectionScope,
    },
    SetReconstructionDisplay {
        reconstruction_label: String,
        change: DisplayChange,
    },
    SetSolo {
        reconstruction_label: Option<String>,
    },
    SetView {
        view: ViewCommand,
    },
    GetLayout,
    SetLayout {
        layout: layout::LayoutTarget,
    },
    ShowPanel {
        panel: crate::dock::Tab,
    },
    HidePanel {
        panel: crate::dock::Tab,
    },
    GetWindow,
    SetWindow {
        change: window::WindowChange,
    },
    Screenshot {
        max_dimension: Option<u32>,
    },
}

/// How a tool named a camera image: by its index in the reconstruction, or by
/// the `.sfmr` relative path that is its name.
///
/// Both, because the surface hands out both — a track observation reports an
/// index, a `list_camera_images` row reports both — and an agent arrives
/// holding whichever it last read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CameraImageSel {
    Index(usize),
    Name(String),
}

/// What `close_reconstruction` was asked to close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CloseTarget {
    One(String),
    All,
}

/// How much of the selection `clear_selection` drops.
///
/// Follows the viewer's own rule that dismissing a photograph says nothing
/// about the lens: [`SelectionScope::CameraImage`] leaves the intrinsics
/// selected, exactly as `AppState::select_image(None)` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectionScope {
    All,
    CameraImage,
    CameraIntrinsics,
    Point,
}

/// A `set_reconstruction_display` request: every field is one of `SceneNode`'s
/// own, and every `None` is left alone.
///
/// `tint` is doubly optional on purpose. The outer `None` is "the call did not
/// mention the tint"; the inner `None` is "clear it back to the node's own
/// colors", which is what a JSON `null` asks for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DisplayChange {
    pub(crate) visible: Option<bool>,
    pub(crate) interactive: Option<bool>,
    pub(crate) show_points: Option<bool>,
    pub(crate) show_camera_images: Option<bool>,
    pub(crate) show_patches: Option<bool>,
    pub(crate) show_points_at_infinity: Option<bool>,
    pub(crate) tint: Option<Option<String>>,
}

/// The five things `set_view` can be asked for.
///
/// One enum rather than a bag of optional fields, because these are *intents*
/// and not representations: "frame the scene" and "put the camera exactly
/// here" are different questions, and a call that carried both would have no
/// answer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ViewCommand {
    /// Frame everything drawn, or one named reconstruction.
    Fit {
        reconstruction_label: Option<String>,
    },
    /// Look through a camera image, as double-click and `Z` do.
    LookThrough {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    /// Leave camera-view mode, keeping the camera where it is.
    ExitCameraView,
    /// The explicit camera, in whatever pieces the call carried.
    Place(Placement),
    /// The field of view alone.
    Fov { fov_short_axis_deg: f64 },
}

/// The explicit camera -- a position, an orientation and a target distance --
/// as the pieces one call carried.
///
/// Every field is optional and every absent one is preserved from the standing
/// view, which is what makes the look-at form (`position` with `target`), the
/// exact form (`orientation_wxyz` with its companions) and a single piece on
/// its own the same command with different fields filled in, rather than three
/// commands that would each need their own placement arithmetic.
///
/// Which combinations determine a camera at all is settled in
/// [`tools::parse`], which sees what the call named; what a combination means
/// is settled in `view::set_view`, which sees the view it is changing.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Placement {
    pub(crate) position: Option<[f64; 3]>,
    pub(crate) target: Option<[f64; 3]>,
    pub(crate) forward: Option<[f64; 3]>,
    pub(crate) orientation_wxyz: Option<[f64; 4]>,
    pub(crate) target_distance: Option<f64>,
    pub(crate) up: Option<[f64; 3]>,
    pub(crate) world_up: Option<[f64; 3]>,
    pub(crate) fov_short_axis_deg: Option<f64>,
}

/// A tool's refusal: everything the viewer can say no to, in the style its own
/// status line uses — what was asked, and what is actually there.
///
/// Distinct from a protocol error. This becomes a `CallToolResult` with
/// `isError: true`, which tells the client the request was well-formed and the
/// *viewer* declined; a malformed request never reaches [`apply_with_window`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolError(pub(crate) String);

impl ToolError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a tool produced.
///
/// Two shapes rather than one, because `screenshot` answers with a picture and
/// the other twenty-one answer with JSON, and squeezing an image through a JSON
/// field would mean a magic key that the transport has to know to look for.
pub(crate) enum ToolOutput {
    Json(Value),
    Png {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        /// One line describing what was in frame, for the text block that
        /// accompanies the image.
        caption: String,
    },
}

/// A tool's answer: what it produced, or a message for `isError: true`.
pub(crate) type Reply = Result<ToolOutput, ToolError>;

/// The answer of the twenty-one tools that speak only JSON.
///
/// Widened to a [`Reply`] at the [`apply_with_window`] dispatch, so nothing below it has to
/// name the shape it is not.
pub(super) type JsonReply = Result<Value, ToolError>;

/// Whether [`apply_with_window`] finished the job, or needs the frame to complete first.
pub(crate) enum Outcome {
    Done(Reply),
    Deferred(Deferred),
}

/// A command whose answer cannot exist until this frame has been rendered and
/// presented.
///
/// Exactly one tool is in here. `App` holds these until the readback phase and
/// answers them there, where the `wgpu::Device` already is — which is what
/// keeps [`apply_with_window`] free of a GPU handle.
pub(crate) enum Deferred {
    Screenshot {
        max_dimension: Option<u32>,
        /// The one-line description of what is in frame, built while the state
        /// was still borrowed. Held here so the readback phase does not have to
        /// reach back into `AppState` to describe a picture it already took.
        caption: String,
    },
}

/// One tool call in flight: what to do, and where the answer goes.
pub(crate) struct Request {
    pub(crate) command: Command,
    pub(crate) reply: tokio::sync::oneshot::Sender<Reply>,
}

/// Apply one command to the viewer, with no window behind it.
///
/// [`apply_with_window`] against a `window::NoWindow` host, which is the
/// windowless case: `get_window` and `set_window` refuse with "no window", and
/// every other tool behaves exactly as it does in the viewer. Kept as its own
/// function for the callers that have no window to offer and should not have to
/// invent one — which today is the tests, the real frame always having a window
/// by the time it drains, so this is compiled for them.
#[cfg(test)]
pub(crate) fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome {
    apply_with_window(state, viewer, &mut window::NoWindow, command)
}

/// Apply one command to the viewer.
///
/// Takes no `App` and no GPU handle, which is what makes twenty-one of the
/// twenty-two tools testable in a headless `cargo test`: `App` owns a
/// `wgpu::Device`, a surface and a window, and constructing one needs a GPU and
/// a display that this crate's lib tests deliberately do without. The one
/// GPU-shaped command leaves through [`Outcome::Deferred`] instead, and the one
/// window-shaped command through [`window::WindowHost`], which a fake can stand
/// in for.
pub(crate) fn apply_with_window(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn window::WindowHost,
    command: Command,
) -> Outcome {
    match command {
        Command::GetScene => done(Ok(render::scene(state, viewer))),
        Command::ListCameraImages {
            reconstruction_label,
            offset,
            limit,
        } => done(read::list_camera_images(
            state,
            reconstruction_label.as_deref(),
            offset,
            limit,
        )),
        Command::GetCameraImage {
            reconstruction_label,
            camera_image,
        } => done(read::get_camera_image(
            state,
            reconstruction_label.as_deref(),
            &camera_image,
        )),
        Command::GetCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => done(read::get_camera_intrinsics(
            state,
            reconstruction_label.as_deref(),
            camera_intrinsics_index,
        )),
        Command::GetPoint { point } => done(read::get_point(state, &point)),
        Command::OpenReconstruction { path } => done(write::open_reconstruction(state, &path)),
        Command::CloseReconstruction { target } => done(write::close_reconstruction(state, target)),
        Command::SelectReconstruction {
            reconstruction_label,
        } => done(write::select_reconstruction(state, &reconstruction_label)),
        Command::SelectCameraImage {
            reconstruction_label,
            camera_image,
        } => done(write::select_camera_image(
            state,
            reconstruction_label.as_deref(),
            &camera_image,
        )),
        Command::SelectCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => done(write::select_camera_intrinsics(
            state,
            reconstruction_label.as_deref(),
            camera_intrinsics_index,
        )),
        Command::SelectPoint { point } => done(write::select_point(state, &point)),
        Command::ClearSelection { scope } => done(write::clear_selection(state, scope)),
        Command::SetReconstructionDisplay {
            reconstruction_label,
            change,
        } => done(write::set_reconstruction_display(
            state,
            &reconstruction_label,
            change,
        )),
        Command::SetSolo {
            reconstruction_label,
        } => done(write::set_solo(state, reconstruction_label.as_deref())),
        Command::SetView { view } => done(view::set_view(state, viewer, view)),
        Command::GetLayout => done(layout::get_layout(state)),
        Command::SetLayout { layout } => done(layout::set_layout(state, &layout)),
        Command::ShowPanel { panel } => done(layout::show_panel(state, panel)),
        Command::HidePanel { panel } => done(layout::hide_panel(state, panel)),
        Command::GetWindow => done(window::get_window(state, host)),
        Command::SetWindow { change } => done(window::set_window(state, host, &change)),
        Command::Screenshot { max_dimension } => {
            // A minimized window renders nothing to photograph, and whether its
            // swapchain still presents at all is platform-dependent — so this
            // is refused rather than attempted, naming the call that fixes it.
            // Checked here, against the window snapshot, so it is under
            // headless test with the rest of the vocabulary.
            if state.window.as_ref().map(|info| info.state) == Some(window::WindowState::Minimized)
            {
                return done(Err(ToolError::new(MINIMIZED)));
            }
            Outcome::Deferred(Deferred::Screenshot {
                max_dimension,
                caption: screenshot_caption(state, viewer),
            })
        }
    }
}

/// Why a `screenshot` of a minimized window is refused.
///
/// A picture of a window the human cannot see answers nothing an agent asked
/// of a shared viewer, so the refusal names the call that makes one possible.
pub(super) const MINIMIZED: &str =
    "The window is minimized, so nothing is being rendered to photograph. Send \
     set_window { \"state\": \"normal\" } first.";

fn done(reply: JsonReply) -> Outcome {
    Outcome::Done(reply.map(ToolOutput::Json))
}

/// The sentence that rides along with a screenshot: what was in frame when it
/// was taken.
///
/// Built here, while `AppState` is still borrowed, rather than at readback:
/// the picture and the description of it should be of the same instant, and
/// the readback phase runs after the UI has had a whole frame to change things.
fn screenshot_caption(state: &AppState, viewer: &Viewer3D) -> String {
    let stats = crate::scene::visible_stats(&state.scene, state.solo);
    let drawn: Vec<&str> = state
        .scene
        .iter()
        .filter(|node| crate::scene::is_visible(node, state.solo))
        .map(|node| node.label.as_str())
        .collect();
    let looking_through = viewer
        .camera_view
        .as_ref()
        .and_then(|camera_view| {
            state
                .node(camera_view.image.recon)
                .and_then(|node| node.recon.images.get(camera_view.image.index()))
        })
        .map(|image| format!(", looking through {}", image.name))
        .unwrap_or_default();
    format!(
        "The 3D viewport: {} drawn ({}), {} points, {} camera images{}.",
        drawn.len(),
        if drawn.is_empty() {
            "nothing loaded".to_string()
        } else {
            drawn.join(", ")
        },
        stats.points,
        stats.images,
        looking_through,
    )
}

// ── Resolution: a wire handle to the thing it names ──────────────────────

/// The reconstruction a tool named, or the selected one when it named none.
///
/// The error names what *is* loaded, because the two ways to get here — a
/// typo, and a reconstruction the human closed in between — both want the same
/// list to recover from.
pub(super) fn resolve_reconstruction(
    state: &AppState,
    label: Option<&str>,
) -> Result<ReconId, ToolError> {
    let Some(label) = label else {
        return state.selected_recon.ok_or_else(|| {
            ToolError::new(
                "No reconstruction is selected — name one with reconstruction_label, or open \
                 a file first.",
            )
        });
    };
    state
        .scene
        .iter()
        .find(|node| node.label == label)
        .map(|node| node.id)
        .ok_or_else(|| {
            ToolError::new(format!(
                "No loaded reconstruction is labelled {label:?}{}",
                loaded_list(state)
            ))
        })
}

/// The camera image a tool named, as an index into `reconstruction`.
pub(super) fn resolve_camera_image(
    state: &AppState,
    reconstruction: ReconId,
    selector: &CameraImageSel,
) -> Result<ImageRef, ToolError> {
    let node = state
        .node(reconstruction)
        .ok_or_else(|| ToolError::new("The reconstruction is no longer loaded."))?;
    let index = match selector {
        CameraImageSel::Index(index) => {
            if *index >= node.recon.images.len() {
                return Err(ToolError::new(format!(
                    "{} has {} camera images — index {index} is out of range.",
                    node.label,
                    node.recon.images.len()
                )));
            }
            *index
        }
        CameraImageSel::Name(name) => node
            .recon
            .images
            .iter()
            .position(|image| image.name == *name)
            .ok_or_else(|| {
                ToolError::new(format!(
                    "{} has no camera image named {name:?} — names are .sfmr relative paths, as \
                     in \"images/IMG_0042.jpg\"; list_camera_images reports them.",
                    node.label
                ))
            })?,
    };
    Ok(ImageRef::new(reconstruction, index))
}

/// The camera intrinsics record a tool named, as an index into
/// `reconstruction`.
pub(super) fn resolve_camera_intrinsics(
    state: &AppState,
    reconstruction: ReconId,
    index: usize,
) -> Result<CameraRef, ToolError> {
    let node = state
        .node(reconstruction)
        .ok_or_else(|| ToolError::new("The reconstruction is no longer loaded."))?;
    if index >= node.recon.cameras.len() {
        return Err(ToolError::new(format!(
            "{} has {} camera intrinsics records — index {index} is out of range.",
            node.label,
            node.recon.cameras.len()
        )));
    }
    Ok(CameraRef::new(reconstruction, index))
}

/// The 3D point a tool named.
///
/// Goes through the same parse and the same lookup the Go to Point dialog uses
/// ([`crate::goto_point`]), so a point id a human copied out of the Point Track
/// panel pastes straight into a tool call, and the two paths cannot disagree
/// about what an id means.
pub(super) fn resolve_point(
    state: &AppState,
    query: &crate::goto_point::PointQuery,
) -> Result<PointRef, ToolError> {
    crate::goto_point::resolve_point_query(&state.scene, state.selected_recon, query)
        .map_err(ToolError)
}

/// `" — loaded: a, b."`, or a note that nothing is, to hang off a
/// "no such reconstruction" message.
fn loaded_list(state: &AppState) -> String {
    if state.scene.is_empty() {
        return " — nothing is loaded.".to_string();
    }
    let labels: Vec<String> = state
        .scene
        .iter()
        .map(|node| format!("{:?}", node.label))
        .collect();
    format!(" — loaded: {}.", labels.join(", "))
}

/// Apply a frame's worth of commands **as the agent**, recording what they did.
///
/// The drain's application phase, with the channel left out so it is reachable
/// from a headless test. Three things happen here that [`apply_with_window`] cannot do for
/// itself:
///
/// - the Action Log's ambient actor is moved to
///   [`Mcp`](crate::action_log::Actor::Mcp) for the whole batch and restored
///   afterwards — one move per frame rather than an `Actor` argument on every
///   `AppState` method;
/// - a **read** is recorded, because it changes no state and so has no state
///   method to log through — and it is recorded from the command *before* the
///   command is applied, so a deferred `screenshot` lines up in order with the
///   commands around it rather than at readback;
/// - a **refusal** is recorded, in the same words the agent receives, because
///   the methods below return their failures rather than logging them.
///
/// A mutating tool that succeeds writes nothing here: the `AppState` and
/// `Viewer3D` methods it called already did, in the same words the GUI's own
/// path produces.
pub(crate) fn apply_as_agent(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn window::WindowHost,
    commands: Vec<Command>,
) -> Vec<Outcome> {
    debug_assert_eq!(
        state.action_log.actor(),
        crate::action_log::Actor::User,
        "the agent's commands were applied with the actor already moved",
    );
    state.action_log.set_actor(crate::action_log::Actor::Mcp);
    let outcomes = commands
        .into_iter()
        .map(|command| {
            log::debug!("MCP: applying {command:?}");
            let tool = command.tool_name();
            let kind = command.kind();
            let query = query_text(state, viewer, &command);
            let outcome = apply_with_window(state, viewer, host, command);
            match (&outcome, query) {
                (Outcome::Done(Err(error)), _) => state
                    .action_log
                    .fail(kind, format!("{tool} failed: {error}")),
                (_, Some(text)) => state.action_log.query(tool, text),
                (_, None) => {}
            }
            outcome
        })
        .collect();
    state.action_log.set_actor(crate::action_log::Actor::User);
    outcomes
}

// ── The Action Log's view of a command ───────────────────────────────────
//
// A mutating tool writes no entry of its own: it calls the same `AppState` and
// `Viewer3D` methods the GUI calls, and those record — which is what makes the
// actor column trustworthy, since two rows reading the same did the same thing.
// What is left for the drain is the two things no state method can know: the
// name of the tool, for a refusal, and the fact that a read happened at all.

impl Command {
    /// The tool this command came from, as the wire names it.
    pub(crate) fn tool_name(&self) -> &'static str {
        match self {
            Command::GetScene => "get_scene",
            Command::ListCameraImages { .. } => "list_camera_images",
            Command::GetCameraImage { .. } => "get_camera_image",
            Command::GetCameraIntrinsics { .. } => "get_camera_intrinsics",
            Command::GetPoint { .. } => "get_point",
            Command::OpenReconstruction { .. } => "open_reconstruction",
            Command::CloseReconstruction { .. } => "close_reconstruction",
            Command::SelectReconstruction { .. } => "select_reconstruction",
            Command::SelectCameraImage { .. } => "select_camera_image",
            Command::SelectCameraIntrinsics { .. } => "select_camera_intrinsics",
            Command::SelectPoint { .. } => "select_point",
            Command::ClearSelection { .. } => "clear_selection",
            Command::SetReconstructionDisplay { .. } => "set_reconstruction_display",
            Command::SetSolo { .. } => "set_solo",
            Command::SetView { .. } => "set_view",
            Command::GetLayout => "get_layout",
            Command::SetLayout { .. } => "set_layout",
            Command::ShowPanel { .. } => "show_panel",
            Command::HidePanel { .. } => "hide_panel",
            Command::GetWindow => "get_window",
            Command::SetWindow { .. } => "set_window",
            Command::Screenshot { .. } => "screenshot",
        }
    }

    /// The Action Log kind a refusal of this command is filed under.
    ///
    /// A failed entry never coalesces, so this is about where the row belongs
    /// rather than about folding — but it should still be the kind the tool's
    /// success would have been.
    pub(crate) fn kind(&self) -> Kind {
        match self {
            Command::GetScene
            | Command::ListCameraImages { .. }
            | Command::GetCameraImage { .. }
            | Command::GetCameraIntrinsics { .. }
            | Command::GetPoint { .. }
            | Command::GetLayout
            | Command::GetWindow
            | Command::Screenshot { .. } => Kind::Query(self.tool_name()),
            Command::OpenReconstruction { .. } | Command::CloseReconstruction { .. } => Kind::File,
            Command::SelectReconstruction { .. }
            | Command::SelectCameraImage { .. }
            | Command::SelectCameraIntrinsics { .. }
            | Command::SelectPoint { .. }
            | Command::ClearSelection { .. } => Kind::Selection,
            Command::SetReconstructionDisplay { .. } | Command::SetSolo { .. } => Kind::Scene,
            Command::SetView { .. } => Kind::View,
            Command::SetLayout { .. } | Command::ShowPanel { .. } | Command::HidePanel { .. } => {
                Kind::Layout
            }
            Command::SetWindow { .. } => Kind::Window,
        }
    }
}

/// The Action Log text for a **read-only** tool, or `None` for a mutating one.
///
/// Written by the drain from the command rather than by the tool, because a
/// read changes no state and so has no state method to log through. Built
/// *before* the command is applied, so a deferred `screenshot` lines up in
/// order with the commands around it rather than at readback.
pub(crate) fn query_text(state: &AppState, viewer: &Viewer3D, command: &Command) -> Option<String> {
    // A call that named no reconstruction read the selected one, and the log
    // should say which that was rather than leaving the row ambiguous.
    let named = |label: &Option<String>| match label {
        Some(label) => label.clone(),
        None => state
            .selected_recon
            .and_then(|id| render::label_of(state, id))
            .unwrap_or_else(|| "(none selected)".to_string()),
    };
    Some(match command {
        Command::GetScene => "get_scene".to_string(),
        Command::ListCameraImages {
            reconstruction_label,
            offset,
            limit,
        } => format!(
            "list_camera_images {} {offset}..{}",
            named(reconstruction_label),
            offset + limit
        ),
        Command::GetCameraImage {
            reconstruction_label,
            camera_image,
        } => format!(
            "get_camera_image {} {}",
            named(reconstruction_label),
            camera_image_text(camera_image)
        ),
        Command::GetCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => format!(
            "get_camera_intrinsics {} #{camera_intrinsics_index}",
            named(reconstruction_label)
        ),
        Command::GetPoint { point } => format!("get_point {}", point_text(point)),
        Command::GetLayout => "get_layout".to_string(),
        Command::GetWindow => "get_window".to_string(),
        Command::Screenshot { max_dimension } => {
            let [width, height] = screenshot_size(viewer, *max_dimension);
            format!("screenshot {width}×{height}")
        }
        _ => return None,
    })
}

/// `images/IMG_0007.jpg`, or `#7` where the call named an index.
fn camera_image_text(selector: &CameraImageSel) -> String {
    match selector {
        CameraImageSel::Index(index) => format!("#{index}"),
        CameraImageSel::Name(name) => name.clone(),
    }
}

/// A point query as the user would have typed it.
fn point_text(query: &crate::goto_point::PointQuery) -> String {
    match query {
        crate::goto_point::PointQuery::Index(index) => format!("#{index}"),
        crate::goto_point::PointQuery::Qualified { hash, index } => format!("pt3d_{hash}_{index}"),
    }
}

/// The size a screenshot taken this frame would come back at.
///
/// The viewport as last laid out, shrunk by `max_dimension` exactly as
/// `App::screenshot` shrinks the pixels — so the log line and the image agree.
fn screenshot_size(viewer: &Viewer3D, max_dimension: Option<u32>) -> [u32; 2] {
    let [width, height] = viewer.panel_size;
    let Some(limit) = max_dimension else {
        return [width, height];
    };
    let longest = width.max(height);
    if longest <= limit || longest == 0 {
        return [width, height];
    }
    let scale = f64::from(limit) / f64::from(longest);
    [
        ((f64::from(width) * scale).round() as u32).max(1),
        ((f64::from(height) * scale).round() as u32).max(1),
    ]
}

/// The reply every selection tool returns: the resulting `selection` block.
///
/// All six return it, and they return it *after* the fact, so the agent sees
/// what the coupling rules in `AppState` did to its request rather than
/// assuming it got what it asked for.
pub(super) fn selection_reply(state: &AppState) -> JsonReply {
    Ok(json!({ "selection": render::selection(state) }))
}
