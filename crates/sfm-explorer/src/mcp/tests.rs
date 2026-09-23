// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the command vocabulary.
//!
//! No GPU and no window, which is what the `apply(&mut AppState, &mut Viewer3D,
//! …)` signature is for: every tool but `screenshot` is exercised here against a
//! two-reconstruction scene, and `screenshot` — the one that needs a real frame
//! — belongs in `ui_basic`.
//!
//! Two things these tests are really about, beyond the JSON shapes:
//!
//! - **MCP cannot route around `AppState`'s invariants.** Selecting a camera
//!   image sets its intrinsics; selecting an intrinsics record the selected
//!   image does not use clears that image; selecting a reconstruction drops
//!   another's point. Those are the guarantees that stop two panels showing two
//!   different files' selections, and a second door into the state that skipped
//!   them would be a bug the panels could not defend against.
//! - **A schema and its parser cannot drift.** Every advertised argument name
//!   is one [`tools::parse`] accepts, checked over the whole catalog rather than
//!   tool by tool, so a tool added later is covered by construction.

use serde_json::{json, Map, Value};
use sfmtool_core::progress::Level;
use sfmtool_core::SfmrReconstruction;

use super::tools::{self, ToolKind};
use super::{apply, apply_as_agent, Command, Outcome, ToolError, ToolOutput};
use crate::action_log::{ActionLog, Actor, Kind};
use crate::dock::Tab;
use crate::layout::WindowLayout;
use crate::progress::{Collector, Detail};
use crate::scene::{PointRef, SceneNode};
use crate::state::AppState;
use crate::test_support::{FakeWindow, NoWindow};
use crate::viewer_3d::Viewer3D;
use crate::window::{WindowHost, WindowState};

// ── Fixtures ────────────────────────────────────────────────────────────

/// A demo reconstruction grown to `images` images named `<prefix>_<i>.jpg`,
/// resolving to **two** intrinsics records: the first half through camera 0,
/// the rest through camera 1.
///
/// Two cameras because a one-camera reconstruction cannot tell the camera-image
/// and camera-intrinsics selections apart — every image uses the only lens, so
/// every coupling rule looks like a no-op.
///
/// Grown and never shortened: `SfmrReconstruction::demo` fixes the camera ring
/// at [`DEMO_IMAGES`] and its tracks observe all of them, so dropping an image
/// would leave observations pointing past the end of the image list.
fn recon(images: usize, prefix: &str) -> SfmrReconstruction {
    assert!(images >= DEMO_IMAGES, "the demo tracks observe every image");
    let mut recon = SfmrReconstruction::demo(64);
    let template = recon.image_table.images[0].clone();
    while recon.image_table.images.len() < images {
        recon.image_table.images.push(template.clone());
    }
    let second_camera = recon.image_table.cameras[0].clone();
    recon.image_table.cameras.push(second_camera);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("images/{prefix}_{i:03}.jpg");
        image.camera_index = if i < images / 2 { 0 } else { 1 };
    }
    recon.metadata.camera_count = recon.image_table.cameras.len() as u32;
    // Resize the per-image derived tables to match the grown image list.
    recon.rebuild_derived_fields();
    recon
}

/// How many images `SfmrReconstruction::demo` builds its camera ring from.
const DEMO_IMAGES: usize = 8;

/// A scene holding two file-backed reconstructions, `alpha` and `beta`, with
/// `alpha` selected — the state most of these tests start from.
fn two_reconstructions() -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/alpha.sfmr"),
        recon(8, "A"),
    ));
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/beta.sfmr"),
        recon(10, "B"),
    ));
    let alpha = state.scene[0].id;
    state.select_recon(alpha);
    // A laid-out viewport, so the view tools have an aspect ratio to work
    // against and the view block can report the two fixed-axis FOVs.
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    // The snapshot the frame would have taken at the top of this frame, so the
    // window block and the minimized check have a window to read even where
    // the test hands no host to change one.
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// A `screenshot` command spelled out, since three of its fields are optional
/// and most tests care about one of them.
fn screenshot(panel: Option<Tab>, hud: bool, max_dimension: Option<u32>) -> Command {
    Command::Screenshot {
        panel,
        hud,
        max_dimension,
    }
}

/// The `Deferred::Screenshot` a command produced, or a panic saying it did not
/// defer.
#[track_caller]
fn deferred_screenshot(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    command: Command,
) -> (super::ScreenshotSource, String) {
    match agent(state, viewer, command) {
        Outcome::Deferred(super::Deferred::Screenshot {
            source, caption, ..
        }) => (source, caption),
        Outcome::Deferred(_) => {
            panic!("expected a screenshot's deferral, got another tool's")
        }
        Outcome::Done(Ok(_)) => panic!("a screenshot must defer, not answer in the frame"),
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
    }
}

/// Apply one command the way the frame does — as the agent, with the Action Log
/// entries that go with it.
///
/// [`apply_as_agent`] rather than bare [`apply`] throughout: attribution is
/// part of what an MCP call *is*, and a test that skipped it would be
/// exercising a path the viewer never takes.
fn agent(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome {
    agent_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host — the two window tools, and anything that
/// wants to see what the window was asked.
fn agent_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> Outcome {
    apply_as_agent(state, viewer, host, vec![command])
        .outcomes
        .pop()
        .expect("one command, one outcome")
}

/// Run one command and unwrap the JSON it produced.
#[track_caller]
fn ok(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Value {
    ok_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host.
#[track_caller]
fn ok_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> Value {
    match agent_with(state, viewer, host, command) {
        Outcome::Done(Ok(ToolOutput::Json(value))) => value,
        Outcome::Done(Ok(ToolOutput::Png { .. })) => panic!("expected JSON, got an image"),
        Outcome::Done(Err(e)) => panic!("expected success, got refusal: {e}"),
        Outcome::Deferred(_) => panic!("expected an answer in this frame"),
    }
}

/// Run one command and unwrap the refusal it produced.
#[track_caller]
fn refused(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> ToolError {
    refused_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host.
#[track_caller]
fn refused_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> ToolError {
    match agent_with(state, viewer, host, command) {
        Outcome::Done(Err(e)) => e,
        Outcome::Done(Ok(_)) => panic!("expected a refusal, got success"),
        Outcome::Deferred(_) => panic!("expected a refusal, got a deferral"),
    }
}

/// Parse a tool call the way the transport would, then apply it.
#[track_caller]
fn call(state: &mut AppState, viewer: &mut Viewer3D, name: &str, arguments: Value) -> Value {
    call_with(state, viewer, &mut NoWindow, name, arguments)
}

/// The same, against a window host.
#[track_caller]
fn call_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    name: &str,
    arguments: Value,
) -> Value {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    let command = tools::parse(name, Some(&map)).unwrap_or_else(|e| panic!("{name}: {e}"));
    ok_with(state, viewer, host, command)
}

/// Parse a tool call the way the transport would and unwrap the refusal, from
/// wherever it came.
///
/// Both halves count as one answer to the agent: whether a call is turned away
/// at the parse or by the viewer is an implementation detail of where the
/// knowledge lives, and a test that fixed which half refused would be asserting
/// that detail rather than the refusal.
#[track_caller]
fn refused_call(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    name: &str,
    arguments: Value,
) -> ToolError {
    refused_call_with(state, viewer, &mut NoWindow, name, arguments)
}

/// The same, against a window host.
#[track_caller]
fn refused_call_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    name: &str,
    arguments: Value,
) -> ToolError {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    match tools::parse(name, Some(&map)) {
        Ok(command) => refused_with(state, viewer, host, command),
        Err(error) => error,
    }
}

mod bench;
mod catalog;
mod display;
mod edit;
mod layout;
mod logged;
mod read;
mod render;
mod server;
mod view;
mod write;

use bench::*;
use edit::*;
use layout::*;
use logged::*;
use render::*;
