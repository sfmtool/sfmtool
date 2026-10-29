// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The names the viewer's own controls are listed under, read from real egui
//! frames of the panels and dialogs that draw them.
//!
//! Each test draws the real code that builds a panel or a dialog with no dock
//! around it and captures the frame as `App::capture_mcp_widgets` does, so a
//! name is checked as `get_widgets` reports it: a control's own label, or the
//! label it is `labelled_by`. A number box, a dropdown or a text input that
//! nothing names is listed with a `null` name, and the tests check that none
//! of the controls they draw is.

use serde_json::{json, Value};
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};
use sfmtool_core::SfmrReconstruction;

use super::super::widgets::WidgetFrame;
use crate::scene::{PointRef, SceneNode};
use crate::state::AppState;

/// The window, in points; one point is one pixel here.
const SCREEN: egui::Vec2 = egui::vec2(1600.0, 1000.0);

/// One context, drawn a frame at a time with whatever a test draws.
struct Drawn {
    ctx: egui::Context,
}

impl Drawn {
    fn new() -> Self {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        Self { ctx }
    }

    /// Draw one frame with `events` delivered, and capture it.
    fn frame(
        &self,
        events: Vec<egui::Event>,
        mut draw: impl FnMut(&egui::Context, &mut egui::Ui),
    ) -> WidgetFrame {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
            events,
            ..Default::default()
        };
        let mut output = self.ctx.run_ui(input, |root_ui| {
            let ctx = root_ui.ctx().clone();
            egui::CentralPanel::default().show(root_ui, |ui| draw(&ctx, ui));
        });
        output.textures_delta.clear();
        let dock = egui_dock::DockState::new(Vec::new());
        WidgetFrame::capture(
            &self.ctx,
            output.platform_output.accesskit_update.as_ref(),
            &dock,
            [SCREEN.x as u32, SCREEN.y as u32],
        )
    }

    /// The window's listing after two frames with no input, the first laying
    /// out what the second is read from.
    fn listing(&self, mut draw: impl FnMut(&egui::Context, &mut egui::Ui)) -> Value {
        self.frame(Vec::new(), &mut draw);
        window(&self.frame(Vec::new(), &mut draw))
    }

    /// Click the middle of `entry`, then settle, and hand back the listing.
    fn click(&self, entry: &Value, mut draw: impl FnMut(&egui::Context, &mut egui::Ui)) -> Value {
        let rect: Vec<f32> = entry["rect_px"]
            .as_array()
            .expect("a rect_px")
            .iter()
            .map(|n| n.as_f64().expect("a number") as f32)
            .collect();
        let pos = egui::pos2(rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0);
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        self.frame(vec![egui::Event::PointerMoved(pos)], &mut draw);
        self.frame(vec![button(true)], &mut draw);
        self.frame(vec![button(false)], &mut draw);
        self.listing(draw)
    }
}

/// The window's listing of a frame.
fn window(frame: &WidgetFrame) -> Value {
    frame.listing(None, None).expect("the window is laid out")
}

/// Every widget entry in a listing: the window's, each dialog's and each
/// menu's.
fn entries(listing: &Value) -> Vec<&Value> {
    let mut all: Vec<&Value> = listing["widgets"].as_array().unwrap().iter().collect();
    for section in ["dialogs", "menus"] {
        for block in listing[section].as_array().unwrap() {
            all.extend(block["widgets"].as_array().unwrap());
        }
    }
    all
}

/// The roles of the controls that take a value, which a name has to say the
/// meaning of.
const CONTROL_ROLES: [&str; 3] = ["spin_button", "combo_box", "text_input"];

/// The names of the controls in `listing` with one of `CONTROL_ROLES`, in
/// drawing order, with `null` for an unnamed one.
fn control_names(listing: &Value) -> Vec<(String, Value)> {
    entries(listing)
        .into_iter()
        .filter(|entry| {
            CONTROL_ROLES
                .iter()
                .any(|role| entry["role"] == json!(role))
        })
        .map(|entry| {
            (
                entry["role"].as_str().unwrap().to_string(),
                entry["name"].clone(),
            )
        })
        .collect()
}

/// The one entry of `role` named `name`.
#[track_caller]
fn the_role<'a>(listing: &'a Value, role: &str, name: &str) -> &'a Value {
    let found: Vec<&Value> = entries(listing)
        .into_iter()
        .filter(|entry| entry["role"] == json!(role) && entry["name"] == json!(name))
        .collect();
    assert_eq!(found.len(), 1, "one {role} {name:?} in {listing:#}");
    found[0]
}

/// No control in `listing` is unnamed.
#[track_caller]
fn assert_all_named(listing: &Value) {
    let unnamed: Vec<(String, Value)> = control_names(listing)
        .into_iter()
        .filter(|(_, name)| name.is_null())
        .collect();
    assert!(unnamed.is_empty(), "unnamed: {unnamed:?} in {listing:#}");
}

// ── Image Detail ────────────────────────────────────────────────────────

/// A camera with radial distortion, so the gear popup offers the scale of
/// the distortion field.
fn distorted_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
        },
        width: 640,
        height: 480,
    }
}

/// The Image Detail toolbar names every control it draws: the two dropdowns
/// by their labels, the size filter's two number boxes by the bound each
/// sets, and in the gear popup the scale dropdown by the field it scales.
#[test]
fn the_image_detail_toolbar_and_its_gear_popup_name_their_controls() {
    use crate::image_detail::ImageDetail;
    use crate::scene::CameraRef;
    use crate::state::{FeatureDisplaySettings, IntrinsicsDisplaySettings, OverlayMode};

    let node = SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        SfmrReconstruction::demo(16),
    );
    let camera = distorted_camera();
    let mut settings = FeatureDisplaySettings {
        overlay_mode: OverlayMode::Features,
        ..Default::default()
    };
    let mut intrinsics = IntrinsicsDisplaySettings {
        enabled: true,
        ..Default::default()
    };
    let mut detail = ImageDetail::new();
    let drawn = Drawn::new();
    let mut draw = |_: &egui::Context, ui: &mut egui::Ui| {
        crate::dock::show_overlay_toolbar(
            ui,
            &mut settings,
            &mut intrinsics,
            &mut detail,
            Some((
                crate::image_detail::ImageIntrinsicsKey::new(
                    CameraRef::new(node.id, 0),
                    &Default::default(),
                ),
                &camera,
            )),
            None,
        );
    };
    let toolbar = drawn.listing(&mut draw);
    assert_eq!(
        control_names(&toolbar),
        [
            ("combo_box".to_string(), json!("Overlay:")),
            ("combo_box".to_string(), json!("Max:")),
            ("spin_button".to_string(), json!("minimum size (px)")),
            ("spin_button".to_string(), json!("maximum size (px)")),
        ],
        "{toolbar:#}"
    );

    // A click opens the box for typing, which draws a text field with the
    // box's id in its place; the box keeps its name while it is open.
    let min = the_role(&toolbar, "spin_button", "minimum size (px)").clone();
    let typing = drawn.click(&min, &mut draw);
    assert!(
        drawn.ctx.memory(|memory| memory.focused()).is_some(),
        "the box is open for typing"
    );
    the_role(&typing, "spin_button", "minimum size (px)");
    assert_all_named(&typing);
    // Escape closes it again, so the gear below is clicked with no box open.
    drawn.frame(
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }],
        &mut draw,
    );

    let gear = the_role(&toolbar, "button", "⚙ Intrinsics settings").clone();
    let popup = drawn.click(&gear, &mut draw);
    assert_eq!(popup["menus"].as_array().unwrap().len(), 1, "{popup:#}");
    let scale = the_role(&popup, "combo_box", "Distortion field scale");
    // Its value is the scale it shows, which is not part of its name.
    assert!(
        scale["value"].as_str().unwrap().starts_with("Auto"),
        "{popup:#}"
    );
    the_role(&popup, "combo_box", "Grid density");
    assert_all_named(&popup);
}

// ── Scene tree ──────────────────────────────────────────────────────────

/// The Scene tree's glyph buttons are named by the glyph each shows and then
/// what it controls, so the name starts with what a person sees and still
/// says what the button is for. No button is named by its glyph alone.
#[test]
fn the_scene_trees_glyph_buttons_are_named_by_glyph_and_purpose() {
    use crate::scene_graph::SceneGraphPanel;

    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        SfmrReconstruction::demo(16),
    ));
    let mut panel = SceneGraphPanel::new();
    let drawn = Drawn::new();
    let tree = drawn.listing(|_, ui| {
        panel.show(ui, &mut state);
    });
    for name in [
        "\u{1f441} Show reconstruction",
        "S Solo",
        "\u{1f5b1} Interactive",
        "\u{1f441} Show camera images",
        "\u{1f441} Show points",
    ] {
        the_role(&tree, "button", name);
    }
    let bare: Vec<&Value> = entries(&tree)
        .into_iter()
        .filter(|entry| {
            entry["role"] == json!("button")
                && ["\u{1f441}", "\u{1f5b1}", "\u{221e}", "S"]
                    .iter()
                    .any(|glyph| entry["name"] == json!(glyph))
        })
        .collect();
    assert!(bare.is_empty(), "named by a glyph alone: {bare:?}");
}

// ── Track View ──────────────────────────────────────────────────────────

/// Track View's body names every number box: the split settings and the
/// threshold row's boxes by what each sets, since the text in or after them
/// is a unit and a reading's name, the geometry search box by the label
/// before it, and the header's two icon buttons by what they do. The field
/// Rename opens is named for what it holds.
#[test]
fn track_view_names_its_boxes_and_its_icon_buttons() {
    use crate::track_view::body::TrackBody;

    let mut state = AppState::new();
    state.append_node(SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, 2), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let drawn = Drawn::new();
    let mut draw = |_: &egui::Context, ui: &mut egui::Ui| {
        panel.show(ui, &state);
    };
    let body = drawn.listing(&mut draw);

    for name in [
        "pieces per axis",
        "piece overlap (%)",
        "geometry search min relative ZNCC (%)",
        "minimum ZNCC whole (%)",
        "minimum ZNCC mid (%)",
        "maximum Self-similarity whole (px)",
        "maximum Shift (px)",
        "maximum Proj. err (px)",
    ] {
        the_role(&body, "spin_button", name);
    }
    // The geometry search box is named by the label drawn before it, which is
    // listed under the same text.
    the_role(&body, "label", "geometry search min relative ZNCC (%)");
    the_role(&body, "button", "Copy Point ID");
    the_role(&body, "button", "Go to Point");
    assert_all_named(&body);

    let rename = the_role(&body, "button", "Rename").clone();
    let renaming = drawn.click(&rename, &mut draw);
    let field = the_role(&renaming, "text_input", "item name");
    assert!(field["value"].as_str().is_some(), "{renaming:#}");
    assert_all_named(&renaming);
}

// ── Dialogs ─────────────────────────────────────────────────────────────

/// The text inputs and dropdowns of the Go to Point, Save As Minimal and
/// Bundle Adjust dialogs are named: each text input by the label before it,
/// and the schedule's preset dropdown, which has no text beside it, by a
/// name of its own. Load Demo Data's number box is checked with the chrome
/// that draws it, in the input tests.
#[test]
fn the_dialogs_name_their_fields() {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        SfmrReconstruction::demo(32),
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    state.open_goto_point();
    state.open_bundle_adjust(id);
    state.save_minimal_prompt.ask(
        id,
        "demo".to_string(),
        std::path::PathBuf::from("/runs/demo-minimal.sfmr"),
        ".".to_string(),
    );
    let drawn = Drawn::new();
    let listing = drawn.listing(|ctx, _| {
        let _ = state
            .goto_point
            .show(ctx, &state.scene, state.selected_recon);
        let _ = state.bundle_adjust_prompt.show(ctx);
        let _ = state.save_minimal_prompt.show(ctx);
    });
    let titles: Vec<&Value> = listing["dialogs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|dialog| &dialog["title"])
        .collect();
    assert_eq!(titles.len(), 3, "{listing:#}");

    let field = the_role(&listing, "text_input", "Point index, or full ID with hash:");
    assert_eq!(field["path"][0], json!("Go to Point"), "{listing:#}");
    let path = the_role(
        &listing,
        "text_input",
        "Workspace path recorded in the copy:",
    );
    assert_eq!(path["value"], json!("."), "{listing:#}");
    the_role(&listing, "text_input", "Schedule");
    the_role(&listing, "combo_box", "Schedule preset");
    assert_all_named(&listing);
}
