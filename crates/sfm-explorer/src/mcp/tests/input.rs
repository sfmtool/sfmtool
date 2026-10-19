// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The input tools, driven through real egui frames.
//!
//! No GPU and no window. Each call is parsed and applied as the drain applies
//! it, and then run a step per frame through the same
//! [`PendingInput`](super::super::input::PendingInput) the viewer runs: its
//! events go into a real egui pass that draws the viewer's own menu bar, its
//! shortcuts and dialogs ([`Chrome`]), and a dock with the real Scene and Image
//! Detail panels in it, and the frame's widgets are captured exactly as
//! `App::capture_mcp_widgets` captures them. So what a click reaches is what
//! egui's hit test says it reaches, and a key goes wherever the viewer reads
//! it.

use std::sync::Arc;

use egui_dock::{DockArea, TabViewer};

use super::super::input::{EguiHeld, WidgetNames};
use super::super::widgets::WidgetFrame;
use super::super::{Deferred, Reply};
use super::*;
use crate::app::Chrome;
use crate::scene_graph::SceneGraphPanel;

/// The window, in points.
const SCREEN: egui::Vec2 = egui::vec2(1200.0, 800.0);

/// Physical pixels per point, not 1, so a tool that forgot to convert would
/// be caught.
const PIXELS_PER_POINT: f32 = 1.5;

/// How often a test button reported each kind of click.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Clicks {
    clicked: u32,
    double_clicked: u32,
    triple_clicked: u32,
}

/// What the test drag pad in the Background Tasks panel saw.
#[derive(Default, Debug, Clone, PartialEq)]
struct Drags {
    started: u32,
    stopped: u32,
    /// The frames it was dragged in, by the button that dragged it.
    by_primary: u32,
    by_secondary: u32,
    /// The frames it was dragged in with Shift held.
    with_shift: u32,
    /// The sum of its drag deltas, in points.
    moved: egui::Vec2,
}

/// Where the test list in the Background Tasks panel is, and how far it has
/// been scrolled, both in points.
#[derive(Debug, Clone, Copy, PartialEq)]
struct List {
    rect: egui::Rect,
    offset: egui::Vec2,
}

impl Default for List {
    fn default() -> Self {
        Self {
            rect: egui::Rect::NOTHING,
            offset: egui::Vec2::ZERO,
        }
    }
}

/// The pixels and features the Image Detail panel draws from, synthesized as
/// its own tests synthesize them.
struct Photograph {
    sift: crate::state::CachedSiftFeatures,
    image: sfmtool_core::camera::image::ImageU8,
}

impl Photograph {
    fn of(node: &SceneNode) -> Self {
        let camera = &node.recon().image_table.cameras[0];
        let (w, h) = (camera.width as f32, camera.height as f32);
        let count = node.recon().point_set.image_feature_to_point[0]
            .len()
            .max(1);
        Self {
            sift: crate::state::CachedSiftFeatures {
                positions_xy: (0..count)
                    .map(|i| {
                        [
                            (i % 8) as f32 * w / 8.0 + 40.0,
                            (i / 8) as f32 * h / 8.0 + 30.0,
                        ]
                    })
                    .collect(),
                affine_shapes: vec![[[6.0, 0.0], [0.0, 6.0]]; count],
                read_count: count,
            },
            image: sfmtool_core::camera::image::ImageU8::new(8, 8, 3, vec![90u8; 8 * 8 * 3]),
        }
    }
}

/// The viewer's chrome and dock, driven a frame at a time.
struct Harness {
    ctx: egui::Context,
    state: AppState,
    viewer: Viewer3D,
    chrome: Chrome,
    scene_graph: SceneGraphPanel,
    photograph: Photograph,
    clicks: Clicks,
    drags: Drags,
    list: List,
    names: WidgetNames,
}

/// The tab bodies: the real Scene and Image Detail panels; a button that
/// counts its clicks and a pad that records its drags in the Background Tasks
/// panel; a label for every other panel.
struct Tabs<'a> {
    state: &'a mut AppState,
    scene_graph: &'a mut SceneGraphPanel,
    image_detail: &'a mut crate::image_detail::ImageDetail,
    photograph: &'a Photograph,
    clicks: &'a mut Clicks,
    drags: &'a mut Drags,
    list: &'a mut List,
}

impl TabViewer for Tabs<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(*tab)
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::SceneGraph => {
                self.scene_graph.show(ui, self.state);
            }
            Tab::ImageDetail => {
                let state = &mut *self.state;
                let node = &state.scene[0];
                let before = crate::state::ImageDetailDisplay::snapshot(
                    &state.feature_display,
                    &state.intrinsics_display,
                );
                self.image_detail.show(
                    ui,
                    node.edited(),
                    node.id,
                    node.history.current_version().serial,
                    Some(0),
                    None,
                    None,
                    None,
                    crate::image_detail::BenchMenu::default(),
                    &[],
                    &crate::platform::ScrollInput::default(),
                    Some(&self.photograph.sift),
                    crate::image_detail::DetailPhotograph::Decoded(&self.photograph.image),
                    &state.feature_display,
                    &mut state.intrinsics_display,
                    None,
                );
                let after = crate::state::ImageDetailDisplay::snapshot(
                    &state.feature_display,
                    &state.intrinsics_display,
                );
                crate::state::record_image_detail_changes(&mut state.action_log, &before, &after);
            }
            Tab::BackgroundTasks => {
                let response = ui.button("Count");
                self.clicks.clicked += u32::from(response.clicked());
                self.clicks.double_clicked += u32::from(response.double_clicked());
                self.clicks.triple_clicked += u32::from(response.triple_clicked());
                let (_, pad) =
                    ui.allocate_exact_size(egui::vec2(200.0, 120.0), egui::Sense::drag());
                pad.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "Pad"));
                let drags = &mut *self.drags;
                drags.started += u32::from(pad.drag_started());
                drags.stopped += u32::from(pad.drag_stopped());
                if pad.dragged() {
                    drags.by_primary += u32::from(pad.dragged_by(egui::PointerButton::Primary));
                    drags.by_secondary += u32::from(pad.dragged_by(egui::PointerButton::Secondary));
                    drags.with_shift += u32::from(ui.input(|input| input.modifiers.shift));
                    drags.moved += pad.drag_delta();
                }
                // A list larger than its area both ways, which a scroll moves.
                let output = egui::ScrollArea::both()
                    .id_salt("test_list")
                    .max_width(160.0)
                    .max_height(80.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.allocate_exact_size(egui::vec2(800.0, 900.0), egui::Sense::hover());
                    });
                self.list.rect = output.inner_rect;
                self.list.offset = output.state.offset;
            }
            other => {
                ui.label(format!("{} body", other.title()));
            }
        }
    }
}

impl Harness {
    /// Two loaded reconstructions, `demo` and `other`, so the `Align to`
    /// submenu has a target, with `demo` selected and the Scene panel in front.
    fn new() -> Self {
        let mut state = AppState::new();
        state.append_node(SceneNode::from_path(
            std::path::Path::new("/runs/demo.sfmr"),
            SfmrReconstruction::demo(32),
        ));
        state.append_node(SceneNode::from_path(
            std::path::Path::new("/runs/other.sfmr"),
            SfmrReconstruction::demo(32),
        ));
        let demo = state.scene[0].id;
        state.select_recon(demo);
        state.show_panel(Tab::SceneGraph);
        state.show_panel(Tab::BackgroundTasks);
        state.show_panel(Tab::ImageDetail);
        let photograph = Photograph::of(&state.scene[0]);
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.set_pixels_per_point(PIXELS_PER_POINT);
        let mut harness = Self {
            ctx,
            state,
            viewer: Viewer3D::new(),
            chrome: Chrome::new(),
            scene_graph: SceneGraphPanel::new(),
            photograph,
            clicks: Clicks::default(),
            drags: Drags::default(),
            list: List::default(),
            names: WidgetNames::default(),
        };
        harness.settle();
        harness
    }

    fn window_px(&self) -> [u32; 2] {
        [
            (SCREEN.x * PIXELS_PER_POINT) as u32,
            (SCREEN.y * PIXELS_PER_POINT) as u32,
        ]
    }

    /// Draw one frame with `events` delivered, and capture it as the viewer
    /// does.
    fn frame(&mut self, events: Vec<egui::Event>) -> Arc<WidgetFrame> {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
            events,
            ..Default::default()
        };
        let Self {
            ctx,
            state,
            viewer,
            chrome,
            scene_graph,
            photograph,
            clicks,
            drags,
            list,
            ..
        } = self;
        let mut output = ctx.run_ui(input, |root_ui| {
            chrome.show(root_ui, state, viewer, &mut NoWindow);
            egui::CentralPanel::default().show(root_ui, |ui| {
                let mut dock =
                    std::mem::replace(&mut state.dock, egui_dock::DockState::new(Vec::new()));
                DockArea::new(&mut dock).show_inside(
                    ui,
                    &mut Tabs {
                        state,
                        scene_graph,
                        image_detail: &mut chrome.image_detail,
                        photograph,
                        clicks,
                        drags,
                        list,
                    },
                );
                state.dock = dock;
            });
        });
        output.textures_delta.clear();
        let window_px = self.window_px();
        let frame = Arc::new(WidgetFrame::capture(
            &self.ctx,
            output.platform_output.accesskit_update.as_ref(),
            &self.state.dock,
            window_px,
        ));
        self.names.remember(&frame);
        frame
    }

    /// Two frames with no input, the second captured.
    fn settle(&mut self) -> Arc<WidgetFrame> {
        self.frame(Vec::new());
        self.frame(Vec::new())
    }

    /// The listing of the window, or of one panel, from a settled frame.
    fn listing(&mut self, panel: Option<Tab>) -> Value {
        self.settle()
            .listing(panel, None)
            .expect("the target is laid out")
    }

    /// Call a tool the way the transport and the drain do, then run it a step
    /// per frame the way the viewer does, to its reply.
    fn call(&mut self, name: &str, arguments: Value) -> Reply {
        let command = tools::parse(name, arguments.as_object())?;
        let outcome = apply_as_agent(
            &mut self.state,
            &mut self.viewer,
            &mut NoWindow,
            vec![command],
        )
        .outcomes
        .pop()
        .expect("one command, one outcome");
        let mut pending = match outcome {
            Outcome::Done(reply) => return reply,
            Outcome::Deferred(Deferred::Input(pending)) => pending,
            Outcome::Deferred(_) => panic!("{name} deferred as something other than input"),
        };
        // A scroll waits out egui's smoothing, which at the 1/60 s egui
        // assumes for a frame with no time of its own is about twenty frames.
        for _ in 0..64 {
            let held = EguiHeld::of(&self.ctx);
            let events = pending.before_pass(&mut self.state, held, &self.names)?;
            let frame = self.frame(events);
            if let Some(reply) = pending.after_pass(frame) {
                return reply;
            }
        }
        panic!("{name} did not answer within 64 frames");
    }

    /// The JSON reply of a call that has to succeed.
    #[track_caller]
    fn ok(&mut self, name: &str, arguments: Value) -> Value {
        match self.call(name, arguments) {
            Ok(ToolOutput::Json(value)) => value,
            Ok(ToolOutput::Png { .. }) => panic!("{name} answered with a picture"),
            Err(error) => panic!("{name} was refused: {error}"),
        }
    }

    /// The refusal of a call that has to be refused.
    #[track_caller]
    fn refused(&mut self, name: &str, arguments: Value) -> ToolError {
        match self.call(name, arguments) {
            Err(error) => error,
            Ok(_) => panic!("{name} was not refused"),
        }
    }

    /// The Scene panel's body in window pixels.
    fn scene_body_px(&self) -> [u32; 4] {
        super::super::panel_rect::panel_crop(
            &self.state.dock,
            Tab::SceneGraph,
            PIXELS_PER_POINT,
            self.window_px(),
        )
        .expect("the Scene panel is laid out")
    }

    /// The `demo` node's row, the widget its context menu opens from.
    fn demo_row(&self) -> egui::Id {
        crate::scene_graph::row_id(self.state.scene[0].id, "node_label")
    }

    /// The Action Log rows written since revision `since`.
    fn rows_since(&self, since: u64) -> Vec<crate::action_log::Entry> {
        self.state.action_log.since(since).cloned().collect()
    }
}

/// The entries of a `widgets` array named `name`.
fn named<'a>(widgets: &'a Value, name: &str) -> Vec<&'a Value> {
    widgets
        .as_array()
        .expect("a widgets array")
        .iter()
        .filter(|entry| entry["name"] == json!(name))
        .collect()
}

/// The one entry named `name`, or a panic listing what there was.
#[track_caller]
fn the<'a>(widgets: &'a Value, name: &str) -> &'a Value {
    let found = named(widgets, name);
    assert_eq!(found.len(), 1, "one {name:?} in {widgets:#}");
    found[0]
}

/// The centre of an entry's `rect_px`.
fn centre(entry: &Value) -> [f64; 2] {
    let r: Vec<f64> = entry["rect_px"]
        .as_array()
        .expect("a rect_px")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    [r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]
}

fn hex(id: egui::Id) -> String {
    format!("{:016x}", id.value())
}

/// A point on the `demo` row of a Scene panel listing, in its pixels: the
/// left end of the row's label, which is drawn over the row's click target
/// and inside the panel's body, where the row itself reaches past it.
fn on_the_demo_row(scene: &Value) -> [f64; 2] {
    let label = named(&scene["widgets"], "demo")
        .into_iter()
        .find(|entry| entry["role"] == json!("label"))
        .expect("the demo row's label");
    let rect = &label["rect_px"];
    let left = rect[0].as_f64().expect("a rect_px");
    let [_, y] = centre(label);
    // Near the start of the text: a label can run past the panel's body.
    [left + 4.0, y.round()]
}

/// The entry with this id in a `widgets` array.
fn with_id<'a>(widgets: &'a Value, id: &str) -> &'a Value {
    widgets
        .as_array()
        .expect("a widgets array")
        .iter()
        .find(|entry| entry["widget"] == json!(id))
        .unwrap_or_else(|| panic!("{id} in {widgets:#}"))
}

// ── click and hover ─────────────────────────────────────────────────────

/// A right click on a Scene row answers with one context menu owned by that
/// row, and a hover on its `Align to` item opens that item's submenu; each is
/// one MCP row, in the words the spec gives.
#[test]
fn a_right_click_on_a_scene_row_opens_its_context_menu_and_a_hover_opens_align_to() {
    let mut harness = Harness::new();
    let scene = harness.listing(Some(Tab::SceneGraph));
    let [x, y] = on_the_demo_row(&scene);
    let at = [x.round(), y.round()];
    let since = harness.state.action_log.revision();

    let reply = harness.ok(
        "click",
        json!({ "panel_name": "scene", "at_px": at, "mouse_button": "right" }),
    );
    assert_eq!(reply["at_px"], json!([at[0] as i64, at[1] as i64]));
    assert_eq!(reply["hit"]["widget"], json!(hex(harness.demo_row())));
    assert_eq!(reply["hit"]["panel_name"], json!("scene"));
    let menus = reply["menus"].as_array().expect("menus");
    assert_eq!(menus.len(), 1, "{reply:#}");
    assert_eq!(menus[0]["kind"], json!("context_menu"));
    assert_eq!(menus[0]["owner"]["widget"], json!(hex(harness.demo_row())));
    assert_eq!(reply["dialogs"], json!([]));
    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert_eq!(
        rows[0].text,
        format!(
            "click right scene {},{} on \"demo\"",
            at[0] as i64, at[1] as i64
        )
    );
    assert_eq!(rows[0].actor, Actor::Mcp);
    assert_eq!(rows[0].kind, Kind::Input);

    // The submenu item, by the id the reply listed it under.
    let align = the(&menus[0]["widgets"], "Align to").clone();
    assert_eq!(align["submenu"], json!(true));
    let reply = harness.ok("hover", json!({ "widget": align["widget"] }));
    assert_eq!(reply["hit"]["widget"], align["widget"]);
    let menus = reply["menus"].as_array().expect("menus");
    assert_eq!(menus.len(), 2, "{reply:#}");
    let submenu = menus
        .iter()
        .find(|menu| menu["kind"] == json!("submenu"))
        .expect("the Align to submenu");
    assert_eq!(submenu["owner"]["widget"], align["widget"]);
    // An item of the context menu, which is over a panel but not in one.
    assert!(submenu["owner"].get("panel_name").is_none(), "{submenu:#}");
    assert_eq!(named(&submenu["widgets"], "other").len(), 1, "{submenu:#}");
    let last = harness.state.action_log.entries().last().unwrap().clone();
    assert!(
        last.text.starts_with("hover window ") && last.text.ends_with(" on \"Align to\""),
        "{}",
        last.text
    );
}

/// A menu bar button opens its menu, and a click on one of its items runs
/// it: Go ▸ Go to Point... opens the dialog, which the reply lists.
#[test]
fn a_click_opens_a_menu_bar_menu_and_a_click_on_an_item_runs_it() {
    let mut harness = Harness::new();
    let window = harness.listing(None);
    let go = the(&window["widgets"], "Go").clone();

    let reply = harness.ok("click", json!({ "widget": go["widget"] }));
    assert_eq!(reply["hit"]["widget"], go["widget"]);
    let menus = reply["menus"].as_array().expect("menus");
    assert_eq!(menus.len(), 1, "{reply:#}");
    assert_eq!(menus[0]["kind"], json!("menu"));
    assert_eq!(menus[0]["owner"]["name"], json!("Go"));
    // AccessKit reads a menu item's shortcut text into its label.
    let item = menus[0]["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| {
            entry["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("Go to Point..."))
        })
        .expect("the Go to Point item")
        .clone();
    assert_eq!(item["path"][0], json!("Go"));

    let reply = harness.ok("click", json!({ "widget": item["widget"] }));
    assert_eq!(reply["menus"], json!([]), "the item closes its menu");
    let dialogs = reply["dialogs"].as_array().expect("dialogs");
    assert_eq!(dialogs.len(), 1, "{reply:#}");
    assert_eq!(dialogs[0]["title"], json!("Go to Point"));
}

// ── drag ────────────────────────────────────────────────────────────────

/// The test pad in a Background Tasks listing.
fn the_pad(harness: &mut Harness) -> Value {
    let panel = harness.listing(Some(Tab::BackgroundTasks));
    the(&panel["widgets"], "Pad").clone()
}

/// A path across the pad, in the panel's pixels: from inside its left edge,
/// right by 30, 60 and 90 pixels, the last 15 pixels lower.
fn across(pad: &Value) -> Vec<[f64; 2]> {
    let left = pad["rect_px"][0].as_f64().expect("a rect_px") + 20.0;
    let [_, y] = centre(pad);
    let y = y.round();
    vec![
        [left, y],
        [left + 30.0, y],
        [left + 60.0, y],
        [left + 90.0, y + 15.0],
    ]
}

/// A drag presses at the first point, moves through the rest one frame each
/// with the button held, and releases at the last: one drag start and one
/// stop, a delta that sums to the path's span, and one MCP row naming both
/// ends.
#[test]
fn a_drag_presses_at_the_first_point_and_releases_at_the_last() {
    let mut harness = Harness::new();
    let pad = the_pad(&mut harness);
    let path = across(&pad);
    let since = harness.state.action_log.revision();

    let reply = harness.ok(
        "drag",
        json!({ "panel_name": Tab::BackgroundTasks.wire_name(), "path_px": path }),
    );
    let [first, last] = [path[0], path[path.len() - 1]];
    assert_eq!(reply["at_px"], json!([last[0] as i64, last[1] as i64]));
    assert_eq!(reply["hit"]["widget"], pad["widget"]);
    assert_eq!(reply["dialogs"], json!([]));
    assert_eq!(reply["menus"], json!([]));

    let drags = harness.drags.clone();
    assert_eq!((drags.started, drags.stopped), (1, 1), "{drags:?}");
    // Dragged from the press through the last move; the release ends it.
    assert_eq!(drags.by_primary, path.len() as u32, "{drags:?}");
    assert_eq!(drags.by_secondary, 0);
    assert_eq!(drags.with_shift, 0);
    let span = egui::vec2(
        ((last[0] - first[0]) as f32) / PIXELS_PER_POINT,
        ((last[1] - first[1]) as f32) / PIXELS_PER_POINT,
    );
    assert!(
        (drags.moved - span).length() < 1e-3,
        "moved {:?}, the path spans {span:?}",
        drags.moved
    );

    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert_eq!(
        rows[0].text,
        format!(
            "drag {} {},{} on \"Pad\" to {},{} through 4 points",
            Tab::BackgroundTasks.wire_name(),
            first[0] as i64,
            first[1] as i64,
            last[0] as i64,
            last[1] as i64
        )
    );
    assert_eq!(rows[0].actor, Actor::Mcp);
    assert_eq!(rows[0].kind, Kind::Input);
}

/// `mouse_button` and `modifiers` are held for the whole drag, and the
/// modifiers are let go afterwards.
#[test]
fn a_drag_holds_its_button_and_modifiers_throughout() {
    let mut harness = Harness::new();
    let pad = the_pad(&mut harness);
    let path = across(&pad);
    let since = harness.state.action_log.revision();

    harness.ok(
        "drag",
        json!({
            "panel_name": Tab::BackgroundTasks.wire_name(),
            "path_px": path,
            "mouse_button": "right",
            "modifiers": ["shift"],
        }),
    );
    let drags = harness.drags.clone();
    assert_eq!((drags.started, drags.stopped), (1, 1), "{drags:?}");
    assert_eq!(drags.by_secondary, path.len() as u32, "{drags:?}");
    assert_eq!(drags.by_primary, 0);
    assert_eq!(drags.with_shift, path.len() as u32, "{drags:?}");
    assert!(!harness.ctx.input(|input| input.modifiers.shift));

    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert!(
        rows[0]
            .text
            .starts_with(&format!("drag right {} ", Tab::BackgroundTasks.wire_name()))
            && rows[0].text.ends_with(" through 4 points with Shift"),
        "{}",
        rows[0].text
    );
}

/// A path with any point outside the target is refused before the button
/// goes down, as one failed row, so nothing is dragged.
#[test]
fn a_drag_with_a_point_outside_its_target_is_refused_before_the_press() {
    let mut harness = Harness::new();
    let pad = the_pad(&mut harness);
    let mut path = across(&pad);
    path[2][1] = -1.0;
    let since = harness.state.action_log.revision();

    let error = harness.refused(
        "drag",
        json!({ "panel_name": Tab::BackgroundTasks.wire_name(), "path_px": path }),
    );
    assert!(error.0.contains("Background Tasks panel's body"), "{error}");
    assert_eq!(harness.drags, Drags::default());
    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert!(rows[0].failed);
    assert_eq!(rows[0].text, format!("drag failed: {error}"));
}

// ── scroll ──────────────────────────────────────────────────────────────

/// The centre of the test list, in window pixels, with the list drawn.
fn on_the_list(harness: &mut Harness) -> [f64; 2] {
    harness.settle();
    let centre = harness.list.rect.center();
    [
        f64::from((centre.x * PIXELS_PER_POINT).round()),
        f64::from((centre.y * PIXELS_PER_POINT).round()),
    ]
}

/// One wheel notch toward the person scrolls a list down by one line, which
/// egui makes 40 points, and the reply comes once the whole line has been
/// scrolled rather than in the middle of egui's smoothing. The row names the
/// point and the notch.
#[test]
fn a_wheel_notch_scrolls_a_list_by_one_line() {
    let mut harness = Harness::new();
    let at = on_the_list(&mut harness);
    assert_eq!(harness.list.offset, egui::Vec2::ZERO);
    let since = harness.state.action_log.revision();

    let reply = harness.ok("scroll", json!({ "at_px": at, "delta_lines": [0, -1] }));
    assert_eq!(reply["at_px"], json!([at[0] as i64, at[1] as i64]));
    assert_eq!(reply["dialogs"], json!([]));
    assert_eq!(reply["menus"], json!([]));
    let offset = harness.list.offset;
    assert!(
        (offset - egui::vec2(0.0, 40.0)).length() < 1e-3,
        "scrolled {offset:?}"
    );
    assert!(!harness.ctx.input(|input| input.is_scrolling()));

    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert_eq!(
        rows[0].text,
        format!(
            "scroll window {},{} by 0,-1 line",
            at[0] as i64, at[1] as i64
        )
    );
    assert_eq!(rows[0].actor, Actor::Mcp);
    assert_eq!(rows[0].kind, Kind::Input);

    // A notch away from the person scrolls back to the top.
    harness.ok("scroll", json!({ "at_px": at, "delta_lines": [0, 1] }));
    assert_eq!(harness.list.offset, egui::Vec2::ZERO);
}

/// A trackpad scroll is in physical pixels, which egui reads in points, and
/// moves the list both ways at once: a negative x shows what is to the right.
#[test]
fn a_trackpad_scroll_moves_a_list_by_its_pixels_in_points() {
    let mut harness = Harness::new();
    let at = on_the_list(&mut harness);
    let since = harness.state.action_log.revision();

    harness.ok("scroll", json!({ "at_px": at, "delta_px": [-30, -45] }));
    let expected = egui::vec2(30.0, 45.0) / PIXELS_PER_POINT;
    let offset = harness.list.offset;
    assert!((offset - expected).length() < 1e-3, "scrolled {offset:?}");
    let rows = harness.rows_since(since);
    assert_eq!(
        rows[0].text,
        format!(
            "scroll window {},{} by -30,-45 px",
            at[0] as i64, at[1] as i64
        )
    );
}

/// Shift turns a wheel's vertical scroll horizontal, as egui does for a
/// person's, and the modifier is let go afterwards.
#[test]
fn shift_turns_a_wheel_scroll_horizontal_and_is_released() {
    let mut harness = Harness::new();
    let at = on_the_list(&mut harness);
    let since = harness.state.action_log.revision();

    harness.ok(
        "scroll",
        json!({ "at_px": at, "delta_lines": [0, -2], "modifiers": ["shift"] }),
    );
    let offset = harness.list.offset;
    assert!(
        (offset - egui::vec2(80.0, 0.0)).length() < 1e-3,
        "scrolled {offset:?}"
    );
    assert!(!harness.ctx.input(|input| input.modifiers.shift));
    let rows = harness.rows_since(since);
    assert!(
        rows[0].text.ends_with(" by 0,-2 lines with Shift"),
        "{}",
        rows[0].text
    );
}

/// The scroll moves the pointer the viewer's own panels test against, not
/// only egui's: the 3D Viewer, Image Detail and the Image Browser decide
/// whether a wheel is theirs by `platform::pointer_in_rect`, which on Windows
/// reads the cursor the window procedure tracks.
#[test]
fn a_scroll_moves_the_pointer_the_panels_test_against() {
    let mut harness = Harness::new();
    let at = on_the_list(&mut harness);
    crate::platform::set_test_pointer_pos(None);
    let list = harness.list.rect;

    harness.ok("scroll", json!({ "at_px": at, "delta_lines": [0, -1] }));
    assert!(crate::platform::pointer_in_rect(&harness.ctx, list));
    harness.ok("hover", json!({ "at_px": [2, 2] }));
    assert!(!crate::platform::pointer_in_rect(&harness.ctx, list));
}

/// A point outside the target is refused before anything is delivered, as one
/// failed row, so nothing scrolls.
#[test]
fn a_scroll_outside_its_target_is_refused_before_it_is_delivered() {
    let mut harness = Harness::new();
    on_the_list(&mut harness);
    let since = harness.state.action_log.revision();
    let error = harness.refused(
        "scroll",
        json!({ "panel_name": "scene", "at_px": [-1, 4], "delta_lines": [0, -1] }),
    );
    assert!(error.0.contains("Scene panel's body"), "{error}");
    assert_eq!(harness.list.offset, egui::Vec2::ZERO);
    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert!(rows[0].failed);
    assert_eq!(rows[0].text, format!("scroll failed: {error}"));
}

/// `count: 2` is one double click: egui reports a click for each press, and
/// the second press, one frame after the first, as the one double click.
#[test]
fn count_two_is_one_double_click() {
    let mut harness = Harness::new();
    let panel = harness.listing(Some(Tab::BackgroundTasks));
    let count = the(&panel["widgets"], "Count").clone();

    harness.ok("click", json!({ "widget": count["widget"] }));
    assert_eq!(
        harness.clicks,
        Clicks {
            clicked: 1,
            double_clicked: 0,
            triple_clicked: 0
        }
    );
    // Well after the first, so the two calls are not one double click.
    for _ in 0..40 {
        harness.frame(Vec::new());
    }
    harness.clicks = Clicks::default();
    harness.ok("click", json!({ "widget": count["widget"], "count": 2 }));
    assert_eq!(
        harness.clicks,
        Clicks {
            clicked: 2,
            double_clicked: 1,
            triple_clicked: 0
        },
        "one double click, and no click past the second press"
    );
    let last = harness.state.action_log.entries().last().unwrap();
    assert!(
        last.text.starts_with("click twice window "),
        "{}",
        last.text
    );
}

/// A widget with a menu open over its centre is refused, naming the menu, and
/// the refusal is one failed row.
#[test]
fn a_click_on_a_widget_under_an_open_menu_is_refused_naming_the_menu() {
    let mut harness = Harness::new();
    let window = harness.listing(None);
    let file = the(&window["widgets"], "File").clone();
    let reply = harness.ok("click", json!({ "widget": file["widget"] }));
    let menu = reply["menus"][0]["rect_px"].clone();
    let rect: Vec<f64> = menu
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap())
        .collect();
    // Any clickable widget of the window whose centre the menu covers.
    let frame = harness.settle();
    let window = frame.listing(None, None).unwrap();
    let covered = window["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| {
            let [x, y] = centre(entry);
            entry["sense"] != json!("none")
                && x > rect[0]
                && x < rect[0] + rect[2]
                && y > rect[1]
                && y < rect[1] + rect[3]
        })
        .expect("the File menu covers a widget of the dock")
        .clone();

    let since = harness.state.action_log.revision();
    let error = harness.refused("click", json!({ "widget": covered["widget"] }));
    assert!(error.0.contains("the File menu"), "{error}");
    assert!(error.0.contains("covered at its centre"), "{error}");
    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert!(rows[0].failed);
    assert_eq!(rows[0].text, format!("click failed: {error}"));
}

// ── press_key ───────────────────────────────────────────────────────────

/// `command` + `Z` undoes through the menu bar's own shortcut: the undo row is
/// the person's, straight after the MCP row that says where the key came from.
#[test]
fn command_z_undoes_through_the_menu_bar_shortcut() {
    let mut harness = Harness::new();
    let demo = harness.state.scene[0].id;
    harness
        .state
        .select_point(crate::scene::PointRef::new(demo, 0));
    harness
        .state
        .delete_selected_point()
        .expect("the point can be deleted");
    assert!(harness.state.can_undo(demo));
    let since = harness.state.action_log.revision();

    let reply = harness.ok("press_key", json!({ "key": "Z", "modifiers": ["command"] }));
    assert!(!harness.state.can_undo(demo), "the delete was undone");
    assert!(harness.state.can_redo(demo));
    assert_eq!(reply["dialogs"], json!([]));
    assert!(reply.get("focused").is_some(), "{reply:#}");
    let rows = harness.rows_since(since);
    let key = rows
        .iter()
        .position(|row| row.actor == Actor::Mcp)
        .expect("the key's row");
    assert!(
        rows[key].text.starts_with("press_key ") && rows[key].text.ends_with("+Z"),
        "{}",
        rows[key].text
    );
    let undo = rows
        .iter()
        .position(|row| row.text.starts_with("Undo: "))
        .expect("the undo's row");
    assert!(key < undo, "{rows:#?}");
    assert_eq!(rows[undo].actor, Actor::User);
    // And nothing is left held down.
    assert_eq!(
        harness.ctx.input(|input| input.modifiers),
        egui::Modifiers::NONE
    );
}

/// Escape cancels Go to Point, and the reply no longer lists it.
#[test]
fn escape_closes_go_to_point_and_the_reply_says_so() {
    let mut harness = Harness::new();
    harness.state.open_goto_point();
    let window = harness.listing(None);
    assert_eq!(window["dialogs"].as_array().unwrap().len(), 1);

    let reply = harness.ok("press_key", json!({ "key": "Escape" }));
    assert_eq!(reply["dialogs"], json!([]), "{reply:#}");
    let last = harness.state.action_log.entries().last().unwrap();
    assert!(
        harness
            .state
            .action_log
            .entries()
            .any(|row| row.text == "press_key Escape"),
        "{}",
        last.text
    );
}

/// `I` with the pointer over Image Detail toggles its intrinsics layer; sent
/// with the pointer elsewhere, it does not.
#[test]
fn i_over_image_detail_toggles_the_intrinsics_layer_and_elsewhere_does_not() {
    let mut harness = Harness::new();
    let before = harness.state.intrinsics_display.enabled;

    harness.ok("press_key", json!({ "key": "I" }));
    assert_eq!(harness.state.intrinsics_display.enabled, before);

    let since = harness.state.action_log.revision();
    harness.ok(
        "press_key",
        json!({ "key": "I", "panel_name": "image_detail" }),
    );
    assert_eq!(harness.state.intrinsics_display.enabled, !before);
    let rows = harness.rows_since(since);
    assert_eq!(rows[0].text, "press_key I over image_detail");
    assert_eq!(rows[0].actor, Actor::Mcp);
    // The panel's own row for the change, as a person's.
    assert!(
        rows.iter()
            .any(|row| row.text.starts_with("Intrinsics ") && row.actor == Actor::User),
        "{rows:#?}"
    );
}

/// The modifiers a call holds are released when it is done, by a click as
/// well as by a key.
#[test]
fn modifiers_are_released_after_the_command() {
    let mut harness = Harness::new();
    let panel = harness.listing(Some(Tab::BackgroundTasks));
    let count = the(&panel["widgets"], "Count").clone();
    harness.ok(
        "click",
        json!({ "widget": count["widget"], "modifiers": ["shift", "alt"] }),
    );
    assert_eq!(
        harness.ctx.input(|input| input.modifiers),
        egui::Modifiers::NONE
    );
    let last = harness.state.action_log.entries().last().unwrap();
    assert!(last.text.contains(" with "), "{}", last.text);
    harness.ok("press_key", json!({ "key": "A", "modifiers": ["shift"] }));
    assert_eq!(
        harness.ctx.input(|input| input.modifiers),
        egui::Modifiers::NONE
    );
}

// ── type_text ───────────────────────────────────────────────────────────

/// Typed into the Go to Point field, the text is the field's value; Enter then
/// selects the point. The row counts the characters and does not carry them.
#[test]
fn type_text_fills_go_to_point_and_enter_selects_the_point() {
    let mut harness = Harness::new();
    harness.state.open_goto_point();
    let window = harness.listing(None);
    let field = window["dialogs"][0]["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["role"] == json!("text_input"))
        .expect("the point field")
        .clone();

    let reply = harness.ok(
        "type_text",
        json!({ "text": "3", "widget": field["widget"] }),
    );
    assert_eq!(reply["focused"]["widget"], field["widget"], "{reply:#}");
    assert_eq!(reply["focused"]["value"], json!("3"), "{reply:#}");
    let last = harness.state.action_log.entries().last().unwrap();
    assert_eq!(
        last.text,
        "type_text 1 character into the text input in Go to Point"
    );

    // Into whatever has focus, which is still the field.
    harness.ok("type_text", json!({ "text": "1" }));
    let window = harness.listing(None);
    assert_eq!(
        with_id(
            &window["dialogs"][0]["widgets"],
            field["widget"].as_str().unwrap()
        )["value"],
        json!("31")
    );

    let reply = harness.ok("press_key", json!({ "key": "Enter" }));
    assert_eq!(reply["dialogs"], json!([]), "{reply:#}");
    let demo = harness.state.scene[0].id;
    assert_eq!(
        harness.state.selected_point,
        Some(crate::scene::PointRef::new(demo, 31))
    );
}

/// A widget that is not a text input is refused, and so is a call with
/// nothing focused, each as one failed row.
#[test]
fn type_text_is_refused_where_the_text_would_go_nowhere() {
    let mut harness = Harness::new();
    let window = harness.listing(None);
    let file = the(&window["widgets"], "File").clone();
    let since = harness.state.action_log.revision();
    let error = harness.refused(
        "type_text",
        json!({ "text": "x", "widget": file["widget"] }),
    );
    assert!(
        error.0.contains("not") || error.0.contains("text input"),
        "{error}"
    );
    assert!(error.0.contains("\"File\""), "{error}");
    let rows = harness.rows_since(since);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].failed);

    let since = harness.state.action_log.revision();
    let error = harness.refused("type_text", json!({ "text": "x" }));
    assert!(error.0.starts_with("Nothing has keyboard focus"), "{error}");
    assert_eq!(harness.rows_since(since).len(), 1);
}

// ── Refusals ────────────────────────────────────────────────────────────

/// An unknown key is refused at the parse, with egui's names for the keys.
#[test]
fn an_unknown_key_is_refused_with_the_names() {
    let error = tools::parse("press_key", json!({ "key": "Ctrl+Z" }).as_object())
        .expect_err("not a key name");
    assert!(
        error
            .0
            .starts_with("press_key does not know the key \"Ctrl+Z\""),
        "{error}"
    );
    for name in ["A to Z", "Enter", "Escape", "OpenBracket", "F1 to F35"] {
        assert!(error.0.contains(name), "{name} in {error}");
    }
    assert!(error.0.contains("modifiers"), "{error}");
    // The spellings the spec promises.
    for key in ["M", "Enter", "ArrowUp", "F12", "Comma", "OpenBracket", "0"] {
        tools::parse("press_key", json!({ "key": key }).as_object())
            .unwrap_or_else(|error| panic!("{key}: {error}"));
    }
}

/// A point outside its target, a closed panel, a panel behind a tab, an
/// unknown widget and a minimized window are each refused in words that say
/// what to do, as one failed row.
#[test]
fn pointer_refusals_say_why_and_record_one_row() {
    let mut harness = Harness::new();
    let [width, height] = harness.window_px();

    let refuse = |harness: &mut Harness, name: &str, arguments: Value| {
        let since = harness.state.action_log.revision();
        let error = harness.refused(name, arguments);
        let rows = harness.rows_since(since);
        assert_eq!(rows.len(), 1, "{name}: {rows:#?}");
        assert!(rows[0].failed);
        assert_eq!(rows[0].actor, Actor::Mcp);
        assert_eq!(rows[0].text, format!("{name} failed: {error}"));
        error
    };

    let error = refuse(
        &mut harness,
        "click",
        json!({ "at_px": [f64::from(width) + 5.0, 10] }),
    );
    assert!(error.0.contains(&format!("{width}×{height}")), "{error}");
    let body = harness.scene_body_px();
    let error = refuse(
        &mut harness,
        "hover",
        json!({ "panel_name": "scene", "at_px": [10, f64::from(body[3])] }),
    );
    assert!(error.0.contains("Scene panel's body"), "{error}");

    harness.state.hide_panel(Tab::ActionLog);
    let error = refuse(
        &mut harness,
        "click",
        json!({ "panel_name": "action_log", "at_px": [1, 1] }),
    );
    assert_eq!(
        error.0,
        "The Action Log panel is closed, so there is nothing of it to point at. Send \
         show_panel { \"panel_name\": \"action_log\" } first."
    );
    // Any node of the layout with a panel behind the one in front.
    let (behind, in_front) = harness
        .state
        .dock
        .iter_leaves()
        .find_map(|(_, leaf)| {
            let front = *leaf.tabs.get(leaf.active.0)?;
            let behind = *leaf.tabs.iter().find(|tab| **tab != front)?;
            Some((behind, front))
        })
        .expect("the layout tabs one panel behind another");
    let error = refuse(
        &mut harness,
        "press_key",
        json!({ "key": "I", "panel_name": behind.wire_name() }),
    );
    assert!(
        error.0.starts_with(&format!(
            "The {} panel is behind {} in its node, so the pointer would land on {} instead.",
            behind.title(),
            in_front.title(),
            in_front.title()
        )),
        "{error}"
    );

    // An id a listing reported, for a widget that has gone: its menu closed.
    let window = harness.listing(None);
    let file = the(&window["widgets"], "File").clone();
    let reply = harness.ok("click", json!({ "widget": file["widget"] }));
    let quit = the(&reply["menus"][0]["widgets"], "Quit").clone();
    harness.ok("press_key", json!({ "key": "Escape" }));
    let error = refuse(&mut harness, "click", json!({ "widget": quit["widget"] }));
    assert!(error.0.contains("It was named \"Quit\""), "{error}");
    assert!(
        error.0.contains("nothing named \"Quit\" is drawn now"),
        "{error}"
    );
    let error = refuse(
        &mut harness,
        "click",
        json!({ "widget": "0123456789abcdef" }),
    );
    assert!(error.0.contains("no listing has reported one"), "{error}");

    harness.state.window =
        Some(crate::test_support::FakeWindow::in_state(WindowState::Minimized).info());
    let error = refuse(&mut harness, "type_text", json!({ "text": "x" }));
    assert!(error.0.starts_with("The window is minimized"), "{error}");
}

/// The parse refuses what no frame could answer: no target, two targets, a
/// panel with a widget, a third click, an unknown button or modifier, and a
/// malformed id.
#[test]
fn the_input_tools_parse_what_they_advertise() {
    let refuse = |name: &str, arguments: Value| {
        tools::parse(name, arguments.as_object()).expect_err("refused at the parse")
    };
    assert!(refuse("click", json!({})).0.contains("needs either at_px"));
    assert!(refuse(
        "click",
        json!({ "at_px": [1, 2], "widget": "0123456789abcdef" })
    )
    .0
    .contains("both widget and at_px"));
    assert!(refuse(
        "hover",
        json!({ "panel_name": "scene", "widget": "0123456789abcdef" })
    )
    .0
    .contains("panel_name only with at_px"));
    assert!(refuse("click", json!({ "at_px": [1, 2], "count": 3 }))
        .0
        .contains("1 or 2"));
    assert!(
        refuse("click", json!({ "at_px": [1, 2], "mouse_button": "back" }))
            .0
            .contains("left, middle and right")
    );
    assert!(
        refuse("click", json!({ "at_px": [1, 2], "modifiers": ["ctrl"] }))
            .0
            .contains("shift, control, alt, command")
    );
    assert!(refuse("click", json!({ "widget": "XYZ" }))
        .0
        .contains("16 lowercase hex digits"));
    // The schema's pattern is lowercase only, and so is the parser.
    assert!(refuse("click", json!({ "widget": "003233F928812D8A" }))
        .0
        .contains("16 lowercase hex digits"));
    assert!(refuse("type_text", json!({ "text": "" }))
        .0
        .contains("nothing to type"));
    assert!(refuse("drag", json!({})).0.contains("needs path_px"));
    assert!(refuse("drag", json!({ "path_px": [[1, 2]] }))
        .0
        .contains("from 2 to 120 points"));
    let long: Vec<[i64; 2]> = (0..121).map(|i| [i, 1]).collect();
    assert!(refuse("drag", json!({ "path_px": long }))
        .0
        .contains("from 2 to 120 points"));
    assert!(refuse("drag", json!({ "path_px": [[1, 2], [1, 2, 3]] }))
        .0
        .contains("an array of points [x, y]"));
    assert!(refuse("drag", json!({ "path_px": [[1, 2], [3, "4"]] }))
        .0
        .contains("an array of points [x, y]"));
    assert!(refuse(
        "drag",
        json!({ "path_px": [[1, 2], [3, 4]], "mouse_button": "back" })
    )
    .0
    .contains("left, middle and right"));
    assert!(refuse("scroll", json!({ "delta_lines": [0, 1] }))
        .0
        .contains("needs either at_px"));
    assert!(refuse("scroll", json!({ "at_px": [1, 2] }))
        .0
        .contains("needs delta_lines"));
    assert!(refuse(
        "scroll",
        json!({ "at_px": [1, 2], "delta_lines": [0, 1], "delta_px": [0, 1] })
    )
    .0
    .contains("both delta_lines and delta_px"));
    assert!(
        refuse("scroll", json!({ "at_px": [1, 2], "delta_px": [0, 0] }))
            .0
            .contains("nothing to scroll")
    );
    assert!(
        refuse("scroll", json!({ "at_px": [1, 2], "delta_lines": [1] }))
            .0
            .contains("an array of 2 numbers")
    );
    assert!(refuse(
        "scroll",
        json!({ "at_px": [1, 2], "delta_lines": [0, "1"] })
    )
    .0
    .contains("an array of 2 numbers"));
    assert!(refuse(
        "scroll",
        json!({ "at_px": [1, 2], "delta_lines": [0, 1], "modifiers": ["meta"] })
    )
    .0
    .contains("shift, control, alt, command"));
    // A scroll has no button.
    assert!(refuse(
        "scroll",
        json!({ "at_px": [1, 2], "delta_lines": [0, 1], "mouse_button": "left" })
    )
    .0
    .contains("mouse_button"));
}

// ── The wire vocabulary ─────────────────────────────────────────────────

/// Every field an input reply carries is one the spec names.
#[test]
fn input_replies_speak_the_wire_vocabulary() {
    let mut harness = Harness::new();
    let scene = harness.listing(Some(Tab::SceneGraph));
    let at = on_the_demo_row(&scene);
    let click = harness.ok(
        "click",
        json!({ "panel_name": "scene", "at_px": [at[0].round(), at[1].round()], "mouse_button": "right" }),
    );
    let hover = harness.ok("hover", json!({ "panel_name": "scene", "at_px": [1, 1] }));
    harness.ok("press_key", json!({ "key": "Escape" }));
    let drag = harness.ok(
        "drag",
        json!({ "panel_name": "scene", "path_px": [[1, 1], [5, 5]] }),
    );
    let scroll = harness.ok(
        "scroll",
        json!({ "panel_name": "scene", "at_px": [1, 1], "delta_lines": [0, -1] }),
    );
    harness.state.open_goto_point();
    harness.settle();
    let key = harness.ok("press_key", json!({ "key": "End" }));
    let text = harness.ok("type_text", json!({ "text": "7" }));

    let allowed = [
        "at_px",
        "hit",
        "focused",
        "dialogs",
        "menus",
        "widgets",
        "widget",
        "role",
        "name",
        "panel_name",
        "rect_px",
        "enabled",
        "toggled",
        "value",
        "sense",
        "path",
        "submenu",
        "title",
        "kind",
        "owner",
    ];
    fn keys(value: &Value, out: &mut std::collections::BTreeSet<String>) {
        match value {
            Value::Object(map) => {
                for (key, inner) in map {
                    out.insert(key.clone());
                    keys(inner, out);
                }
            }
            Value::Array(items) => items.iter().for_each(|item| keys(item, out)),
            _ => {}
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for reply in [&click, &hover, &drag, &scroll, &key, &text] {
        keys(reply, &mut seen);
    }
    for key in &seen {
        assert!(
            allowed.contains(&key.as_str()),
            "{key:?} is not in the vocabulary"
        );
    }
    for required in ["at_px", "hit", "focused", "dialogs", "menus"] {
        assert!(seen.contains(required), "no reply carried {required:?}");
    }
    let mut click_keys: Vec<&str> = click
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    click_keys.sort_unstable();
    assert_eq!(click_keys, ["at_px", "dialogs", "hit", "menus"]);
    let mut drag_keys: Vec<&str> = drag
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    drag_keys.sort_unstable();
    assert_eq!(drag_keys, click_keys);
    let mut scroll_keys: Vec<&str> = scroll
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    scroll_keys.sort_unstable();
    assert_eq!(scroll_keys, click_keys);
    let mut key_keys: Vec<&str> = key
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    key_keys.sort_unstable();
    assert_eq!(key_keys, ["dialogs", "focused", "menus"]);
}
