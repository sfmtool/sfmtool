// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The widget listing, read from real egui frames.
//!
//! No GPU and no window: egui lays a frame out without either, and the
//! AccessKit tree is part of its output once `enable_accesskit` is on. So these
//! draw a real `egui_dock::DockArea` over the viewer's own dock state, with the
//! real Scene panel in it, the viewer's own Go to Point and Bundle Adjust
//! dialogs, and a menu bar built the way `app/menu.rs` builds one, and capture
//! the frame exactly as `App::capture_mcp_widgets` does. The panels other than
//! Scene are drawn as a label with their title: what is under test is the
//! listing, not those panels.
//!
//! Clicks aim at rectangles the listing itself reported, since that is what an
//! agent will do with it.

use egui_dock::{DockArea, TabViewer};

use super::super::widgets::WidgetFrame;
use super::*;
use crate::scene_graph::SceneGraphPanel;

/// The window, in points.
const SCREEN: egui::Vec2 = egui::vec2(1200.0, 800.0);

/// Physical pixels per point: not 1, so a listing that forgot to convert
/// would be caught.
const PIXELS_PER_POINT: f32 = 1.5;

/// One viewer-shaped UI, driven a frame at a time.
struct Harness {
    ctx: egui::Context,
    state: AppState,
    scene_graph: SceneGraphPanel,
}

/// The tab bodies: the real Scene panel, and a label for every other panel.
struct Tabs<'a> {
    state: &'a mut AppState,
    scene_graph: &'a mut SceneGraphPanel,
}

impl TabViewer for Tabs<'_> {
    type Tab = Tab;

    // The viewer's own tab id, so the body ids are the ones the viewer has.
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
            // A combo box inside a panel, as Image Detail's toolbar has.
            Tab::BackgroundTask => {
                egui::ComboBox::from_id_salt("test_combo")
                    .selected_text("One")
                    .show_ui(ui, |ui| {
                        let _ = ui.selectable_label(true, "One");
                        let _ = ui.selectable_label(false, "Two");
                    });
            }
            other => {
                ui.label(format!("{} body", other.title()));
            }
        }
    }
}

impl Harness {
    /// Two loaded reconstructions, `demo` and `other`, so the `Align to`
    /// submenu has a target, with the Scene panel in front.
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
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.set_pixels_per_point(PIXELS_PER_POINT);
        let mut harness = Self {
            ctx,
            state,
            scene_graph: SceneGraphPanel::new(),
        };
        // egui resolves input against the previous pass's rectangles, so
        // nothing can be pressed before a frame has been laid out.
        harness.settle();
        harness
    }

    fn window_px(&self) -> [u32; 2] {
        [
            (SCREEN.x * PIXELS_PER_POINT) as u32,
            (SCREEN.y * PIXELS_PER_POINT) as u32,
        ]
    }

    /// Draw one frame with `events` delivered, and capture it.
    fn frame(&mut self, events: Vec<egui::Event>) -> WidgetFrame {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
            events,
            ..Default::default()
        };
        let Self {
            ctx,
            state,
            scene_graph,
        } = self;
        let mut output = ctx.run_ui(input, |root_ui| {
            egui::Panel::top("menu_bar").show(root_ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        let _ = ui.button("Open...");
                        let _ = ui.button("Close");
                    });
                });
            });
            let ctx = root_ui.ctx().clone();
            let _ = state.bundle_adjust_prompt.show(&ctx);
            let _ = state
                .goto_point
                .show(&ctx, &state.scene, state.selected_recon);
            egui::CentralPanel::default().show(root_ui, |ui| {
                let mut dock =
                    std::mem::replace(&mut state.dock, egui_dock::DockState::new(Vec::new()));
                DockArea::new(&mut dock).show_inside(ui, &mut Tabs { state, scene_graph });
                state.dock = dock;
            });
        });
        output.textures_delta.clear();
        let window_px = self.window_px();
        WidgetFrame::capture(
            &self.ctx,
            output.platform_output.accesskit_update.as_ref(),
            &self.state.dock,
            window_px,
        )
    }

    fn settle(&mut self) -> WidgetFrame {
        self.frame(Vec::new());
        self.frame(Vec::new())
    }

    /// A point in window pixels as the window point egui's input takes.
    fn to_points(&self, [x, y]: [f64; 2]) -> egui::Pos2 {
        egui::pos2(x as f32 / PIXELS_PER_POINT, y as f32 / PIXELS_PER_POINT)
    }

    /// Press and release `button` at a point of the window, in window pixels,
    /// the pointer having moved there first, and settle.
    fn click_at(&mut self, at_px: [f64; 2], button: egui::PointerButton) -> WidgetFrame {
        let pos = self.to_points(at_px);
        let event = |pressed| egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        self.frame(vec![egui::Event::PointerMoved(pos)]);
        self.frame(vec![event(true)]);
        self.frame(vec![event(false)]);
        self.settle()
    }

    /// Move the pointer to a point of the window, in window pixels, and settle.
    fn hover_at(&mut self, at_px: [f64; 2]) -> WidgetFrame {
        let pos = self.to_points(at_px);
        self.frame(vec![egui::Event::PointerMoved(pos)]);
        self.settle()
    }

    /// The Scene panel body's rectangle in window pixels, as a screenshot of
    /// the panel crops it.
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
}

/// The listing of `panel`, whole.
#[track_caller]
fn listing(frame: &WidgetFrame, panel: Option<Tab>) -> Value {
    frame.listing(panel, None).expect("the target is laid out")
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

/// The centre of an entry's `rect_px`, in the pixels of its target.
fn centre(entry: &Value) -> [f64; 2] {
    let rect: Vec<f64> = entry["rect_px"]
        .as_array()
        .expect("a rect_px")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    [rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0]
}

fn widget_ids(widgets: &Value) -> Vec<String> {
    widgets
        .as_array()
        .expect("a widgets array")
        .iter()
        .map(|entry| entry["widget"].as_str().expect("an id").to_string())
        .collect()
}

fn hex(id: egui::Id) -> String {
    format!("{:016x}", id.value())
}

// ── The coordinate space ────────────────────────────────────────────────

/// A panel listing is in the panel body's pixels: a widget's rectangle there,
/// offset by the body's origin, is its rectangle in the window's listing, and
/// the target's size is the size a screenshot of the panel comes back at.
#[test]
fn a_panel_listing_is_in_the_panel_body_s_pixels() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let body = harness.scene_body_px();
    let window = listing(&frame, None);
    let scene = listing(&frame, Some(Tab::SceneGraph));

    assert_eq!(scene["target"]["panel_name"], json!("scene"));
    assert_eq!(scene["target"]["size_px"], json!([body[2], body[3]]));
    assert_eq!(window["target"]["panel_name"], Value::Null);
    assert_eq!(window["target"]["size_px"], json!(harness.window_px()));

    let row = hex(harness.demo_row());
    let in_panel = scene["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(row))
        .expect("the demo row is in the Scene panel's listing");
    let in_window = window["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(row))
        .expect("the demo row is in the window's listing");
    let offset = |entry: &Value, dx: i64, dy: i64| {
        let r: Vec<i64> = entry["rect_px"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect();
        [r[0] + dx, r[1] + dy, r[2], r[3]]
    };
    assert_eq!(
        offset(in_panel, i64::from(body[0]), i64::from(body[1])),
        offset(in_window, 0, 0),
    );
    assert_eq!(in_panel["sense"], json!("click"));
    assert_eq!(in_panel["enabled"], json!(true));
    // Converted to pixels, not left in points: the panel's own record of the
    // row, in points, scaled by the frame's pixels per point.
    let points = harness
        .scene_graph
        .hit_rect(harness.demo_row())
        .expect("the panel recorded the row");
    let expected = [
        points.min.x * PIXELS_PER_POINT,
        points.min.y * PIXELS_PER_POINT,
        points.width() * PIXELS_PER_POINT,
        points.height() * PIXELS_PER_POINT,
    ];
    for (axis, want) in expected.into_iter().enumerate() {
        let got = in_window["rect_px"][axis].as_f64().unwrap() as f32;
        assert!(
            (got - want).abs() <= 1.0,
            "rect_px {} against {expected:?} from the panel's points",
            in_window["rect_px"]
        );
    }

    // And the listing of the panel holds only what is in it: nothing from the
    // menu bar.
    assert!(named(&scene["widgets"], "File").is_empty(), "{scene:#}");
    assert_eq!(named(&window["widgets"], "File").len(), 1);
}

/// `crop_px` keeps the widgets that overlap it, each whole, and one that
/// reaches outside the target is refused with the target's size.
#[test]
fn crop_px_narrows_the_listing_and_is_refused_outside_the_target() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let whole = listing(&frame, Some(Tab::SceneGraph));
    let row = whole["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(hex(harness.demo_row())))
        .unwrap()
        .clone();
    let [x, y] = centre(&row);
    let crop = [x as u32, y as u32, 2, 2];
    let cropped = frame
        .listing(Some(Tab::SceneGraph), Some(crop))
        .expect("a crop inside the body");
    assert_eq!(cropped["target"]["crop_px"], json!(crop));
    let ids = widget_ids(&cropped["widgets"]);
    assert!(ids.contains(&hex(harness.demo_row())), "{cropped:#}");
    assert!(ids.len() < widget_ids(&whole["widgets"]).len());
    // Listed with its whole rectangle, not the part inside the crop.
    let listed = cropped["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(hex(harness.demo_row())))
        .unwrap();
    assert_eq!(listed["rect_px"], row["rect_px"]);

    let body = harness.scene_body_px();
    let error = frame
        .listing(Some(Tab::SceneGraph), Some([body[2] - 10, 0, 20, 20]))
        .expect_err("reaches past the right edge");
    assert!(
        error.0.contains(&format!("{}×{}", body[2], body[3])),
        "{error}"
    );
    assert!(error.0.contains("Scene panel"), "{error}");
}

// ── Dock tabs ───────────────────────────────────────────────────────────

/// Each dock tab is listed as a `tab` named for its panel and carrying the
/// wire name, found through `egui_dock`'s own id for it.
#[test]
fn dock_tabs_are_listed_with_the_panel_they_raise() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let window = listing(&frame, None);
    let tabs: Vec<&Value> = window["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["role"] == json!("tab"))
        .collect();
    let open = harness.state.dock.iter_all_tabs().count();
    assert_eq!(tabs.len(), open, "one entry per open tab: {tabs:#?}");
    let scene = tabs
        .iter()
        .find(|entry| entry["panel_name"] == json!("scene"))
        .expect("the Scene tab");
    assert_eq!(scene["name"], json!("Scene"));
    assert_eq!(scene["sense"], json!("click_and_drag"));
    // Tabs are above the bodies, so a panel listing has none.
    let panel = listing(&frame, Some(Tab::SceneGraph));
    assert!(panel["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .all(|entry| entry["role"] != json!("tab")));
}

// ── Menus ───────────────────────────────────────────────────────────────

/// A right click on a Scene row opens one context menu, owned by that row and
/// placed by where the click landed, since the row's click target has no name
/// of its own; its items are in the menu and not in the main list.
#[test]
fn a_right_click_on_a_scene_row_lists_one_context_menu_owned_by_the_row() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let window = listing(&frame, None);
    let row = window["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(hex(harness.demo_row())))
        .unwrap()
        .clone();
    let at = centre(&row);
    let frame = harness.click_at(at, egui::PointerButton::Secondary);

    let scene = listing(&frame, Some(Tab::SceneGraph));
    let menus = scene["menus"].as_array().unwrap();
    assert_eq!(menus.len(), 1, "{scene:#}");
    let menu = &menus[0];
    assert_eq!(menu["kind"], json!("context_menu"));
    assert_eq!(menu["owner"]["widget"], json!(hex(harness.demo_row())));
    assert_eq!(menu["owner"]["panel_name"], json!("scene"));
    assert_eq!(menu["owner"]["name"], Value::Null);
    // In the panel's pixels, where the click was aimed.
    let body = harness.scene_body_px();
    let at_px = &menu["owner"]["at_px"];
    let expected = [at[0] - f64::from(body[0]), at[1] - f64::from(body[1])];
    for axis in 0..2 {
        let got = at_px[axis].as_f64().expect("at_px");
        assert!(
            (got - expected[axis]).abs() <= 1.0,
            "{at_px} vs {expected:?}"
        );
    }

    let items = &menu["widgets"];
    let close = the(items, "Close");
    assert_eq!(close["role"], json!("button"));
    assert_eq!(close["enabled"], json!(true));
    let align = the(items, "Align to");
    assert_eq!(align["submenu"], json!(true), "{align:#}");
    assert!(close.get("submenu").is_none());
    // The menu's widgets are its own, not repeated in the panel's list.
    let main = widget_ids(&scene["widgets"]);
    for id in widget_ids(items) {
        assert!(!main.contains(&id), "{id} is listed twice");
    }
    // And the caption a screenshot would lead with says so.
    assert_eq!(
        frame.open_sentence().as_deref(),
        Some("A context menu in the Scene panel is open.")
    );

    // Hovering `Align to` opens its submenu, owned by that item.
    let align_centre = centre(align);
    let frame = harness.hover_at([
        align_centre[0] + f64::from(body[0]),
        align_centre[1] + f64::from(body[1]),
    ]);
    let window = listing(&frame, None);
    let menus = window["menus"].as_array().unwrap();
    assert_eq!(menus.len(), 2, "{window:#}");
    let submenu = menus
        .iter()
        .find(|menu| menu["kind"] == json!("submenu"))
        .expect("a submenu");
    assert_eq!(submenu["owner"]["name"], json!("Align to"));
    assert_eq!(submenu["owner"]["widget"], align["widget"]);
    // Its items carry the chain that reached them.
    let other = the(&submenu["widgets"], "other");
    let path: Vec<&str> = other["path"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert_eq!(path.last(), Some(&"other"));
    assert!(path.contains(&"Align to"), "{path:?}");
}

/// A menu bar menu is a `menu` owned by its button, and its items' paths start
/// with the menu's name, which is what tells `File ▸ Close` from any other
/// `Close`.
#[test]
fn a_menu_bar_menu_is_owned_by_its_button_and_names_its_items() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let file = the(&listing(&frame, None)["widgets"], "File").clone();
    assert_eq!(file["role"], json!("button"));
    let frame = harness.click_at(centre(&file), egui::PointerButton::Primary);

    let window = listing(&frame, None);
    let menus = window["menus"].as_array().unwrap();
    assert_eq!(menus.len(), 1, "{window:#}");
    assert_eq!(menus[0]["kind"], json!("menu"));
    assert_eq!(menus[0]["owner"]["name"], json!("File"));
    assert_eq!(menus[0]["owner"]["widget"], file["widget"]);
    assert!(menus[0]["owner"].get("panel_name").is_none());
    let close = the(&menus[0]["widgets"], "Close");
    assert_eq!(close["path"], json!(["File", "Close"]));
    assert_eq!(
        frame.open_sentence().as_deref(),
        Some("The File menu is open.")
    );
    // A panel the menu does not reach does not list it.
    let scene = listing(&frame, Some(Tab::SceneGraph));
    let covers = |rect: &Value| {
        let body = harness.scene_body_px();
        let r: Vec<i64> = rect
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect();
        r[0] < i64::from(body[2]) && r[0] + r[2] > 0 && r[1] < i64::from(body[3]) && r[1] + r[3] > 0
    };
    let in_window = &menus[0]["rect_px"];
    let body = harness.scene_body_px();
    let shifted = json!([
        in_window[0].as_i64().unwrap() - i64::from(body[0]),
        in_window[1].as_i64().unwrap() - i64::from(body[1]),
        in_window[2],
        in_window[3],
    ]);
    assert_eq!(
        scene["menus"].as_array().unwrap().len(),
        usize::from(covers(&shifted)),
        "{scene:#}"
    );
}

/// A combo box inside a panel opens a `dropdown`, owned by the box and placed
/// in its panel.
#[test]
fn a_combo_box_in_a_panel_opens_a_dropdown() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let panel = listing(&frame, Some(Tab::BackgroundTask));
    let combo = panel["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["role"] == json!("combo_box"))
        .expect("the combo box")
        .clone();
    let body = super::super::panel_rect::panel_crop(
        &harness.state.dock,
        Tab::BackgroundTask,
        PIXELS_PER_POINT,
        harness.window_px(),
    )
    .expect("the panel is laid out");
    let [x, y] = centre(&combo);
    let frame = harness.click_at(
        [x + f64::from(body[0]), y + f64::from(body[1])],
        egui::PointerButton::Primary,
    );
    let window = listing(&frame, None);
    let menus = window["menus"].as_array().unwrap();
    assert_eq!(menus.len(), 1, "{window:#}");
    assert_eq!(menus[0]["kind"], json!("dropdown"));
    assert_eq!(menus[0]["owner"]["widget"], combo["widget"]);
    assert_eq!(menus[0]["owner"]["panel_name"], json!("background_task"));
    assert_eq!(named(&menus[0]["widgets"], "Two").len(), 1, "{window:#}");
}

// ── Dialogs ─────────────────────────────────────────────────────────────

/// An open dialog is in `dialogs`, with its widgets, and none of them is in
/// the main list; a floating dock window would not be.
#[test]
fn a_dialog_is_listed_in_dialogs_and_not_in_widgets() {
    let mut harness = Harness::new();
    harness.state.open_goto_point();
    let frame = harness.settle();
    let window = listing(&frame, None);
    let dialogs = window["dialogs"].as_array().unwrap();
    assert_eq!(dialogs.len(), 1, "{window:#}");
    assert_eq!(dialogs[0]["title"], json!("Go to Point"));
    let own = &dialogs[0]["widgets"];
    let field = own
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["role"] == json!("text_input"))
        .expect("the point field");
    assert!(field.get("value").is_some(), "{field:#}");
    // Go is greyed until something is typed; the field was left empty.
    let go = the(own, "Go");
    assert_eq!(go["path"], json!(["Go to Point", "Go"]));
    let main = widget_ids(&window["widgets"]);
    for id in widget_ids(own) {
        assert!(!main.contains(&id), "{id} is listed twice");
    }
    assert_eq!(
        frame.open_sentence().as_deref(),
        Some("The Go to Point dialog is open.")
    );
    // The dialog opens with the field focused, which is what the key tools
    // of the next phase report.
    let focused = frame.focused().expect("a focused widget");
    assert_eq!(hex(focused.id), field["widget"].as_str().unwrap());
}

/// Two buttons with one name in two containers are told apart by `path`.
#[test]
fn path_tells_two_buttons_of_one_name_apart() {
    let mut harness = Harness::new();
    let demo = harness.state.scene[0].id;
    harness.state.open_bundle_adjust(demo);
    harness.state.open_goto_point();
    let frame = harness.settle();
    let window = listing(&frame, None);
    let dialogs = window["dialogs"].as_array().unwrap();
    assert_eq!(dialogs.len(), 2, "{window:#}");
    let mut paths: Vec<Value> = dialogs
        .iter()
        .flat_map(|dialog| named(&dialog["widgets"], "Cancel"))
        .map(|entry| entry["path"].clone())
        .collect();
    paths.sort_by_key(|path| path.to_string());
    assert_eq!(
        paths,
        [
            json!(["Bundle Adjust", "Cancel"]),
            json!(["Go to Point", "Cancel"])
        ]
    );
    let sentence = frame.open_sentence().expect("two dialogs are open");
    assert!(sentence.ends_with("dialogs are open."), "{sentence}");
}

// ── What the input tools will resolve through ───────────────────────────

/// A widget id finds its rectangle in any target, and a point finds the
/// topmost widget that can be clicked there — the row, not the label drawn on
/// it, and the menu, once one covers the row.
#[test]
fn a_point_finds_the_topmost_clickable_widget() {
    let mut harness = Harness::new();
    let frame = harness.settle();
    let row = harness.demo_row();
    let in_panel = frame
        .rect_px(row, Some(Tab::SceneGraph))
        .expect("the row is drawn");
    let listed = listing(&frame, Some(Tab::SceneGraph));
    let entry = listed["widgets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["widget"] == json!(hex(row)))
        .unwrap();
    assert_eq!(entry["rect_px"], json!(in_panel));

    let window = frame.rect_px(row, None).unwrap();
    let centre_px = [
        window[0] as f32 + window[2] as f32 / 2.0,
        window[1] as f32 + window[3] as f32 / 2.0,
    ];
    assert_eq!(frame.topmost_at(centre_px).map(|w| w.id), Some(row));

    // Once a context menu is open, a point inside it finds a menu item.
    let frame = harness.click_at(
        [f64::from(centre_px[0]), f64::from(centre_px[1])],
        egui::PointerButton::Secondary,
    );
    let window = listing(&frame, None);
    let menu = &window["menus"][0];
    let close = the(&menu["widgets"], "Close");
    let [x, y] = centre(close);
    let hit = frame.topmost_at([x as f32, y as f32]).expect("an item");
    assert_eq!(hex(hit.id), close["widget"].as_str().unwrap());
}

// ── The tool ────────────────────────────────────────────────────────────

/// `get_widgets` parses, defers to the frame, and is recorded as a read.
#[test]
fn get_widgets_defers_and_is_recorded_as_a_query() {
    let (mut state, mut viewer) = two_reconstructions();
    state.show_panel(Tab::SceneGraph);
    let command = tools::parse(
        "get_widgets",
        json!({ "panel_name": "scene", "crop_px": [0, 0, 100, 50] }).as_object(),
    )
    .expect("a valid call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Widgets { panel, crop, frame }) => {
            assert_eq!(panel, Some(Tab::SceneGraph));
            assert_eq!(crop, Some([0, 0, 100, 50]));
            assert!(frame.is_none());
        }
        _ => panic!("get_widgets must defer to the frame"),
    }
    let entry = state.action_log.entries().last().expect("one row");
    assert_eq!(entry.text, "get_widgets scene crop 0,0 100×50");
    assert_eq!(entry.actor, Actor::Mcp);
}

/// The refusals: a closed panel and one behind a tab, in screenshot's words
/// for a listing; a minimized window; a crop outside the window, naming its
/// size; a crop with no area; `widgets: true` without the HUD.
#[test]
fn get_widgets_and_screenshot_refuse_what_they_cannot_list() {
    let (mut state, mut viewer) = two_reconstructions();
    state.hide_panel(Tab::ActionLog);
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_widgets",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(
        error.0,
        "The Action Log panel is closed, so there is nothing of it to list. Send show_panel \
         { \"panel_name\": \"action_log\" } first."
    );
    let failed = state.action_log.entries().last().expect("a row");
    assert!(failed.failed, "a refusal writes a failed row");

    // Behind a tab: any node of the default layout with two tabs in it.
    let (behind, in_front) = state
        .dock
        .iter_leaves()
        .find_map(|(_, leaf)| {
            let front = *leaf.tabs.get(leaf.active.0)?;
            let behind = *leaf.tabs.iter().find(|tab| **tab != front)?;
            Some((behind, front))
        })
        .expect("the default layout tabs one panel behind another");
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_widgets",
        json!({ "panel_name": behind.wire_name() }),
    );
    assert!(
        error.0.starts_with(&format!(
            "The {} panel is behind {} in its node, so a listing of it would be a listing of \
             {}.",
            behind.title(),
            in_front.title(),
            in_front.title()
        )),
        "{error}"
    );

    let [width, height] = state.window.as_ref().unwrap().inner_size;
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_widgets",
        json!({ "crop_px": [width - 10, 0, 20, 20] }),
    );
    assert!(error.0.contains(&format!("{width}×{height}")), "{error}");
    let error = refused_call(
        &mut state,
        &mut viewer,
        "screenshot",
        json!({ "crop_px": [0, height - 1, 10, 10] }),
    );
    assert!(error.0.contains(&format!("{width}×{height}")), "{error}");
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_widgets",
        json!({ "crop_px": [0, 0, 0, 10] }),
    );
    assert!(error.0.contains("no area"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "screenshot",
        json!({ "panel_name": "viewer_3d", "hud": false, "widgets": true }),
    );
    assert!(error.0.contains("no widgets on it"), "{error}");

    state.window = Some(crate::test_support::FakeWindow::in_state(WindowState::Minimized).info());
    let error = refused_call(&mut state, &mut viewer, "get_widgets", json!({}));
    assert!(error.0.starts_with("The window is minimized"), "{error}");
}

/// A screenshot with `crop_px` and `widgets` defers with both, and its log row
/// says what was asked for.
#[test]
fn a_screenshot_with_a_crop_and_widgets_defers_with_both() {
    let (mut state, mut viewer) = two_reconstructions();
    let command = tools::parse(
        "screenshot",
        json!({ "crop_px": [10, 20, 300, 200], "widgets": true }).as_object(),
    )
    .expect("a valid call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Screenshot {
            crop,
            widgets,
            caption,
            ..
        }) => {
            assert_eq!(crop, Some([10, 20, 300, 200]));
            assert!(widgets);
            assert!(
                caption.contains("cropped to [10, 20, 300, 200]"),
                "{caption}"
            );
        }
        _ => panic!("a screenshot must defer"),
    }
    let entry = state.action_log.entries().last().expect("one row");
    assert_eq!(
        entry.text,
        "screenshot window crop 10,20 300×200 300×200 with widgets"
    );
}

// ── The wire vocabulary ─────────────────────────────────────────────────

/// Every field name a listing carries is one the spec settles on, spelled
/// out: no abbreviation, and pixel quantities ending `_px`.
#[test]
fn a_listing_speaks_the_wire_vocabulary() {
    let mut harness = Harness::new();
    harness.state.open_goto_point();
    harness.settle();
    let frame = harness.settle();
    let window = listing(&frame, None);
    let row = frame.rect_px(harness.demo_row(), None).unwrap();
    let frame = harness.click_at(
        [
            f64::from(row[0] as i32) + row[2] as f64 / 2.0,
            f64::from(row[1] as i32) + row[3] as f64 / 2.0,
        ],
        egui::PointerButton::Secondary,
    );
    let with_menu = listing(&frame, None);

    let allowed = [
        "target",
        "panel_name",
        "size_px",
        "crop_px",
        "dialogs",
        "menus",
        "widgets",
        "widget",
        "role",
        "name",
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
        "at_px",
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
    keys(&window, &mut seen);
    keys(&with_menu, &mut seen);
    for key in &seen {
        assert!(
            allowed.contains(&key.as_str()),
            "{key:?} is not in the vocabulary"
        );
    }
    for required in ["dialogs", "menus", "owner", "at_px", "rect_px", "path"] {
        assert!(seen.contains(required), "no listing carried {required:?}");
    }
    // The ids are 16 lowercase hex digits.
    for id in widget_ids(&window["widgets"]) {
        assert_eq!(id.len(), 16, "{id}");
        assert!(id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
    // Roles are AccessKit's in snake case, or null.
    for entry in window["widgets"].as_array().unwrap() {
        if let Some(role) = entry["role"].as_str() {
            assert!(
                role.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "{role}"
            );
        }
    }
}
