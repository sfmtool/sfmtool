// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The widget listing: what egui drew in one frame, as `get_widgets` and a
//! `screenshot` with `widgets: true` report it.
//!
//! A [`WidgetFrame`] is captured once, after an egui pass, in a frame that has
//! a listing waiting for it and in no other. It is built from three things egui
//! already holds, all public API:
//!
//! - the pass's `WidgetRects` (every widget's id, layer, clipped interaction
//!   rectangle, `Sense` and `enabled` flag), read from `prev_pass` because
//!   `Context::run_ui` swaps the finished pass there when it ends;
//! - the frame's AccessKit `TreeUpdate`, which carries each widget's role,
//!   label, toggled state and value under `Id::accesskit_id`, and the parent
//!   links that make up a widget's `path`;
//! - egui's layer order and area memory, which say which layers are popups and
//!   windows and where they are.
//!
//! Everything is kept in window points until a listing is asked for, and only
//! then converted into the physical pixels of one target (the window or a
//! panel's body). That keeps one captured frame usable by every request that
//! was waiting on it, whatever each one targets.
//!
//! The listing is also what the input tools in [`super::input`] resolve
//! against (a widget id to its rectangle, a point to the widget on top of it,
//! the focused widget), which is why those lookups are here rather than in the
//! tools that use them.

use std::collections::HashMap;

use egui::accesskit::{self, NodeId, Role, Toggled};
use serde_json::{json, Map, Value};

use super::panel_rect::panel_crop;
use super::ToolError;
use crate::dock::Tab;

/// One captured frame of widgets, dialogs and menus.
pub(crate) struct WidgetFrame {
    /// Physical pixels per logical point, as this frame used them.
    pixels_per_point: f32,
    /// The window's size in physical pixels, the space `rect_px` of a
    /// window-scoped listing is in.
    window_px: [u32; 2],
    /// Every listed widget, in drawing order: back layer first, and within a
    /// layer in the order egui registered them.
    widgets: Vec<Widget>,
    /// The dialogs and menus drawn above the dock, in drawing order.
    surfaces: Vec<Surface>,
    /// The body of each panel that is in front in its node, in window pixels,
    /// already clipped to the window as a screenshot crop is.
    panels: Vec<(Tab, [u32; 4])>,
    /// The widget with keyboard focus, if any.
    focused: Option<egui::Id>,
}

/// One widget of a captured frame.
pub(crate) struct Widget {
    pub(crate) id: egui::Id,
    /// Which dialog or menu the widget is in, as an index into
    /// `WidgetFrame::surfaces`; `None` for the menu bar, the dock and its
    /// panels, floating dock windows included.
    surface: Option<usize>,
    /// The interaction rectangle after egui's clipping, in window points.
    rect: egui::Rect,
    sense: egui::Sense,
    enabled: bool,
    /// `None` where the widget has no AccessKit node, or one whose role is
    /// `Unknown`: nothing in the tree says what it is.
    role: Option<Role>,
    name: Option<String>,
    toggled: Option<Toggled>,
    value: Option<String>,
    /// Named AccessKit ancestors, outermost first, ending with the widget's own
    /// name where it has one. For a widget in a menu, the owner's path comes
    /// first, since egui gives a popup no AccessKit link to what opened it.
    path: Vec<String>,
    /// Whether this item opens a submenu.
    submenu: bool,
    /// For a dock tab, the panel it raises.
    tab: Option<Tab>,
}

impl Widget {
    /// The AccessKit label, if it has one.
    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Whether this is a text input, the one kind of widget text can be typed
    /// into.
    pub(crate) fn takes_text(&self) -> bool {
        self.role == Some(Role::TextInput)
    }

    /// The role as the listing spells it, or `None` where the tree says
    /// nothing.
    pub(crate) fn role_text(&self) -> Option<String> {
        self.role.map(role_name)
    }
}

/// A dialog or a menu: one egui area above the dock.
struct Surface {
    rect: egui::Rect,
    kind: SurfaceKind,
}

enum SurfaceKind {
    Dialog {
        title: String,
    },
    Menu {
        kind: MenuKind,
        /// The widget the menu hangs from, where one could be found.
        owner: Option<egui::Id>,
        /// Where the right click that opened a context menu landed, in window
        /// points.
        at: Option<egui::Pos2>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuKind {
    /// Opened from the menu bar.
    Menu,
    /// Opened by a right click.
    ContextMenu,
    /// Hanging from an item of another menu.
    Submenu,
    /// Opened from a combo box or a menu button inside a panel.
    Dropdown,
}

impl MenuKind {
    fn wire_name(self) -> &'static str {
        match self {
            MenuKind::Menu => "menu",
            MenuKind::ContextMenu => "context_menu",
            MenuKind::Submenu => "submenu",
            MenuKind::Dropdown => "dropdown",
        }
    }
}

/// The arrow `egui::containers::menu::SubMenuButton` puts to the right of a
/// submenu item's text. AccessKit reads the button's text atoms joined by a
/// space, so the arrow arrives at the end of the label.
const SUBMENU_ARROW: &str = egui::containers::menu::SubMenuButton::RIGHT_ARROW;

impl WidgetFrame {
    /// Read the frame egui has just finished.
    ///
    /// Called after `Context::run_ui` and before `handle_platform_output`
    /// consumes the AccessKit update. `window_px` is the size of the surface
    /// the frame is presented on, so a panel's body is clipped to it exactly as
    /// the screenshot crop is.
    pub(crate) fn capture(
        ctx: &egui::Context,
        update: Option<&accesskit::TreeUpdate>,
        dock: &egui_dock::DockState<Tab>,
        window_px: [u32; 2],
    ) -> Self {
        let pixels_per_point = ctx.pixels_per_point();
        let tree = Tree::new(update);

        // Layers in the order egui paints them. The background layer is not
        // an area, so it is not in egui's area order; it is painted first.
        let area_order: Vec<egui::LayerId> = ctx.memory(|memory| memory.layer_ids().collect());
        let mut layers: Vec<(egui::LayerId, Vec<egui::WidgetRect>)> = ctx.viewport(|viewport| {
            viewport
                .prev_pass
                .widgets
                .layers()
                .map(|(layer, rects)| (*layer, rects.to_vec()))
                .collect()
        });
        layers.sort_by_key(|(layer, _)| {
            (
                layer.order,
                area_order
                    .iter()
                    .position(|other| other == layer)
                    .map_or(0, |position| position + 1),
            )
        });

        // What opened each popup: `egui::Popup` names its area after the
        // widget it was built from, `id.with("popup")`, and a submenu names
        // its own `id.with("submenu")`. So the owner is found by id, over
        // every widget of the frame.
        let mut popup_owner: HashMap<egui::Id, (egui::Id, bool)> = HashMap::new();
        for (_, rects) in &layers {
            for rect in rects {
                popup_owner.insert(popup_id(rect.id), (rect.id, false));
                popup_owner.insert(
                    egui::containers::menu::SubMenu::id_from_widget_id(rect.id),
                    (rect.id, true),
                );
            }
        }

        // The floating dock windows are egui windows too, but they hold
        // panels, not a dialog. `egui_dock` names each one after its surface.
        let dock_windows: Vec<egui::Id> = dock
            .iter_surfaces_indexed()
            .filter(|(_, surface)| matches!(surface, egui_dock::Surface::Window(..)))
            .map(|(index, _)| egui::Id::new(format!("window {index:?}")))
            .collect();

        let panels: Vec<(Tab, [u32; 4])> = Tab::ALL
            .into_iter()
            .filter(|panel| panel_in_front(dock, *panel))
            .filter_map(|panel| {
                panel_crop(dock, panel, pixels_per_point, window_px).map(|rect| (panel, rect))
            })
            .collect();
        // The same bodies in points, for deciding which panel a widget is in.
        let panel_points: Vec<(Tab, egui::Rect)> = Tab::ALL
            .into_iter()
            .filter(|panel| panel_in_front(dock, *panel))
            .filter_map(|panel| crate::dock::panel_body_points(dock, panel).map(|r| (panel, r)))
            .collect();

        let mut surfaces: Vec<Surface> = Vec::new();
        let mut widgets: Vec<Widget> = Vec::new();
        for (layer, rects) in &layers {
            // Tooltips appear after a hover delay and cannot be clicked, and
            // the debug layer is egui's own.
            if matches!(layer.order, egui::Order::Tooltip | egui::Order::Debug) {
                continue;
            }
            // An area registers itself as a widget under this id: the area as
            // a whole, which a listing reports as the dialog or menu rather
            // than as one of its widgets.
            let area_widget = layer.id.with("move");
            let area_rect = ctx
                .memory(|memory| memory.area_rect(layer.id))
                .map(|rect| to_global(ctx, *layer, rect));
            let surface = match layer.order {
                // A foreground area that no widget of the frame opened is not
                // a popup, and its widgets stay in the main list rather than
                // being reported as a menu nobody opened.
                egui::Order::Foreground => popup_owner.get(&layer.id).map(|&(owner, sub)| {
                    let at = egui::Popup::position_of_id(ctx, layer.id);
                    let kind = if sub {
                        MenuKind::Submenu
                    } else if at.is_some() {
                        // Opened at the pointer: `Popup::context_menu` and
                        // `context_menu::on_secondary_click` both do that, and
                        // a menu or combo box opens against its button.
                        MenuKind::ContextMenu
                    } else if find_rect(&layers, owner)
                        .is_some_and(|rect| in_a_panel(&panel_points, &dock_windows, ctx, rect))
                    {
                        MenuKind::Dropdown
                    } else {
                        MenuKind::Menu
                    };
                    Surface {
                        rect: area_rect.unwrap_or(egui::Rect::NOTHING),
                        kind: SurfaceKind::Menu {
                            kind,
                            owner: Some(owner),
                            at,
                        },
                    }
                }),
                egui::Order::Middle if !dock_windows.contains(&layer.id) => tree
                    .node(area_widget)
                    .filter(|node| node.role() == Role::Window)
                    .and_then(|node| node.label().map(str::to_string))
                    .filter(|title| !title.is_empty())
                    .map(|title| Surface {
                        rect: area_rect.unwrap_or(egui::Rect::NOTHING),
                        kind: SurfaceKind::Dialog { title },
                    }),
                _ => None,
            };
            let surface_index = surface.map(|surface| {
                surfaces.push(surface);
                surfaces.len() - 1
            });
            let transform = ctx.layer_transform_to_global(*layer);
            let listed_before = widgets.len();
            for rect in rects {
                if rect.id == area_widget && layer.order != egui::Order::Background {
                    continue;
                }
                let node = tree.node(rect.id);
                // Every `Ui` registers a rectangle too. One with no node and
                // nothing to click is layout, not a widget.
                if node.is_none() && !rect.sense.interactive() {
                    continue;
                }
                let global = transform.map_or(rect.interact_rect, |t| t * rect.interact_rect);
                if !global.is_positive() || !global.is_finite() {
                    continue;
                }
                let role = node.map(|node| node.role()).filter(|r| *r != Role::Unknown);
                let raw_name = node.and_then(node_name);
                // egui gives most `Ui`s a node of their own, as a generic
                // container, so that its children have a parent in the tree.
                // One with no name and nothing to click is still layout.
                if role == Some(Role::GenericContainer)
                    && raw_name.is_none()
                    && !rect.sense.interactive()
                {
                    continue;
                }
                let (name, arrow) = match raw_name {
                    Some(name) => match name.strip_suffix(SUBMENU_ARROW) {
                        Some(stripped) => (Some(stripped.trim_end().to_string()), true),
                        None => (Some(name), false),
                    },
                    None => (None, false),
                };
                // Read from how egui built the item: a `SubMenuButton` ends its
                // text with the arrow, and an open submenu is an area named
                // after the item.
                let submenu_layer = egui::containers::menu::SubMenu::id_from_widget_id(rect.id);
                let submenu = arrow || layers.iter().any(|(other, _)| other.id == submenu_layer);
                let path = tree.path(rect.id, name.as_deref());
                widgets.push(Widget {
                    id: rect.id,
                    surface: surface_index,
                    rect: global,
                    sense: rect.sense,
                    enabled: rect.enabled,
                    role,
                    name,
                    toggled: node.and_then(|node| node.toggled()),
                    value: node.and_then(|node| node_value(node, role)),
                    path,
                    submenu,
                    tab: None,
                });
            }
            // A menu an item has just closed is still drawn for the frame
            // after, as an empty area. Nothing in it can be pressed, and a
            // reply listing it would say a menu is open that is closing.
            if let Some(index) = surface_index {
                let empty = widgets.len() == listed_before;
                if empty && matches!(surfaces[index].kind, SurfaceKind::Menu { .. }) {
                    surfaces.pop();
                }
            }
        }

        // Dock tabs: `egui_dock` interacts with each tab under an id built
        // from its surface, node and position in the node, so the dock's own
        // arrangement names each tab's widget.
        let dock_area = egui::Id::new("egui_dock::DockArea");
        for (path, tab) in dock.iter_all_tabs() {
            let id = dock_area
                .with((path.surface, "surface"))
                .with((path.node, "node"))
                .with((path.tab.0, "tab"));
            if let Some(widget) = widgets.iter_mut().find(|widget| widget.id == id) {
                widget.role = Some(Role::Tab);
                widget.name = Some(tab.title().to_string());
                widget.path = vec![tab.title().to_string()];
                widget.tab = Some(*tab);
            }
        }

        let mut frame = WidgetFrame {
            pixels_per_point,
            window_px,
            widgets,
            surfaces,
            panels,
            focused: ctx.memory(|memory| memory.focused()),
        };
        frame.prefix_menu_paths();
        frame
    }

    /// Put each menu item's owner chain in front of its own path.
    ///
    /// egui parents a popup's widgets to the popup area, and the area to the
    /// AccessKit root: nothing in the tree says that `Save As...` is in the
    /// File menu. The owner found by id does, so the listing adds it.
    fn prefix_menu_paths(&mut self) {
        let prefixes: Vec<Vec<String>> = (0..self.surfaces.len())
            .map(|surface| self.owner_path(surface, 0))
            .collect();
        for widget in &mut self.widgets {
            if let Some(prefix) = widget.surface.and_then(|surface| prefixes.get(surface)) {
                if !prefix.is_empty() {
                    let mut path = prefix.clone();
                    path.append(&mut widget.path);
                    widget.path = path;
                }
            }
        }
    }

    /// The full path of the widget a menu hangs from, or nothing for a dialog
    /// or an ownerless popup.
    fn owner_path(&self, surface: usize, depth: usize) -> Vec<String> {
        // A menu cannot own itself, but a malformed frame should not recurse
        // for ever finding out.
        if depth > 8 {
            return Vec::new();
        }
        let SurfaceKind::Menu {
            owner: Some(owner), ..
        } = &self.surfaces[surface].kind
        else {
            return Vec::new();
        };
        let Some(widget) = self.find(*owner) else {
            return Vec::new();
        };
        let mut path = widget
            .surface
            .map(|outer| self.owner_path(outer, depth + 1))
            .unwrap_or_default();
        path.extend(widget.path.iter().cloned());
        path
    }

    /// The widget with this id, if the frame drew it.
    pub(crate) fn find(&self, id: egui::Id) -> Option<&Widget> {
        self.widgets.iter().find(|widget| widget.id == id)
    }

    /// The widget whose id has this value, which is how the wire spells an id.
    pub(crate) fn find_value(&self, value: u64) -> Option<&Widget> {
        self.widgets
            .iter()
            .find(|widget| widget.id.value() == value)
    }

    /// Every named widget of the frame, as its id's value and its name.
    pub(crate) fn names(&self) -> impl Iterator<Item = (u64, &str)> {
        self.widgets
            .iter()
            .filter_map(|widget| Some((widget.id.value(), widget.name.as_deref()?)))
    }

    /// The widgets of the frame named `name`.
    pub(crate) fn named(&self, name: &str) -> Vec<&Widget> {
        self.widgets
            .iter()
            .filter(|widget| widget.name.as_deref() == Some(name))
            .collect()
    }

    /// The topmost widget that can be clicked under a point in window pixels.
    ///
    /// Topmost is last drawn, which is how egui resolves a press.
    pub(crate) fn topmost_at(&self, point_px: [f32; 2]) -> Option<&Widget> {
        let point = self.to_points(point_px);
        self.widgets
            .iter()
            .rev()
            .find(|widget| widget.sense.interactive() && widget.rect.contains(point))
    }

    /// The name a person reads at a point in window pixels: the topmost
    /// clickable widget's own, or where that has none, the topmost named
    /// widget drawn over the point. A Scene row's click target is unnamed and
    /// its label is a separate widget on top of it.
    pub(crate) fn name_at(&self, point_px: [f32; 2]) -> Option<&str> {
        if let Some(name) = self
            .topmost_at(point_px)
            .and_then(|widget| widget.name.as_deref())
        {
            return Some(name);
        }
        let point = self.to_points(point_px);
        self.widgets
            .iter()
            .rev()
            .filter(|widget| widget.rect.contains(point))
            .find_map(|widget| widget.name.as_deref())
    }

    /// What is drawn over `widget` at a point in window pixels, described, or
    /// `None` when a press there would reach it.
    ///
    /// Two things can be over it: a clickable widget drawn after it, which
    /// egui's hit test prefers, and a dialog or menu drawn above the one it is
    /// in, whose background takes the press even where it has no widget. A
    /// widget drawn before it that cannot be clicked, such as the label on a
    /// Scene row, does not cover it.
    pub(crate) fn covering(&self, widget: &Widget, point_px: [f32; 2]) -> Option<String> {
        let own = self
            .widgets
            .iter()
            .position(|other| other.id == widget.id)?;
        if let Some(top) = self.topmost_at(point_px) {
            let top_index = self.widgets.iter().position(|other| other.id == top.id)?;
            if top_index > own {
                return Some(self.describe(top));
            }
        }
        let point = self.to_points(point_px);
        let first_above = widget.surface.map_or(0, |surface| surface + 1);
        self.surfaces
            .iter()
            .enumerate()
            .skip(first_above)
            .find(|(_, surface)| surface.rect.contains(point))
            .map(|(index, _)| self.surface_phrase(index))
    }

    /// A widget as a sentence names it: `"Close" in a context menu in the Scene
    /// panel`, or `an unnamed widget in the Scene panel`.
    pub(crate) fn describe(&self, widget: &Widget) -> String {
        let what = match (&widget.name, widget.role) {
            (Some(name), _) => format!("\"{name}\""),
            (None, Some(role)) => format!("an unnamed {}", role_name(role).replace('_', " ")),
            (None, None) => "an unnamed widget".to_string(),
        };
        format!("{what} {}", self.place(widget))
    }

    /// Where a widget is, as the end of a sentence: `in the Go to Point
    /// dialog`, `in the File menu`, `in the Scene panel`, or `in the window`.
    pub(crate) fn place(&self, widget: &Widget) -> String {
        match widget.surface {
            Some(surface) => format!("in {}", self.surface_phrase(surface)),
            None => match self.panel_of(widget.rect.center()) {
                Some(panel) => format!("in the {} panel", panel.title()),
                None => "in the window".to_string(),
            },
        }
    }

    /// The `hit` block of a pointer tool's reply: the widget's id, role, name
    /// and the panel it is in, or `null` for empty space.
    pub(crate) fn hit_block(&self, widget: Option<&Widget>) -> Value {
        let Some(widget) = widget else {
            return Value::Null;
        };
        let panel = widget
            .surface
            .is_none()
            .then(|| self.panel_of(widget.rect.center()))
            .flatten();
        json!({
            "widget": widget_id_text(widget.id),
            "role": widget.role.map_or(Value::Null, |role| json!(role_name(role))),
            "name": widget.name,
            "panel_name": panel.map_or(Value::Null, |panel| json!(panel.wire_name())),
        })
    }

    /// The widget with keyboard focus, if the frame drew it.
    pub(crate) fn focused(&self) -> Option<&Widget> {
        self.focused.and_then(|id| self.find(id))
    }

    /// The focused widget as a listing entry of `panel`'s target, or `null`.
    pub(crate) fn focused_entry(&self, panel: Option<Tab>) -> Value {
        let Some(widget) = self.focused() else {
            return Value::Null;
        };
        match self.target_px(panel).or_else(|_| self.target_px(None)) {
            Ok(target) => self.entry(widget, target),
            Err(_) => Value::Null,
        }
    }

    /// A widget's rectangle in the target's pixels. The input tools aim at a
    /// widget's centre in window pixels ([`Self::centre_px`]); the tests read
    /// rectangles in a panel's pixels through this.
    #[cfg(test)]
    pub(crate) fn rect_px(&self, id: egui::Id, panel: Option<Tab>) -> Option<[i64; 4]> {
        let target = self.target_px(panel).ok()?;
        self.find(id)
            .map(|widget| self.to_target(widget.rect, target))
    }

    /// The centre of a widget's rectangle, in window pixels.
    pub(crate) fn centre_px(&self, widget: &Widget) -> [f32; 2] {
        let centre = widget.rect.center();
        let scale = self.scale();
        [centre.x * scale, centre.y * scale]
    }

    /// A point in window pixels as the window point egui's input takes.
    pub(crate) fn to_points(&self, [x, y]: [f32; 2]) -> egui::Pos2 {
        let scale = self.scale();
        egui::pos2(x / scale, y / scale)
    }

    /// The target's rectangle in window pixels: the whole window, or the body
    /// of one panel as the screenshot crops it.
    pub(crate) fn target_px(&self, panel: Option<Tab>) -> Result<[u32; 4], ToolError> {
        match panel {
            None => Ok([0, 0, self.window_px[0], self.window_px[1]]),
            Some(panel) => self
                .panels
                .iter()
                .find(|(tab, _)| *tab == panel)
                .map(|(_, rect)| *rect)
                .ok_or_else(|| {
                    ToolError::new(format!(
                        "The {} panel has not been laid out in the window, so there is no \
                         rectangle to list the widgets of. It may have been closed since the \
                         call was made.",
                        panel.title()
                    ))
                }),
        }
    }

    fn scale(&self) -> f32 {
        if self.pixels_per_point > 0.0 {
            self.pixels_per_point
        } else {
            1.0
        }
    }

    /// A rectangle in window points as whole pixels of `target`, each edge
    /// rounded on its own as the panel crop rounds them.
    fn to_target(&self, rect: egui::Rect, target: [u32; 4]) -> [i64; 4] {
        let scale = self.scale();
        let left = (rect.min.x * scale).round() as i64 - i64::from(target[0]);
        let top = (rect.min.y * scale).round() as i64 - i64::from(target[1]);
        let right = (rect.max.x * scale).round() as i64 - i64::from(target[0]);
        let bottom = (rect.max.y * scale).round() as i64 - i64::from(target[1]);
        [left, top, right - left, bottom - top]
    }

    /// The listing for one target, as `get_widgets` returns it.
    ///
    /// `crop` is a rectangle of the target, already checked against the size
    /// the target had when the call was applied; it is checked again here,
    /// against the frame, since a layout change can land in between.
    pub(crate) fn listing(
        &self,
        panel: Option<Tab>,
        crop: Option<[u32; 4]>,
    ) -> Result<Value, ToolError> {
        let target = self.target_px(panel)?;
        let area = match crop {
            None => target,
            Some(crop) => {
                check_crop(crop, [target[2], target[3]], panel)?;
                [target[0] + crop[0], target[1] + crop[1], crop[2], crop[3]]
            }
        };
        let overlaps = |rect: egui::Rect| {
            let scale = self.scale();
            let left = rect.min.x * scale;
            let top = rect.min.y * scale;
            let right = rect.max.x * scale;
            let bottom = rect.max.y * scale;
            left < (area[0] + area[2]) as f32
                && right > area[0] as f32
                && top < (area[1] + area[3]) as f32
                && bottom > area[1] as f32
        };

        let widgets: Vec<Value> = self
            .widgets
            .iter()
            .filter(|widget| widget.surface.is_none() && overlaps(widget.rect))
            .map(|widget| self.entry(widget, target))
            .collect();

        let mut dialogs = Vec::new();
        let mut menus = Vec::new();
        for (index, surface) in self.surfaces.iter().enumerate() {
            if !overlaps(surface.rect) {
                continue;
            }
            let own: Vec<Value> = self
                .widgets
                .iter()
                .filter(|widget| widget.surface == Some(index))
                .map(|widget| self.entry(widget, target))
                .collect();
            let rect_px = self.to_target(surface.rect, target);
            match &surface.kind {
                SurfaceKind::Dialog { title } => dialogs.push(json!({
                    "title": title,
                    "rect_px": rect_px,
                    "widgets": own,
                })),
                SurfaceKind::Menu { kind, owner, at } => menus.push(json!({
                    "kind": kind.wire_name(),
                    "owner": self.owner_block(*kind, *owner, *at, target),
                    "rect_px": rect_px,
                    "widgets": own,
                })),
            }
        }

        let mut target_block = Map::new();
        target_block.insert(
            "panel_name".into(),
            panel.map_or(Value::Null, |panel| json!(panel.wire_name())),
        );
        target_block.insert("size_px".into(), json!([target[2], target[3]]));
        if let Some(crop) = crop {
            target_block.insert("crop_px".into(), json!(crop));
        }
        Ok(json!({
            "target": Value::Object(target_block),
            "dialogs": dialogs,
            "menus": menus,
            "widgets": widgets,
        }))
    }

    /// One widget as a listing entry.
    fn entry(&self, widget: &Widget, target: [u32; 4]) -> Value {
        let mut entry = Map::new();
        entry.insert("widget".into(), json!(widget_id_text(widget.id)));
        entry.insert(
            "role".into(),
            widget
                .role
                .map_or(Value::Null, |role| json!(role_name(role))),
        );
        entry.insert("name".into(), json!(widget.name));
        entry.insert("rect_px".into(), json!(self.to_target(widget.rect, target)));
        entry.insert("enabled".into(), json!(widget.enabled));
        if matches!(widget.role, Some(Role::CheckBox | Role::RadioButton)) {
            entry.insert(
                "toggled".into(),
                match widget.toggled {
                    Some(Toggled::True) => json!(true),
                    Some(Toggled::False) => json!(false),
                    // An indeterminate check box is neither.
                    Some(Toggled::Mixed) | None => Value::Null,
                },
            );
        }
        if matches!(
            widget.role,
            Some(Role::TextInput | Role::Slider | Role::SpinButton)
        ) {
            entry.insert("value".into(), json!(widget.value));
        }
        entry.insert("sense".into(), json!(sense_name(widget.sense)));
        entry.insert("path".into(), json!(widget.path));
        if widget.submenu {
            entry.insert("submenu".into(), json!(true));
        }
        if let Some(tab) = widget.tab {
            entry.insert("panel_name".into(), json!(tab.wire_name()));
        }
        Value::Object(entry)
    }

    /// The `owner` block of a menu.
    fn owner_block(
        &self,
        kind: MenuKind,
        owner: Option<egui::Id>,
        at: Option<egui::Pos2>,
        target: [u32; 4],
    ) -> Value {
        let widget = owner.and_then(|id| self.find(id));
        let Some(widget) = widget else {
            return Value::Null;
        };
        let mut block = Map::new();
        // A submenu's owner is an item of another menu, which may lie over a
        // panel without being in it.
        if let Some(panel) = widget
            .surface
            .is_none()
            .then(|| self.panel_of(widget.rect.center()))
            .flatten()
        {
            block.insert("panel_name".into(), json!(panel.wire_name()));
        }
        block.insert("widget".into(), json!(widget_id_text(widget.id)));
        block.insert("name".into(), json!(widget.name));
        // A context menu opened on something with no name of its own -- the
        // Image Browser strip, the viewport, the photograph -- is placed by
        // where the click landed.
        if kind == MenuKind::ContextMenu && widget.name.is_none() {
            if let Some(at) = at {
                let scale = self.scale();
                block.insert(
                    "at_px".into(),
                    json!([
                        (at.x * scale).round() as i64 - i64::from(target[0]),
                        (at.y * scale).round() as i64 - i64::from(target[1]),
                    ]),
                );
            }
        }
        Value::Object(block)
    }

    /// The panel whose body holds a point in window points.
    fn panel_of(&self, point: egui::Pos2) -> Option<Tab> {
        let scale = self.scale();
        let [x, y] = [point.x * scale, point.y * scale];
        self.panels
            .iter()
            .find(|(_, [left, top, width, height])| {
                x >= *left as f32
                    && y >= *top as f32
                    && x < (left + width) as f32
                    && y < (top + height) as f32
            })
            .map(|(tab, _)| *tab)
    }

    /// The sentence a screenshot caption leads with when a dialog or a menu
    /// is open: `The Bundle Adjust dialog is open.`, or `None` when nothing is.
    pub(crate) fn open_sentence(&self) -> Option<String> {
        let dialogs: Vec<&str> = self
            .surfaces
            .iter()
            .filter_map(|surface| match &surface.kind {
                SurfaceKind::Dialog { title } => Some(title.as_str()),
                SurfaceKind::Menu { .. } => None,
            })
            .collect();
        let mut sentences = Vec::new();
        match dialogs.as_slice() {
            [] => {}
            [one] => sentences.push(format!("The {one} dialog is open.")),
            [first @ .., last] => sentences.push(format!(
                "The {} and {last} dialogs are open.",
                first.join(", ")
            )),
        }
        for (index, surface) in self.surfaces.iter().enumerate() {
            if !matches!(surface.kind, SurfaceKind::Menu { .. }) {
                continue;
            }
            let phrase = self.surface_phrase(index);
            let mut letters = phrase.chars();
            let capitalized: String = letters
                .next()
                .map(|first| first.to_uppercase().chain(letters).collect())
                .unwrap_or_default();
            sentences.push(format!("{capitalized} is open."));
        }
        (!sentences.is_empty()).then(|| sentences.join(" "))
    }

    /// A dialog or a menu as a sentence names it: `the Go to Point dialog`,
    /// `the File menu`, `a context menu in the Scene panel`.
    fn surface_phrase(&self, index: usize) -> String {
        let (kind, owner) = match &self.surfaces[index].kind {
            SurfaceKind::Dialog { title } => return format!("the {title} dialog"),
            SurfaceKind::Menu { kind, owner, .. } => (*kind, owner),
        };
        let owner = owner.and_then(|id| self.find(id));
        let name = owner.and_then(|widget| widget.name.as_deref());
        let panel = owner.and_then(|widget| self.panel_of(widget.rect.center()));
        match (kind, name) {
            (MenuKind::Menu, Some(name)) => format!("the {name} menu"),
            (MenuKind::Submenu, Some(name)) => format!("the {name} submenu"),
            (MenuKind::ContextMenu, Some(name)) => format!("a context menu on \"{name}\""),
            (MenuKind::ContextMenu, None) => match panel {
                Some(panel) => format!("a context menu in the {} panel", panel.title()),
                None => "a context menu".to_string(),
            },
            (MenuKind::Dropdown, Some(name)) => format!("the \"{name}\" dropdown"),
            (_, None) => format!("a {}", kind.wire_name().replace('_', " ")),
        }
    }
}

/// Refuse a crop that reaches outside the target, naming the target's size.
///
/// A crop that was silently clipped would hand back a smaller picture, or a
/// shorter listing, than the one asked for, with nothing to say so.
pub(crate) fn check_crop(
    crop: [u32; 4],
    size: [u32; 2],
    panel: Option<Tab>,
) -> Result<(), ToolError> {
    let [x, y, width, height] = crop.map(u64::from);
    if x + width <= u64::from(size[0]) && y + height <= u64::from(size[1]) {
        return Ok(());
    }
    let target = match panel {
        None => "The window".to_string(),
        Some(panel) => format!("The {} panel's body", panel.title()),
    };
    Err(ToolError::new(format!(
        "crop_px [{x}, {y}, {width}, {height}] reaches outside the target. {target} is \
         {}×{} pixels, so a crop has to fit inside [0, 0, {}, {}].",
        size[0], size[1], size[0], size[1]
    )))
}

/// Whether `panel` is open and the tab in front in its node.
fn panel_in_front(dock: &egui_dock::DockState<Tab>, panel: Tab) -> bool {
    let Some(path) = dock.find_tab(&panel) else {
        return false;
    };
    dock.leaf(path.node_path())
        .is_ok_and(|leaf| leaf.tabs.get(leaf.active.0) == Some(&panel))
}

/// A widget rectangle's layer and interaction rectangle, found by id.
fn find_rect(
    layers: &[(egui::LayerId, Vec<egui::WidgetRect>)],
    id: egui::Id,
) -> Option<(egui::LayerId, egui::Rect)> {
    layers
        .iter()
        .flat_map(|(_, rects)| rects.iter())
        .find(|rect| rect.id == id)
        .map(|rect| (rect.layer_id, rect.interact_rect))
}

/// Whether a widget sits in a docked panel's body or in a floating dock
/// window, rather than in the menu bar.
fn in_a_panel(
    panels: &[(Tab, egui::Rect)],
    dock_windows: &[egui::Id],
    ctx: &egui::Context,
    (layer, rect): (egui::LayerId, egui::Rect),
) -> bool {
    if dock_windows.contains(&layer.id) {
        return true;
    }
    let center = to_global(ctx, layer, rect).center();
    panels.iter().any(|(_, body)| body.contains(center))
}

fn to_global(ctx: &egui::Context, layer: egui::LayerId, rect: egui::Rect) -> egui::Rect {
    ctx.layer_transform_to_global(layer)
        .map_or(rect, |transform| transform * rect)
}

/// The area id of a popup opened from the widget `owner`.
///
/// `egui::Popup::default_response_id` and `egui::ComboBox` both name a popup
/// `owner.with("popup")`. The former takes a whole `Response`, which a captured
/// frame does not have, so the rule is restated here; a headless test opens a
/// real context menu and checks that its owner is found through it.
fn popup_id(owner: egui::Id) -> egui::Id {
    owner.with("popup")
}

/// The id as the wire spells it: 16 lowercase hex digits of egui's `Id`,
/// which is also the widget's AccessKit node id.
pub(crate) fn widget_id_text(id: egui::Id) -> String {
    format!("{:016x}", id.value())
}

/// An AccessKit role in snake case: `CheckBox` is `check_box`.
fn role_name(role: Role) -> String {
    let camel = format!("{role:?}");
    let mut snake = String::with_capacity(camel.len() + 4);
    for (i, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                snake.push('_');
            }
            snake.push(c.to_ascii_lowercase());
        } else {
            snake.push(c);
        }
    }
    snake
}

/// What a click on a widget with this sense can do.
///
/// egui's `Sense` has no bit for hovering (`Sense::HOVER` is the empty set),
/// so a widget that neither clicks nor drags is `none`, whether it shows a
/// tooltip or only text.
fn sense_name(sense: egui::Sense) -> &'static str {
    match (sense.senses_click(), sense.senses_drag()) {
        (true, true) => "click_and_drag",
        (true, false) => "click",
        (false, true) => "drag",
        (false, false) => "none",
    }
}

/// A node's name: its label, or for a label widget its text, which egui puts
/// in the value.
fn node_name(node: &accesskit::Node) -> Option<String> {
    let name = match node.role() {
        Role::Label => node.value().or_else(|| node.label()),
        _ => node.label(),
    }?;
    let name = name.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// A text input's text, or a slider's or drag value's number, as a string.
fn node_value(node: &accesskit::Node, role: Option<Role>) -> Option<String> {
    match role {
        Some(Role::TextInput) => node.value().map(str::to_string),
        Some(Role::Slider | Role::SpinButton) => node
            .numeric_value()
            .map(|value| value.to_string())
            .or_else(|| node.value().map(str::to_string)),
        _ => None,
    }
}

/// The frame's AccessKit tree, indexed both ways.
struct Tree<'a> {
    nodes: HashMap<NodeId, &'a accesskit::Node>,
    parents: HashMap<NodeId, NodeId>,
}

impl<'a> Tree<'a> {
    fn new(update: Option<&'a accesskit::TreeUpdate>) -> Self {
        let mut nodes = HashMap::new();
        let mut parents = HashMap::new();
        for (id, node) in update.map(|update| update.nodes.as_slice()).unwrap_or(&[]) {
            nodes.insert(*id, node);
            for child in node.children() {
                parents.insert(*child, *id);
            }
        }
        Self { nodes, parents }
    }

    fn node(&self, id: egui::Id) -> Option<&'a accesskit::Node> {
        self.nodes.get(&id.accesskit_id()).copied()
    }

    /// The names of a widget's named ancestors, outermost first, and its own.
    fn path(&self, id: egui::Id, own: Option<&str>) -> Vec<String> {
        let mut path: Vec<String> = Vec::new();
        let mut current = id.accesskit_id();
        // The tree is acyclic by construction; the bound is for a malformed
        // update, which should give a short path rather than a hang.
        for _ in 0..64 {
            let Some(parent) = self.parents.get(&current) else {
                break;
            };
            if let Some(name) = self.nodes.get(parent).and_then(|node| node_name(node)) {
                let name = name
                    .strip_suffix(SUBMENU_ARROW)
                    .map(|stripped| stripped.trim_end().to_string())
                    .unwrap_or(name);
                path.push(name);
            }
            current = *parent;
        }
        path.reverse();
        path.extend(own.map(str::to_string));
        path
    }
}
