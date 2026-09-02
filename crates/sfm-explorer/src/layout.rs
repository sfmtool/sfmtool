// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Panel layout: the arrangement of the dock, and its file.
//!
//! See `specs/gui/panel-layout.md`. Three things live here:
//!
//! - [`Layout`], a description of a dock arrangement that is independent of
//!   `egui_dock`'s node indices — a split tree of panel names, readable and
//!   writable by hand, and the schema of the layout file.
//! - The conversions [`Layout::from_dock`] / [`Layout::to_dock`], the one place
//!   in the crate that knows how `egui_dock` represents a tree.
//! - The panel operations on [`AppState`] — [`AppState::show_panel`],
//!   [`AppState::hide_panel`], [`AppState::reset_layout`],
//!   [`AppState::apply_layout`] — which is where the Action Log is in reach,
//!   and the Panels menu that drives them ([`panels_menu`]).
//!
//! The JSON is read and written by hand rather than through `serde`: a node is
//! a leaf or a split by which keys it carries, and `serde` does not honour
//! `deny_unknown_fields` on an untagged enum, so a derive could not refuse a
//! typo in `"fraction"`.

use egui_dock::{DockState, LeafNode, Node, NodeIndex, Split, Surface, TabIndex, Tree};
use serde_json::Value;

use crate::action_log::Kind;
use crate::dock::Tab;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// The `sfm_explorer_layout` value written, and the only one read.
pub(crate) const LAYOUT_VERSION: u64 = 1;

/// The tab a placeholder leaf carries while [`Layout::to_dock`] grows the tree.
///
/// `Tree::split` refuses a leaf with no tabs, so a subtree cannot be installed
/// empty and then filled; every placeholder is overwritten before the dock is
/// handed back.
const PLACEHOLDER: Tab = Tab::Viewer3D;

// ── The schema ───────────────────────────────────────────────────────────

/// A dock arrangement, as the layout file spells it.
///
/// Serializable, buildable by hand, and independent of `egui_dock`'s node
/// indices. `main` is the docked arrangement — `None` when nothing is docked in
/// the main surface — and `windows` the floating surfaces.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Layout {
    pub main: Option<LayoutNode>,
    pub windows: Vec<LayoutWindow>,
}

/// One node of a layout tree: a split of two children, or a leaf of tabs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayoutNode {
    /// `fraction` is `first`'s share of the node, in `(0, 1)` exclusive.
    Split {
        split: SplitDirection,
        fraction: f32,
        first: Box<LayoutNode>,
        second: Box<LayoutNode>,
    },
    /// The tabs in tab-bar order, and which one is in front.
    Leaf { tabs: Vec<Tab>, active: Tab },
}

/// How a split arranges its children — *not* which way its divider runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitDirection {
    /// `first` on the left, `second` on the right.
    LeftRight,
    /// `first` on top, `second` below.
    TopBottom,
}

impl SplitDirection {
    /// The wire spelling.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            SplitDirection::LeftRight => "left_right",
            SplitDirection::TopBottom => "top_bottom",
        }
    }

    /// The `egui_dock` split that keeps `fraction` meaning "`first`'s share".
    ///
    /// `Split::Left` and `Split::Above` swap the children round and would
    /// invert it.
    fn as_egui(self) -> Split {
        match self {
            SplitDirection::LeftRight => Split::Right,
            SplitDirection::TopBottom => Split::Below,
        }
    }
}

/// A floating surface: its tree, and where it sits when it has been laid out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayoutWindow {
    pub tree: LayoutNode,
    /// Screen-anchored, in logical points. `None` for a window that has never
    /// been drawn and so has no rect to report.
    pub rect: Option<LayoutRect>,
}

/// A window's rectangle in logical points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LayoutRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// One violation, with the path to the node it was found at.
///
/// `Display` is the message the Action Log records: `main.second.first: unknown
/// key "fracton"`, or just the message for a violation of the document itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayoutError {
    pub path: String,
    pub message: String,
}

impl LayoutError {
    fn at(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

impl std::error::Error for LayoutError {}

// ── `Tab`'s wire spelling, and where a panel calls home ──────────────────

/// Where a panel goes when it is opened into a layout that holds no home for
/// it. See `specs/gui/panel-layout.md` § "Home positions", rule 3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Home {
    /// The 3D viewport, which no edge is home to: it takes the root.
    Root,
    /// A split of the main surface's root along `edge`, the new node taking
    /// `share` of it.
    Edge { edge: Split, share: f32 },
}

/// The panels that share a node in the default layout, in default order.
///
/// A panel opened from the menu goes home to whichever of its group-mates is
/// still on screen (rule 2), which is what keeps the Point Track panel behind
/// Image Detail rather than beside it.
const GROUPS: [&[Tab]; 2] = [
    &[
        Tab::ImageDetail,
        Tab::PointTrackDetail,
        Tab::IntrinsicsDetail,
    ],
    &[Tab::ImageBrowser, Tab::ActionLog],
];

impl Tab {
    /// Every panel, in default-layout order — which is the Panels menu's order.
    pub(crate) const ALL: [Tab; 7] = [
        Tab::SceneGraph,
        Tab::Viewer3D,
        Tab::ImageBrowser,
        Tab::ImageDetail,
        Tab::PointTrackDetail,
        Tab::IntrinsicsDetail,
        Tab::ActionLog,
    ];

    /// The panel's name on the wire: its title, lower-cased and joined with
    /// underscores.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Tab::SceneGraph => "scene",
            Tab::Viewer3D => "viewer_3d",
            Tab::ImageBrowser => "image_browser",
            Tab::ImageDetail => "image_detail",
            Tab::PointTrackDetail => "point_track",
            Tab::IntrinsicsDetail => "camera_intrinsics",
            Tab::ActionLog => "action_log",
        }
    }

    /// The panel `name` spells, or `None`. Exact: neither `"Scene"` nor
    /// `"viewer3d"` is a panel.
    pub(crate) fn from_wire_name(name: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|tab| tab.wire_name() == name)
    }

    /// Every panel name, as an error message lists them.
    pub(crate) fn all_wire_names() -> String {
        Tab::ALL
            .iter()
            .map(|tab| tab.wire_name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Where this panel is put when nothing of its default group is open.
    pub(crate) fn home(self) -> Home {
        match self {
            Tab::Viewer3D => Home::Root,
            Tab::SceneGraph => Home::Edge {
                edge: Split::Left,
                share: 0.18,
            },
            Tab::ImageBrowser | Tab::ActionLog => Home::Edge {
                edge: Split::Below,
                share: 0.20,
            },
            Tab::ImageDetail | Tab::PointTrackDetail | Tab::IntrinsicsDetail => Home::Edge {
                edge: Split::Right,
                share: 0.33,
            },
        }
    }

    /// The panels this one shares a node with in the default layout, itself
    /// included. Empty for a panel that has a node to itself.
    fn group(self) -> &'static [Tab] {
        GROUPS
            .into_iter()
            .find(|group| group.contains(&self))
            .unwrap_or(&[])
    }
}

// ── The stock layout ─────────────────────────────────────────────────────

impl Default for Layout {
    /// The stock seven-panel grid:
    ///
    /// ```text
    /// ┌────────┬──────────────────┬───────────────┐
    /// │        │    3D Viewer     │ Image Detail  │
    /// │ Scene  ├──────────────────┴───────────────┤
    /// │        │ Image Browser │ Action Log       │
    /// └────────┴──────────────────────────────────┘
    /// ```
    ///
    /// The Scene tab takes a narrow left split of the root — narrow because the
    /// tree is a list of short labels and everything else in the window wants
    /// the width. Two nodes hold more than one tab, and in both the first is
    /// the active one: the bottom node opens on the Image Browser with the
    /// Action Log behind it, and the right-hand node on Image Detail.
    ///
    /// `Layout::default().to_dock()` is what the viewer starts with, and what
    /// Panels ▸ Reset Layout restores.
    fn default() -> Self {
        Layout {
            main: Some(LayoutNode::Split {
                split: SplitDirection::LeftRight,
                fraction: 0.18,
                first: Box::new(LayoutNode::leaf(&[Tab::SceneGraph])),
                second: Box::new(LayoutNode::Split {
                    split: SplitDirection::TopBottom,
                    fraction: 0.8,
                    first: Box::new(LayoutNode::Split {
                        split: SplitDirection::LeftRight,
                        fraction: 0.67,
                        first: Box::new(LayoutNode::leaf(&[Tab::Viewer3D])),
                        second: Box::new(LayoutNode::leaf(&[
                            Tab::ImageDetail,
                            Tab::PointTrackDetail,
                            Tab::IntrinsicsDetail,
                        ])),
                    }),
                    second: Box::new(LayoutNode::leaf(&[Tab::ImageBrowser, Tab::ActionLog])),
                }),
            }),
            windows: Vec::new(),
        }
    }
}

impl LayoutNode {
    /// A leaf opening on its first tab, which is how a `Vec`-built
    /// `egui_dock` leaf opens too.
    fn leaf(tabs: &[Tab]) -> Self {
        LayoutNode::Leaf {
            tabs: tabs.to_vec(),
            active: tabs[0],
        }
    }
}

// ── Reading and writing a live dock ──────────────────────────────────────

impl Layout {
    /// Read the arrangement out of a live dock.
    ///
    /// Tolerates what `egui_dock` leaves behind: the node heap is padded to a
    /// full level with `Node::Empty`, and a surface emptied of tabs reports no
    /// root at all.
    pub(crate) fn from_dock(dock: &DockState<Tab>) -> Self {
        let main = read_node(dock.main_surface(), NodeIndex::root());
        let mut windows = Vec::new();
        for (index, surface) in dock.iter_surfaces_indexed() {
            if index.is_main() {
                continue;
            }
            let Surface::Window(tree, window_state) = surface else {
                continue;
            };
            let Some(tree) = read_node(tree, NodeIndex::root()) else {
                continue;
            };
            let rect = window_state.rect();
            windows.push(LayoutWindow {
                tree,
                // `Rect::NOTHING` is what a window that has never been laid out
                // reports; there is no rect to write, so the field is omitted
                // rather than invented.
                rect: rect.is_finite().then(|| LayoutRect {
                    x: rect.min.x,
                    y: rect.min.y,
                    width: rect.width(),
                    height: rect.height(),
                }),
            });
        }
        Layout { main, windows }
    }

    /// Build a dock from a layout.
    ///
    /// Assumes [`Layout::validate`] has passed: an empty leaf would trip
    /// `Tree::split`'s own assertion, and a duplicated panel would give two
    /// tabs one `egui::Id`.
    pub(crate) fn to_dock(&self) -> DockState<Tab> {
        let mut dock = DockState::new(vec![PLACEHOLDER]);
        match &self.main {
            Some(node) => place_node(dock.main_surface_mut(), NodeIndex::root(), node),
            // The all-closed state, spelled the way closing the last tab
            // spells it: `remove_leaf` on the root clears the node vector.
            None => dock.main_surface_mut().remove_leaf(NodeIndex::root()),
        }
        for window in &self.windows {
            let surface = dock.add_window(vec![PLACEHOLDER]);
            place_node(&mut dock[surface], NodeIndex::root(), &window.tree);
            if let Some(rect) = window.rect {
                if let Some(state) = dock.get_window_state_mut(surface) {
                    state
                        .set_position(egui::pos2(rect.x, rect.y))
                        .set_size(egui::vec2(rect.width, rect.height));
                }
            }
        }
        dock
    }

    /// Check every rule of `specs/gui/panel-layout.md` § "Validation" that
    /// survives parsing, naming the first violation.
    pub(crate) fn validate(&self) -> Result<(), LayoutError> {
        let mut seen = Vec::new();
        if let Some(main) = &self.main {
            validate_node(main, "main", &mut seen)?;
        }
        for (index, window) in self.windows.iter().enumerate() {
            validate_node(&window.tree, &format!("windows[{index}].tree"), &mut seen)?;
        }
        Ok(())
    }
}

/// Read the subtree rooted at `index`, or `None` where there is nothing to read.
fn read_node(tree: &Tree<Tab>, index: NodeIndex) -> Option<LayoutNode> {
    if index.0 >= tree.len() {
        return None;
    }
    match &tree[index] {
        Node::Empty => None,
        Node::Leaf(leaf) => {
            let &first = leaf.tabs.first()?;
            Some(LayoutNode::Leaf {
                tabs: leaf.tabs.clone(),
                active: leaf.tabs.get(leaf.active.0).copied().unwrap_or(first),
            })
        }
        Node::Horizontal(split) => join(tree, index, SplitDirection::LeftRight, split.fraction),
        Node::Vertical(split) => join(tree, index, SplitDirection::TopBottom, split.fraction),
    }
}

/// Read both children of the split at `index`.
///
/// A split with only one readable child collapses to that child: it cannot
/// arise from a dock `egui_dock` is maintaining, and a layout is better off
/// describing the panel that is there than a split around nothing.
fn join(
    tree: &Tree<Tab>,
    index: NodeIndex,
    split: SplitDirection,
    fraction: f32,
) -> Option<LayoutNode> {
    let first = read_node(tree, index.left());
    let second = read_node(tree, index.right());
    match (first, second) {
        (Some(first), Some(second)) => Some(LayoutNode::Split {
            split,
            fraction,
            first: Box::new(first),
            second: Box::new(second),
        }),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// Write `node` into `index`, growing the tree as the shape demands.
///
/// `egui_dock` has no constructor that takes a subtree: a `Tree` is grown by
/// `split`, which moves what is at `index` to the first child and installs the
/// new node as the second. So a split is written by splitting `index` with a
/// placeholder leaf and then letting each child overwrite its own index.
fn place_node(tree: &mut Tree<Tab>, index: NodeIndex, node: &LayoutNode) {
    match node {
        LayoutNode::Leaf { tabs, active } => {
            let mut leaf = LeafNode::new(tabs.clone());
            leaf.active = TabIndex(tabs.iter().position(|tab| tab == active).unwrap_or(0));
            tree[index] = Node::Leaf(leaf);
        }
        LayoutNode::Split {
            split,
            fraction,
            first,
            second,
        } => {
            let [a, b] = tree.split(index, split.as_egui(), *fraction, Node::leaf(PLACEHOLDER));
            place_node(tree, a, first);
            place_node(tree, b, second);
        }
    }
}

/// Validate one subtree, accumulating the panels seen so far across the whole
/// document so that a panel in `main` and in a window is caught too.
fn validate_node(node: &LayoutNode, path: &str, seen: &mut Vec<Tab>) -> Result<(), LayoutError> {
    match node {
        LayoutNode::Leaf { tabs, active } => {
            if tabs.is_empty() {
                return Err(LayoutError::at(path, "a leaf must have at least one tab"));
            }
            for tab in tabs {
                if seen.contains(tab) {
                    return Err(LayoutError::at(
                        path,
                        format!("panel \"{}\" appears more than once", tab.wire_name()),
                    ));
                }
                seen.push(*tab);
            }
            if !tabs.contains(active) {
                return Err(LayoutError::at(
                    path,
                    format!(
                        "active \"{}\" is not one of this leaf's tabs",
                        active.wire_name()
                    ),
                ));
            }
            Ok(())
        }
        LayoutNode::Split {
            fraction,
            first,
            second,
            ..
        } => {
            if !(*fraction > 0.0 && *fraction < 1.0) {
                return Err(LayoutError::at(
                    path,
                    format!("fraction must be strictly between 0 and 1, not {fraction}"),
                ));
            }
            validate_node(first, &format!("{path}.first"), seen)?;
            validate_node(second, &format!("{path}.second"), seen)
        }
    }
}

// ── The layout file ──────────────────────────────────────────────────────

impl Layout {
    /// Parse and validate a layout file.
    ///
    /// The document is refused as a whole: a caller that gets an `Err` has a
    /// layout it can leave exactly as it was.
    pub(crate) fn from_json(text: &str) -> Result<Self, LayoutError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|error| LayoutError::at("", format!("not valid JSON: {error}")))?;
        Layout::from_value(&value)
    }

    /// Validate an already-parsed layout document.
    ///
    /// The half of [`Layout::from_json`] below the JSON parse, so that a
    /// document that arrived as a `serde_json::Value` — an MCP `set_layout`
    /// argument — meets exactly the rules, and exactly the messages, a file on
    /// disk meets.
    pub(crate) fn from_value(value: &Value) -> Result<Self, LayoutError> {
        let Some(object) = value.as_object() else {
            return Err(LayoutError::at("", "the document must be a JSON object"));
        };
        // The version first, so a JSON file that is not a layout at all says so
        // rather than complaining about its own perfectly good keys.
        let Some(version) = object.get("sfm_explorer_layout").and_then(Value::as_u64) else {
            return Err(LayoutError::at("", "Not a layout file"));
        };
        if version > LAYOUT_VERSION {
            return Err(LayoutError::at(
                "",
                format!(
                    "Layout version {version} is newer than this viewer reads ({LAYOUT_VERSION})"
                ),
            ));
        }
        if version != LAYOUT_VERSION {
            return Err(LayoutError::at(
                "",
                format!("Layout version {version} is not one this viewer reads ({LAYOUT_VERSION})"),
            ));
        }
        known_keys(object, "", &["sfm_explorer_layout", "main", "windows"])?;

        let main = match object.get("main") {
            None | Some(Value::Null) => None,
            Some(value) => Some(node_from_json(value, "main")?),
        };
        let windows = match object.get("windows") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut windows = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    windows.push(window_from_json(item, &format!("windows[{index}]"))?);
                }
                windows
            }
            Some(_) => return Err(LayoutError::at("", "\"windows\" must be an array")),
        };

        let layout = Layout { main, windows };
        layout.validate()?;
        Ok(layout)
    }

    /// The layout as a file: pretty-printed, keys in schema order, one trailing
    /// newline.
    pub(crate) fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"sfm_explorer_layout\": {LAYOUT_VERSION},\n"));
        out.push_str("  \"main\": ");
        match &self.main {
            Some(node) => write_node(&mut out, node, 1),
            None => out.push_str("null"),
        }
        out.push_str(",\n");
        if self.windows.is_empty() {
            out.push_str("  \"windows\": []\n");
        } else {
            out.push_str("  \"windows\": [\n");
            for (index, window) in self.windows.iter().enumerate() {
                out.push_str("    ");
                write_window(&mut out, window, 2);
                out.push_str(if index + 1 == self.windows.len() {
                    "\n"
                } else {
                    ",\n"
                });
            }
            out.push_str("  ]\n");
        }
        out.push_str("}\n");
        out
    }
}

/// Refuse any key that is not one of `allowed`.
///
/// A typo silently applying a default would leave the author believing the file
/// says something it does not.
fn known_keys(
    object: &serde_json::Map<String, Value>,
    path: &str,
    allowed: &[&str],
) -> Result<(), LayoutError> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(LayoutError::at(path, format!("unknown key \"{key}\"")));
        }
    }
    Ok(())
}

/// One node: a leaf if it carries `tabs`, a split if it carries `split`.
fn node_from_json(value: &Value, path: &str) -> Result<LayoutNode, LayoutError> {
    let Some(object) = value.as_object() else {
        return Err(LayoutError::at(path, "a node must be a JSON object"));
    };
    if object.contains_key("tabs") {
        known_keys(object, path, &["tabs", "active"])?;
        let Some(Value::Array(items)) = object.get("tabs") else {
            return Err(LayoutError::at(path, "\"tabs\" must be an array of panels"));
        };
        let mut tabs = Vec::with_capacity(items.len());
        for item in items {
            tabs.push(tab_from_json(item, path)?);
        }
        let active = match object.get("active") {
            None | Some(Value::Null) => *tabs
                .first()
                .ok_or_else(|| LayoutError::at(path, "a leaf must have at least one tab"))?,
            Some(value) => tab_from_json(value, path)?,
        };
        return Ok(LayoutNode::Leaf { tabs, active });
    }
    if object.contains_key("split") {
        known_keys(object, path, &["split", "fraction", "first", "second"])?;
        let split = match object.get("split").and_then(Value::as_str) {
            Some("left_right") => SplitDirection::LeftRight,
            Some("top_bottom") => SplitDirection::TopBottom,
            _ => {
                return Err(LayoutError::at(
                    path,
                    format!(
                        "\"split\" must be \"left_right\" or \"top_bottom\", not {}",
                        describe(object.get("split")),
                    ),
                ))
            }
        };
        let Some(fraction) = object.get("fraction").and_then(Value::as_f64) else {
            return Err(LayoutError::at(path, "\"fraction\" must be a number"));
        };
        let first = object
            .get("first")
            .ok_or_else(|| LayoutError::at(path, "a split needs a \"first\" child"))?;
        let second = object
            .get("second")
            .ok_or_else(|| LayoutError::at(path, "a split needs a \"second\" child"))?;
        return Ok(LayoutNode::Split {
            split,
            fraction: fraction as f32,
            first: Box::new(node_from_json(first, &format!("{path}.first"))?),
            second: Box::new(node_from_json(second, &format!("{path}.second"))?),
        });
    }
    Err(LayoutError::at(
        path,
        "a node must have either \"tabs\" (a leaf) or \"split\" (a split)",
    ))
}

/// One panel name.
fn tab_from_json(value: &Value, path: &str) -> Result<Tab, LayoutError> {
    let Some(name) = value.as_str() else {
        return Err(LayoutError::at(path, "a panel name must be a string"));
    };
    Tab::from_wire_name(name).ok_or_else(|| {
        LayoutError::at(
            path,
            format!(
                "unknown panel \"{name}\"; the panels are {}",
                Tab::all_wire_names()
            ),
        )
    })
}

/// One floating window: its tree, and optionally where it sits.
fn window_from_json(value: &Value, path: &str) -> Result<LayoutWindow, LayoutError> {
    let Some(object) = value.as_object() else {
        return Err(LayoutError::at(path, "a window must be a JSON object"));
    };
    known_keys(object, path, &["tree", "rect"])?;
    let tree = object
        .get("tree")
        .ok_or_else(|| LayoutError::at(path, "a window needs a \"tree\""))?;
    let tree = node_from_json(tree, &format!("{path}.tree"))?;
    let rect = match object.get("rect") {
        None | Some(Value::Null) => None,
        Some(value) => Some(rect_from_json(value, &format!("{path}.rect"))?),
    };
    Ok(LayoutWindow { tree, rect })
}

/// A window rectangle, all four fields required.
fn rect_from_json(value: &Value, path: &str) -> Result<LayoutRect, LayoutError> {
    let Some(object) = value.as_object() else {
        return Err(LayoutError::at(path, "a rect must be a JSON object"));
    };
    known_keys(object, path, &["x", "y", "width", "height"])?;
    let field = |name: &str| -> Result<f32, LayoutError> {
        object
            .get(name)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .ok_or_else(|| LayoutError::at(path, format!("\"{name}\" must be a number")))
    };
    Ok(LayoutRect {
        x: field("x")?,
        y: field("y")?,
        width: field("width")?,
        height: field("height")?,
    })
}

/// A JSON value as an error message names it.
fn describe(value: Option<&Value>) -> String {
    match value {
        None => "nothing".to_string(),
        Some(value) => value.to_string(),
    }
}

/// Write `node` at the current position, indented for a parent at `depth`.
fn write_node(out: &mut String, node: &LayoutNode, depth: usize) {
    let inner = "  ".repeat(depth + 1);
    let outer = "  ".repeat(depth);
    out.push_str("{\n");
    match node {
        LayoutNode::Leaf { tabs, active } => {
            let names: Vec<String> = tabs
                .iter()
                .map(|tab| format!("\"{}\"", tab.wire_name()))
                .collect();
            out.push_str(&format!("{inner}\"tabs\": [{}],\n", names.join(", ")));
            out.push_str(&format!("{inner}\"active\": \"{}\"\n", active.wire_name()));
        }
        LayoutNode::Split {
            split,
            fraction,
            first,
            second,
        } => {
            out.push_str(&format!("{inner}\"split\": \"{}\",\n", split.wire_name()));
            out.push_str(&format!("{inner}\"fraction\": {fraction},\n"));
            out.push_str(&format!("{inner}\"first\": "));
            write_node(out, first, depth + 1);
            out.push_str(",\n");
            out.push_str(&format!("{inner}\"second\": "));
            write_node(out, second, depth + 1);
            out.push('\n');
        }
    }
    out.push_str(&outer);
    out.push('}');
}

/// Write one floating window at the current position.
fn write_window(out: &mut String, window: &LayoutWindow, depth: usize) {
    let inner = "  ".repeat(depth + 1);
    let outer = "  ".repeat(depth);
    out.push_str("{\n");
    out.push_str(&format!("{inner}\"tree\": "));
    write_node(out, &window.tree, depth + 1);
    match window.rect {
        Some(rect) => {
            out.push_str(",\n");
            out.push_str(&format!(
                "{inner}\"rect\": {{ \"x\": {}, \"y\": {}, \"width\": {}, \"height\": {} }}\n",
                rect.x, rect.y, rect.width, rect.height
            ));
        }
        None => out.push('\n'),
    }
    out.push_str(&outer);
    out.push('}');
}

// ── The panel operations ─────────────────────────────────────────────────

impl AppState {
    /// Whether `tab` is docked anywhere — the tick in the Panels menu.
    pub(crate) fn is_panel_open(&self, tab: Tab) -> bool {
        self.dock.find_tab(&tab).is_some()
    }

    /// Show `tab`, at its home position if it is not already open.
    ///
    /// The three rules of `specs/gui/panel-layout.md` § "Home positions", in
    /// order: raise it where it is; else push it in behind a default
    /// group-mate; else split the main surface's root along its home edge.
    /// Records `Raised …` or `Opened …`.
    pub(crate) fn show_panel(&mut self, tab: Tab) {
        // Rule 1: already open. Nothing moves; it comes to the front.
        if let Some(path) = self.dock.find_tab(&tab) {
            let _ = self.dock.set_active_tab(path);
            self.dock.set_focused_node_and_surface(path.node_path());
            self.action_log
                .record(Kind::Layout, format!("Raised {} panel", tab.title()));
            return;
        }
        // Rule 2: a default group-mate is open, so go in behind its tabs.
        let mate = tab
            .group()
            .iter()
            .filter(|&&mate| mate != tab)
            .find_map(|&mate| self.dock.find_tab(&mate));
        if let Some(path) = mate {
            if let Ok(leaf) = self.dock.leaf_mut(path.node_path()) {
                leaf.append_tab(tab);
                self.dock.set_focused_node_and_surface(path.node_path());
                self.record_opened(tab);
                return;
            }
        }
        // Rule 3: a split of the root along the panel's home edge. Any panel
        // opened into an empty dock becomes its root leaf, whatever its home
        // says — there is nothing to split.
        let surface = self.dock.main_surface_mut();
        if surface.root_node().is_none_or(Node::is_empty) {
            *surface = Tree::new(vec![tab]);
            self.record_opened(tab);
            return;
        }
        match tab.home() {
            // The viewport is what everything else is arranged around, and no
            // edge is home to it: it joins the root's first leaf instead.
            Home::Root => surface.push_to_first_leaf(tab),
            Home::Edge { edge, share } => {
                // `Tree::split` takes the *first* child's share, and the new
                // node is first only for Left and Above.
                let fraction = match edge {
                    Split::Left | Split::Above => share,
                    Split::Right | Split::Below => 1.0 - share,
                };
                surface.split(NodeIndex::root(), edge, fraction, Node::leaf(tab));
            }
        }
        self.record_opened(tab);
    }

    /// Close `tab`. A no-op, logged or otherwise, on a panel that is not open.
    pub(crate) fn hide_panel(&mut self, tab: Tab) {
        let Some(path) = self.dock.find_tab(&tab) else {
            return;
        };
        self.dock.remove_tab(path);
        self.record_closed(tab);
    }

    /// Replace the whole arrangement with the stock grid. Records
    /// `Reset layout`.
    pub(crate) fn reset_layout(&mut self) {
        self.dock = Layout::default().to_dock();
        self.action_log.record(Kind::Layout, "Reset layout");
    }

    /// Validate `layout`, then replace the whole dock with it.
    ///
    /// Records nothing itself: two callers word the same operation differently
    /// — the menu says where the file came from, an MCP tool would say which
    /// tool — and a method that logged would have to be told which.
    pub(crate) fn apply_layout(&mut self, layout: &Layout) -> Result<(), LayoutError> {
        layout.validate()?;
        self.dock = layout.to_dock();
        Ok(())
    }

    /// The current arrangement, as the layout file spells it.
    pub(crate) fn layout(&self) -> Layout {
        Layout::from_dock(&self.dock)
    }

    /// The entry a panel's arrival writes, wherever it landed.
    fn record_opened(&mut self, tab: Tab) {
        self.action_log
            .record(Kind::Layout, format!("Opened {} panel", tab.title()));
    }

    /// The entry a panel's departure writes, whether the tab's close button or
    /// the Panels menu asked for it.
    pub(crate) fn record_closed(&mut self, tab: Tab) {
        self.action_log
            .record(Kind::Layout, format!("Closed {} panel", tab.title()));
    }
}

// ── The Panels menu ──────────────────────────────────────────────────────

/// The body of the **Panels** menu: a checkbox per panel, then the three
/// layout-wide items.
///
/// Split out of `app.rs` so it can be drawn — and read back — in a headless
/// frame.
pub(crate) fn panels_menu(ui: &mut egui::Ui, state: &mut AppState) {
    for tab in Tab::ALL {
        let mut open = state.is_panel_open(tab);
        if ui.checkbox(&mut open, tab.title()).clicked() {
            if open {
                state.show_panel(tab);
            } else {
                state.hide_panel(tab);
            }
            ui.close();
        }
    }
    ui.separator();
    if ui
        .button("Reset Layout")
        .on_hover_text("Put every panel back in its default place")
        .clicked()
    {
        state.reset_layout();
        ui.close();
    }
    ui.separator();
    if ui.button("Save Layout...").clicked() {
        save_layout(state);
        ui.close();
    }
    if ui.button("Load Layout...").clicked() {
        load_layout(state);
        ui.close();
    }
}

/// Panels ▸ Save Layout…: a save dialog, then the file.
fn save_layout(state: &mut AppState) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("Layout", &["json"])
        .set_file_name(DEFAULT_LAYOUT_FILE_NAME)
        .save_file()
    else {
        return;
    };
    match std::fs::write(&path, state.layout().to_json()) {
        Ok(()) => state
            .action_log
            .record(Kind::Layout, format!("Saved layout to {}", path.display())),
        Err(error) => state.action_log.fail(
            Kind::Layout,
            format!("Save layout to {}: {error}", path.display()),
        ),
    }
}

/// Panels ▸ Load Layout…: an open dialog, then the file — or a refusal that
/// leaves the arrangement on screen exactly as it was.
fn load_layout(state: &mut AppState) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("Layout", &["json"])
        .pick_file()
    else {
        return;
    };
    let parsed = std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|text| Layout::from_json(&text).map_err(|error| error.to_string()));
    let outcome = parsed.and_then(|layout| {
        state
            .apply_layout(&layout)
            .map_err(|error| error.to_string())
    });
    match outcome {
        Ok(()) => state.action_log.record(
            Kind::Layout,
            format!("Loaded layout from {}", path.display()),
        ),
        Err(message) => state.action_log.fail(
            Kind::Layout,
            format!("Load layout from {}: {message}", path.display()),
        ),
    }
}

/// What the save dialog offers to call the file.
const DEFAULT_LAYOUT_FILE_NAME: &str = "layout.json";
