// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Panel layout: the arrangement of the dock, and its file.
//!
//! See `specs/gui/panel-layout.md`. Four things live here:
//!
//! - [`WindowLayout`], the layout **document**: the window's placement
//!   ([`crate::window`]) and the panel arrangement, either of them optional. It
//!   is what the file holds, what Panels ▸ Save Layout… writes, and what one
//!   MCP `set_window_layout` carries.
//! - [`Layout`], a description of a dock arrangement that is independent of
//!   `egui_dock`'s node indices — a split tree of panel names, readable and
//!   writable by hand, and the `layout` section of that document.
//! - The conversions [`Layout::from_dock`] / [`Layout::to_dock`], the one place
//!   in the crate that knows how `egui_dock` represents a tree.
//! - The operations on [`AppState`] — [`AppState::show_panel`],
//!   [`AppState::hide_panel`], [`AppState::reset_layout`],
//!   [`AppState::apply_layout`], [`AppState::apply_window_layout`],
//!   [`AppState::load_layout_file`] — which is where the Action Log is in
//!   reach, and the Panels menu that drives them ([`panels_menu`]).
//!
//! The JSON is read and written by hand rather than through `serde`: a node is
//! a leaf or a split by which keys it carries, and `serde` does not honour
//! `deny_unknown_fields` on an untagged enum, so a derive could not refuse a
//! typo in `"fraction"`.

use std::path::{Path, PathBuf};

use egui_dock::{DockState, LeafNode, Node, NodeIndex, Split, Surface, TabIndex, Tree};
use serde_json::Value;

use crate::action_log::Kind;
use crate::dock::Tab;
use crate::state::AppState;
use crate::window::{MonitorRect, NormalRect, WindowChange, WindowHost, WindowState};

#[cfg(test)]
mod tests;

/// The `sfm_explorer_layout` value written, and the only one read.
pub(crate) const LAYOUT_VERSION: u64 = 2;

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
    pub(crate) fn at(path: impl Into<String>, message: impl Into<String>) -> Self {
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
    ///
    /// `path` is where the arrangement sits in the document — `"layout"` for a
    /// document's section — so a violation reads
    /// `layout.main.second: unknown key "fracton"`.
    pub(crate) fn validate(&self, path: &str) -> Result<(), LayoutError> {
        let mut seen = Vec::new();
        if let Some(main) = &self.main {
            validate_node(main, &format!("{path}.main"), &mut seen)?;
        }
        for (index, window) in self.windows.iter().enumerate() {
            validate_node(
                &window.tree,
                &format!("{path}.windows[{index}].tree"),
                &mut seen,
            )?;
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

// ── The layout document ──────────────────────────────────────────────────

/// One document's worth of panels: an arrangement, or the stock grid by name.
///
/// `"default"` is legal in a file as well as on the wire — a file that says it
/// is a reset — and the viewer never writes that form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayoutSection {
    Layout(Layout),
    Default,
}

/// The layout document: the window's placement and the panel arrangement,
/// either of them optional.
///
/// One document rather than two, because the two are one thing to the person
/// sitting at the window: "my viewer, maximized on the left monitor, with the
/// Action Log along the bottom" is one arrangement, and saving half of it is
/// not saving it. A section that is `None` is one the document does not
/// describe and applying it leaves alone.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowLayout {
    pub(crate) window: Option<WindowChange>,
    pub(crate) layout: Option<LayoutSection>,
}

impl Default for WindowLayout {
    /// The stock grid, and nothing about the window — which is what a headless
    /// [`AppState`] has to say about where its window is.
    fn default() -> Self {
        WindowLayout {
            window: None,
            layout: Some(LayoutSection::Layout(Layout::default())),
        }
    }
}

impl WindowLayout {
    /// Parse and validate a layout file.
    ///
    /// The document is refused as a whole: a caller that gets an `Err` has a
    /// window and a layout it can leave exactly as they were.
    pub(crate) fn from_json(text: &str) -> Result<Self, LayoutError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|error| LayoutError::at("", format!("not valid JSON: {error}")))?;
        WindowLayout::from_value(&value)
    }

    /// Validate an already-parsed document.
    ///
    /// The half of [`WindowLayout::from_json`] below the JSON parse, so a
    /// document that arrived as a `serde_json::Value` — a `set_window_layout`
    /// argument — meets exactly the rules, and exactly the messages, a file on
    /// disk meets.
    ///
    /// The version tag is optional and checked when present: a file carries it,
    /// and so does a reply sent back whole, but a call that asks for one thing
    /// should not have to.
    pub(crate) fn from_value(value: &Value) -> Result<Self, LayoutError> {
        let Some(object) = value.as_object() else {
            return Err(LayoutError::at("", "the document must be a JSON object"));
        };
        // The version first, so a JSON file that is not a layout at all says so
        // rather than complaining about its own perfectly good keys. With no
        // tag and no section, there is nothing here that claims to be a layout.
        match object.get("sfm_explorer_layout") {
            Some(value) => {
                let Some(version) = value.as_u64() else {
                    return Err(LayoutError::at("", "Not a layout file"));
                };
                if version > LAYOUT_VERSION {
                    return Err(LayoutError::at(
                        "",
                        format!(
                            "Layout version {version} is newer than this viewer reads \
                             ({LAYOUT_VERSION})"
                        ),
                    ));
                }
                if version != LAYOUT_VERSION {
                    return Err(LayoutError::at(
                        "",
                        format!(
                            "Layout version {version} is not one this viewer reads \
                             ({LAYOUT_VERSION})"
                        ),
                    ));
                }
            }
            None if !object.is_empty()
                && !object.contains_key("window")
                && !object.contains_key("layout") =>
            {
                return Err(LayoutError::at("", "Not a layout file"))
            }
            None => {}
        }
        known_keys(object, "", &["sfm_explorer_layout", "window", "layout"])?;

        let window = match object.get("window") {
            None | Some(Value::Null) => None,
            Some(value) => Some(WindowChange::from_json(value, "window")?),
        };
        let layout = match object.get("layout") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) if name == "default" => Some(LayoutSection::Default),
            Some(Value::String(_)) => {
                return Err(LayoutError::at(
                    "layout",
                    "the only named layout is \"default\"",
                ))
            }
            Some(value @ Value::Object(_)) => {
                Some(LayoutSection::Layout(Layout::from_value(value, "layout")?))
            }
            Some(_) => {
                return Err(LayoutError::at(
                    "layout",
                    "must be an arrangement, null, or \"default\"",
                ))
            }
        };
        Ok(WindowLayout { window, layout })
    }

    /// The document as a file: pretty-printed, keys in schema order, one
    /// trailing newline.
    pub(crate) fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"sfm_explorer_layout\": {LAYOUT_VERSION}"));
        if let Some(window) = &self.window {
            out.push_str(",\n  \"window\": ");
            window.write_json(&mut out, 1);
        }
        match &self.layout {
            Some(LayoutSection::Layout(layout)) => {
                out.push_str(",\n  \"layout\": ");
                layout.write_json(&mut out, 1);
            }
            Some(LayoutSection::Default) => out.push_str(",\n  \"layout\": \"default\""),
            None => {}
        }
        out.push_str("\n}\n");
        out
    }

    /// Whether the document asks for anything at all.
    ///
    /// A file that asks for nothing loads as a no-op; a tool call that asks for
    /// nothing is a request with no request in it, and is refused — which is
    /// the only caller, so a build without the MCP surface has none.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn is_empty(&self) -> bool {
        self.layout.is_none() && self.window.as_ref().is_none_or(WindowChange::is_empty)
    }
}

/// What the window portion of a document did.
///
/// `change` is what reached the window, which is the *fitted* rectangle where
/// one was fitted, and `fitted` the monitor it came from — the two things an
/// Action Log row needs to say what happened rather than what was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppliedWindow {
    pub(crate) change: WindowChange,
    pub(crate) fitted: Option<MonitorRect>,
}

/// Where the default layout file lives, or `None` without a home directory.
///
/// The name is fixed and the directory is the one place the viewer can rely on
/// without knowing a workspace.
pub(crate) fn default_layout_path() -> Option<PathBuf> {
    #[allow(deprecated)] // Un-deprecated in 1.85, below the workspace MSRV.
    std::env::home_dir().map(|home| home.join(DEFAULT_LAYOUT_FILE_NAME))
}

/// What the viewer reads at startup, and what the save dialog offers to call
/// the file.
pub(crate) const DEFAULT_LAYOUT_FILE_NAME: &str = ".sfm-explorer-default-layout.json";

// ── The `layout` section ─────────────────────────────────────────────────

impl Layout {
    /// Validate the `layout` section of a document.
    ///
    /// `path` is where the section sits in the document, so a violation below a
    /// node reads `layout.main.second: unknown key "fracton"`.
    pub(crate) fn from_value(value: &Value, path: &str) -> Result<Self, LayoutError> {
        let Some(object) = value.as_object() else {
            return Err(LayoutError::at(path, "must be a JSON object"));
        };
        known_keys(object, path, &["main", "windows"])?;

        let main = match object.get("main") {
            None | Some(Value::Null) => None,
            Some(value) => Some(node_from_json(value, &format!("{path}.main"))?),
        };
        let windows = match object.get("windows") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut windows = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    windows.push(window_from_json(item, &format!("{path}.windows[{index}]"))?);
                }
                windows
            }
            Some(_) => return Err(LayoutError::at(path, "\"windows\" must be an array")),
        };

        let layout = Layout { main, windows };
        layout.validate(path)?;
        Ok(layout)
    }

    /// The arrangement as the file writes it, indented for a parent at `depth`.
    pub(crate) fn write_json(&self, out: &mut String, depth: usize) {
        let inner = "  ".repeat(depth + 1);
        let outer = "  ".repeat(depth);
        out.push_str("{\n");
        out.push_str(&format!("{inner}\"main\": "));
        match &self.main {
            Some(node) => write_node(out, node, depth + 1),
            None => out.push_str("null"),
        }
        out.push_str(",\n");
        if self.windows.is_empty() {
            out.push_str(&format!("{inner}\"windows\": []\n"));
        } else {
            out.push_str(&format!("{inner}\"windows\": [\n"));
            for (index, window) in self.windows.iter().enumerate() {
                out.push_str(&"  ".repeat(depth + 2));
                write_window(out, window, depth + 2);
                out.push_str(if index + 1 == self.windows.len() {
                    "\n"
                } else {
                    ",\n"
                });
            }
            out.push_str(&format!("{inner}]\n"));
        }
        out.push_str(&outer);
        out.push('}');
    }
}

/// Refuse any key that is not one of `allowed`.
///
/// A typo silently applying a default would leave the author believing the file
/// says something it does not.
pub(crate) fn known_keys(
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
        layout.validate("layout")?;
        self.dock = layout.to_dock();
        Ok(())
    }

    /// The current arrangement, as the layout file spells it.
    pub(crate) fn layout(&self) -> Layout {
        Layout::from_dock(&self.dock)
    }

    /// Refresh the window snapshot from `host`, remembering the rectangle
    /// whenever the window reads as `normal`.
    ///
    /// Called at the top of every frame. The remembering is why: `winit`
    /// reports only the *current* rectangle, so once a window is maximized its
    /// restored rectangle is unreadable, and the memory has to have been
    /// current at the moment of the maximize.
    pub(crate) fn observe_window(&mut self, host: &dyn WindowHost) {
        let observed = host.observe().map(|(info, _)| info);
        if let Some(info) = &observed {
            if info.state == WindowState::Normal {
                self.window_normal_rect = Some(NormalRect {
                    outer_position: info.outer_position,
                    inner_size: info.inner_size,
                });
            }
        }
        self.window = observed;
    }

    /// The document Panels ▸ Save Layout… writes: the window's placement, and
    /// the panel arrangement.
    ///
    /// The placement is the snapshot's state with the *remembered* normal
    /// rectangle beside it, so a maximized window saves as `maximized` plus the
    /// rectangle it will come back to. Without a snapshot — a headless
    /// `AppState` — there is no window section at all.
    pub(crate) fn window_layout(&self) -> WindowLayout {
        WindowLayout {
            window: self.window.as_ref().map(|info| WindowChange {
                state: Some(info.state),
                outer_position: self.window_normal_rect.and_then(|rect| rect.outer_position),
                inner_size: self.window_normal_rect.map(|rect| rect.inner_size),
                monitor: info.monitor.as_ref().map(MonitorRect::of),
                focus: false,
            }),
            layout: Some(LayoutSection::Layout(self.layout())),
        }
    }

    /// Apply a whole document: **window portion first, panel portion second.**
    ///
    /// In that order so the call reads "make the window like this, then arrange
    /// the panels", and so a panel tree is laid out into the window it was
    /// meant for. The document is validated whole before any of it is applied,
    /// so a validation refusal touches nothing; a *platform* refusal can only
    /// come from the host, and stops the call before the panels.
    ///
    /// Records nothing: the menu, the startup load and the MCP tool each word
    /// their own entry. What it hands back is what the *window* portion did
    /// ([`AppliedWindow`]), because a fitted rectangle is not the one the
    /// caller sent and the Action Log row has to say the numbers that reached
    /// the window.
    pub(crate) fn apply_window_layout(
        &mut self,
        host: &mut dyn WindowHost,
        document: &WindowLayout,
    ) -> Result<Option<AppliedWindow>, LayoutError> {
        if let Some(LayoutSection::Layout(layout)) = &document.layout {
            layout.validate("layout")?;
        }
        let mut applied = None;
        if let Some(change) = document.window.as_ref().filter(|c| !c.is_empty()) {
            // The fit happens here rather than in the host, so it is one pure
            // function under headless test and every host sees a plain
            // rectangle.
            let monitors = host.observe().map(|(_, monitors)| monitors);
            let (change, fitted) = crate::window::fit_to_monitor(
                change,
                monitors.as_deref().unwrap_or(&[]),
                self.window.as_ref().and_then(|info| info.monitor.as_ref()),
            );
            host.apply(&change)
                .map_err(|error| LayoutError::at("window", error.0))?;
            // So the reply, and a later call in the same batch, see the change
            // rather than the window as it was.
            self.observe_window(host);
            self.remember_applied_rect(&change);
            applied = Some(AppliedWindow { change, fitted });
        }
        match &document.layout {
            Some(LayoutSection::Layout(layout)) => self.apply_layout(layout)?,
            Some(LayoutSection::Default) => self.apply_layout(&Layout::default())?,
            None => {}
        }
        Ok(applied)
    }

    /// Remember a rectangle that was applied to a window that is not showing
    /// it.
    ///
    /// [`AppState::observe_window`] can only remember what it can read, and a
    /// window that came out of the apply maximized, minimized or fullscreen
    /// does not report the rectangle underneath. But the viewer *just set* that
    /// rectangle, so it knows it — and without this the document would keep
    /// describing the rectangle from before the call as the one the window
    /// restores to. A normal window needs none of this: the observation above
    /// already read the truth, clamps included.
    fn remember_applied_rect(&mut self, change: &WindowChange) {
        if !change.has_geometry()
            || self
                .window
                .as_ref()
                .is_none_or(|info| info.state == WindowState::Normal)
        {
            return;
        }
        let previous = self.window_normal_rect;
        let Some(inner_size) = change.inner_size.or(previous.map(|rect| rect.inner_size)) else {
            return;
        };
        self.window_normal_rect = Some(NormalRect {
            outer_position: change
                .outer_position
                .or(previous.and_then(|rect| rect.outer_position)),
            inner_size,
        });
    }

    /// Load and apply one layout file, recording what happened.
    ///
    /// Shared by Panels ▸ Load Layout… and the startup load of the default
    /// file, so the two cannot come to treat a file differently. A file that
    /// does not parse, does not validate, or names something the platform will
    /// not do is refused as a whole and the failed entry says why — which puts
    /// it on the viewport status line, where the human sees *why* their layout
    /// did not come back.
    pub(crate) fn load_layout_file(&mut self, host: &mut dyn WindowHost, path: &Path) {
        let outcome = std::fs::read_to_string(path)
            .map_err(|error| error.to_string())
            .and_then(|text| WindowLayout::from_json(&text).map_err(|error| error.to_string()))
            .and_then(|document| {
                self.apply_window_layout(host, &document)
                    .map_err(|error| error.to_string())
            });
        match outcome {
            Ok(_) => self.action_log.record(
                Kind::Layout,
                format!("Loaded layout from {}", path.display()),
            ),
            Err(message) => self.action_log.fail(
                Kind::Layout,
                format!("Load layout from {}: {message}", path.display()),
            ),
        }
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
/// frame. It takes the window host because Save and Load carry the window's
/// placement as well as the panels; the frame passes a clone of its
/// `Arc<Window>` and the headless test passes a fake. A menu load applies the
/// window change mid-frame rather than at the top of one, so the *next* frame
/// is the first laid out at the new size — right for a human click, and not
/// worth a deferral.
pub(crate) fn panels_menu(ui: &mut egui::Ui, state: &mut AppState, host: &mut dyn WindowHost) {
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
        load_layout(state, host);
        ui.close();
    }
}

/// Panels ▸ Save Layout…: a save dialog, then the file.
///
/// The dialog opens on the default file (§ "The default layout file"), so the
/// common case — "keep it like this" — is Save Layout…, Enter, and the viewer
/// comes up this way next time.
fn save_layout(state: &mut AppState) {
    let mut dialog = rfd::FileDialog::new()
        .add_filter("Layout", &["json"])
        .set_file_name(DEFAULT_LAYOUT_FILE_NAME);
    if let Some(directory) =
        default_layout_path().and_then(|path| path.parent().map(Path::to_owned))
    {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.save_file() else {
        return;
    };
    match std::fs::write(&path, state.window_layout().to_json()) {
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
/// leaves the window and the arrangement exactly as they were.
fn load_layout(state: &mut AppState, host: &mut dyn WindowHost) {
    let mut dialog = rfd::FileDialog::new().add_filter("Layout", &["json"]);
    if let Some(directory) =
        default_layout_path().and_then(|path| path.parent().map(Path::to_owned))
    {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.pick_file() else {
        return;
    };
    state.load_layout_file(host, &path);
}
