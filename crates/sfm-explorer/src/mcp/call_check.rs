// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a caller outside the viewer can check about a tool call before it
//! sends one: the tool catalog as `tools/list` advertises it, and the parse
//! `tools/call` runs on the arguments.
//!
//! This is the crate's public MCP interface, and the only one. `sfm
//! explorer-call` reads it through the `sfmtool.explorer` binding module, so a
//! call it refuses before connecting is refused with the message the viewer
//! would send, from the same parser, and the command can tell whether the
//! viewer it reaches was built from the same catalog. Neither function touches
//! application state: what the parse cannot know (whether a label names a
//! loaded reconstruction, whether an index is in range) is decided only by the
//! running viewer.

use serde_json::{Map, Value};

use super::tools::{self, ToolKind};

/// One tool of the viewer's catalog.
#[derive(Debug, Clone)]
pub struct CatalogTool {
    /// The tool's name, which is what `tools/call` takes.
    pub name: &'static str,
    /// What the tool does to the viewer: `"read"`, `"write"`, `"save"` (writes
    /// a file) or `"input"` (presses what a person could press).
    pub kind: &'static str,
    /// The tool's entry in a `tools/list` reply, as this viewer serves it:
    /// `name`, `description`, `inputSchema` and `annotations`.
    pub advertised: Value,
}

/// Every tool the viewer advertises, in the order `tools/list` reports them.
pub fn tool_catalog() -> Vec<CatalogTool> {
    tools::catalog()
        .iter()
        .map(|spec| CatalogTool {
            name: spec.name,
            kind: match spec.kind {
                ToolKind::Read => "read",
                ToolKind::Write => "write",
                ToolKind::Save => "save",
                ToolKind::Input => "input",
            },
            advertised: serde_json::to_value(super::server::advertise(spec))
                .expect("an advertised tool serializes to JSON"),
        })
        .collect()
}

/// Run the parse `tools/call` runs on `arguments` for the tool `name`, and
/// return the message the viewer would refuse the call with, if any.
///
/// `Ok(())` means the arguments fit the tool; the running viewer may still
/// refuse the call for what it finds in the scene. The message of an `Err` is
/// the one the viewer's endpoint sends as its JSON-RPC `invalid_params` error,
/// including for a tool name it does not know.
pub fn validate_tool_call(
    name: &str,
    arguments: Option<&Map<String, Value>>,
) -> Result<(), String> {
    tools::parse(name, arguments)
        .map(|_| ())
        .map_err(|error| error.0)
}
