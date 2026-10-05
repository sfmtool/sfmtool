// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The command line:
//! `[--mcp [PORT]] [--no-default-layout] [--demo] [path.sfmr ...]`, after the
//! program name, which differs between the `sfm-explorer` binary and
//! `python -m sfmtool._explorer` and so is not named in the help or errors.
//!
//! Hand-rolled rather than `clap`, because there are three flags and a list of
//! paths. A dozen lines keeps the viewer's dependency tree as it was; reach for
//! an argument parser if this grows options that take values, not before.

use std::path::PathBuf;

/// The port `--mcp` binds when given no number.
///
/// Fixed rather than ephemeral so a client config can be written once and keep
/// working — the shape desktop applications with in-process MCP servers use.
/// Two viewers cannot both take it; `--mcp 0` is the answer to that.
pub(crate) const DEFAULT_MCP_PORT: u16 = 8787;

/// What the command line asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Args {
    /// Files to load, in the order given, each as its own scene node.
    pub(crate) paths: Vec<PathBuf>,
    /// The port `--mcp` asked for, if it was given at all. `Some(0)` means an
    /// ephemeral port, which the endpoint line then reports.
    pub(crate) mcp_port: Option<u16>,
    /// Skip the startup load of `~/.sfm-explorer-default-layout.json`, and come
    /// up with the stock grid whatever is saved there.
    pub(crate) no_default_layout: bool,
    /// Append a node of generated demo data at startup, after any files — the
    /// same node File > Load Demo Data… makes, at the dialog's default point
    /// count.
    pub(crate) demo: bool,
    /// `--help` was asked for; print [`USAGE`] and exit without opening a
    /// window.
    pub(crate) help: bool,
}

/// What `--help` prints.
pub(crate) const USAGE: &str = "\
SfM Explorer — the SfM Tool 3D reconstruction viewer

USAGE:
    [OPTIONS] [FILE.sfmr ...]

Every file given is loaded as its own node in the scene graph, so several
reconstructions can be compared side by side in one 3D space.

OPTIONS:
    --mcp [PORT]    Host a Model Context Protocol endpoint on 127.0.0.1, so an
                    agent can drive this window. Off unless asked for. PORT
                    defaults to 8787; 0 takes an ephemeral port, reported on
                    stdout at startup.
    --no-default-layout
                    Start with the stock panel grid, ignoring any layout saved
                    at ~/.sfm-explorer-default-layout.json.
    --demo          Load a generated demo reconstruction, after any files. The
                    same node File > Load Demo Data... makes at its default
                    point count, without the two dialogs.
    -h, --help      Print this message and exit.
";

/// Recognize the command line.
///
/// `--mcp` takes its port as either `--mcp=PORT` or a following bare number.
/// The following-argument form has to look at what comes next, because
/// `--mcp scene.sfmr` is the common invocation and means the default port and a
/// file — so a next argument that is not all digits is left alone rather than
/// consumed. One that is all digits is the port, and an error if it does not
/// fit in one, rather than a file named `70000`.
pub(crate) fn parse(argv: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = argv.into_iter().peekable();
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "-h" | "--help" => args.help = true,
            "--no-default-layout" => args.no_default_layout = true,
            "--demo" => args.demo = true,
            "--mcp" => {
                // A next word of digits is a port, and one too large to be a
                // port is an error rather than a file name: `--mcp 70000`
                // means a port, and binding 8787 instead would be wrong.
                let port = match argv.next_if(|next| is_number(next)) {
                    Some(value) => value.parse::<u16>().map_err(|_| port_error(&value))?,
                    None => DEFAULT_MCP_PORT,
                };
                args.mcp_port = Some(port);
            }
            other => {
                if let Some(value) = other.strip_prefix("--mcp=") {
                    let port = value.parse::<u16>().map_err(|_| port_error(value))?;
                    args.mcp_port = Some(port);
                } else if other.starts_with('-') && other != "-" {
                    return Err(format!(
                        "SfM Explorer has no option {other:?}. Run --help for what it takes."
                    ));
                } else {
                    args.paths.push(PathBuf::from(other));
                }
            }
        }
    }
    Ok(args)
}

/// Whether the word after a bare `--mcp` is meant as its port: one or more
/// ASCII digits, whether or not the number fits in a port.
fn is_number(word: &str) -> bool {
    !word.is_empty() && word.bytes().all(|b| b.is_ascii_digit())
}

/// The error for a value given as `--mcp`'s port that is not one.
fn port_error(value: &str) -> String {
    format!("--mcp wants a port number from 0 to 65535, not {value:?}.")
}

#[cfg(test)]
mod tests;
