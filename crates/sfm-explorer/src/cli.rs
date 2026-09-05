// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The command line:
//! `sfm-explorer [--mcp [PORT]] [--no-default-layout] [path.sfmr ...]`.
//!
//! Hand-rolled rather than `clap`, because there are two flags and a list of
//! paths. A dozen lines keeps the binary's dependency tree as it was; reach for
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
    /// `--help` was asked for; print [`USAGE`] and exit without opening a
    /// window.
    pub(crate) help: bool,
}

/// What `--help` prints.
pub(crate) const USAGE: &str = "\
sfm-explorer — the SfM Tool 3D reconstruction viewer

USAGE:
    sfm-explorer [OPTIONS] [FILE.sfmr ...]

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
    -h, --help      Print this message and exit.
";

/// Recognize the command line.
///
/// `--mcp` takes its port as either `--mcp=PORT` or a following bare number.
/// The following-argument form has to look at what comes next, because
/// `--mcp scene.sfmr` is the common invocation and means the default port and a
/// file — so a next argument that is not a port is left alone rather than
/// consumed.
pub(crate) fn parse(argv: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = argv.into_iter().peekable();
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "-h" | "--help" => args.help = true,
            "--no-default-layout" => args.no_default_layout = true,
            "--mcp" => {
                let port = match argv.peek().and_then(|next| next.parse::<u16>().ok()) {
                    Some(port) => {
                        argv.next();
                        port
                    }
                    None => DEFAULT_MCP_PORT,
                };
                args.mcp_port = Some(port);
            }
            other => {
                if let Some(value) = other.strip_prefix("--mcp=") {
                    let port = value.parse::<u16>().map_err(|_| {
                        format!("--mcp wants a port number from 0 to 65535, not {value:?}.")
                    })?;
                    args.mcp_port = Some(port);
                } else if other.starts_with('-') && other != "-" {
                    return Err(format!(
                        "sfm-explorer has no option {other:?}. Run --help for what it takes."
                    ));
                } else {
                    args.paths.push(PathBuf::from(other));
                }
            }
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests;
