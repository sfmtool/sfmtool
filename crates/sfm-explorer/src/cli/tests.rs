// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Argument parsing, exercised straight through [`super::parse`].

use super::*;

fn parse_ok(argv: &[&str]) -> Args {
    parse(argv.iter().map(|s| s.to_string())).expect("parses")
}

#[test]
fn bare_arguments_are_paths() {
    let args = parse_ok(&["a.sfmr", "b.sfmr"]);
    assert_eq!(
        args.paths,
        vec![PathBuf::from("a.sfmr"), PathBuf::from("b.sfmr")]
    );
    assert_eq!(args.mcp_port, None);
}

#[test]
fn mcp_alone_takes_the_default_port() {
    assert_eq!(parse_ok(&["--mcp"]).mcp_port, Some(8787));
}

/// The invocation the docs lead with: a flag with no port, then the file.
/// The file must not be eaten as the port.
#[test]
fn mcp_does_not_swallow_a_following_path() {
    let args = parse_ok(&["--mcp", "scene.sfmr"]);
    assert_eq!(args.mcp_port, Some(8787));
    assert_eq!(args.paths, vec![PathBuf::from("scene.sfmr")]);
}

#[test]
fn mcp_takes_a_port_either_way() {
    assert_eq!(parse_ok(&["--mcp", "9000"]).mcp_port, Some(9000));
    assert_eq!(parse_ok(&["--mcp=9000"]).mcp_port, Some(9000));
}

/// Zero is a real request — bind an ephemeral port — and not "no port".
#[test]
fn port_zero_is_a_port() {
    let args = parse_ok(&["--mcp", "0", "scene.sfmr"]);
    assert_eq!(args.mcp_port, Some(0));
    assert_eq!(args.paths, vec![PathBuf::from("scene.sfmr")]);
}

#[test]
fn an_unparseable_explicit_port_is_an_error() {
    let error = parse(["--mcp=nine".to_string()]).expect_err("rejected");
    assert!(error.contains("--mcp"), "{error}");
}

#[test]
fn an_unknown_option_is_an_error_rather_than_a_path() {
    let error = parse(["--verbose".to_string()]).expect_err("rejected");
    assert!(error.contains("--verbose"), "{error}");
}

/// The flag the UI tests pass, so a developer's own saved layout cannot
/// make the suite's panel assertions fail on their machine.
#[test]
fn the_default_layout_can_be_skipped() {
    assert!(!parse_ok(&["scene.sfmr"]).no_default_layout);
    let args = parse_ok(&["--no-default-layout", "scene.sfmr"]);
    assert!(args.no_default_layout);
    assert_eq!(args.paths, vec![PathBuf::from("scene.sfmr")]);
}

#[test]
fn help_is_recognized_both_ways() {
    assert!(parse_ok(&["-h"]).help);
    assert!(parse_ok(&["--help"]).help);
}
