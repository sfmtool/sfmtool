// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The errors [`run_with_args`] returns before it touches the process: a
//! command line it cannot act on is reported, not exited on, and no event loop
//! is created for it, so these run headless and in any order.

use super::*;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|arg| arg.to_string()).collect()
}

#[test]
fn an_unknown_option_is_a_status_2_error() {
    let error = run_with_args(args(&["--no-such-option"])).unwrap_err();
    assert_eq!(error.exit_status(), 2);
    assert!(
        error.message().contains("--no-such-option"),
        "{}",
        error.message()
    );
    assert_eq!(error.to_string(), error.message());
}

#[test]
fn an_mcp_value_that_is_not_a_port_is_a_status_2_error() {
    let error = run_with_args(args(&["--mcp=not-a-port"])).unwrap_err();
    assert_eq!(error.exit_status(), 2);
}

#[test]
fn help_returns_ok_without_opening_a_window() {
    assert_eq!(run_with_args(args(&["--help"])), Ok(()));
    assert_eq!(run_with_args(args(&["-h", "scene.sfmr"])), Ok(()));
}

#[cfg(not(feature = "mcp"))]
#[test]
fn mcp_in_a_build_without_it_is_a_status_2_error() {
    let error = run_with_args(args(&["--mcp", "0"])).unwrap_err();
    assert_eq!(error.exit_status(), 2);
    assert!(
        error.message().contains("\"mcp\" feature"),
        "{}",
        error.message()
    );
}

#[test]
fn a_second_event_loop_is_explained() {
    let error = event_loop_error(winit::error::EventLoopError::RecreationAttempt);
    assert_eq!(error.exit_status(), 1);
    assert!(error.message().contains("already run in this process"));
}
