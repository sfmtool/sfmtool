// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! No parameter, field or binding typed as an `OrientedPatch` is named bare
//! `frame`.
//!
//! A patch's geometry, its centre and its two half-extent vectors, is a
//! **patch frame** (`patch_frame`), never shortened to `frame`: bare `frame`
//! means a video or rig frame, a UI frame or a coordinate frame elsewhere in
//! this tree (`specs/GLOSSARY.md`, "The bench"). A name is easy to shorten
//! by hand, so this scan reads every Rust source under `crates/` and fails
//! with each `frame: …OrientedPatch` or `frames: …OrientedPatch` it finds.
//! It sees only names that state their type, which is where a new parameter
//! or field is written; a `let frame = …` whose type is inferred is left to
//! review.

use std::path::{Path, PathBuf};

/// The repository root: `crates/sfmtool-core` → `crates` → the checkout.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the checkout is two levels above crates/sfmtool-core")
        .to_path_buf()
}

/// Every `.rs` file under `dir`, skipping build output and hidden directories.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                sources(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// Whether `code`, one line with its `//` comment removed, names something
/// `frame` or `frames` with a type that mentions `OrientedPatch`.
fn names_a_patch_frame(code: &str) -> bool {
    let bytes = code.as_bytes();
    for word in ["frames", "frame"] {
        let mut from = 0;
        while let Some(at) = code[from..].find(word).map(|i| i + from) {
            from = at + word.len();
            let before = at.checked_sub(1).map(|i| bytes[i]);
            if before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_') {
                continue;
            }
            let rest = code[from..].trim_start();
            let Some(ty) = rest.strip_prefix(':') else {
                continue;
            };
            if ty.starts_with(':') {
                continue;
            }
            let end = ty.find([',', ')', '=', '{', ';']).unwrap_or(ty.len());
            if ty[..end].contains("OrientedPatch") {
                return true;
            }
        }
    }
    false
}

#[test]
fn no_oriented_patch_is_named_bare_frame() {
    let root = repo_root();
    let mut files = Vec::new();
    sources(&root.join("crates"), &mut files);
    let mut found = Vec::new();
    // This file spells the forbidden form in its own examples.
    for path in files
        .iter()
        .filter(|p| !p.ends_with("patch_frame_names.rs"))
    {
        let text = std::fs::read_to_string(path).expect("readable UTF-8 source");
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            if names_a_patch_frame(code) {
                let shown = path.strip_prefix(&root).unwrap_or(path).display();
                found.push(format!("{shown}:{}: {}", number + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "name these patch_frame, not frame (specs/GLOSSARY.md, \"patch frame\"):\n{}",
        found.join("\n")
    );
}

#[test]
fn the_scan_tells_a_patch_frame_from_other_frames() {
    assert!(names_a_patch_frame("fn f(frame: &OrientedPatch) {"));
    assert!(names_a_patch_frame("    frame: Option<&OrientedPatch>,"));
    assert!(names_a_patch_frame(
        "    frames: &[crate::patch::cloud::OrientedPatch],"
    ));
    assert!(!names_a_patch_frame("fn f(patch_frame: &OrientedPatch) {"));
    assert!(!names_a_patch_frame(
        "fn f(frame: &egui::Frame, patch: &OrientedPatch) {"
    ));
    assert!(!names_a_patch_frame("    rig_frame: usize,"));
    assert!(!names_a_patch_frame("use crate::frame::OrientedPatch;"));
}
