// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Every source file in this repository opens with the license header.
//!
//! The convention held everywhere except Rust `tests.rs`, where it had drifted
//! to roughly two files in three — a convention that is followed by hand is
//! followed unevenly, and a header is the easiest line in a new file to
//! forget. So it is read back out of the sources instead: this walks the Rust
//! and Python trees and fails with the list of files whose first two lines are
//! not the header.
//!
//! It lives in `sfmtool-core/tests/` rather than beside a module because the
//! subject is the repository, not any one crate's behaviour: an integration
//! test has no library module it implicitly belongs to, and this crate is the
//! one the existing workspace-wide scan
//! (`numeric/tests.rs::the_workspace_has_one_median`) already runs from.

use std::io::BufRead;
use std::path::{Path, PathBuf};

/// The header, without the per-language comment marker.
const HEADER: [&str; 2] = [
    "Copyright The SfM Tool Authors",
    "SPDX-License-Identifier: Apache-2.0",
];

/// The repository root: `crates/sfmtool-core` → `crates` → the checkout.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the checkout is two levels above crates/sfmtool-core")
        .to_path_buf()
}

/// Every file under `dir` with extension `ext`, skipping build output and
/// hidden directories (`.git`, `.pixi`, `.claude`, …).
fn sources(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
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
                sources(&path, ext, out);
            }
        } else if name.ends_with(ext) {
            out.push(path);
        }
    }
}

/// The header lines `path` is missing, if any.
///
/// Only the first few lines are read, so the whole scan costs one page per
/// file. A leading `#!` shebang or `# -*- coding:` line is allowed to come
/// first — Python needs the shebang on line 1 to stay executable — and the
/// header must then start on the line after it.
fn missing_header(path: &Path, comment: &str) -> Option<String> {
    let file = std::fs::File::open(path).expect("readable source");
    let mut lines = std::io::BufReader::new(file)
        .lines()
        .take(4)
        .map(|line| line.expect("valid UTF-8 source"))
        .peekable();
    if lines
        .peek()
        .is_some_and(|l| l.starts_with("#!") || l.contains("coding:"))
    {
        lines.next();
    }
    for expected in HEADER {
        let expected = format!("{comment} {expected}");
        if lines.next().as_deref() != Some(expected.as_str()) {
            return Some(expected);
        }
    }
    None
}

/// The mechanical half of "every source file carries the license header".
#[test]
fn every_source_file_opens_with_the_license_header() {
    let root = repo_root();
    let mut scanned = Vec::new();
    let mut offenders = Vec::new();

    for (dir, ext, comment) in [
        ("crates", ".rs", "//"),
        ("src", ".py", "#"),
        ("tests", ".py", "#"),
        ("scripts", ".py", "#"),
    ] {
        let mut paths = Vec::new();
        sources(&root.join(dir), ext, &mut paths);
        paths.sort();
        assert!(
            !paths.is_empty(),
            "expected to find {ext} sources under {dir}/"
        );
        for path in &paths {
            if let Some(expected) = missing_header(path, comment) {
                let rel = path
                    .strip_prefix(&root)
                    .expect("scanned under the checkout")
                    .to_string_lossy()
                    .replace('\\', "/");
                offenders.push(format!("{rel}: expected {expected:?}"));
            }
        }
        scanned.extend(paths);
    }

    assert!(
        scanned.len() > 500,
        "expected to scan the whole repository, found {} files",
        scanned.len()
    );
    assert!(
        offenders.is_empty(),
        "source files without the license header:\n  {}\n\
         Every .rs and .py file in this repository opens with\n\
         \x20 <comment> Copyright The SfM Tool Authors\n\
         \x20 <comment> SPDX-License-Identifier: Apache-2.0\n\
         (after the shebang, where there is one). Add those two lines.",
        offenders.join("\n  ")
    );
}
