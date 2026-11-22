// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rust test code writes only under a temporary directory of its own.
//!
//! A test that builds its path from `std::env::temp_dir()` names a directory
//! that every other run on the machine names too: two checkouts, two worktrees
//! or two `cargo test` invocations then read and write the same files, and a
//! directory one run leaves behind changes what the next one reads. A test
//! takes a `tempfile::TempDir` instead, which is unique and removed when it
//! drops. This walks the workspace's Rust test code and fails with every line
//! that names the system temporary directory, since that is how a fixed path
//! under it is built.
//!
//! Test code is a `tests.rs` file, any file under a `tests/` directory, and the
//! body of a `#[cfg(test)] mod … { … }` block in any other file. Production
//! code that uses the temporary directory on purpose is not test code and is
//! not read. Comment lines are skipped, so a doc comment may name the function.
//!
//! It lives in `sfmtool-core/tests/` beside `license_headers.rs`, the other
//! scan whose subject is the repository rather than one crate.

use std::path::{Path, PathBuf};

/// What a use of the system temporary directory contains, whether written
/// `std::env::…` or `env::…` or imported by name. Built from two parts so this
/// file, which is itself test code, does not contain it.
const NEEDLE: &str = concat!("env::", "temp_dir");

/// The repository root: `crates/sfmtool-core` → `crates` → the checkout.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the checkout is two levels above crates/sfmtool-core")
        .to_path_buf()
}

/// Every `.rs` file under `dir`, skipping build output and hidden directories.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
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
                rust_sources(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// Whether the whole of `path` is test code: a `tests.rs` file or a file under
/// a `tests/` directory.
fn is_test_file(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n == "tests.rs")
        || path.components().any(|c| c.as_os_str() == "tests")
}

/// The 1-based line ranges of `source` inside `#[cfg(test)] mod … { … }`
/// blocks, found by matching braces from the `mod` line. Braces inside string
/// literals would throw the count off; test modules in this workspace hold
/// none that matter, and a miscount only widens or narrows the range read.
fn cfg_test_module_lines(source: &str) -> Vec<(usize, usize)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "#[cfg(test)]" {
            // The attribute may be followed by more attributes before the item.
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with("#[") {
                j += 1;
            }
            let item = lines.get(j).map(|l| l.trim_start()).unwrap_or_default();
            let is_inline_mod = (item.starts_with("mod ") || item.starts_with("pub mod "))
                && item.trim_end().ends_with('{');
            if is_inline_mod {
                let mut depth = 0i64;
                let mut k = j;
                while k < lines.len() {
                    depth += lines[k].matches('{').count() as i64;
                    depth -= lines[k].matches('}').count() as i64;
                    if depth <= 0 {
                        break;
                    }
                    k += 1;
                }
                ranges.push((j + 1, k + 1));
                i = k;
            }
        }
        i += 1;
    }
    ranges
}

/// The lines of `source` that name the system temporary directory, as
/// `(line number, text)`, among the lines `in_test` accepts.
fn temp_dir_uses(source: &str, in_test: impl Fn(usize) -> bool) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line))
        .filter(|&(n, line)| {
            in_test(n) && !line.trim_start().starts_with("//") && line.contains(NEEDLE)
        })
        .map(|(n, line)| (n, line.trim().to_string()))
        .collect()
}

/// Every use of the system temporary directory in the Rust test code under
/// `crates`, as `path:line: text`.
fn shared_temp_paths(crates: &Path) -> Vec<String> {
    let mut files = Vec::new();
    rust_sources(crates, &mut files);
    files.sort();
    let mut found = Vec::new();
    for path in files {
        let source = std::fs::read_to_string(&path).expect("valid UTF-8 source");
        let uses = if is_test_file(&path) {
            temp_dir_uses(&source, |_| true)
        } else {
            let ranges = cfg_test_module_lines(&source);
            temp_dir_uses(&source, |n| {
                ranges.iter().any(|&(a, b)| (a..=b).contains(&n))
            })
        };
        for (line, text) in uses {
            found.push(format!("{}:{line}: {text}", path.display()));
        }
    }
    found
}

#[test]
fn rust_tests_write_only_under_a_temporary_directory_of_their_own() {
    let crates = repo_root().join("crates");
    let found = shared_temp_paths(&crates);
    assert!(
        found.is_empty(),
        "{} line(s) of Rust test code name the system temporary directory, so \
         concurrent runs share their files; take a `tempfile::tempdir()` and \
         keep it alive while the test uses it (specs/STANDARDS.md, \"A test \
         writes only under a temporary directory of its own\"):\n{}",
        found.len(),
        found.join("\n")
    );
}

/// The scan itself finds a use in each kind of test code and none in
/// production code or a comment.
#[test]
fn the_scan_reads_test_code_and_only_test_code() {
    let in_mod = "fn production() {
    let _ = std::T();
}
                  #[cfg(test)]
mod tests {
    // std::T() in a comment
                      fn t() {
        let _ = std::T();
    }
}
                  fn after() {
    let _ = std::T();
}
"
    .replace('T', NEEDLE);
    let in_mod = in_mod.as_str();
    let ranges = cfg_test_module_lines(in_mod);
    let uses = temp_dir_uses(in_mod, |n| {
        ranges.iter().any(|&(a, b)| (a..=b).contains(&n))
    });
    assert_eq!(
        uses.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        [8],
        "{uses:?}"
    );

    assert!(is_test_file(Path::new("crates/x/src/foo/tests.rs")));
    assert!(is_test_file(Path::new("crates/x/src/foo/tests/bar.rs")));
    assert!(is_test_file(Path::new("crates/x/tests/it.rs")));
    assert!(!is_test_file(Path::new("crates/x/src/foo.rs")));
}
