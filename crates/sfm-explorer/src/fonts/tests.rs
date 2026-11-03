// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Every non-ASCII character the viewer's code writes has a glyph in each
//! font family a text can be drawn in.
//!
//! The characters are read from the viewer's own source rather than listed
//! here, so a new string with a new character is checked without anyone
//! remembering to add it. Test files are left out: their strings are
//! inputs, such as image names in other scripts, not text the viewer writes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::*;

/// Every `.rs` file under `dir` that is not a test file.
fn source_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the viewer's source directory") {
        let path = entry.expect("a directory entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if name != "tests" {
                source_files(&path, files);
            }
        } else if name.ends_with(".rs") && name != "tests.rs" && name != "test_support.rs" {
            files.push(path);
        }
    }
}

/// The non-ASCII characters in `source` outside comments: those written
/// literally in strings and character literals, and those a string or a
/// character literal spells as a `\u{…}` escape.
///
/// A small scanner rather than a Rust parser: it knows where a string starts
/// and ends, so a `//` inside one is not taken for a comment, and it steps
/// over a character literal whole, so `'"'` does not open a string. A
/// comment's characters are left out because a comment is never drawn.
fn written_characters(source: &str, found: &mut BTreeSet<char>) {
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            match c {
                '\\' if chars.get(i + 1) == Some(&'u') && chars.get(i + 2) == Some(&'{') => {
                    let close = (i + 3..chars.len())
                        .find(|&j| chars[j] == '}')
                        .expect("a closed \\u{} escape");
                    let hex: String = chars[i + 3..close].iter().collect();
                    let code = u32::from_str_radix(&hex, 16).expect("a hex \\u{} escape");
                    let escaped = char::from_u32(code).expect("a valid \\u{} escape");
                    if !escaped.is_ascii() {
                        found.insert(escaped);
                    }
                    i = close + 1;
                    continue;
                }
                '\\' => i += 1,
                '"' => in_string = false,
                _ if !c.is_ascii() => {
                    found.insert(c);
                }
                _ => {}
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        } else if c == '"' {
            in_string = true;
        } else if c == '\'' {
            // A character literal is `'x'` or an escape such as `'\''`; a
            // lifetime has no closing quote and is stepped over as code.
            let len = if chars.get(i + 1) == Some(&'\\') {
                (i + 3..chars.len().min(i + 12)).find(|&j| chars[j] == '\'')
            } else if chars.get(i + 2) == Some(&'\'') {
                Some(i + 2)
            } else {
                None
            };
            if let Some(close) = len {
                let body: String = chars[i + 1..close].iter().collect();
                if let Some(hex) = body.strip_prefix("\\u{").and_then(|b| b.strip_suffix('}')) {
                    let code = u32::from_str_radix(hex, 16).expect("a hex \\u{} escape");
                    let escaped = char::from_u32(code).expect("a valid \\u{} escape");
                    if !escaped.is_ascii() {
                        found.insert(escaped);
                    }
                }
                for d in body.chars() {
                    if !d.is_ascii() {
                        found.insert(d);
                    }
                }
                i = close + 1;
                continue;
            }
        } else if !c.is_ascii() {
            found.insert(c);
        }
        i += 1;
    }
}

#[test]
fn the_scanner_reads_strings_and_skips_comments() {
    let mut found = BTreeSet::new();
    written_characters(
        "// a comment's × is never drawn\n\
         let a = \"0.1 → 0.2 // still a string ·\"; // but this — is a comment\n\
         let b = '\"'; let c = \"\\u{2212}\"; let d = '°'; fn f<'a>(x: &'a str) {}\n\
         let e = '\\u{25b8}'; let f = '\\'';\n\
         /* a block comment's ≤ */",
        &mut found,
    );
    assert_eq!(found, BTreeSet::from(['→', '·', '−', '°', '▸']));
}

#[test]
fn every_character_the_viewer_writes_has_a_glyph_in_each_family() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    source_files(&src, &mut files);
    let mut found = BTreeSet::new();
    for file in &files {
        let source = std::fs::read_to_string(file).expect("a readable source file");
        written_characters(&source, &mut found);
    }
    // Characters the viewer is known to write, the `→` the Action Log row's
    // tooltip drew as an empty box among them, so a scanner that stopped
    // finding characters cannot pass this test by finding none.
    for c in ['→', '·', '×', '°', '…', '—'] {
        assert!(found.contains(&c), "the scan found no {c:?} in {src:?}");
    }

    let text: String = found.iter().collect();
    let mut missing: BTreeMap<String, String> = BTreeMap::new();
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let lacking = lacking_glyphs(&family, &text);
        if !lacking.is_empty() {
            missing.insert(format!("{family:?}"), lacking);
        }
    }
    assert!(
        missing.is_empty(),
        "characters the viewer writes that a font family has no glyph for: {missing:?}"
    );
}
