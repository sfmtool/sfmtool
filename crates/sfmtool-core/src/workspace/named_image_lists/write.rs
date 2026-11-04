// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The writer of the lists file: [`NamedImageLists::write_lists`] adds lists
//! to a workspace's `.sfm-named-image-lists.json` or replaces lists in it.
//!
//! The file is written whole in one layout, the one
//! `specs/workspace/named-image-lists.md` shows: two-space indents, one line
//! per entry. The reader allows nothing in the file but the version, the
//! lists, and each entry's pattern and range, so writing those back keeps all
//! of its content; each range is written from the text the file held, not
//! from the parsed `RangeExpr`, whose `Display` can spell the same frame
//! numbers differently.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::json::Json;
use super::{
    is_named_image_list_name, validate_entry, NamedImageList, NamedImageLists,
    NamedImageListsError, NAMED_IMAGE_LISTS_FILE_NAME, NAMED_IMAGE_LISTS_VERSION, NAME_RULE,
};

impl NamedImageLists {
    /// Add `lists` to the lists file of the workspace whose root is
    /// `workspace_dir`, or replace lists of the same names in it, creating the
    /// file when the workspace has none. Returns the file's path.
    ///
    /// The new lists are checked first, by the rules the reader applies to a
    /// file, with every problem reported in `InvalidNewLists`; a name given
    /// twice in `lists` is one of them. Then the existing file is read and
    /// checked as [`Self::read`] does, with its errors, `NotAWorkspace`
    /// included. A list the file already defines is refused with
    /// `ListExists`, naming every such list, unless `replace` is true.
    ///
    /// The file is rewritten whole: every list it held keeps its place and
    /// order, a replaced list takes the place of the one it replaces, and new
    /// lists follow in the order given. The bytes go to a temporary file
    /// beside the lists file, which is then renamed over it, so a write that
    /// fails at any point leaves the old file as it was. Nothing is written
    /// when any check fails.
    pub fn write_lists(
        workspace_dir: &Path,
        lists: &[NamedImageList],
        replace: bool,
    ) -> Result<PathBuf, NamedImageListsError> {
        let path = workspace_dir.join(NAMED_IMAGE_LISTS_FILE_NAME);
        let new_lists = check_new_lists(lists, &path)?;
        let mut written = Self::read(workspace_dir)?
            .map(|existing| existing.lists)
            .unwrap_or_default();
        if !replace {
            let names: Vec<String> = new_lists
                .iter()
                .filter(|list| written.iter().any(|old| old.name == list.name))
                .map(|list| list.name.clone())
                .collect();
            if !names.is_empty() {
                return Err(NamedImageListsError::ListExists { path, names });
            }
        }
        for list in new_lists {
            match written.iter_mut().find(|old| old.name == list.name) {
                Some(old) => *old = list,
                None => written.push(list),
            }
        }
        let text = render(&written);
        sfmtool_archive_io::write_atomically(&path, |file| file.write_all(text.as_bytes()))
            .map_err(|source| NamedImageListsError::Io {
                lists_path: path.clone(),
                path: path.clone(),
                source,
            })?;
        Ok(path)
    }
}

/// Check the lists to write by the reader's rules, and return them as the
/// reader would have built them; `InvalidNewLists` with every problem when any
/// breaks a rule.
///
/// Each entry goes through the reader's own entry check, with its range as
/// the text that will be written, so a range whose `Display` would not parse
/// back is refused here rather than written.
fn check_new_lists(
    lists: &[NamedImageList],
    path: &Path,
) -> Result<Vec<NamedImageList>, NamedImageListsError> {
    let mut problems = Vec::new();
    let mut names = BTreeSet::new();
    let mut checked = Vec::with_capacity(lists.len());
    for list in lists {
        let name = &list.name;
        if !names.insert(name.as_str()) {
            problems.push(format!("the new list '{name}' is given more than once"));
            continue;
        }
        let before = problems.len();
        if !is_named_image_list_name(name) {
            problems.push(format!(
                "the new list name '{name}' is not valid; {NAME_RULE}"
            ));
        }
        let mut entries = Vec::with_capacity(list.entries.len());
        for (index, entry) in list.entries.iter().enumerate() {
            let mut members = vec![("pattern".to_string(), Json::String(entry.pattern.clone()))];
            if let Some((_, text)) = &entry.range {
                members.push(("range".to_string(), Json::String(text.clone())));
            }
            let at = format!("new list '{name}' entry {}", index + 1);
            if let Some(entry) = validate_entry(&at, &Json::Object(members), &mut problems) {
                entries.push(entry);
            }
        }
        if problems.len() == before {
            checked.push(NamedImageList {
                name: name.clone(),
                entries,
            });
        }
    }
    if problems.is_empty() {
        Ok(checked)
    } else {
        Err(NamedImageListsError::InvalidNewLists {
            path: path.to_path_buf(),
            problems,
        })
    }
}

/// The text of a lists file holding `lists`, in the file's layout.
fn render(lists: &[NamedImageList]) -> String {
    let mut out =
        format!("{{\n  \"version\": {NAMED_IMAGE_LISTS_VERSION},\n  \"named_image_lists\": {{");
    for (i, list) in lists.iter().enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        out.push_str("    ");
        out.push_str(&json_string(&list.name));
        out.push_str(": [");
        for (j, entry) in list.entries.iter().enumerate() {
            out.push_str(if j == 0 { "\n" } else { ",\n" });
            out.push_str("      { \"pattern\": ");
            out.push_str(&json_string(&entry.pattern));
            if let Some((_, text)) = &entry.range {
                out.push_str(", \"range\": ");
                out.push_str(&json_string(text));
            }
            out.push_str(" }");
        }
        if !list.entries.is_empty() {
            out.push_str("\n    ");
        }
        out.push(']');
    }
    if !lists.is_empty() {
        out.push_str("\n  ");
    }
    out.push_str("}\n}\n");
    out
}

/// `text` as a JSON string, quoted and escaped.
fn json_string(text: &str) -> String {
    serde_json::Value::String(text.to_string()).to_string()
}
