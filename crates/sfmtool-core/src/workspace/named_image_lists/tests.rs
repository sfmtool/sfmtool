// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::AtomicBool;

fn ws() -> PathBuf {
    PathBuf::from("ws")
}

fn lists_path() -> String {
    ws().join(NAMED_IMAGE_LISTS_FILE_NAME).display().to_string()
}

fn parse(text: &str) -> Result<NamedImageLists, NamedImageListsError> {
    NamedImageLists::parse(&ws(), text.as_bytes())
}

fn parse_ok(text: &str) -> NamedImageLists {
    parse(text).unwrap_or_else(|e| panic!("expected a valid file, got: {e}"))
}

/// The problems of a file the validator rejects, after checking that the
/// message names the file on every line.
fn problems(text: &str) -> Vec<String> {
    match parse(text) {
        Err(e @ NamedImageListsError::Invalid { .. }) => {
            let message = e.to_string();
            for line in message.lines() {
                assert!(
                    line.starts_with(&format!("{}: ", lists_path())),
                    "line does not name the file: {line}"
                );
            }
            let NamedImageListsError::Invalid { problems, .. } = e else {
                unreachable!()
            };
            problems
        }
        Err(e) => panic!("expected Invalid, got: {e}"),
        Ok(_) => panic!("expected Invalid, got a valid file"),
    }
}

fn one_list(lists: &str) -> String {
    format!(r#"{{"version": 1, "named_image_lists": {lists}}}"#)
}

fn range(text: &str) -> RangeExpr {
    RangeExpr::from_str(text).unwrap()
}

fn paths(members: &[NamedImageListMember]) -> Vec<&str> {
    members.iter().map(|m| m.path.as_str()).collect()
}

// ── Names ────────────────────────────────────────────────────────────────

#[test]
fn name_rule() {
    for good in ["a", "walk1_sharp", "rig-sharp.v2", "A", "1a", "_.-x"] {
        assert!(is_named_image_list_name(good), "{good}");
    }
    for bad in [
        "", "1", "1-10", "10-50:2", "_-.", "walk 1", "wälk", "a/b", "a,b",
    ] {
        assert!(!is_named_image_list_name(bad), "{bad}");
    }
}

// ── The file as a whole ──────────────────────────────────────────────────

#[test]
fn a_valid_file_keeps_its_lists_and_entries_in_file_order() {
    let lists = parse_ok(&one_list(
        r#"{
            "walk1_sharp": [{"pattern": "frames/walk1_%04d.jpg", "range": "1-120,131-400"}],
            "rig_sharp": [{"pattern": "rig/*/frame_%05d.jpg", "range": "10-900:2"}],
            "hero_shots": [{"pattern": "photos/DSC_0042.JPG"}, {"pattern": "photos/DSC_0057.JPG"}]
        }"#,
    ));
    let names: Vec<&str> = lists.lists().iter().map(NamedImageList::name).collect();
    assert_eq!(names, ["walk1_sharp", "rig_sharp", "hero_shots"]);
    assert_eq!(lists.path(), ws().join(NAMED_IMAGE_LISTS_FILE_NAME));
    let hero = lists.get("hero_shots").unwrap();
    assert_eq!(hero.entries().len(), 2);
    assert_eq!(hero.entries()[1].pattern(), "photos/DSC_0057.JPG");
    assert!(!hero.entries()[0].has_frame_field());
    let walk = lists.get("walk1_sharp").unwrap();
    assert!(walk.entries()[0].has_frame_field());
    assert_eq!(walk.entries()[0].range(), Some(&range("1-120,131-400")));
    // Names are compared exactly.
    assert!(lists.get("Walk1_sharp").is_none());
}

#[test]
fn an_empty_lists_object_is_valid() {
    let lists = parse_ok(r#"{"version": 1, "named_image_lists": {}}"#);
    assert!(lists.lists().is_empty());
}

#[test]
fn a_byte_order_mark_is_skipped() {
    let mut bytes = b"\xEF\xBB\xBF".to_vec();
    bytes.extend_from_slice(br#"{"version": 1, "named_image_lists": {}}"#);
    assert!(NamedImageLists::parse(&ws(), &bytes).is_ok());
}

#[test]
fn invalid_json_is_reported_with_line_and_column_and_nothing_else() {
    // Line 3 is missing a comma after the version; the unknown key and the
    // bad version would be problems, but nothing past the parse is checked.
    let text = "{\n  \"bogus\": 1,\n  \"version\": 7\n  \"named_image_lists\": {}\n}";
    let e = parse(text).unwrap_err();
    let NamedImageListsError::InvalidJson { line, column, .. } = &e else {
        panic!("expected InvalidJson, got: {e}");
    };
    assert_eq!((*line, *column), (4, 3));
    let message = e.to_string();
    assert!(
        message.starts_with(&format!(
            "{}: not valid JSON at line 4, column 3: ",
            lists_path()
        )),
        "{message}"
    );
    assert!(!message.contains(" at line 4 column 3"), "{message}");
}

#[test]
fn an_unknown_version_gives_the_exact_message_and_nothing_else() {
    let text = r#"{"version": 2, "bogus": true, "named_image_lists": {"1": 3}}"#;
    let e = parse(text).unwrap_err();
    assert!(matches!(e, NamedImageListsError::UnsupportedVersion { .. }));
    assert_eq!(
        e.to_string(),
        format!(
            "{}: named image lists version 2 is not supported; this sfmtool reads version 1",
            lists_path()
        )
    );
}

#[test]
fn a_top_level_that_is_not_an_object() {
    assert_eq!(
        problems("[1, 2]"),
        ["the file holds an array, not an object"]
    );
}

#[test]
fn missing_and_wrongly_typed_top_level_keys() {
    assert_eq!(
        problems("{}"),
        ["missing key 'version'", "missing key 'named_image_lists'"]
    );
    assert_eq!(
        problems(r#"{"version": "1", "named_image_lists": []}"#),
        [
            "'version' is a string, not an integer",
            "'named_image_lists' is an array, not an object"
        ]
    );
    assert_eq!(
        problems(r#"{"version": 1.5, "named_image_lists": {}}"#),
        ["'version' is a number, not an integer"]
    );
}

#[test]
fn unknown_and_repeated_top_level_keys() {
    assert_eq!(
        problems(r#"{"version": 1, "named_image_lists": {}, "lists": {}, "version": 1}"#),
        [
            "unknown top-level key 'lists'; the keys are 'version' and 'named_image_lists'",
            "the top-level key 'version' appears more than once",
        ]
    );
}

#[test]
fn bad_and_repeated_list_names() {
    assert_eq!(
        problems(&one_list(
            r#"{"walk 1": [{"pattern": "a.jpg"}], "10-20": [{"pattern": "b.jpg"}],
                "ok": [{"pattern": "c.jpg"}], "ok": [{"pattern": "d.jpg"}]}"#
        )),
        [
            format!("named_image_lists: 'walk 1' is not a valid list name; {NAME_RULE}"),
            format!("named_image_lists: '10-20' is not a valid list name; {NAME_RULE}"),
            "named_image_lists: the list 'ok' is defined more than once".to_string(),
        ]
    );
}

#[test]
fn wrongly_typed_lists_and_entries() {
    assert_eq!(
        problems(&one_list(r#"{"a": {"pattern": "x.jpg"}, "b": [3, null]}"#)),
        [
            "list 'a' is an object, not an array of entries",
            "list 'b' entry 1 is a number, not an object",
            "list 'b' entry 2 is null, not an object",
        ]
    );
}

#[test]
fn missing_wrongly_typed_unknown_and_repeated_entry_keys() {
    assert_eq!(
        problems(&one_list(
            r#"{"a": [
                {"range": "1-3"},
                {"pattern": 4},
                {"pattern": "x_%d.jpg", "range": 5},
                {"pattern": "y.jpg", "frames": "1"},
                {"pattern": "z.jpg", "pattern": "z.jpg"}
            ]}"#
        )),
        [
            "list 'a' entry 1: missing key 'pattern'",
            "list 'a' entry 2: 'pattern' is a number, not a string",
            "list 'a' entry 3: 'range' is a number, not a string",
            "list 'a' entry 4: unknown key 'frames'; an entry's keys are 'pattern' and 'range'",
            "list 'a' entry 5: the key 'pattern' appears more than once",
        ]
    );
}

#[test]
fn bad_patterns_use_the_camrig_checker() {
    let found = problems(&one_list(
        r#"{"a": [
            {"pattern": "/abs/x.jpg"},
            {"pattern": "frames/../x.jpg"},
            {"pattern": "f_%d_%04d.jpg"},
            {"pattern": "a**/x.jpg"},
            {"pattern": ""}
        ]}"#,
    ));
    let expected: Vec<String> = [
        "/abs/x.jpg",
        "frames/../x.jpg",
        "f_%d_%04d.jpg",
        "a**/x.jpg",
        "",
    ]
    .iter()
    .enumerate()
    .map(|(i, p)| {
        format!(
            "list 'a' entry {}: {}",
            i + 1,
            sfmtool_camrig_format::validate_pattern(p).unwrap_err()
        )
    })
    .collect();
    assert_eq!(found, expected);
    // `%%` is a literal percent, not a frame field.
    let lists = parse_ok(&one_list(
        r#"{"a": [{"pattern": "100%%/x_%d.jpg", "range": "1"}]}"#,
    ));
    assert!(lists.get("a").unwrap().entries()[0].has_frame_field());
}

#[test]
fn bad_ranges_and_a_range_without_a_frame_field() {
    let found = problems(&one_list(
        r#"{"a": [
            {"pattern": "x_%d.jpg", "range": "1-"},
            {"pattern": "photos/DSC_0042.JPG", "range": "1-3"},
            {"pattern": "100%%.jpg", "range": "1"}
        ]}"#,
    ));
    assert_eq!(found.len(), 3, "{found:?}");
    assert!(
        found[0].starts_with("list 'a' entry 1: 'range' '1-' is not a range expression: "),
        "{}",
        found[0]
    );
    assert_eq!(
        found[1],
        "list 'a' entry 2: has a 'range', but its pattern 'photos/DSC_0042.JPG' has no frame \
         field (%d or %0Nd) for the range to apply to"
    );
    assert!(
        found[2].starts_with("list 'a' entry 3: has a 'range'"),
        "{}",
        found[2]
    );
}

#[test]
fn every_problem_is_reported_in_one_pass() {
    let found = problems(
        r#"{
            "version": 1,
            "extra": 0,
            "named_image_lists": {
                "bad name": [{"pattern": "a.jpg"}],
                "a": [{"pattern": "../x.jpg"}, {"pattern": "y.jpg", "range": "1"}],
                "b": [{"pattern": "z_%d.jpg", "range": "x"}, {"pattern": "w.jpg", "colour": 1}]
            }
        }"#,
    );
    assert_eq!(found.len(), 6, "{found:#?}");
    assert!(found[0].starts_with("unknown top-level key 'extra'"));
    assert!(found[1].starts_with("named_image_lists: 'bad name' is not a valid list name"));
    assert!(found[2].starts_with("list 'a' entry 1: "));
    assert!(found[3].starts_with("list 'a' entry 2: has a 'range'"));
    assert!(found[4].starts_with("list 'b' entry 1: 'range' 'x'"));
    assert!(found[5].starts_with("list 'b' entry 2: unknown key 'colour'"));
}

// ── Reading from a workspace ─────────────────────────────────────────────

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(WORKSPACE_FILE_NAME), "{}").unwrap();
    dir
}

fn touch(root: &Path, relative: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b"").unwrap();
}

fn write_lists(root: &Path, lists: &str) -> NamedImageLists {
    std::fs::write(root.join(NAMED_IMAGE_LISTS_FILE_NAME), one_list(lists)).unwrap();
    NamedImageLists::read(root).unwrap().unwrap()
}

#[test]
fn a_missing_file_is_no_lists_and_not_an_empty_file() {
    let dir = workspace();
    assert!(NamedImageLists::read(dir.path()).unwrap().is_none());
    let lists = write_lists(dir.path(), "{}");
    assert!(lists.lists().is_empty());
    assert_eq!(lists.workspace_dir(), dir.path());
}

#[test]
fn a_directory_without_a_workspace_file_is_not_a_workspace() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME), one_list("{}")).unwrap();
    let e = NamedImageLists::read(dir.path()).unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::NotAWorkspace { .. }),
        "{e}"
    );
}

#[test]
fn a_read_error_names_the_file() {
    let dir = workspace();
    // A directory where the file should be cannot be read as one.
    std::fs::create_dir(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME)).unwrap();
    let e = NamedImageLists::read(dir.path()).unwrap_err();
    assert!(matches!(e, NamedImageListsError::Io { .. }), "{e}");
    assert!(e.to_string().contains(NAMED_IMAGE_LISTS_FILE_NAME), "{e}");
}

// ── Resolving ────────────────────────────────────────────────────────────

fn resolve(
    lists: &NamedImageLists,
    name: &str,
    narrow: Option<&str>,
) -> Result<Vec<NamedImageListMember>, NamedImageListsError> {
    let narrow = narrow.map(range);
    lists.resolve(name, narrow.as_ref(), &Progress::none())
}

#[test]
fn a_list_is_the_union_of_its_entries_in_entry_then_frame_order() {
    let dir = workspace();
    for i in 1..=12 {
        touch(dir.path(), &format!("frames/walk1_{i:04}.jpg"));
    }
    touch(dir.path(), "frames/walk1_0005.png");
    touch(dir.path(), "photos/DSC_0042.JPG");
    let lists = write_lists(
        dir.path(),
        r#"{"mixed": [
            {"pattern": "frames/walk1_%04d.jpg", "range": "8-10"},
            {"pattern": "photos/DSC_0042.JPG"},
            {"pattern": "frames/walk1_%04d.jpg", "range": "1,3"}
        ]}"#,
    );
    let members = resolve(&lists, "mixed", None).unwrap();
    assert_eq!(
        paths(&members),
        [
            "frames/walk1_0008.jpg",
            "frames/walk1_0009.jpg",
            "frames/walk1_0010.jpg",
            "photos/DSC_0042.JPG",
            "frames/walk1_0001.jpg",
            "frames/walk1_0003.jpg",
        ]
    );
    assert_eq!(
        members[3],
        NamedImageListMember {
            path: "photos/DSC_0042.JPG".to_string(),
            entry: 1,
            frame: None
        }
    );
    assert_eq!(members[0].frame, Some(8));
    assert_eq!(members[4].entry, 2);
}

#[test]
fn frames_order_numerically_not_by_name() {
    let dir = workspace();
    for i in [1, 2, 10, 100] {
        touch(dir.path(), &format!("f/img_{i}.jpg"));
    }
    let lists = write_lists(dir.path(), r#"{"a": [{"pattern": "f/img_%d.jpg"}]}"#);
    let members = resolve(&lists, "a", None).unwrap();
    assert_eq!(
        paths(&members),
        [
            "f/img_1.jpg",
            "f/img_2.jpg",
            "f/img_10.jpg",
            "f/img_100.jpg"
        ]
    );
}

#[test]
fn rig_sensors_of_one_frame_order_by_path() {
    let dir = workspace();
    for frame in 1..=3 {
        for cam in ["cam1", "cam0"] {
            touch(dir.path(), &format!("rig/{cam}/frame_{frame:05}.jpg"));
        }
    }
    let lists = write_lists(
        dir.path(),
        r#"{"rig": [{"pattern": "rig/*/frame_%05d.jpg", "range": "2-3"}]}"#,
    );
    let members = resolve(&lists, "rig", None).unwrap();
    assert_eq!(
        paths(&members),
        [
            "rig/cam0/frame_00002.jpg",
            "rig/cam1/frame_00002.jpg",
            "rig/cam0/frame_00003.jpg",
            "rig/cam1/frame_00003.jpg",
        ]
    );
}

#[test]
fn an_entry_without_a_frame_field_orders_by_path() {
    let dir = workspace();
    for name in ["c.jpg", "a.jpg", "b/x.jpg", "b.jpg"] {
        touch(dir.path(), &format!("p/{name}"));
    }
    let lists = write_lists(dir.path(), r#"{"a": [{"pattern": "p/**/*.jpg"}]}"#);
    let members = resolve(&lists, "a", None).unwrap();
    assert_eq!(
        paths(&members),
        ["p/a.jpg", "p/b.jpg", "p/b/x.jpg", "p/c.jpg"]
    );
}

#[test]
fn a_file_two_entries_match_is_an_error_naming_both() {
    let dir = workspace();
    for i in 1..=4 {
        touch(dir.path(), &format!("frames/walk1_{i:04}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"a": [
            {"pattern": "frames/walk1_%04d.jpg", "range": "1-3"},
            {"pattern": "frames/*.jpg"}
        ]}"#,
    );
    let e = resolve(&lists, "a", None).unwrap_err();
    let NamedImageListsError::DuplicateMatches { duplicates, .. } = &e else {
        panic!("expected DuplicateMatches, got: {e}");
    };
    // Frame 4 is outside entry 1's range, so only entry 2 matches it.
    let files: Vec<&str> = duplicates.iter().map(|d| d.file.as_str()).collect();
    assert_eq!(
        files,
        [
            "frames/walk1_0001.jpg",
            "frames/walk1_0002.jpg",
            "frames/walk1_0003.jpg"
        ]
    );
    assert!(duplicates
        .iter()
        .all(|d| d.first_entry == 0 && d.second_entry == 1));
    assert!(e.to_string().starts_with(&format!(
        "{}: list 'a' matches 'frames/walk1_0001.jpg' by both entry 1 and entry 2",
        dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME).display()
    )));
}

#[test]
fn matching_is_case_sensitive_on_every_platform() {
    let dir = workspace();
    touch(dir.path(), "photos/DSC_0042.JPG");
    let lists = write_lists(
        dir.path(),
        r#"{"lower": [{"pattern": "photos/dsc_0042.jpg"}],
            "dir": [{"pattern": "Photos/DSC_0042.JPG"}],
            "exact": [{"pattern": "photos/DSC_0042.JPG"}]}"#,
    );
    for name in ["lower", "dir"] {
        let e = resolve(&lists, name, None).unwrap_err();
        assert!(
            matches!(e, NamedImageListsError::NoMatch { .. }),
            "{name}: {e}"
        );
    }
    assert_eq!(
        paths(&resolve(&lists, "exact", None).unwrap()),
        ["photos/DSC_0042.JPG"]
    );
    let names = ["photos/DSC_0042.JPG"];
    assert!(lists.resolve_in_names("lower", &names, None).is_err());
}

#[test]
fn entries_match_files_not_directories() {
    let dir = workspace();
    std::fs::create_dir_all(dir.path().join("frames/walk1_0001.jpg")).unwrap();
    touch(dir.path(), "frames/walk1_0002.jpg");
    let lists = write_lists(
        dir.path(),
        r#"{"a": [{"pattern": "frames/walk1_%04d.jpg"}]}"#,
    );
    assert_eq!(
        paths(&resolve(&lists, "a", None).unwrap()),
        ["frames/walk1_0002.jpg"]
    );
}

#[test]
fn narrowing_keeps_the_members_the_range_also_holds() {
    let dir = workspace();
    for i in 1..=10 {
        touch(dir.path(), &format!("frames/walk1_{i:04}.jpg"));
        touch(dir.path(), &format!("rig/cam0/frame_{i:05}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"a": [
            {"pattern": "frames/walk1_%04d.jpg", "range": "2-9"},
            {"pattern": "rig/cam0/frame_%05d.jpg"}
        ]}"#,
    );
    let members = resolve(&lists, "a", Some("1-3,9-10")).unwrap();
    assert_eq!(
        paths(&members),
        [
            "frames/walk1_0002.jpg",
            "frames/walk1_0003.jpg",
            "frames/walk1_0009.jpg",
            "rig/cam0/frame_00001.jpg",
            "rig/cam0/frame_00002.jpg",
            "rig/cam0/frame_00003.jpg",
            "rig/cam0/frame_00009.jpg",
            "rig/cam0/frame_00010.jpg",
        ]
    );
    // A narrowing that leaves nothing is an empty match.
    let e = resolve(&lists, "a", Some("50")).unwrap_err();
    assert!(matches!(e, NamedImageListsError::NoMatch { .. }), "{e}");
}

#[test]
fn narrowing_a_list_with_an_entry_without_a_frame_field_is_an_error() {
    let dir = workspace();
    touch(dir.path(), "frames/walk1_0001.jpg");
    touch(dir.path(), "photos/DSC_0042.JPG");
    let lists = write_lists(
        dir.path(),
        r#"{"a": [{"pattern": "frames/walk1_%04d.jpg"}, {"pattern": "photos/DSC_0042.JPG"}]}"#,
    );
    let e = resolve(&lists, "a", Some("1")).unwrap_err();
    let NamedImageListsError::NarrowedEntryWithoutFrameField { entries, .. } = &e else {
        panic!("expected NarrowedEntryWithoutFrameField, got: {e}");
    };
    assert_eq!(entries, &[(1, "photos/DSC_0042.JPG".to_string())]);
    assert!(e
        .to_string()
        .contains("list 'a' entry 2 cannot be narrowed by a range"));
    // The same rule holds against names.
    let names = ["frames/walk1_0001.jpg"];
    let narrow = range("1");
    assert!(matches!(
        lists.resolve_in_names("a", &names, Some(&narrow)),
        Err(NamedImageListsError::NarrowedEntryWithoutFrameField { .. })
    ));
    // Without a range the list resolves.
    assert_eq!(resolve(&lists, "a", None).unwrap().len(), 2);
}

#[test]
fn a_list_that_matches_nothing_is_an_error_when_used() {
    let dir = workspace();
    let lists = write_lists(
        dir.path(),
        r#"{"empty": [], "none": [{"pattern": "frames/*.jpg"}]}"#,
    );
    for name in ["empty", "none"] {
        let e = resolve(&lists, name, None).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!(
                "{}: named image list '{name}' matches no image",
                lists.path().display()
            )
        );
    }
}

#[test]
fn an_unknown_name_lists_the_defined_names() {
    let lists = parse_ok(&one_list(
        r#"{"b": [{"pattern": "x.jpg"}], "a": [{"pattern": "y.jpg"}]}"#,
    ));
    let e = lists.resolve_in_names("c", &["x.jpg"], None).unwrap_err();
    assert_eq!(
        e.to_string(),
        format!(
            "{}: no named image list is called 'c'; the lists it defines are 'b', 'a'",
            lists_path()
        )
    );
    let empty = parse_ok(&one_list("{}"));
    assert!(empty
        .resolve_in_names("c", &["x.jpg"], None)
        .unwrap_err()
        .to_string()
        .ends_with("the file defines no lists"));
}

#[test]
fn names_resolve_without_files_and_files_resolve_without_names() {
    let text = r#"{"rig": [{"pattern": "rig/*/frame_%05d.jpg", "range": "1-2"}]}"#;
    // Against names: no file exists, and a name outside the workspace layout
    // the pattern describes is not a member.
    let lists = parse_ok(&one_list(text));
    let names = [
        "rig/cam1/frame_00002.jpg",
        "rig/cam0/frame_00001.jpg",
        "rig/cam0/frame_00003.jpg",
        "other/frame_00001.jpg",
        "rig/cam0/frame_00001.jpg",
    ];
    let members = lists.resolve_in_names("rig", &names, None).unwrap();
    assert_eq!(
        paths(&members),
        ["rig/cam0/frame_00001.jpg", "rig/cam1/frame_00002.jpg"]
    );

    // Against the filesystem: the same list finds the files that exist,
    // including ones no name above mentions.
    let dir = workspace();
    for path in ["rig/cam0/frame_00001.jpg", "rig/cam2/frame_00001.jpg"] {
        touch(dir.path(), path);
    }
    let lists = write_lists(dir.path(), text);
    assert_eq!(
        paths(&resolve(&lists, "rig", None).unwrap()),
        ["rig/cam0/frame_00001.jpg", "rig/cam2/frame_00001.jpg"]
    );
}

#[test]
fn double_star_reaches_every_depth_once() {
    let dir = workspace();
    for path in ["a/x.jpg", "a/b/x.jpg", "a/b/c/x.jpg", "x.jpg", "a/b/y.png"] {
        touch(dir.path(), path);
    }
    let lists = write_lists(
        dir.path(),
        r#"{"one": [{"pattern": "**/x.jpg"}],
            "two": [{"pattern": "**/**/x.jpg"}],
            "tail": [{"pattern": "a/**"}]}"#,
    );
    let all = ["a/b/c/x.jpg", "a/b/x.jpg", "a/x.jpg", "x.jpg"];
    assert_eq!(paths(&resolve(&lists, "one", None).unwrap()), all);
    assert_eq!(paths(&resolve(&lists, "two", None).unwrap()), all);
    assert_eq!(
        paths(&resolve(&lists, "tail", None).unwrap()),
        ["a/b/c/x.jpg", "a/b/x.jpg", "a/b/y.png", "a/x.jpg"]
    );
}

#[test]
fn a_literal_percent_matches_a_literal_percent() {
    let dir = workspace();
    touch(dir.path(), "100%/x_1.jpg");
    touch(dir.path(), "100a/x_1.jpg");
    let lists = write_lists(dir.path(), r#"{"a": [{"pattern": "100%%/x_%d.jpg"}]}"#);
    assert_eq!(
        paths(&resolve(&lists, "a", None).unwrap()),
        ["100%/x_1.jpg"]
    );
}

#[test]
fn a_frame_number_too_large_is_an_error() {
    let lists = parse_ok(&one_list(r#"{"a": [{"pattern": "f_%d.jpg"}]}"#));
    let names = ["f_99999999999999999999.jpg"];
    let e = lists.resolve_in_names("a", &names, None).unwrap_err();
    assert!(
        matches!(
            e,
            NamedImageListsError::FrameNumberTooLarge { entry: 0, .. }
        ),
        "{e}"
    );
}

#[test]
fn resolving_reports_progress_and_stops_when_cancelled() {
    let dir = workspace();
    for i in 1..=5 {
        touch(dir.path(), &format!("frames/walk1_{i:04}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"a": [{"pattern": "frames/walk1_%04d.jpg"}]}"#,
    );

    let counts = std::sync::Mutex::new(Vec::new());
    let sink = |event: sfmtool_progress::Event<'_>| {
        if let sfmtool_progress::Event::Count { done, .. } = event {
            counts.lock().unwrap().push(done);
        }
    };
    lists.resolve("a", None, &Progress::to(&sink)).unwrap();
    // The root holds the workspace file, the lists file and `frames`; then
    // `frames` holds five files.
    assert_eq!(*counts.lock().unwrap(), [3, 8]);

    let stop = AtomicBool::new(true);
    let e = lists
        .resolve("a", None, &Progress::none().cancelled_by(&stop))
        .unwrap_err();
    assert!(matches!(e, NamedImageListsError::Cancelled), "{e}");
}
