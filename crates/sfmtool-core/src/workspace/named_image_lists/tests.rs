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
            {"pattern": ""},
            {"pattern": "f_%4d.jpg"}
        ]}"#,
    ));
    let expected: Vec<String> = [
        "/abs/x.jpg",
        "frames/../x.jpg",
        "f_%d_%04d.jpg",
        "a**/x.jpg",
        "",
        "f_%4d.jpg",
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
    // A narrowing that leaves nothing says the range did it.
    let e = resolve(&lists, "a", Some("50")).unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::NarrowedToNothing { .. }),
        "{e}"
    );
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
fn a_file_an_entry_misses_only_by_the_padding_of_its_number_is_an_error() {
    // A pattern written when a frame field matched any run of digits: `%d`
    // over an ffmpeg sequence that crosses from 0999 to 1000.
    let dir = workspace();
    for path in [
        "f/x_0998.jpg",
        "f/x_0999.jpg",
        "f/x_1000.jpg",
        "f/x_12345.jpg",
        "f/y_007.jpg",
    ] {
        touch(dir.path(), path);
    }
    let lists = write_lists(
        dir.path(),
        r#"{"loose": [{"pattern": "f/x_%d.jpg"}],
            "padded": [{"pattern": "f/x_%04d.jpg"}],
            "ranged": [{"pattern": "f/x_%d.jpg", "range": "1000-2000"}],
            "y": [{"pattern": "f/y_%04d.jpg"}]}"#,
    );
    let e = resolve(&lists, "loose", None).unwrap_err();
    let NamedImageListsError::FramePadding { entries, .. } = &e else {
        panic!("expected FramePadding, got: {e}");
    };
    assert_eq!(
        entries,
        &[(
            0,
            "f/x_%d.jpg".to_string(),
            vec!["f/x_0998.jpg".to_string(), "f/x_0999.jpg".to_string()]
        )]
    );
    let message = e.to_string();
    assert!(
        message.contains("entry 1 leaves out 'f/x_0998.jpg' and 1 other file"),
        "{message}"
    );
    // The same list against a reconstruction's names.
    let names = ["f/x_0999.jpg", "f/x_1000.jpg"];
    let e = lists.resolve_in_names("loose", &names, None).unwrap_err();
    assert!(
        matches!(&e, NamedImageListsError::FramePadding { entries, .. }
            if entries[0].2 == ["f/x_0999.jpg"]),
        "{e}"
    );
    // `%04d` writes every number of the sequence, 12345 included.
    assert_eq!(
        paths(&resolve(&lists, "padded", None).unwrap()),
        [
            "f/x_0998.jpg",
            "f/x_0999.jpg",
            "f/x_1000.jpg",
            "f/x_12345.jpg"
        ]
    );
    // A file outside the entry's range was never the entry's to take.
    assert_eq!(
        paths(&resolve(&lists, "ranged", None).unwrap()),
        ["f/x_1000.jpg"]
    );
    // A list that only misses files names them rather than matching nothing.
    let e = resolve(&lists, "y", None).unwrap_err();
    assert!(
        matches!(&e, NamedImageListsError::FramePadding { entries, .. }
            if entries[0].2 == ["f/y_007.jpg"]),
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

#[test]
fn literal_entries_are_looked_up_by_their_exact_name() {
    let dir = workspace();
    for name in [
        "photos/DSC_0042.JPG",
        "photos/dsc_0057.jpg",
        "p%/a.jpg",
        "photos/x/y.jpg",
    ] {
        touch(dir.path(), name);
    }
    for i in 0..50 {
        touch(dir.path(), &format!("photos/other_{i}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"picked": [
            {"pattern": "photos/DSC_0042.JPG"},
            {"pattern": "photos/DSC_0057.JPG"},
            {"pattern": "p%%/a.jpg"},
            {"pattern": "photos/x"},
            {"pattern": "photos/*/y.jpg"}
        ]}"#,
    );
    let expected = vec!["photos/DSC_0042.JPG", "p%/a.jpg", "photos/x/y.jpg"];
    let members = resolve(&lists, "picked", None).unwrap();
    assert_eq!(paths(&members), expected);
    assert_eq!(
        members.iter().map(|m| m.entry).collect::<Vec<_>>(),
        vec![0, 2, 4]
    );

    let mut names: Vec<String> = (0..50).map(|i| format!("photos/other_{i}.jpg")).collect();
    names.extend(
        [
            "photos/x/y.jpg",
            "p%/a.jpg",
            "photos/dsc_0057.jpg",
            "photos/DSC_0042.JPG",
        ]
        .map(String::from),
    );
    let members = lists.resolve_in_names("picked", &names, None).unwrap();
    assert_eq!(paths(&members), expected);
}

#[test]
fn a_narrowing_range_that_leaves_nothing_names_the_list_and_the_range() {
    let dir = workspace();
    for i in 1..=5 {
        touch(dir.path(), &format!("frames/walk_{i:04}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"walk": [{"pattern": "frames/walk_%04d.jpg"}]}"#,
    );
    let e = resolve(&lists, "walk", Some("100-200")).unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::NarrowedToNothing { .. }),
        "{e}"
    );
    assert_eq!(
        e.to_string(),
        format!(
            "{}: the range '100-200' left no members of named image list 'walk'",
            lists.path().display()
        )
    );
    let names = ["frames/walk_0001.jpg"];
    let e = lists
        .resolve_in_names("walk", &names, Some(&range("7")))
        .unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::NarrowedToNothing { .. }),
        "{e}"
    );
    // A list that matches nothing before narrowing is still that error.
    let e = lists
        .resolve_in_names("walk", &["other.jpg"], Some(&range("1")))
        .unwrap_err();
    assert!(matches!(e, NamedImageListsError::NoMatch { .. }), "{e}");
}

// ── Slicing ──────────────────────────────────────────────────────────────

fn slice(text: &str) -> NamedImageListSlice {
    NamedImageListSlice::parse(text).unwrap()
}

#[test]
fn a_slice_indexes_the_resolved_members_by_position_not_frame() {
    let dir = workspace();
    // Irregular frame numbers, as a picked list holds.
    for i in [3, 7, 8, 20, 41, 42] {
        touch(dir.path(), &format!("frames/walk1_{i:04}.jpg"));
    }
    let lists = write_lists(
        dir.path(),
        r#"{"walk1": [{"pattern": "frames/walk1_%04d.jpg"}]}"#,
    );
    let members = resolve(&lists, "walk1", None).unwrap();
    let frames = |kept: Vec<NamedImageListMember>| -> Vec<Option<i64>> {
        kept.into_iter().map(|m| m.frame).collect()
    };
    let sliced = |text: &str| {
        lists
            .slice_members("walk1", None, &slice(text), members.clone())
            .unwrap()
    };
    assert_eq!(frames(sliced("[:2]")), [Some(3), Some(7)]);
    assert_eq!(frames(sliced("[-2:]")), [Some(41), Some(42)]);
    assert_eq!(frames(sliced("[::2]")), [Some(3), Some(8), Some(41)]);
    assert_eq!(
        frames(sliced("[::-1]")),
        [Some(42), Some(41), Some(20), Some(8), Some(7), Some(3)]
    );
    // [3:8] is by position: the 4th and later, not frames 3 to 7.
    assert_eq!(frames(sliced("[3:8]")), [Some(20), Some(41), Some(42)]);
    assert_eq!(frames(sliced("[:30]")).len(), 6);

    // Narrowed first, then sliced.
    let narrow = range("5-41");
    let narrowed = resolve(&lists, "walk1", Some("5-41")).unwrap();
    let kept = lists
        .slice_members("walk1", Some(&narrow), &slice("[1:3]"), narrowed)
        .unwrap();
    assert_eq!(frames(kept), [Some(8), Some(20)]);

    // Any item type: paths as well as members.
    let paths: Vec<&str> = vec!["a.jpg", "b.jpg", "c.jpg"];
    assert_eq!(
        lists
            .slice_members("walk1", None, &slice("[1:]"), paths)
            .unwrap(),
        ["b.jpg", "c.jpg"]
    );
}

#[test]
fn a_rig_slice_counts_files_not_frames() {
    let dir = workspace();
    for i in [10, 12, 14] {
        for cam in ["cam0", "cam1"] {
            touch(dir.path(), &format!("rig/{cam}/frame_{i:05}.jpg"));
        }
    }
    let lists = write_lists(
        dir.path(),
        r#"{"rig": [{"pattern": "rig/*/frame_%05d.jpg"}]}"#,
    );
    let members = resolve(&lists, "rig", None).unwrap();
    let kept = |text: &str| {
        let kept = lists
            .slice_members("rig", None, &slice(text), members.clone())
            .unwrap();
        kept.into_iter().map(|m| m.path).collect::<Vec<_>>()
    };
    assert_eq!(
        kept("[:3]"),
        [
            "rig/cam0/frame_00010.jpg",
            "rig/cam1/frame_00010.jpg",
            "rig/cam0/frame_00012.jpg"
        ]
    );
    assert_eq!(
        kept("[1::2]"),
        [
            "rig/cam1/frame_00010.jpg",
            "rig/cam1/frame_00012.jpg",
            "rig/cam1/frame_00014.jpg"
        ]
    );
}

#[test]
fn a_slice_that_keeps_nothing_names_the_file_list_slice_and_length() {
    let dir = workspace();
    let lists = write_lists(
        dir.path(),
        r#"{"walk1": [{"pattern": "frames/walk1_%04d.jpg"}]}"#,
    );
    let items: Vec<usize> = (0..30).collect();
    let e = lists
        .slice_members("walk1", None, &slice("[40:]"), items.clone())
        .unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::SlicedToNothing { len: 30, .. }),
        "{e:?}"
    );
    assert_eq!(
        e.to_string(),
        format!(
            "{}: the slice [40:] of named image list 'walk1' leaves none of its 30 images",
            lists.path().display()
        )
    );
    let narrow = range("100-900");
    let e = lists
        .slice_members("walk1", Some(&narrow), &slice("[5:2]"), items)
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        format!(
            "{}: the slice [5:2] of named image list 'walk1' narrowed by the range '100-900' \
             leaves none of its 30 images",
            lists.path().display()
        )
    );
    let e = lists
        .slice_members("walk1", None, &slice("[1:]"), vec![0])
        .unwrap_err();
    assert!(e.to_string().ends_with("leaves none of its 1 image"), "{e}");
    let e = lists
        .slice_members("nope", None, &slice("[:]"), vec![0])
        .unwrap_err();
    assert!(matches!(e, NamedImageListsError::UnknownList { .. }), "{e}");
}

// ── Writing ──────────────────────────────────────────────────────────────

/// The example file of `specs/workspace/named-image-lists.md`, in the layout
/// the writer writes.
const SPEC_EXAMPLE: &str = r#"{
  "version": 1,
  "named_image_lists": {
    "walk1_sharp": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-120,131-400" }
    ],
    "rig_sharp": [
      { "pattern": "rig/*/frame_%05d.jpg", "range": "10-900:2" }
    ],
    "hero_shots": [
      { "pattern": "photos/DSC_0042.JPG" },
      { "pattern": "photos/DSC_0057.JPG" }
    ]
  }
}
"#;

fn entry(pattern: &str, range_text: Option<&str>) -> NamedImageListEntry {
    NamedImageListEntry::new(pattern, range_text.map(range))
}

fn list(name: &str, entries: Vec<NamedImageListEntry>) -> NamedImageList {
    NamedImageList::new(name, entries)
}

fn lists_file(root: &Path) -> String {
    std::fs::read_to_string(root.join(NAMED_IMAGE_LISTS_FILE_NAME)).unwrap()
}

fn names(lists: &NamedImageLists) -> Vec<&str> {
    lists.lists().iter().map(|l| l.name()).collect()
}

/// The names in the workspace root, sorted, so a test can see that no
/// temporary file was left beside the lists file.
fn root_names(root: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

#[test]
fn writing_creates_the_file_when_the_workspace_has_none() {
    let dir = workspace();
    let path = NamedImageLists::write_lists(
        dir.path(),
        &[
            list("walk1", vec![entry("frames/walk1_%04d.jpg", Some("1-5,9"))]),
            list("hero", vec![entry("photos/DSC_0042.JPG", None)]),
            list("empty", vec![]),
        ],
        false,
    )
    .unwrap();
    assert_eq!(path, dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME));
    assert_eq!(
        lists_file(dir.path()),
        r#"{
  "version": 1,
  "named_image_lists": {
    "walk1": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-5,9" }
    ],
    "hero": [
      { "pattern": "photos/DSC_0042.JPG" }
    ],
    "empty": []
  }
}
"#
    );
    let read = NamedImageLists::read(dir.path()).unwrap().unwrap();
    assert_eq!(names(&read), ["walk1", "hero", "empty"]);
    assert_eq!(read.lists()[1].entries()[0].range(), None);
    assert_eq!(
        root_names(dir.path()),
        [NAMED_IMAGE_LISTS_FILE_NAME, WORKSPACE_FILE_NAME]
    );
}

#[test]
fn writing_no_lists_to_a_new_file_writes_an_empty_lists_object() {
    let dir = workspace();
    NamedImageLists::write_lists(dir.path(), &[], false).unwrap();
    assert_eq!(
        lists_file(dir.path()),
        "{\n  \"version\": 1,\n  \"named_image_lists\": {}\n}\n"
    );
}

#[test]
fn a_new_list_goes_at_the_end_and_the_existing_text_is_unchanged() {
    let dir = workspace();
    std::fs::write(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME), SPEC_EXAMPLE).unwrap();
    NamedImageLists::write_lists(
        dir.path(),
        &[list(
            "walk2",
            vec![entry("frames/walk2_%d.jpg", Some("3-30:3"))],
        )],
        false,
    )
    .unwrap();
    let before_end = SPEC_EXAMPLE.strip_suffix("\n  }\n}\n").unwrap();
    assert_eq!(
        lists_file(dir.path()),
        format!(
            "{before_end},\n    \"walk2\": [\n      {{ \"pattern\": \"frames/walk2_%d.jpg\", \
             \"range\": \"3-30:3\" }}\n    ]\n  }}\n}}\n"
        )
    );
}

#[test]
fn replacing_a_list_with_itself_leaves_the_file_byte_for_byte() {
    let dir = workspace();
    std::fs::write(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME), SPEC_EXAMPLE).unwrap();
    let read = NamedImageLists::read(dir.path()).unwrap().unwrap();
    let rig = read.get("rig_sharp").unwrap().clone();
    NamedImageLists::write_lists(dir.path(), &[rig], true).unwrap();
    assert_eq!(lists_file(dir.path()), SPEC_EXAMPLE);
}

#[test]
fn an_existing_range_is_written_from_its_text_and_a_new_one_by_display() {
    let dir = workspace();
    // A range that `RangeExpr`'s `Display` would write as "1-3,7", in a
    // layout other than the writer's.
    std::fs::write(
        dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME),
        one_list(r#"{"a": [{"range": " 1,2,3, 7", "pattern": "f_%d.jpg"}]}"#),
    )
    .unwrap();
    NamedImageLists::write_lists(
        dir.path(),
        &[list("b", vec![entry("g_%d.jpg", Some("1,2,3, 7"))])],
        false,
    )
    .unwrap();
    let text = lists_file(dir.path());
    assert!(
        text.contains(r#"{ "pattern": "f_%d.jpg", "range": " 1,2,3, 7" }"#),
        "{text}"
    );
    assert!(
        text.contains(r#"{ "pattern": "g_%d.jpg", "range": "1-3,7" }"#),
        "{text}"
    );
    let read = NamedImageLists::read(dir.path()).unwrap().unwrap();
    assert_eq!(read.lists()[0].entries()[0].range(), Some(&range("1-3,7")));
}

#[test]
fn a_name_the_file_defines_is_refused_without_replace_and_nothing_is_written() {
    let dir = workspace();
    std::fs::write(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME), SPEC_EXAMPLE).unwrap();
    let e = NamedImageLists::write_lists(
        dir.path(),
        &[
            list("hero_shots", vec![entry("photos/DSC_0001.JPG", None)]),
            list("new", vec![entry("photos/DSC_0002.JPG", None)]),
            list("walk1_sharp", vec![entry("frames/walk1_%04d.jpg", None)]),
        ],
        false,
    )
    .unwrap_err();
    let NamedImageListsError::ListExists { names, .. } = &e else {
        panic!("expected ListExists, got: {e}");
    };
    assert_eq!(names, &["hero_shots", "walk1_sharp"]);
    let path = dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME);
    assert_eq!(
        e.to_string(),
        format!(
            "{0}: the named image list 'hero_shots' is already defined, so nothing was \
             written\n{0}: the named image list 'walk1_sharp' is already defined, so nothing \
             was written",
            path.display()
        )
    );
    assert_eq!(lists_file(dir.path()), SPEC_EXAMPLE);
}

#[test]
fn a_replaced_list_keeps_its_place_and_new_lists_follow_in_order() {
    let dir = workspace();
    std::fs::write(dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME), SPEC_EXAMPLE).unwrap();
    NamedImageLists::write_lists(
        dir.path(),
        &[
            list("zeta", vec![entry("z.jpg", None)]),
            list(
                "rig_sharp",
                vec![entry("rig/cam0/frame_%05d.jpg", Some("1-9"))],
            ),
            list("alpha", vec![]),
        ],
        true,
    )
    .unwrap();
    let read = NamedImageLists::read(dir.path()).unwrap().unwrap();
    assert_eq!(
        names(&read),
        ["walk1_sharp", "rig_sharp", "hero_shots", "zeta", "alpha"]
    );
    let rig = read.get("rig_sharp").unwrap();
    assert_eq!(rig.entries().len(), 1);
    assert_eq!(rig.entries()[0].pattern(), "rig/cam0/frame_%05d.jpg");
    assert_eq!(rig.entries()[0].range(), Some(&range("1-9")));
    // The lists that were not replaced are written as they were.
    let text = lists_file(dir.path());
    assert!(
        text.starts_with(
            "{\n  \"version\": 1,\n  \"named_image_lists\": {\n    \"walk1_sharp\": [\n      \
             { \"pattern\": \"frames/walk1_%04d.jpg\", \"range\": \"1-120,131-400\" }\n    ],\n"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "    \"hero_shots\": [\n      { \"pattern\": \"photos/DSC_0042.JPG\" },\n      \
             { \"pattern\": \"photos/DSC_0057.JPG\" }\n    ],\n"
        ),
        "{text}"
    );
}

#[test]
fn new_lists_that_break_the_rules_are_refused_and_nothing_is_written() {
    let dir = workspace();
    // Values whose gaps alternate between 1 and more than 1 make a range whose
    // `Display` has more sub-ranges than a range expression may hold.
    let squares: Vec<i64> = (0..30_000).map(|i: i64| i * i).collect();
    let too_long = RangeExpr::from_values(squares).unwrap();
    let e = NamedImageLists::write_lists(
        dir.path(),
        &[
            list("1-10", vec![entry("a_%d.jpg", None)]),
            list("bad_pattern", vec![entry("../a.jpg", None)]),
            list("no_field", vec![entry("a.jpg", Some("1-3"))]),
            list(
                "empty_range",
                vec![NamedImageListEntry::new(
                    "a_%d.jpg",
                    Some(RangeExpr::from_values(vec![]).unwrap()),
                )],
            ),
            list(
                "too_long",
                vec![NamedImageListEntry::new("a_%d.jpg", Some(too_long))],
            ),
            list("good", vec![entry("a_%d.jpg", None)]),
            list("good", vec![entry("b_%d.jpg", None)]),
        ],
        false,
    )
    .unwrap_err();
    let path = dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME);
    for line in e.to_string().lines() {
        assert!(
            line.starts_with(&format!("{}: nothing was written: ", path.display())),
            "{line}"
        );
    }
    let NamedImageListsError::InvalidNewLists { problems, .. } = e else {
        panic!("expected InvalidNewLists, got: {e}");
    };
    assert_eq!(problems.len(), 6, "{problems:#?}");
    assert!(problems[0].starts_with("the new list name '1-10' is not valid; "));
    assert!(problems[1].starts_with("new list 'bad_pattern' entry 1: "));
    assert!(problems[2].starts_with("new list 'no_field' entry 1: has a 'range'"));
    assert!(problems[3].starts_with("new list 'empty_range' entry 1: 'range' '' "));
    assert!(problems[4].starts_with("new list 'too_long' entry 1: 'range' '0,1,4,9,"));
    assert!(
        problems[4].contains("too many sub-ranges"),
        "{}",
        &problems[4][problems[4].len() - 200..]
    );
    assert_eq!(problems[5], "the new list 'good' is given more than once");
    assert!(!path.exists());

    // An existing file is left as it was.
    std::fs::write(&path, SPEC_EXAMPLE).unwrap();
    let e = NamedImageLists::write_lists(
        dir.path(),
        &[list("x", vec![entry("a.jpg", Some("1"))])],
        true,
    )
    .unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::InvalidNewLists { .. }),
        "{e}"
    );
    assert_eq!(lists_file(dir.path()), SPEC_EXAMPLE);
}

#[test]
fn a_lists_file_that_fails_the_readers_checks_is_refused_and_left_untouched() {
    let dir = workspace();
    let path = dir.path().join(NAMED_IMAGE_LISTS_FILE_NAME);
    let new = [list("new", vec![entry("a.jpg", None)])];
    let invalid_entry_key = one_list(r#"{"a": [{"pattern": "a.jpg", "colour": "red"}]}"#);
    let unknown_top_level_key = r#"{"version": 1, "named_image_lists": {}, "comment": "x"}"#;
    let other_version = r#"{"version": 2, "anything": []}"#;
    let not_json = "{\"version\": 1,";
    for text in [
        invalid_entry_key.as_str(),
        unknown_top_level_key,
        other_version,
        not_json,
    ] {
        std::fs::write(&path, text).unwrap();
        let e = NamedImageLists::write_lists(dir.path(), &new, true).unwrap_err();
        let expected = if text == other_version {
            matches!(e, NamedImageListsError::UnsupportedVersion { .. })
        } else if text == not_json {
            matches!(e, NamedImageListsError::InvalidJson { .. })
        } else {
            matches!(e, NamedImageListsError::Invalid { .. })
        };
        assert!(expected, "{text}: {e}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    }
    assert_eq!(
        root_names(dir.path()),
        [NAMED_IMAGE_LISTS_FILE_NAME, WORKSPACE_FILE_NAME]
    );
}

#[test]
fn writing_into_a_directory_that_is_not_a_workspace_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let e = NamedImageLists::write_lists(dir.path(), &[list("a", vec![])], false).unwrap_err();
    assert!(
        matches!(e, NamedImageListsError::NotAWorkspace { .. }),
        "{e}"
    );
    assert!(root_names(dir.path()).is_empty());
}

#[test]
fn a_pattern_that_json_escapes_reads_back_the_same() {
    let dir = workspace();
    let pattern = "say \"hi\"/caf\u{e9}\ttab_%d.jpg";
    NamedImageLists::write_lists(dir.path(), &[list("q", vec![entry(pattern, None)])], false)
        .unwrap();
    let text = lists_file(dir.path());
    assert!(text.contains(r#""say \"hi\"/café\ttab_%d.jpg""#), "{text}");
    let read = NamedImageLists::read(dir.path()).unwrap().unwrap();
    assert_eq!(read.get("q").unwrap().entries()[0].pattern(), pattern);
}
