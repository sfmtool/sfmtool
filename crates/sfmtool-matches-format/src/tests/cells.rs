// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-cell columns of `cluster_patches/` (format version 8, with the
//! `refused_outlier` status from version 9).

use super::*;

/// The member count of `make_cluster_patch_test_data`.
const K: usize = 5;

const SHIFT: &str = "cluster_patches/member_cell_shift_px.5.3.3.2.float32.zst";
const ZNCC: &str = "cluster_patches/member_cell_zncc.5.3.3.float32.zst";
const STATUS: &str = "cluster_patches/member_cell_status.5.3.3.uint8.zst";
const ITERATIONS: &str = "cluster_patches/member_cell_iterations.5.uint8.zst";

/// `make_cluster_patch_test_data` with per-cell columns: member 1, the one
/// `kept` member, carries readings with every cell status; the others carry
/// none.
fn make_cell_test_data() -> MatchesData {
    let mut data = make_cluster_patch_test_data();
    let mut cells = MemberCellData::not_attempted(K);
    let statuses = [
        [
            ClusterCellStatus::Fitted,
            ClusterCellStatus::Fitted,
            ClusterCellStatus::RefusedCurvature,
        ],
        [
            ClusterCellStatus::Fitted,
            ClusterCellStatus::RefusedZncc,
            ClusterCellStatus::RefusedOutlier,
        ],
        [
            ClusterCellStatus::RefusedBound,
            ClusterCellStatus::Fitted,
            ClusterCellStatus::NotAttempted,
        ],
    ];
    for (row, row_statuses) in statuses.iter().enumerate() {
        for (col, &status) in row_statuses.iter().enumerate() {
            cells.status[[1, row, col]] = status as u8;
            let measured = matches!(
                status,
                ClusterCellStatus::Fitted
                    | ClusterCellStatus::RefusedZncc
                    | ClusterCellStatus::RefusedOutlier
            );
            if measured {
                cells.shift_px[[1, row, col, 0]] = 0.125 * row as f32 - 0.0625;
                cells.shift_px[[1, row, col, 1]] = -0.25 * col as f32 + 0.03125;
            }
            if status != ClusterCellStatus::NotAttempted {
                cells.zncc[[1, row, col]] = 0.9 - 0.05 * (row * 3 + col) as f32;
            }
        }
    }
    cells.iterations[1] = 3;
    data.cluster_patches.as_mut().unwrap().member_cells = Some(cells);
    data
}

/// Bit-exact comparison of two cell column sets (NaN == NaN).
fn assert_cells_eq(actual: &MemberCellData, expected: &MemberCellData) {
    let bits = |v: &f32| v.to_bits();
    assert_eq!(actual.shift_px.shape(), expected.shift_px.shape());
    assert_eq!(
        actual.shift_px.iter().map(bits).collect::<Vec<_>>(),
        expected.shift_px.iter().map(bits).collect::<Vec<_>>()
    );
    assert_eq!(actual.zncc.shape(), expected.zncc.shape());
    assert_eq!(
        actual.zncc.iter().map(bits).collect::<Vec<_>>(),
        expected.zncc.iter().map(bits).collect::<Vec<_>>()
    );
    assert_eq!(actual.status, expected.status);
    assert_eq!(actual.iterations, expected.iterations);
}

/// Write `make_cell_test_data`, apply `mutate` to the archive entries and
/// rebuild it with fresh hashes.
fn craft_cell_file(
    name: &str,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u8>)>),
) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("valid.matches");
    let dst = dir.join("crafted.matches");
    write_matches(&src, &make_cell_test_data(), 3).unwrap();
    let mut entries = load_archive_entries(&src);
    mutate(&mut entries);
    rebuild_matches_archive(&entries, &dst);
    (dir, dst)
}

/// The `member_cell_status_names` value of a file (`Null` when absent).
fn stored_cell_status_names(path: &std::path::Path) -> serde_json::Value {
    let entries = load_archive_entries(path);
    let cp_meta: serde_json::Value =
        serde_json::from_slice(&entries.iter().find(|(n, _)| n == CP_METADATA).unwrap().1).unwrap();
    cp_meta["member_cell_status_names"].clone()
}

#[test]
fn test_member_cells_round_trip() {
    let data = make_cell_test_data();
    let (dir, path) = write_to_temp("matches_test_member_cells_round_trip", &data);

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "{errors:?}");
    assert_eq!(
        read_matches_metadata(&path).unwrap().version,
        MATCHES_FORMAT_VERSION
    );
    assert_eq!(
        stored_cell_status_names(&path),
        serde_json::json!([
            "fitted",
            "refused_curvature",
            "refused_zncc",
            "not_attempted",
            "refused_bound",
            "refused_outlier"
        ])
    );
    let names: Vec<String> = load_archive_entries(&path)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for entry in [SHIFT, ZNCC, STATUS, ITERATIONS] {
        assert!(names.iter().any(|n| n == entry), "{entry} in {names:?}");
    }

    let loaded = read_matches(&path).unwrap();
    let cp = loaded.cluster_patches.as_ref().unwrap();
    assert_cells_eq(
        cp.member_cells.as_ref().expect("cells read back"),
        data.cluster_patches
            .as_ref()
            .unwrap()
            .member_cells
            .as_ref()
            .unwrap(),
    );

    // Writing the loaded data back gives the same bytes.
    let again = dir.join("again.matches");
    write_matches(&again, &loaded, 3).unwrap();
    assert_eq!(
        read_matches(&again).unwrap().content_hash.content_xxh128,
        loaded.content_hash.content_xxh128
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_cluster_patches_without_cells_write_no_cell_entries() {
    let (dir, path) = write_to_temp(
        "matches_test_member_cells_absent",
        &make_cluster_patch_test_data(),
    );
    assert!(!load_archive_entries(&path)
        .iter()
        .any(|(n, _)| n.starts_with("cluster_patches/member_cell_")));
    assert_eq!(stored_cell_status_names(&path), serde_json::Value::Null);
    let loaded = read_matches(&path).unwrap();
    assert!(loaded.cluster_patches.unwrap().member_cells.is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_7_file_reads_with_no_cells() {
    // A version 7 file has no per-cell columns: it reads and verifies, and
    // reports no cells.
    let dir = std::env::temp_dir().join("matches_test_member_cells_v7");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let current = dir.join("current.matches");
    let v7 = dir.join("v7.matches");
    write_matches(&current, &make_cell_test_data(), 3).unwrap();
    rewrite_matches_version(&current, &v7, 7);
    assert!(!load_archive_entries(&v7)
        .iter()
        .any(|(n, _)| n.starts_with("cluster_patches/member_cell_")));

    let (valid, errors) = verify_matches(&v7).unwrap();
    assert!(valid, "{errors:?}");
    let loaded = read_matches(&v7).unwrap();
    assert_eq!(loaded.metadata.version, MATCHES_FORMAT_VERSION);
    let cp = loaded.cluster_patches.unwrap();
    assert!(cp.member_cells.is_none());
    assert_eq!(
        cp.member_status,
        make_cluster_patch_test_data()
            .cluster_patches
            .unwrap()
            .member_status
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_8_file_reads_unchanged() {
    // A version 8 file's legend cannot name refused_outlier: with the five
    // statuses version 8 defined, it reads and verifies as before.
    let mut expected = make_cell_test_data()
        .cluster_patches
        .unwrap()
        .member_cells
        .unwrap();
    // The one refused_outlier cell becomes fitted, which version 8 knew.
    expected.status[[1, 1, 2]] = ClusterCellStatus::Fitted as u8;
    let dir = std::env::temp_dir().join("matches_test_member_cells_v8");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let current = dir.join("current.matches");
    let v8 = dir.join("v8.matches");
    let mut data = make_cell_test_data();
    data.cluster_patches.as_mut().unwrap().member_cells = Some(expected.clone());
    write_matches(&current, &data, 3).unwrap();
    rewrite_matches_version(&current, &v8, 8);
    assert_eq!(
        stored_cell_status_names(&v8),
        serde_json::json!(ClusterCellStatus::NAMES[..5])
    );

    let (valid, errors) = verify_matches(&v8).unwrap();
    assert!(valid, "{errors:?}");
    let loaded = read_matches(&v8).unwrap();
    assert_eq!(loaded.metadata.version, MATCHES_FORMAT_VERSION);
    assert_cells_eq(
        loaded
            .cluster_patches
            .as_ref()
            .unwrap()
            .member_cells
            .as_ref()
            .unwrap(),
        &expected,
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_member_cell_status_read_through_another_legend() {
    // The reader resolves each cell code through the file's legend and hands
    // back the canonical numbering; writing it back states the canonical
    // legend again.
    let expected = make_cell_test_data()
        .cluster_patches
        .unwrap()
        .member_cells
        .unwrap();
    let mut reversed = ClusterCellStatus::NAMES;
    reversed.reverse();
    let (dir, path) = craft_cell_file("matches_test_member_cell_legend_reversed", |entries| {
        mutate_cp_metadata(entries, |json| {
            json["member_cell_status_names"] = serde_json::json!(reversed)
        });
        mutate_entry(entries, STATUS, |bytes| {
            for code in bytes.iter_mut() {
                *code = (ClusterCellStatus::ALL.len() - 1) as u8 - *code;
            }
        });
    });
    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "{errors:?}");
    let loaded = read_matches(&path).unwrap();
    assert_cells_eq(
        loaded
            .cluster_patches
            .as_ref()
            .unwrap()
            .member_cells
            .as_ref()
            .unwrap(),
        &expected,
    );

    let rewritten = dir.join("rewritten.matches");
    write_matches(&rewritten, &loaded, 3).unwrap();
    assert_eq!(
        stored_cell_status_names(&rewritten),
        serde_json::json!(ClusterCellStatus::NAMES)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_member_cell_status_partial_legend() {
    // Like `member_status_names`, the cell legend may name any subset of the
    // defined statuses: a file whose cells use five of them may state only
    // those five, and its codes are read through that shorter legend.
    let mut expected = make_cell_test_data()
        .cluster_patches
        .unwrap()
        .member_cells
        .unwrap();
    // The one refused_bound cell becomes refused_curvature, so no cell uses
    // refused_bound and the legend can leave it out.
    expected.status[[1, 2, 0]] = ClusterCellStatus::RefusedCurvature as u8;
    let legend = [
        ClusterCellStatus::NotAttempted,
        ClusterCellStatus::Fitted,
        ClusterCellStatus::RefusedOutlier,
        ClusterCellStatus::RefusedZncc,
        ClusterCellStatus::RefusedCurvature,
    ];
    let names: Vec<&str> = legend.iter().map(|s| s.as_str()).collect();
    let stored: Vec<u8> = expected
        .status
        .iter()
        .map(|&code| legend.iter().position(|s| *s as u8 == code).unwrap() as u8)
        .collect();
    let (dir, path) = craft_cell_file("matches_test_member_cell_legend_partial", |entries| {
        mutate_cp_metadata(entries, |json| {
            json["member_cell_status_names"] = serde_json::json!(names)
        });
        mutate_entry(entries, STATUS, |bytes| bytes.copy_from_slice(&stored));
    });
    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "{errors:?}");
    let loaded = read_matches(&path).unwrap();
    assert_cells_eq(
        loaded
            .cluster_patches
            .as_ref()
            .unwrap()
            .member_cells
            .as_ref()
            .unwrap(),
        &expected,
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_malformed_member_cells_rejected() {
    // Each case is refused by the reader and reported by the verifier.
    type Mutation = Box<dyn FnOnce(&mut Vec<(String, Vec<u8>)>)>;
    let cases: Vec<(&str, Mutation, &str)> = vec![
        (
            "column_without_legend",
            Box::new(|entries| {
                mutate_cp_metadata(entries, |json| {
                    json.as_object_mut()
                        .unwrap()
                        .remove("member_cell_status_names");
                })
            }),
            "is present but cluster_patches/metadata.json carries no member_cell_status_names",
        ),
        (
            "legend_without_column",
            Box::new(|entries| entries.retain(|(n, _)| n != ZNCC)),
            "carries member_cell_status_names but the member_cell_zncc column is missing",
        ),
        (
            "wrong_member_count",
            Box::new(|entries| {
                let entry = entries.iter_mut().find(|(n, _)| n == SHIFT).unwrap();
                entry.0 = "cluster_patches/member_cell_shift_px.4.3.3.2.float32.zst".into();
                entry.1.truncate(4 * 18 * 4);
            }),
            "member_cell_shift_px.4.3.3.2.float32.zst does not match the member count",
        ),
        (
            "short_column",
            Box::new(|entries| mutate_entry(entries, ZNCC, |bytes| bytes.truncate(4 * 9 * 4))),
            "member_cell_zncc",
        ),
        (
            "status_past_legend",
            Box::new(|entries| mutate_entry(entries, STATUS, |bytes| bytes[9] = 6)),
            "member_cell_status[1][0][0] is 6, past the 6 names its legend gives",
        ),
        (
            "unknown_legend_name",
            Box::new(|entries| {
                mutate_cp_metadata(entries, |json| {
                    json["member_cell_status_names"] =
                        serde_json::json!(["fitted", "refused", "refused_zncc", "not_attempted"])
                })
            }),
            "member_cell_status_names[1] is \"refused\", not one of",
        ),
        (
            "reading_on_a_member_not_kept",
            Box::new(|entries| mutate_entry(entries, STATUS, |bytes| bytes[2 * 9 + 4] = 0)),
            "member_cell_status[2][1][1] is fitted, but member 2 is rejected_low_zncc, not kept",
        ),
        (
            "zncc_on_a_member_not_kept",
            Box::new(|entries| {
                mutate_entry(entries, ZNCC, |bytes| {
                    bytes[0..4].copy_from_slice(&0.5f32.to_le_bytes())
                })
            }),
            "member_cell_zncc[0][0][0] is 0.5, but member 0 is reference, not kept",
        ),
        (
            "shift_on_a_member_not_kept",
            Box::new(|entries| {
                mutate_entry(entries, SHIFT, |bytes| {
                    let at = 4 * (2 * 18 + 2 * 6 + 2 + 1);
                    bytes[at..at + 4].copy_from_slice(&0.25f32.to_le_bytes())
                })
            }),
            "member_cell_shift_px[2][2][1][1] is 0.25, but member 2 is rejected_low_zncc, not kept",
        ),
        (
            "iterations_on_a_member_not_kept",
            Box::new(|entries| mutate_entry(entries, ITERATIONS, |bytes| bytes[0] = 2)),
            "member_cell_iterations[0] is 2, but member 0 is reference, not kept",
        ),
        (
            "version_7_with_legend",
            Box::new(|entries| {
                mutate_metadata(entries, |json| json["version"] = serde_json::json!(7))
            }),
            "version 7 file carries cluster_patches/metadata.json member_cell_status_names \
             (introduced in version 8)",
        ),
        (
            "version_8_names_refused_outlier",
            Box::new(|entries| {
                mutate_metadata(entries, |json| json["version"] = serde_json::json!(8))
            }),
            "version 8 file names refused_outlier in cluster_patches/metadata.json \
             member_cell_status_names (introduced in version 9)",
        ),
    ];
    for (label, mutate, expected) in cases {
        let (dir, path) =
            craft_cell_file(&format!("matches_test_bad_member_cells_{label}"), mutate);
        let msg = format!("{}", read_matches(&path).err().unwrap());
        assert!(msg.contains(expected), "{label}: read said {msg}");
        let (valid, errors) = verify_matches(&path).unwrap();
        assert!(!valid, "{label}");
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "{label}: verify said {errors:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn test_write_validation_member_cells() {
    let mut wrong_shape = make_cell_test_data();
    wrong_shape
        .cluster_patches
        .as_mut()
        .unwrap()
        .member_cells
        .as_mut()
        .unwrap()
        .zncc = ndarray::Array3::from_elem((K - 1, 3, 3), f32::NAN);
    expect_write_error(
        "matches_test_write_member_cells_shape",
        &wrong_shape,
        "member_cell_zncc shape [4, 3, 3] != [5, 3, 3]",
    );

    let mut bad_status = make_cell_test_data();
    bad_status
        .cluster_patches
        .as_mut()
        .unwrap()
        .member_cells
        .as_mut()
        .unwrap()
        .status[[1, 0, 0]] = 9;
    expect_write_error(
        "matches_test_write_member_cells_status",
        &bad_status,
        "member_cell_status[1][0][0] = 9 is not a valid ClusterCellStatus discriminant",
    );

    let mut zncc_not_kept = make_cell_test_data();
    zncc_not_kept
        .cluster_patches
        .as_mut()
        .unwrap()
        .member_cells
        .as_mut()
        .unwrap()
        .zncc[[4, 1, 2]] = 0.75;
    expect_write_error(
        "matches_test_write_member_cells_zncc_not_kept",
        &zncc_not_kept,
        "member_cell_zncc[4][1][2] is 0.75, but member 4 is",
    );

    let mut not_kept = make_cell_test_data();
    not_kept
        .cluster_patches
        .as_mut()
        .unwrap()
        .member_cells
        .as_mut()
        .unwrap()
        .iterations[3] = 1;
    expect_write_error(
        "matches_test_write_member_cells_not_kept",
        &not_kept,
        "member_cell_iterations[3] is 1, but member 3 is not_evaluated, not kept",
    );
}

#[test]
fn test_select_clusters_gathers_member_cells() {
    // A selection keeps each surviving member's cells beside it.
    let data = make_cell_test_data();
    let selected = data
        .select_clusters(&ClusterSelect {
            accepted_statuses: vec![ClusterMemberStatus::Reference, ClusterMemberStatus::Kept],
            ..ClusterSelect::default()
        })
        .unwrap();
    let cp = selected.cluster_patches.as_ref().unwrap();
    let cells = cp.member_cells.as_ref().unwrap();
    assert_eq!(cells.iterations.len(), cp.member_status.len());
    let src = data
        .cluster_patches
        .as_ref()
        .unwrap()
        .member_cells
        .as_ref()
        .unwrap();
    let kept = cp
        .member_status
        .iter()
        .position(|&s| s == ClusterMemberStatus::Kept as u8)
        .unwrap();
    assert_eq!(
        cells.status.index_axis(ndarray::Axis(0), kept),
        src.status.index_axis(ndarray::Axis(0), 1)
    );
    assert_eq!(cells.iterations[kept], 3);
}
