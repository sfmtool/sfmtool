// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::*;

#[test]
fn each_tool_records_its_own_conversion() {
    let row = |tool| tool_image_to_gray(tool).map(|c| c.to_json());
    assert_eq!(
        row("sfmtool"),
        Some(json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B"}))
    );
    assert_eq!(
        row("COLMAP"),
        Some(json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}))
    );
    assert_eq!(
        row("opencv"),
        Some(json!({"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}))
    );
    assert_eq!(row("sfmtool-undistort"), None);
    assert_eq!(row("mytool"), None);
}

#[test]
fn an_sfmtool_upgrade_keeps_every_option_and_converts_the_gray_formula() {
    let v1 = json!({
        "peak_ratio": 0.7,
        "max_num_features": 2500,
        "gray_formula": "0.5*R + 0.5*G",
    });
    let v2 = upgrade_feature_options("sfmtool", &v1).unwrap();
    assert_eq!(
        v2,
        json!({
            "peak_ratio": 0.7,
            "max_num_features": 2500,
            "image_to_gray": {"formula": "0.5*R + 0.5*G"},
        })
    );
    // Options with no gray_formula took the extractor's default.
    assert_eq!(
        upgrade_feature_options("sfmtool", &json!({})).unwrap(),
        json!({"image_to_gray": {"formula": "0.2126*R + 0.7152*G + 0.0722*B"}})
    );
}

#[test]
fn a_colmap_or_opencv_upgrade_gains_the_tool_conversion() {
    let v1 = json!({"max_image_size": 2000, "use_gpu": false});
    assert_eq!(
        upgrade_feature_options("colmap", &v1).unwrap(),
        json!({
            "max_image_size": 2000,
            "use_gpu": false,
            "image_to_gray": {"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8},
        })
    );
    assert_eq!(
        upgrade_feature_options("opencv", &json!({"nfeatures": 0})).unwrap(),
        json!({
            "nfeatures": 0,
            "image_to_gray": {"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8},
        })
    );
}

#[test]
fn an_upgrade_refuses_what_it_cannot_convert() {
    let err = |tool: &str, options: serde_json::Value| {
        upgrade_feature_options(tool, &options).unwrap_err()
    };
    assert_eq!(
        err("sfmtool", json!({"image_to_gray": {"formula": "G"}})),
        FeatureOptionsError::AlreadyVersion2
    );
    assert_eq!(
        err("sfmtool-undistort", json!({})),
        FeatureOptionsError::Undistort
    );
    assert!(err("sfmtool-undistort", json!({}))
        .to_string()
        .contains("run 'sfm undistort' again"));
    assert_eq!(
        err("mytool", json!({})),
        FeatureOptionsError::UnknownTool("mytool".into())
    );
    assert!(matches!(
        err("sfmtool", json!({"gray_formula": "R +"})),
        FeatureOptionsError::GrayFormula(_)
    ));
    assert!(matches!(
        err("sfmtool", json!({"gray_formula": 3})),
        FeatureOptionsError::GrayFormula(_)
    ));
    assert_eq!(err("sfmtool", json!([])), FeatureOptionsError::NotAnObject);
}

#[test]
fn the_recorded_conversion_reads_either_version() {
    let read = |tool: &str, options: serde_json::Value| {
        recorded_image_to_gray(tool, &options).unwrap().to_json()
    };
    assert_eq!(
        read(
            "sfmtool",
            json!({"image_to_gray": {"formula": "G", "quantize_bits": 4}})
        ),
        json!({"formula": "G", "quantize_bits": 4})
    );
    assert_eq!(
        read("sfmtool", json!({"gray_formula": "G"})),
        json!({"formula": "G"})
    );
    assert_eq!(
        read("sfmtool", json!({})),
        json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B"})
    );
    assert_eq!(
        read("opencv", json!({})),
        json!({"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8})
    );
    // An undistort record states its own object, or else its source's.
    assert_eq!(
        read(
            "sfmtool-undistort",
            json!({"image_to_gray": {"formula": "B"}, "source_feature_tool": "colmap",
                   "source_feature_options": {}})
        ),
        json!({"formula": "B"})
    );
    assert_eq!(
        read(
            "sfmtool-undistort",
            json!({"source_feature_tool": "sfmtool",
                   "source_feature_options": {"gray_formula": "R"}})
        ),
        json!({"formula": "R"})
    );
    // A version 1 undistort of a version 1 undistort.
    assert_eq!(
        read(
            "sfmtool-undistort",
            json!({"source_feature_tool": "sfmtool-undistort",
                   "source_feature_options": {"source_feature_tool": "colmap",
                                              "source_feature_options": {}}})
        ),
        json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8})
    );
    assert_eq!(
        recorded_image_to_gray("sfmtool-undistort", &json!({})).unwrap_err(),
        FeatureOptionsError::NoSource
    );
    assert!(matches!(
        recorded_image_to_gray("sfmtool", &json!({"image_to_gray": {"formula": "R +"}})),
        Err(FeatureOptionsError::ImageToGray(_))
    ));
}

#[test]
fn the_folder_name_leaves_out_use_gpu_and_changes_with_the_conversion() {
    let options = json!({"max_image_size": 4096, "image_to_gray": {"formula": "G"}});
    let mut with_gpu = options.clone();
    with_gpu["use_gpu"] = json!(true);
    let a = feature_prefix_dir("colmap", "sift-colmap", &options);
    assert_eq!(a, feature_prefix_dir("colmap", "sift-colmap", &with_gpu));
    assert!(a.starts_with("features/sift-colmap-"));
    assert_eq!(a.len(), "features/sift-colmap-".len() + 32);
    let mut other = options.clone();
    other["image_to_gray"] = json!({"formula": "G", "quantize_bits": 8});
    assert_ne!(a, feature_prefix_dir("colmap", "sift-colmap", &other));
}
