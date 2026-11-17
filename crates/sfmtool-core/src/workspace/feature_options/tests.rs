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

// The expected names and JSON below were computed by the Python
// implementation that named every folder before `feature_prefix_dir` did
// (`get_feature_tool_xxh128` in `sfmtool.sift.file`, removed since), so these
// tests hold the Rust function to the names already on disk.

#[test]
fn the_folder_names_of_the_tool_defaults_are_the_ones_python_gave() {
    let colmap = json!({
        "domain_size_pooling": false, "edge_threshold": 10.0,
        "estimate_affine_shape": false,
        "image_to_gray": {"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8},
        "max_image_size": 4096, "max_num_features": null, "normalization": "L1_ROOT",
        "peak_threshold": 0.006666666666666667, "upright": false, "use_gpu": true,
    });
    assert_eq!(
        feature_prefix_dir("colmap", "sift-colmap", &colmap),
        "features/sift-colmap-c2405a72d0875a97223cbeccadab3291"
    );
    let opencv = json!({
        "contrastThreshold": 0.04, "edgeThreshold": 10,
        "image_to_gray": {"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8},
        "nOctaveLayers": 3, "nfeatures": 0, "sigma": 1.6,
    });
    assert_eq!(
        feature_prefix_dir("opencv", "sift-opencv", &opencv),
        "features/sift-opencv-3bb5fdfd886e9635e4d821bb73a6c743"
    );
    assert_eq!(
        feature_prefix_dir("opencv", "sift-opencv", &json!({})),
        "features/sift-opencv-9abd05623e163b58c61a1b726b5ddb99"
    );
}

#[test]
fn a_version_1_record_and_its_upgrade_name_the_folders_python_gave() {
    // The record of a real workspace `sfm ws init` wrote with a cap of 2500
    // features, and the same record after `upgrade_feature_options`.
    let v1 = json!({
        "blur_radius_factor": 2.25, "contrast_threshold": 0.0067, "descriptor_bins": 8,
        "descriptor_clamp": 0.2, "descriptor_magnification": 3.0, "descriptor_width": 4,
        "double_image": true, "edge_threshold": 10.0,
        "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B", "input_sigma": 0.5,
        "max_num_features": 2500, "octave_layers": 3, "orientation_bins": 36,
        "peak_ratio": 0.8, "sigma": 1.6,
    });
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &v1),
        "features/sift-sfmtool-fcc10056c50160cb2ca677365606978e"
    );
    let v2 = upgrade_feature_options("sfmtool", &v1).unwrap();
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &v2),
        "features/sift-sfmtool-37ce76962479ffbf2752277c1fdcd2ba"
    );
}

/// The compact JSON `feature_prefix_dir` hashes for these options.
fn hashed_json(feature_tool: &str, feature_type: &str, options: &serde_json::Value) -> String {
    let mut options = options.clone();
    options.as_object_mut().unwrap().remove("use_gpu");
    let mut out = String::new();
    write_python_json(
        &json!({"feature_tool": feature_tool, "feature_type": feature_type,
                "feature_options": options}),
        &mut out,
    );
    out
}

#[test]
fn floats_are_written_as_python_writes_them() {
    let options = json!({
        "tiny": 1e-05, "small": 0.0001, "smaller": 0.00001234, "c": 0.0067,
        "third": 0.006666666666666667, "integral": 3.0, "frac": 2.25, "ten": 10.0,
        "e15": 1e15, "e16": 1e16, "e22": 1e22, "huge": 1.5e300, "neg": -2.5e-07,
        "negzero": -0.0, "zero": 0.0, "long": 123456789012345680.0, "min": 5e-324,
    });
    assert_eq!(
        hashed_json("sfmtool", "sift-sfmtool", &options),
        r#"{"feature_options":{"c":0.0067,"e15":1000000000000000.0,"e16":1e+16,"e22":1e+22,"frac":2.25,"huge":1.5e+300,"integral":3.0,"long":1.2345678901234568e+17,"min":5e-324,"neg":-2.5e-07,"negzero":-0.0,"small":0.0001,"smaller":1.234e-05,"ten":10.0,"third":0.006666666666666667,"tiny":1e-05,"zero":0.0},"feature_tool":"sfmtool","feature_type":"sift-sfmtool"}"#
    );
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
        "features/sift-sfmtool-3dd52f7569e403295d50176251b5d1d7"
    );
}

#[test]
fn a_float_halfway_between_two_shortest_strings_takes_the_even_digit_as_python_does() {
    // 1997913866023742.25 exactly: `.2` and `.3` both read back, and are
    // equally close.
    let options = json!({"tie": 1997913866023742.2, "tie2": 0.30000000000000004});
    assert_eq!(
        hashed_json("sfmtool", "sift-sfmtool", &options),
        r#"{"feature_options":{"tie":1997913866023742.2,"tie2":0.30000000000000004},"feature_tool":"sfmtool","feature_type":"sift-sfmtool"}"#
    );
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
        "features/sift-sfmtool-75f3eebccafb7e88d78631666c87d34e"
    );
}

#[test]
fn ints_bools_null_and_the_top_level_use_gpu_are_written_as_python_writes_them() {
    let options = json!({
        "int": 7, "negint": -3, "u64max": u64::MAX, "i64min": i64::MIN,
        "t": true, "f": false, "n": null, "use_gpu": true,
    });
    assert_eq!(
        hashed_json("colmap", "sift-colmap", &options),
        r#"{"feature_options":{"f":false,"i64min":-9223372036854775808,"int":7,"n":null,"negint":-3,"t":true,"u64max":18446744073709551615},"feature_tool":"colmap","feature_type":"sift-colmap"}"#
    );
    assert_eq!(
        feature_prefix_dir("colmap", "sift-colmap", &options),
        "features/sift-colmap-7a8e2882182d26867b58cd10b9c8faab"
    );
}

#[test]
fn nested_objects_and_lists_are_written_with_sorted_keys_and_keep_a_nested_use_gpu() {
    let options = json!({
        "zeta": 1,
        "alpha": {"y": 0.5, "b": [1, 2.0, "x", null, {"q": 1e-07}], "use_gpu": false},
        "image_to_gray": {"quantize_bits": 8, "formula": "0.299*R + 0.587*G + 0.114*B"},
    });
    assert_eq!(
        hashed_json("sfmtool", "sift-sfmtool", &options),
        r#"{"feature_options":{"alpha":{"b":[1,2.0,"x",null,{"q":1e-07}],"use_gpu":false,"y":0.5},"image_to_gray":{"formula":"0.299*R + 0.587*G + 0.114*B","quantize_bits":8},"zeta":1},"feature_tool":"sfmtool","feature_type":"sift-sfmtool"}"#
    );
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
        "features/sift-sfmtool-7e734e8215cf5096e78a73379c675064"
    );
}

#[test]
fn strings_and_keys_are_escaped_and_sorted_as_python_writes_them() {
    let options = json!({
        "s": "\u{e9} \u{fc} \u{2603} \u{1F600} \"q\" \\ / \n\r\t\u{8}\u{c}\u{1}\u{1f}\u{7f}",
        "Z": "", "_": "a", "\u{e9}": "key", "a": "x",
    });
    // Python's output, with `|` standing for a backslash before each `u`
    // escape so the expected text is plain ASCII.
    let expected = r#"{"feature_options":{"Z":"","_":"a","a":"x","s":"|u00e9 |u00fc |u2603 |ud83d|ude00 \"q\" \\ / \n\r\t\b\f|u0001|u001f|u007f","|u00e9":"key"},"feature_tool":"sfmtool","feature_type":"sift-sfmtool"}"#;
    assert_eq!(
        hashed_json("sfmtool", "sift-sfmtool", &options),
        expected.replace('|', "\\")
    );
    assert_eq!(
        feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
        "features/sift-sfmtool-06f6970d3ddc032fb0a1348752c32818"
    );
}
