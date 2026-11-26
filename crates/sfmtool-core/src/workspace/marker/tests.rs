// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// What `sfm ws init` writes for a default `sfmtool` workspace: Python's
/// `json.dump(record, f, indent=2)` of the record `init_workspace` built
/// before this writer existed, with `\n` line breaks, and the
/// `mipmap_options` object after it.
const DEFAULT_MARKER: &str = r#"{
  "version": 2,
  "feature_tool": "sfmtool",
  "feature_type": "sift-sfmtool",
  "feature_options": {
    "octave_layers": 3,
    "sigma": 1.6,
    "blur_radius_factor": 2.25,
    "input_sigma": 0.5,
    "double_image": true,
    "double_image_below_long_side": 4096,
    "contrast_threshold": 0.0067,
    "edge_threshold": 10.0,
    "max_num_features": 8192,
    "orientation_bins": 36,
    "peak_ratio": 0.8,
    "descriptor_width": 4,
    "descriptor_bins": 8,
    "descriptor_magnification": 3.0,
    "descriptor_clamp": 0.2,
    "image_to_gray": {
      "formula": "0.2126*R + 0.7152*G + 0.0722*B"
    }
  },
  "feature_prefix_dir": "features/sift-sfmtool-fc19c88d9c98d94ebff1b97f2a345bb6",
  "mipmap_options": {
    "out_of_core_long_side": 1536,
    "jpeg_quality": 95,
    "full_resolution_jpeg_quality": 95
  }
}"#;

/// The same for `sfm ws init --max-features 4000 --image-to-gray G
/// --quantize-bits 6`.
const CUSTOM_MARKER: &str = r#"{
  "version": 2,
  "feature_tool": "sfmtool",
  "feature_type": "sift-sfmtool",
  "feature_options": {
    "octave_layers": 3,
    "sigma": 1.6,
    "blur_radius_factor": 2.25,
    "input_sigma": 0.5,
    "double_image": true,
    "double_image_below_long_side": 4096,
    "contrast_threshold": 0.0067,
    "edge_threshold": 10.0,
    "max_num_features": 4000,
    "orientation_bins": 36,
    "peak_ratio": 0.8,
    "descriptor_width": 4,
    "descriptor_bins": 8,
    "descriptor_magnification": 3.0,
    "descriptor_clamp": 0.2,
    "image_to_gray": {
      "formula": "G",
      "quantize_bits": 6
    }
  },
  "feature_prefix_dir": "features/sift-sfmtool-3c0cb5ee50910f836de546984da60395",
  "mipmap_options": {
    "out_of_core_long_side": 1536,
    "jpeg_quality": 95,
    "full_resolution_jpeg_quality": 95
  }
}"#;

fn custom() -> SfmtoolWorkspaceOptions {
    SfmtoolWorkspaceOptions {
        max_num_features: 4000,
        image_to_gray: ImageToGray::new("G", Some(6)).unwrap(),
    }
}

#[test]
fn the_marker_is_what_python_wrote_for_the_same_options() {
    assert_eq!(
        sfmtool_workspace_marker(&SfmtoolWorkspaceOptions::default()),
        DEFAULT_MARKER
    );
    assert_eq!(sfmtool_workspace_marker(&custom()), CUSTOM_MARKER);
}

#[test]
fn the_feature_options_read_back_to_the_parameters_they_record() {
    for options in [SfmtoolWorkspaceOptions::default(), custom()] {
        let read = SiftParams::from_feature_options(&options.feature_options()).unwrap();
        let expected = options.sift_params();
        assert_eq!(format!("{read:?}"), format!("{expected:?}"));
        assert_eq!(read.double_image_below_long_side, Some(4096));
        assert_eq!(read.max_num_features, Some(options.max_num_features));
    }
}

#[test]
fn a_new_workspace_reads_back_the_new_workspace_mipmap_options() {
    let dir = tempfile::tempdir().unwrap();
    for options in [SfmtoolWorkspaceOptions::default(), custom()] {
        write_sfmtool_workspace_marker(dir.path(), &options).unwrap();
        assert_eq!(
            crate::workspace::read_mipmap_options(dir.path()).unwrap(),
            Some(NEW_WORKSPACE_MIPMAP_OPTIONS)
        );
    }
    assert_eq!(
        NEW_WORKSPACE_MIPMAP_OPTIONS,
        crate::camera::mipmap::MipmapOptions {
            out_of_core_long_side: 1536,
            jpeg_quality: 95,
            full_resolution_jpeg_quality: 95,
        }
    );
}

#[test]
fn write_creates_the_directory_and_replaces_an_old_marker() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("new").join("workspace");
    let text = write_sfmtool_workspace_marker(&workspace, &custom()).unwrap();
    assert_eq!(text, CUSTOM_MARKER);
    let marker = workspace.join(".sfm-workspace.json");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), CUSTOM_MARKER);

    let text =
        write_sfmtool_workspace_marker(&workspace, &SfmtoolWorkspaceOptions::default()).unwrap();
    assert_eq!(text, DEFAULT_MARKER);
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), DEFAULT_MARKER);

    // A path under a file cannot be a workspace directory.
    let under_a_file = marker.join("inner");
    assert!(write_sfmtool_workspace_marker(&under_a_file, &custom()).is_err());
}

#[test]
fn the_indented_writer_writes_empty_containers_lists_and_escapes_as_python() {
    let value = serde_json::json!({
        "zeta": [1, 2.5, [], {}],
        "version": "caf\u{e9} \"q\"",
        "alpha": null,
    });
    let mut text = String::new();
    write_indented(&value, 0, &mut text);
    // json.dumps({"version": "café \"q\"", "alpha": None,
    //             "zeta": [1, 2.5, [], {}]}, indent=2)
    assert_eq!(
        text,
        r#"{
  "version": "caf\u00e9 \"q\"",
  "alpha": null,
  "zeta": [
    1,
    2.5,
    [],
    {}
  ]
}"#
    );
}
