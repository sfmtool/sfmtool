// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The proposal banner's words: what the 3D view says at its top centre while
//! a camera model proposal is open.
//!
//! See `specs/gui/viewport-hud.md` § "The proposal banner". The first line
//! names the proposal; while Preview retriangulation is ticked the next says
//! where the preview's triangulation stands and its counts, and a third, in
//! the failure colour, how many points lose their position and why. The
//! reasons are worded as the Retriangulate edits' Action Log row words them, so
//! the banner, the wire and the row Apply writes say the same thing about one
//! point.

use super::preview::{lost_reason, PreviewCounts};
use super::{CameraModelProposal, TriangulationState};

/// One line of the banner, and whether it is drawn in the failure colour.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BannerLine {
    /// The words.
    pub(crate) text: String,
    /// Drawn in the failure colour rather than the banner's own.
    pub(crate) failure: bool,
}

impl BannerLine {
    fn plain(text: String) -> Self {
        BannerLine {
            text,
            failure: false,
        }
    }
}

/// What the second line says while the job has no answer for the current
/// generation.
pub(crate) const TRIANGULATING: &str = "Triangulating…";

impl CameraModelProposal {
    /// The banner's lines. `selected` is the selected point's index in the
    /// version, when it is on this proposal's node: where the preview has that
    /// point lose its position, a last line in the failure colour gives the
    /// reason from the proposed run's verdict.
    pub(crate) fn banner_lines(&self, selected: Option<u32>) -> Vec<BannerLine> {
        let mut lines = vec![BannerLine::plain(self.banner_title())];
        match self.triangulation_state() {
            TriangulationState::Off => {}
            TriangulationState::Pending => {
                lines.push(BannerLine::plain(TRIANGULATING.to_string()));
            }
            TriangulationState::Refused(why) => {
                lines.push(BannerLine::plain(format!("No preview: {why}")));
            }
            TriangulationState::Ready(answer) => {
                lines.extend(count_lines(&answer.counts));
                let lost = selected.and_then(|index| {
                    let at = answer
                        .counts
                        .lost_points
                        .binary_search_by_key(&index, |p| p.index)
                        .ok()?;
                    Some(answer.counts.lost_points[at])
                });
                if let Some(point) = lost {
                    lines.push(BannerLine {
                        text: format!("Point {}: {}", point.index, point.verdict.label()),
                        failure: true,
                    });
                }
            }
        }
        lines
    }

    /// The first line: the camera, the node, the source and target models and
    /// the coefficient count, or for a refit of a spline the count and domain
    /// end before and after, as the refit's version label names them.
    fn banner_title(&self) -> String {
        let who = format!("camera {} of {}", self.camera, self.label);
        let target = self.inputs.camera_model;
        match &self.fit {
            Ok(refit) => match crate::state::edits::refit_change(&self.source, &refit.camera) {
                Some(change) => format!("Proposal for {who}: refit spline, {change}"),
                None => {
                    let count = refit
                        .camera
                        .model
                        .radial_spline()
                        .map_or(String::new(), |(b, _, _)| {
                            format!(", {} coefficients", b.len())
                        });
                    format!(
                        "Proposal for {who}: {} → {target}{count}",
                        self.source.model_name()
                    )
                }
            },
            Err(_) => format!(
                "Proposal for {who}: {} → {target}, fit refused",
                self.source.model_name()
            ),
        }
    }
}

/// The preview's counts in the banner's words: `3020 tracks: 2950 move,
/// median 0.0123, p90 0.0450; 3 left where they were for too few usable
/// rays`, then, in the failure colour where any lose their position, `37 lose
/// their position: 37 behind a camera that sees them`.
pub(crate) fn count_lines(counts: &PreviewCounts) -> Vec<BannerLine> {
    let mut first = format!("{} tracks: {} move", counts.tracks, counts.moves);
    if let (Some(median), Some(p90)) = (counts.median_shift, counts.p90_shift) {
        first.push_str(&format!(", median {median:.4}, p90 {p90:.4}"));
    }
    if counts.kept > 0 {
        first.push_str(&format!(
            "; {} left where they were for too few usable rays",
            counts.kept
        ));
    }
    let mut lines = vec![BannerLine::plain(first)];
    let lost = counts.lost();
    if lost > 0 {
        let reasons: Vec<String> = counts
            .lost_by_verdict
            .iter()
            .map(|(verdict, n)| format!("{n} {}", lost_reason(*verdict)))
            .collect();
        lines.push(BannerLine {
            text: format!("{lost} lose their position: {}", reasons.join(", ")),
            failure: true,
        });
    }
    lines
}

#[cfg(test)]
mod tests {
    use sfmtool_core::reconstruction::triangulation::PointVerdict;

    use super::*;
    use crate::camera_model_proposal::preview::LostPoint;

    fn counts() -> PreviewCounts {
        PreviewCounts {
            tracks: 3020,
            moves: 2950,
            lost_by_verdict: vec![(PointVerdict::Behind, 37), (PointVerdict::OverBar, 2)],
            kept: 3,
            held: 0,
            median_shift: Some(0.0123),
            p90_shift: Some(0.045),
            lost_points: vec![LostPoint {
                index: 5121,
                verdict: PointVerdict::Behind,
            }],
        }
    }

    /// The counts in the banner's words, the lost points in the failure
    /// colour and worded as the Retriangulate edits' row words them.
    #[test]
    fn the_counts_read_as_the_retriangulate_rows_do() {
        let lines = count_lines(&counts());
        assert_eq!(
            lines[0].text,
            "3020 tracks: 2950 move, median 0.0123, p90 0.0450; 3 left where they were for \
             too few usable rays"
        );
        assert!(!lines[0].failure);
        assert_eq!(
            lines[1].text,
            "39 lose their position: 37 behind a camera that sees them, 2 past the \
             reprojection bar"
        );
        assert!(lines[1].failure);
        let calm = PreviewCounts {
            lost_by_verdict: Vec::new(),
            lost_points: Vec::new(),
            kept: 0,
            ..counts()
        };
        assert_eq!(count_lines(&calm).len(), 1, "a line for nothing lost");
    }

    /// The banner's lines as an open proposal on Kerry-like data would have
    /// them, at their longest.
    fn long_lines() -> Vec<BannerLine> {
        let mut lines = vec![BannerLine::plain(
            "Proposal for camera 0 of kerry_park_ground_truth_candidate_tk107: \
             OPENCV_FISHEYE → SFMTOOL_FISHEYE, 8 coefficients"
                .to_string(),
        )];
        lines.extend(count_lines(&counts()));
        lines.push(BannerLine {
            text: format!("Point 5121: {}", PointVerdict::Behind.label()),
            failure: true,
        });
        lines
    }

    /// At the stock 3D view widths and a wide one, every line of the banner
    /// stays inside the view and no line runs into another.
    #[test]
    fn the_banner_stays_inside_the_view_and_its_lines_apart() {
        let ctx = egui::Context::default();
        let lines = long_lines();
        for width in [341.0_f32, 482.0, 1800.0] {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    egui::vec2(width, 720.0),
                )),
                ..Default::default()
            };
            let mut panel = None;
            let painted = crate::test_support::painted_text_rects(&ctx, input, |ui| {
                let rect = ui.max_rect();
                panel = crate::viewer_3d::overlay::draw_proposal_banner(ui.painter(), rect, &lines);
            });
            let panel = panel.expect("a banner");
            assert!(
                panel.left() >= 0.0 && panel.right() <= width,
                "the banner's frame {panel:?} leaves a {width} px view"
            );
            assert_eq!(painted.len(), lines.len(), "at {width} px");
            for text in &painted {
                assert!(
                    text.rect.left() >= panel.left() && text.rect.right() <= panel.right(),
                    "{:?} leaves the banner at {width} px",
                    text.text
                );
            }
            for (i, a) in painted.iter().enumerate() {
                for b in &painted[i + 1..] {
                    assert!(
                        !a.rect.shrink(1.0).intersects(b.rect.shrink(1.0)),
                        "{:?} and {:?} overlap at {width} px",
                        a.text,
                        b.text
                    );
                }
            }
        }
    }

    /// Every character the banner can show is drawn from a font that has a
    /// glyph for it: the `→` of a model change, which the proportional font
    /// lacks, from the monospace one, and the rest, refusal sentences' `θ`,
    /// `≤`, `°`, `…` and `—` included, from the proportional font.
    #[test]
    fn every_banner_character_has_a_glyph() {
        let ctx = egui::Context::default();
        let mut text: String = long_lines().iter().map(|l| l.text.clone()).collect();
        text.push_str(&format!(
            "{TRIANGULATING} No preview: θ ≤ 84.5°; domain 150.1° → 108.8° — n/a"
        ));
        let mut checked = false;
        crate::test_support::painted_text_rects(&ctx, egui::RawInput::default(), |ui| {
            let job =
                crate::viewer_3d::overlay::banner_job(ui.ctx(), &text, egui::Color32::WHITE, 400.0);
            ui.ctx().fonts_mut(|fonts| {
                for section in &job.sections {
                    let font = &section.format.font_id;
                    let range = section.byte_range.start.0..section.byte_range.end.0;
                    for c in job.text[range].chars() {
                        if font.family == egui::FontFamily::Proportional {
                            assert!(
                                c.is_whitespace() || fonts.has_glyph(font, c),
                                "{c:?} has no glyph in {font:?}"
                            );
                        } else {
                            // egui's `has_glyph` answers no for every
                            // character of the monospace font, whose first
                            // face also holds the replacement glyph, so the
                            // characters sent to it are checked by name:
                            // only the arrow, which Hack draws, as the
                            // Action Log shows.
                            assert_eq!(c, '→', "{c:?} was sent to the monospace font");
                        }
                    }
                }
            });
            checked = true;
        });
        assert!(checked);
    }

    /// The banners start below the 3D view's status line and the camera
    /// position's own row under it, so neither covers the status message or
    /// the camera position.
    #[test]
    fn the_banners_start_below_the_status_line() {
        use crate::viewer_3d::overlay::{BANNER_TOP, POSITION_THIRD_ROW_TOP, STATUS_TOP};
        let ctx = egui::Context::default();
        let mut height = 0.0;
        crate::test_support::painted_text_rects(&ctx, egui::RawInput::default(), |ui| {
            let galley = ui.painter().layout_no_wrap(
                "MCP: Proposed SFMTOOL_FISHEYE".to_string(),
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
            height = galley.size().y;
        });
        assert!(height > 0.0);
        assert!(
            STATUS_TOP + height <= POSITION_THIRD_ROW_TOP,
            "{STATUS_TOP} + {height} > {POSITION_THIRD_ROW_TOP}"
        );
        assert!(
            POSITION_THIRD_ROW_TOP + height <= BANNER_TOP,
            "{POSITION_THIRD_ROW_TOP} + {height} > {BANNER_TOP}"
        );
    }
}
