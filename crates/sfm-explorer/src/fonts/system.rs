// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Fonts from the operating system, for the Han, Kana and Hangul characters
//! that none of egui's bundled faces has.
//!
//! Image names, reconstruction labels and paths can be in any script, and
//! the bundled faces stop short of Chinese, Japanese and Korean, so such a
//! name draws as a row of empty boxes. Nothing is shipped for it: the
//! fallback uses a face the operating system already has, from a fixed list
//! of files per system, and reads it only once text that needs it appears.
//!
//! See `specs/gui/architecture.md` § "Fonts".

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};

use super::sfnt;
use crate::action_log::{ActionLog, Actor, Kind};
use crate::state::AppState;

/// A group of scripts one system face is looked for to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Script {
    /// Chinese and Japanese: Han ideographs, Hiragana and Katakana.
    HanKana,
    /// Korean: Hangul syllables.
    Hangul,
}

impl Script {
    /// Each group, in the order its face is added to the families: a face
    /// found for Han and Kana goes before a Korean one, so that a Han
    /// character a Korean face also has is drawn in the Chinese or Japanese
    /// face, which has more of them.
    const ALL: [Script; 2] = [Script::HanKana, Script::Hangul];

    /// Characters a face has to map, every one, to count as drawing the
    /// group. Code points rather than a string, so that the glyph test of
    /// the viewer's own strings does not take them for text the viewer draws.
    fn sample(self) -> &'static [u32] {
        match self {
            // 中文日本語漢字 あいう アイウ
            Script::HanKana => &[
                0x4E2D, 0x6587, 0x65E5, 0x672C, 0x8A9E, 0x6F22, 0x5B57, 0x3042, 0x3044, 0x3046,
                0x30A2, 0x30A4, 0x30A6,
            ],
            // 서울 황소 한국어
            Script::Hangul => &[0xC11C, 0xC6B8, 0xD669, 0xC18C, 0xD55C, 0xAD6D, 0xC5B4],
        }
    }

    fn name(self) -> &'static str {
        match self {
            Script::HanKana => "Han and Kana",
            Script::Hangul => "Hangul",
        }
    }
}

/// Whether `code` is in a block a system face is loaded for: Hangul, Han,
/// Kana, and the CJK punctuation and full-width forms that go with them.
fn is_cjk(code: u32) -> bool {
    matches!(
        code,
        0x1100..=0x11FF       // Hangul Jamo
        | 0x2E80..=0x2FDF     // CJK and Kangxi radicals
        | 0x3000..=0x303F     // CJK symbols and punctuation
        | 0x3040..=0x30FF     // Hiragana, Katakana
        | 0x3130..=0x318F     // Hangul compatibility Jamo
        | 0x31F0..=0x31FF     // Katakana phonetic extensions
        | 0x3400..=0x4DBF     // CJK extension A
        | 0x4E00..=0x9FFF     // CJK unified ideographs
        | 0xA960..=0xA97F     // Hangul Jamo extended A
        | 0xAC00..=0xD7FF     // Hangul syllables, Jamo extended B
        | 0xF900..=0xFAFF     // CJK compatibility ideographs
        | 0xFF00..=0xFFEF     // half-width and full-width forms
        | 0x20000..=0x3134F // CJK extensions B to G
    )
}

/// The code points egui's bundled faces map, as sorted, disjoint, inclusive
/// ranges; read from the bundled files once.
fn bundled_ranges() -> &'static [(u32, u32)] {
    static RANGES: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    RANGES.get_or_init(|| {
        let mut ranges: Vec<(u32, u32)> = egui::FontDefinitions::default()
            .font_data
            .values()
            .filter_map(|data| sfnt::faces(&data.font))
            .flatten()
            .flat_map(|face| face.ranges().to_vec())
            .collect();
        ranges.sort_unstable();
        ranges
    })
}

/// Whether any of egui's bundled faces maps `code`.
fn bundled_has(code: u32) -> bool {
    let ranges = bundled_ranges();
    // The faces' ranges overlap one another, so look at every range that
    // starts at or before `code`.
    let i = ranges.partition_point(|&(start, _)| start <= code);
    ranges[..i].iter().any(|&(_, end)| end >= code)
}

/// Whether `text` has a Han, Kana or Hangul character that none of the
/// bundled faces has a glyph for: text the system fallback would draw.
fn needs_system_font(text: &str) -> bool {
    text.chars()
        .map(u32::from)
        .any(|code| is_cjk(code) && !bundled_has(code))
}

/// The operating systems with a candidate list of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Platform {
    Windows,
    MacOs,
    /// Linux and the other Unix systems, which keep fonts in the same
    /// places.
    Unix,
}

impl Platform {
    fn current() -> Platform {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Unix
        }
    }
}

/// A font file the fallback may use, and the groups it is listed for.
#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    path: PathBuf,
    scripts: &'static [Script],
}

const HAN_KANA: &[Script] = &[Script::HanKana];
const HANGUL: &[Script] = &[Script::Hangul];
const ALL_CJK: &[Script] = &[Script::HanKana, Script::Hangul];

/// The font files `platform` is looked in, in the order they are tried, with
/// environment variables read through `env`.
///
/// A face that draws every group is listed first, so that one face is loaded
/// where one will do.
fn candidates(platform: Platform, env: &dyn Fn(&str) -> Option<OsString>) -> Vec<Candidate> {
    let var = |name: &str| {
        env(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let listed = |dir: &Path, files: &[(&str, &'static [Script])]| -> Vec<Candidate> {
        files
            .iter()
            .map(|&(file, scripts)| Candidate {
                path: dir.join(file),
                scripts,
            })
            .collect()
    };
    match platform {
        Platform::Windows => {
            let Some(windows) = var("WINDIR").or_else(|| var("SystemRoot")) else {
                return Vec::new();
            };
            listed(
                &windows.join("Fonts"),
                &[
                    ("msyh.ttc", HAN_KANA),     // Microsoft YaHei
                    ("YuGothR.ttc", HAN_KANA),  // Yu Gothic Regular
                    ("YuGothM.ttc", HAN_KANA),  // Yu Gothic Medium
                    ("msgothic.ttc", HAN_KANA), // MS Gothic
                    ("simsun.ttc", HAN_KANA),   // SimSun
                    ("malgun.ttf", HANGUL),     // Malgun Gothic
                    ("gulim.ttc", HANGUL),      // Gulim
                ],
            )
        }
        Platform::MacOs => {
            let system = Path::new("/System/Library/Fonts");
            let supplemental = system.join("Supplemental");
            let mut list = listed(
                system,
                &[
                    ("PingFang.ttc", HAN_KANA),
                    ("Hiragino Sans GB.ttc", HAN_KANA),
                ],
            );
            list.extend(listed(
                &supplemental,
                &[("Songti.ttc", HAN_KANA), ("Arial Unicode.ttf", ALL_CJK)],
            ));
            list.extend(listed(
                Path::new("/Library/Fonts"),
                &[("Arial Unicode.ttf", ALL_CJK)],
            ));
            list.extend(listed(system, &[("AppleSDGothicNeo.ttc", HANGUL)]));
            list.extend(listed(&supplemental, &[("AppleGothic.ttf", HANGUL)]));
            list
        }
        Platform::Unix => {
            let noto = [("NotoSansCJK-Regular.ttc", ALL_CJK)];
            let mut list = Vec::new();
            for dir in [
                "/usr/share/fonts/opentype/noto",
                "/usr/share/fonts/noto-cjk",
                "/usr/share/fonts/google-noto-cjk",
                "/usr/share/fonts/truetype/noto",
            ] {
                list.extend(listed(Path::new(dir), &noto));
            }
            list.extend(listed(
                Path::new("/usr/share/fonts/google-noto-sans-cjk-vf-fonts"),
                &[("NotoSansCJK-VF.ttc", ALL_CJK)],
            ));
            let home = var("HOME");
            let mut user_dirs: Vec<PathBuf> = Vec::new();
            user_dirs.extend(var("XDG_DATA_HOME").map(|dir| dir.join("fonts")));
            user_dirs.extend(home.iter().map(|home| home.join(".local/share/fonts")));
            user_dirs.extend(home.iter().map(|home| home.join(".fonts")));
            user_dirs.dedup();
            for dir in &user_dirs {
                list.extend(listed(dir, &noto));
            }
            list.extend(listed(
                Path::new("/usr/share/fonts/truetype/wqy"),
                &[("wqy-microhei.ttc", ALL_CJK), ("wqy-zenhei.ttc", ALL_CJK)],
            ));
            list.extend(listed(
                Path::new("/usr/share/fonts/truetype/droid"),
                &[("DroidSansFallbackFull.ttf", ALL_CJK)],
            ));
            list.extend(listed(
                Path::new("/usr/share/fonts/truetype/nanum"),
                &[("NanumGothic.ttf", HANGUL)],
            ));
            list
        }
    }
}

/// A face picked from a system font file.
#[derive(Debug)]
struct ChosenFace {
    path: PathBuf,
    /// The face's index in the file.
    index: u32,
    /// The groups the face draws.
    scripts: Vec<Script>,
    /// The whole file, which egui keeps for as long as the face is in use.
    bytes: Vec<u8>,
}

/// A candidate file that was read and opens as a font: its bytes and faces.
type OpenedFile = (Vec<u8>, Vec<sfnt::Face>);

/// What a search of the candidates found.
#[derive(Debug, Default)]
struct Found {
    /// The faces to add, in the order they go into the families.
    faces: Vec<ChosenFace>,
    /// A line for each file that was there but could not be used.
    problems: Vec<String>,
}

/// The face of `faces` that maps all of `script`'s sample, nearest to a
/// regular weight, and the first of those on a tie.
fn best_face(faces: &[sfnt::Face], script: Script) -> Option<&sfnt::Face> {
    faces
        .iter()
        .filter(|face| script.sample().iter().all(|&code| face.maps(code)))
        .min_by_key(|face| face.weight.map_or(u16::MAX, |weight| weight.abs_diff(400)))
}

/// Go through `candidates` for each group in turn and take, for a group no
/// face already taken draws, the first file `read` returns that has a face
/// drawing it.
///
/// A file that is missing is passed over without a word; one that cannot be
/// read, does not open as a font, or has no face for the group it is listed
/// for is passed over with a line in [`Found::problems`].
fn choose(candidates: &[Candidate], read: &mut dyn FnMut(&Path) -> io::Result<Vec<u8>>) -> Found {
    let mut found = Found::default();
    // Each file is read at most once, and kept until it is taken or the
    // search ends.
    let mut opened: Vec<(PathBuf, Option<OpenedFile>)> = Vec::new();
    for script in Script::ALL {
        if found
            .faces
            .iter()
            .any(|face| face.scripts.contains(&script))
        {
            continue;
        }
        for candidate in candidates.iter().filter(|c| c.scripts.contains(&script)) {
            let slot = match opened.iter().position(|(path, _)| *path == candidate.path) {
                Some(slot) => slot,
                None => {
                    let file = match read(&candidate.path) {
                        Ok(bytes) => match sfnt::faces(&bytes) {
                            Some(faces) => Some((bytes, faces)),
                            None => {
                                found.problems.push(format!(
                                    "{} does not open as a font",
                                    candidate.path.display()
                                ));
                                None
                            }
                        },
                        Err(err) if err.kind() == io::ErrorKind::NotFound => None,
                        Err(err) => {
                            found.problems.push(format!(
                                "{} could not be read: {err}",
                                candidate.path.display()
                            ));
                            None
                        }
                    };
                    opened.push((candidate.path.clone(), file));
                    opened.len() - 1
                }
            };
            let Some((_, faces)) = &opened[slot].1 else {
                continue;
            };
            let Some(face) = best_face(faces, script) else {
                found.problems.push(format!(
                    "{} has no face with {} glyphs",
                    candidate.path.display(),
                    script.name()
                ));
                continue;
            };
            let index = face.index;
            let scripts = Script::ALL
                .into_iter()
                .filter(|&s| s.sample().iter().all(|&code| face.maps(code)))
                .collect();
            let Some((bytes, _)) = opened[slot].1.take() else {
                continue;
            };
            found.faces.push(ChosenFace {
                path: candidate.path.clone(),
                index,
                scripts,
                bytes,
            });
            break;
        }
    }
    found
}

/// Look for the faces on this machine's candidate files.
fn search_system() -> Found {
    let candidates = candidates(Platform::current(), &|name| std::env::var_os(name));
    choose(&candidates, &mut |path| std::fs::read(path))
}

/// The Action Log line that reports what a search found.
fn report(found: &Found) -> String {
    let mut line = if found.faces.is_empty() {
        "No system font with Han, Kana or Hangul glyphs was found; \
         those characters are drawn as boxes"
            .to_owned()
    } else {
        let faces: Vec<String> = found
            .faces
            .iter()
            .enumerate()
            .map(|(i, face)| {
                // A face draws a group only where no face before it does.
                let scripts: Vec<&str> = face
                    .scripts
                    .iter()
                    .filter(|s| !found.faces[..i].iter().any(|f| f.scripts.contains(s)))
                    .map(|s| s.name())
                    .collect();
                format!(
                    "{} from {} (face {})",
                    scripts.join(" and "),
                    face.path.display(),
                    face.index
                )
            })
            .collect();
        let mut line = format!("Drawing {}", faces.join(", "));
        for script in Script::ALL {
            if !found
                .faces
                .iter()
                .any(|face| face.scripts.contains(&script))
            {
                line.push_str(&format!("; no system font has {} glyphs", script.name()));
            }
        }
        line
    };
    for problem in &found.problems {
        line.push_str("; ");
        line.push_str(problem);
    }
    line
}

/// The viewer's font definitions with `faces` added, in order, as the last
/// faces of both families.
fn definitions_with(faces: Vec<ChosenFace>) -> egui::FontDefinitions {
    let mut fonts = super::font_definitions();
    for (i, face) in faces.into_iter().enumerate() {
        let name = format!("system-{i}");
        let mut data = egui::FontData::from_owned(face.bytes);
        data.index = face.index;
        fonts.font_data.insert(name.clone(), Arc::new(data));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push(name.clone());
        }
    }
    fonts
}

/// Where the fallback is in its one-time life.
enum Stage {
    /// No text needing a system face has been seen yet. Holds what has been
    /// scanned, so each frame scans only what changed.
    Watching {
        /// Each node and the version it showed when last scanned.
        scene: Vec<(crate::scene::ReconId, crate::document::VersionSerial)>,
        /// The newest Action Log revision scanned.
        log_revision: u64,
    },
    /// A search is running on its own thread.
    Searching(mpsc::Receiver<Found>),
    /// The search has ended and its faces, if any, are installed. Nothing
    /// more is done.
    Done,
}

/// The system font fallback: watches the text the viewer is given for
/// characters only a system face draws, and then searches for and installs
/// such faces, once.
pub(crate) struct SystemFallback {
    stage: Stage,
    search: Arc<dyn Fn() -> Found + Send + Sync>,
}

impl SystemFallback {
    /// A fallback that searches this machine's font files.
    pub(crate) fn new() -> Self {
        Self::with_search(Arc::new(search_system))
    }

    fn with_search(search: Arc<dyn Fn() -> Found + Send + Sync>) -> Self {
        Self {
            stage: Stage::Watching {
                scene: Vec::new(),
                log_revision: 0,
            },
            search,
        }
    }

    /// Called once a frame, before the egui pass.
    ///
    /// While watching, scans the text of each reconstruction whose version
    /// changed and each Action Log entry written since the last call, and
    /// starts the search on its own thread when one has a character only a
    /// system face draws. When the search has ended, installs its faces on
    /// `ctx` and writes one Action Log line about it.
    pub(crate) fn update(&mut self, ctx: &egui::Context, state: &mut AppState) {
        match &mut self.stage {
            Stage::Watching {
                scene,
                log_revision,
            } => {
                let mut needed = false;
                let now: Vec<_> = state
                    .scene
                    .iter()
                    .map(|node| (node.id, node.history.current_version().serial))
                    .collect();
                if now != *scene {
                    needed = state
                        .scene
                        .iter()
                        .filter(|node| {
                            !scene.contains(&(node.id, node.history.current_version().serial))
                        })
                        .any(node_needs_system_font);
                    *scene = now;
                }
                needed |= state
                    .action_log
                    .since(*log_revision)
                    .any(|entry| needs_system_font(&entry.text));
                *log_revision = state.action_log.revision();
                if needed {
                    let (tx, rx) = mpsc::channel();
                    let search = Arc::clone(&self.search);
                    let wake = state.wake.clone();
                    let spawned = std::thread::Builder::new()
                        .name("system fonts".to_owned())
                        .spawn(move || {
                            let _ = tx.send(search());
                            if let Some(wake) = wake {
                                wake();
                            }
                        });
                    self.stage = match spawned {
                        Ok(_) => Stage::Searching(rx),
                        Err(err) => {
                            state.action_log.record_as(
                                Actor::Viewer,
                                Kind::Session,
                                format!("The search for system fonts did not start: {err}"),
                            );
                            Stage::Done
                        }
                    };
                }
            }
            Stage::Searching(rx) => match rx.try_recv() {
                Ok(found) => {
                    self.install(ctx, &mut state.action_log, found);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    state.action_log.record_as(
                        Actor::Viewer,
                        Kind::Session,
                        "The search for system fonts ended without an answer",
                    );
                    self.stage = Stage::Done;
                }
            },
            Stage::Done => {}
        }
    }

    /// Install what a search found and report it; the fallback is done.
    fn install(&mut self, ctx: &egui::Context, log: &mut ActionLog, found: Found) {
        log.record_as(Actor::Viewer, Kind::Session, report(&found));
        if !found.faces.is_empty() {
            ctx.set_fonts(definitions_with(found.faces));
        }
        self.stage = Stage::Done;
    }
}

/// Whether any text of `node` that the viewer shows has a character only a
/// system face draws: its label, its path, its image names, and its rig and
/// sensor names.
fn node_needs_system_font(node: &crate::scene::SceneNode) -> bool {
    let recon = node.recon();
    needs_system_font(&node.label)
        || node
            .path
            .as_ref()
            .is_some_and(|path| needs_system_font(&path.to_string_lossy()))
        || recon
            .image_set
            .images
            .iter()
            .any(|image| needs_system_font(&image.name))
        || recon.image_set.rig_frame_data.as_ref().is_some_and(|rigs| {
            rigs.rigs_metadata.rigs.iter().any(|rig| {
                needs_system_font(&rig.name)
                    || rig.sensor_names.iter().any(|name| needs_system_font(name))
            })
        })
}

#[cfg(test)]
mod tests;
