// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The system font fallback: the candidate lists, the choice among them, the
//! check for text that needs a system face, and the one-time search.
//!
//! The fonts a test reads are built here, not read from the machine, which
//! may have none of the candidates. The one test that reads this machine's
//! fonts passes over itself when none is there.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::*;

// ── Fonts built for the tests ──

/// One face's tables: a format 12 `cmap` mapping each of `codes` to a glyph,
/// and an `OS/2` table carrying `weight`, when there is one.
fn tables(codes: &[u32], weight: Option<u16>) -> Vec<([u8; 4], Vec<u8>)> {
    let mut cmap = Vec::new();
    cmap.extend(0u16.to_be_bytes()); // version
    cmap.extend(1u16.to_be_bytes()); // one encoding record
    cmap.extend(3u16.to_be_bytes()); // Windows
    cmap.extend(10u16.to_be_bytes()); // full Unicode
    cmap.extend(12u32.to_be_bytes()); // the subtable's offset
    cmap.extend(12u16.to_be_bytes()); // format 12
    cmap.extend(0u16.to_be_bytes());
    cmap.extend((16 + 12 * codes.len() as u32).to_be_bytes());
    cmap.extend(0u32.to_be_bytes());
    cmap.extend((codes.len() as u32).to_be_bytes());
    for (i, &code) in codes.iter().enumerate() {
        cmap.extend(code.to_be_bytes());
        cmap.extend(code.to_be_bytes());
        cmap.extend((i as u32 + 1).to_be_bytes());
    }
    let mut tables = vec![(*b"cmap", cmap)];
    if let Some(weight) = weight {
        let mut os2 = vec![0u8; 4];
        os2.extend(weight.to_be_bytes());
        tables.push((*b"OS/2", os2));
    }
    tables
}

/// The table directory and tables of one face, placed at `base` in the file.
fn face_bytes(base: usize, tables: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(0x0001_0000u32.to_be_bytes());
    out.extend((tables.len() as u16).to_be_bytes());
    out.extend([0u8; 6]);
    let mut at = base + 12 + 16 * tables.len();
    for (tag, data) in tables {
        out.extend(tag);
        out.extend(0u32.to_be_bytes());
        out.extend((at as u32).to_be_bytes());
        out.extend((data.len() as u32).to_be_bytes());
        at += data.len();
    }
    for (_, data) in tables {
        out.extend(data);
    }
    out
}

/// A plain font file with one face.
fn font(codes: &[u32], weight: Option<u16>) -> Vec<u8> {
    face_bytes(0, &tables(codes, weight))
}

/// A `.ttc` collection of the given faces.
fn collection(faces: &[(&[u32], Option<u16>)]) -> Vec<u8> {
    let header = 12 + 4 * faces.len();
    let mut bodies = Vec::new();
    let mut offsets = Vec::new();
    let mut at = header;
    for &(codes, weight) in faces {
        let body = face_bytes(at, &tables(codes, weight));
        offsets.push(at as u32);
        at += body.len();
        bodies.push(body);
    }
    let mut out = b"ttcf".to_vec();
    out.extend(0x0001_0000u32.to_be_bytes());
    out.extend((faces.len() as u32).to_be_bytes());
    for offset in offsets {
        out.extend(offset.to_be_bytes());
    }
    for body in bodies {
        out.extend(body);
    }
    out
}

fn han_kana() -> Vec<u32> {
    Script::HanKana.sample().to_vec()
}

fn hangul() -> Vec<u32> {
    Script::Hangul.sample().to_vec()
}

fn both() -> Vec<u32> {
    [han_kana(), hangul()].concat()
}

/// A reader over `files`, which counts the reads of each path in `reads`; a
/// path not in `files` is missing.
fn reader<'a>(
    files: &'a HashMap<PathBuf, io::Result<Vec<u8>>>,
    reads: &'a mut Vec<PathBuf>,
) -> impl FnMut(&Path) -> io::Result<Vec<u8>> + 'a {
    move |path| {
        reads.push(path.to_path_buf());
        match files.get(path) {
            Some(Ok(bytes)) => Ok(bytes.clone()),
            Some(Err(err)) => Err(io::Error::new(err.kind(), err.to_string())),
            None => Err(io::ErrorKind::NotFound.into()),
        }
    }
}

fn candidate(path: &str, scripts: &'static [Script]) -> Candidate {
    Candidate {
        path: PathBuf::from(path),
        scripts,
    }
}

// ── The file format reader ──

#[test]
fn the_reader_maps_what_egui_maps_in_the_bundled_faces() {
    let ctx = egui::Context::default();
    super::super::install(&ctx);
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |_| {});
    let egui_has: BTreeSet<u32> = ctx.fonts_mut(|fonts| {
        let mut font = fonts.fonts.font(&egui::FontFamily::Monospace);
        font.characters().keys().map(|&c| u32::from(c)).collect()
    });
    let mut ours = BTreeSet::new();
    for &(start, end) in bundled_ranges() {
        ours.extend(start..=end);
    }
    // egui leaves out of the bundled faces a few code points it never draws.
    let dropped = BTreeSet::from([0x534D, 0x5350, 0xE0FF, 0xEFFD, 0xF0FF, 0xF200]);
    assert!(egui_has.len() > 1000, "egui's faces map {}", egui_has.len());
    let only_egui: Vec<_> = egui_has.difference(&ours).collect();
    let only_ours: Vec<_> = ours
        .difference(&egui_has)
        .filter(|c| !dropped.contains(c))
        .collect();
    assert!(only_egui.is_empty(), "mapped by egui alone: {only_egui:x?}");
    assert!(
        only_ours.is_empty(),
        "mapped by this reader alone: {only_ours:x?}"
    );
}

#[test]
fn the_reader_opens_a_font_and_a_collection_and_refuses_damaged_files() {
    let one = sfnt::faces(&font(&[0x41, 0xAC00], Some(400))).expect("a font");
    assert_eq!(one.len(), 1);
    assert!(one[0].maps(0xAC00) && one[0].maps(0x41) && !one[0].maps(0x42));
    assert_eq!(one[0].weight, Some(400));

    let ttc = collection(&[(&[0x41], Some(300)), (&[0x42], None)]);
    let faces = sfnt::faces(&ttc).expect("a collection");
    assert_eq!(
        faces
            .iter()
            .map(|f| (f.index, f.weight))
            .collect::<Vec<_>>(),
        [(0, Some(300)), (1, None)]
    );
    assert!(faces[1].maps(0x42) && !faces[1].maps(0x41));

    let good = font(&[0x41], None);
    for (what, bytes) in [
        ("empty", Vec::new()),
        ("text", b"not a font at all".to_vec()),
        ("cut short", good[..good.len() - 3].to_vec()),
        ("collection header cut short", ttc[..14].to_vec()),
    ] {
        assert!(sfnt::faces(&bytes).is_none(), "{what} opened");
    }
}

// ── The candidate lists ──

fn env_of(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
    let vars: HashMap<String, OsString> = vars
        .iter()
        .map(|&(k, v)| (k.to_owned(), OsString::from(v)))
        .collect();
    move |name| vars.get(name).cloned()
}

#[test]
fn the_windows_list_is_under_windir() {
    let list = candidates(Platform::Windows, &env_of(&[("WINDIR", r"D:\Win")]));
    let root = Path::new(r"D:\Win").join("Fonts");
    assert_eq!(
        list[0],
        Candidate {
            path: root.join("msyh.ttc"),
            scripts: HAN_KANA
        }
    );
    assert!(list.iter().all(|c| c.path.starts_with(&root)), "{list:?}");
    assert!(list.contains(&Candidate {
        path: root.join("malgun.ttf"),
        scripts: HANGUL
    }));
    for script in Script::ALL {
        assert!(
            list.iter().any(|c| c.scripts.contains(&script)),
            "{script:?}"
        );
    }

    let list = candidates(Platform::Windows, &env_of(&[("SystemRoot", r"E:\Sys")]));
    assert!(list[0].path.starts_with(Path::new(r"E:\Sys").join("Fonts")));
    assert!(candidates(Platform::Windows, &env_of(&[("WINDIR", "")])).is_empty());
}

#[test]
fn the_macos_list_has_a_face_for_each_group() {
    let list = candidates(Platform::MacOs, &env_of(&[]));
    assert_eq!(
        list[0].path,
        Path::new("/System/Library/Fonts/PingFang.ttc")
    );
    assert!(list.contains(&candidate(
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        HANGUL
    )));
    for script in Script::ALL {
        assert!(
            list.iter().any(|c| c.scripts.contains(&script)),
            "{script:?}"
        );
    }
}

#[test]
fn the_unix_list_starts_with_noto_and_looks_in_the_user_font_folders() {
    let list = candidates(
        Platform::Unix,
        &env_of(&[("HOME", "/h"), ("XDG_DATA_HOME", "/x")]),
    );
    assert_eq!(
        list[0],
        candidate(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ALL_CJK
        )
    );
    for dir in ["/x/fonts", "/h/.local/share/fonts", "/h/.fonts"] {
        let path = Path::new(dir).join("NotoSansCJK-Regular.ttc");
        assert!(
            list.iter().any(|c| c.path == path),
            "no {path:?} in {list:?}"
        );
    }
    // With no XDG_DATA_HOME, its default is listed once.
    let list = candidates(Platform::Unix, &env_of(&[("HOME", "/h")]));
    let local = Path::new("/h/.local/share/fonts/NotoSansCJK-Regular.ttc");
    assert_eq!(list.iter().filter(|c| c.path == local).count(), 1);
}

// ── The choice ──

#[test]
fn the_first_file_found_for_each_group_is_taken() {
    let list = [
        candidate("/missing.ttc", HAN_KANA),
        candidate("/first.ttc", HAN_KANA),
        candidate("/second.ttc", HAN_KANA),
        candidate("/korean.ttf", HANGUL),
    ];
    let files = HashMap::from([
        (
            PathBuf::from("/first.ttc"),
            Ok(font(&han_kana(), Some(400))),
        ),
        (
            PathBuf::from("/second.ttc"),
            Ok(font(&han_kana(), Some(400))),
        ),
        (PathBuf::from("/korean.ttf"), Ok(font(&hangul(), Some(400)))),
    ]);
    let mut reads = Vec::new();
    let found = choose(&list, &mut reader(&files, &mut reads));
    let taken: Vec<_> = found
        .faces
        .iter()
        .map(|f| (f.path.clone(), f.index))
        .collect();
    assert_eq!(
        taken,
        [
            (PathBuf::from("/first.ttc"), 0),
            (PathBuf::from("/korean.ttf"), 0)
        ]
    );
    assert_eq!(found.faces[0].scripts, [Script::HanKana]);
    assert_eq!(found.faces[1].scripts, [Script::Hangul]);
    assert!(!reads.contains(&PathBuf::from("/second.ttc")), "{reads:?}");
    assert!(found.problems.is_empty(), "{:?}", found.problems);
    assert_eq!(
        report(&found),
        format!(
            "Drawing Han and Kana from {} (face 0), Hangul from {} (face 0)",
            Path::new("/first.ttc").display(),
            Path::new("/korean.ttf").display()
        )
    );
}

#[test]
fn a_korean_face_with_han_too_is_reported_for_hangul_alone() {
    let list = [candidate("/cj.ttc", HAN_KANA), candidate("/k.ttf", HANGUL)];
    let files = HashMap::from([
        (PathBuf::from("/cj.ttc"), Ok(font(&han_kana(), Some(400)))),
        (PathBuf::from("/k.ttf"), Ok(font(&both(), Some(400)))),
    ]);
    let mut reads = Vec::new();
    let found = choose(&list, &mut reader(&files, &mut reads));
    assert_eq!(found.faces.len(), 2);
    assert_eq!(found.faces[1].scripts, Script::ALL);
    let line = report(&found);
    assert!(line.ends_with("Hangul from /k.ttf (face 0)"), "{line}");
}

#[test]
fn one_face_that_draws_every_group_is_the_only_face_taken() {
    let list = [
        candidate("/noto.ttc", ALL_CJK),
        candidate("/korean.ttf", HANGUL),
    ];
    let files = HashMap::from([
        (PathBuf::from("/noto.ttc"), Ok(font(&both(), Some(400)))),
        (PathBuf::from("/korean.ttf"), Ok(font(&hangul(), Some(400)))),
    ]);
    let mut reads = Vec::new();
    let found = choose(&list, &mut reader(&files, &mut reads));
    assert_eq!(found.faces.len(), 1);
    assert_eq!(found.faces[0].scripts, Script::ALL);
    assert_eq!(reads, [PathBuf::from("/noto.ttc")]);
}

#[test]
fn a_collection_gives_its_regular_face_that_draws_the_group() {
    let kana = han_kana();
    let ttc = collection(&[
        (&[0x41], Some(400)),
        (&kana, Some(700)),
        (&kana, Some(400)),
        (&kana, Some(400)),
    ]);
    let files = HashMap::from([(PathBuf::from("/c.ttc"), Ok(ttc))]);
    let mut reads = Vec::new();
    let found = choose(
        &[candidate("/c.ttc", HAN_KANA)],
        &mut reader(&files, &mut reads),
    );
    assert_eq!(found.faces.len(), 1);
    assert_eq!(found.faces[0].index, 2);
}

#[test]
fn damaged_and_unreadable_files_are_passed_over_with_a_line_each() {
    let list = [
        candidate("/gone.ttc", HAN_KANA),
        candidate("/locked.ttc", HAN_KANA),
        candidate("/junk.ttc", HAN_KANA),
        candidate("/latin.ttf", HAN_KANA),
        candidate("/good.ttc", HAN_KANA),
    ];
    let files = HashMap::from([
        (
            PathBuf::from("/locked.ttc"),
            Err(io::ErrorKind::PermissionDenied.into()),
        ),
        (PathBuf::from("/junk.ttc"), Ok(b"ttcf\0\0".to_vec())),
        (PathBuf::from("/latin.ttf"), Ok(font(&[0x41], Some(400)))),
        (PathBuf::from("/good.ttc"), Ok(font(&han_kana(), Some(400)))),
    ]);
    let mut reads = Vec::new();
    let found = choose(&list, &mut reader(&files, &mut reads));
    assert_eq!(found.faces.len(), 1);
    assert_eq!(found.faces[0].path, Path::new("/good.ttc"));
    assert_eq!(found.problems.len(), 3, "{:?}", found.problems);
    assert!(
        found.problems.iter().all(|p| !p.contains("gone")),
        "{:?}",
        found.problems
    );
    // No Hangul face, so the line says so as well as naming the faces.
    let line = report(&found);
    assert!(line.contains("no system font has Hangul glyphs"), "{line}");
    assert!(line.contains("/good.ttc"), "{line}");
}

// ── Which text needs a system face ──

#[test]
fn han_kana_and_hangul_need_a_system_face_and_other_text_does_not() {
    for name in ["서울_황소_04.jpg", "東京タワー.jpg", "カメラ", "名前"] {
        assert!(needs_system_font(name), "{name}");
    }
    for name in [
        "IMG_0001.jpg",
        "café → ü × 3°",
        "",
        "images/seoul_bull/04.jpg",
    ] {
        assert!(!needs_system_font(name), "{name}");
    }
}

// ── The one-time search ──

/// A fallback whose search finds `found` and counts its runs in `runs`.
fn counting(
    runs: &Arc<AtomicUsize>,
    found: impl Fn() -> Found + Send + Sync + 'static,
) -> SystemFallback {
    let runs = Arc::clone(runs);
    SystemFallback::with_search(Arc::new(move || {
        runs.fetch_add(1, Ordering::SeqCst);
        found()
    }))
}

/// Update `fallback` until its search has ended, or fail after a while.
fn until_done(fallback: &mut SystemFallback, ctx: &egui::Context, state: &mut AppState) {
    let start = Instant::now();
    while !matches!(fallback.stage, Stage::Done) {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the search did not end"
        );
        fallback.update(ctx, state);
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn fallback_lines(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .filter(|e| e.text.contains("system font"))
        .map(|e| e.text.clone())
        .collect()
}

#[test]
fn ascii_reconstructions_never_start_a_search() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut fallback = counting(&runs, Found::default);
    let ctx = egui::Context::default();
    let mut state = AppState::new();
    state.load_demo(10);
    state.scene[0].recon_mut().image_set.images[0].name = "café_01.jpg".to_owned();
    for _ in 0..3 {
        fallback.update(&ctx, &mut state);
    }
    assert!(matches!(fallback.stage, Stage::Watching { .. }));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[test]
fn a_hangul_image_name_starts_one_search_and_no_more() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut fallback = counting(&runs, Found::default);
    let ctx = egui::Context::default();
    super::super::install(&ctx);
    let mut state = AppState::new();
    state.load_demo(10);
    fallback.update(&ctx, &mut state);
    assert!(matches!(fallback.stage, Stage::Watching { .. }));

    state.load_demo(10);
    state.scene[1].recon_mut().image_set.images[3].name = "서울_황소_04.jpg".to_owned();
    fallback.update(&ctx, &mut state);
    assert!(matches!(fallback.stage, Stage::Searching(_)));
    until_done(&mut fallback, &ctx, &mut state);

    state.load_demo(10);
    state.scene[2].recon_mut().image_set.images[0].name = "東京.jpg".to_owned();
    state.action_log.record(Kind::File, "Opened 서울.sfmr");
    for _ in 0..3 {
        fallback.update(&ctx, &mut state);
    }
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let lines = fallback_lines(&state);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("No system font"), "{lines:?}");
}

#[test]
fn an_action_log_line_starts_the_search() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut fallback = counting(&runs, Found::default);
    let ctx = egui::Context::default();
    let mut state = AppState::new();
    fallback.update(&ctx, &mut state);
    state
        .action_log
        .record(Kind::File, "Could not open C:/写真/a.sfmr");
    fallback.update(&ctx, &mut state);
    until_done(&mut fallback, &ctx, &mut state);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[test]
fn a_search_that_finds_nothing_leaves_the_fonts_as_they_were() {
    let ctx = egui::Context::default();
    super::super::install(&ctx);
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |_| {});
    let before = ctx.fonts(|fonts| fonts.definitions().clone());

    // Every candidate is missing, unreadable or damaged.
    let files = Arc::new(HashMap::from([
        (PathBuf::from("/junk.ttc"), b"OTTO\0\x01".to_vec()),
        (PathBuf::from("/empty.ttf"), Vec::new()),
    ]));
    let list = vec![
        candidate("/gone.ttc", ALL_CJK),
        candidate("/junk.ttc", ALL_CJK),
        candidate("/empty.ttf", HANGUL),
    ];
    let mut fallback = SystemFallback::with_search(Arc::new(move || {
        choose(&list, &mut |path: &Path| {
            files
                .get(path)
                .cloned()
                .ok_or_else(|| io::ErrorKind::NotFound.into())
        })
    }));
    let mut state = AppState::new();
    state.action_log.record(Kind::File, "Opened 서울.sfmr");
    until_done(&mut fallback, &ctx, &mut state);
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |_| {});

    assert_eq!(ctx.fonts(|fonts| fonts.definitions().clone()), before);
    let lines = fallback_lines(&state);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].contains("/junk.ttc does not open as a font"),
        "{lines:?}"
    );
    assert!(
        lines[0].contains("/empty.ttf does not open as a font"),
        "{lines:?}"
    );
}

#[test]
fn found_faces_go_after_hack_in_both_families() {
    // A bundled face's bytes stand in for a system file, so that egui can
    // build fonts from the definitions in a frame.
    let hack = egui::FontDefinitions::default().font_data["Hack"]
        .font
        .to_vec();
    let face = ChosenFace {
        path: PathBuf::from("/hack.ttf"),
        index: 0,
        scripts: vec![Script::Hangul],
        bytes: hack,
    };
    let fonts = definitions_with(vec![face]);
    let proportional = &fonts.families[&egui::FontFamily::Proportional];
    assert_eq!(
        &proportional[proportional.len() - 2..],
        ["Hack", "system-0"]
    );
    let monospace = &fonts.families[&egui::FontFamily::Monospace];
    assert_eq!(monospace[0], "Hack");
    assert_eq!(monospace.last().map(String::as_str), Some("system-0"));
    let ctx = egui::Context::default();
    ctx.set_fonts(fonts);
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("ok");
    });
}

#[test]
fn a_system_face_on_this_machine_draws_hangul() {
    let found = search_system();
    if !found
        .faces
        .iter()
        .any(|face| face.scripts.contains(&Script::Hangul))
    {
        eprintln!(
            "skipped: no system face with Hangul glyphs here ({})",
            report(&found)
        );
        return;
    }
    let ctx = egui::Context::default();
    ctx.set_fonts(definitions_with(found.faces));
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("서울 황소");
    });
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let lacking = ctx.fonts_mut(|fonts| {
            let mut font = fonts.fonts.font(&family);
            let characters = font.characters();
            "서울황소"
                .chars()
                .filter(|c| !characters.contains_key(c))
                .collect::<String>()
        });
        assert!(lacking.is_empty(), "{family:?} lacks {lacking}");
    }
}
