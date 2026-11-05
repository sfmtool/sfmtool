# Picking Images: Open Questions

**Status:** Draft

Amends [cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md)
and [core/analysis/pick-images.md](../core/analysis/pick-images.md). Decided
with the maintainer: the command's name and its one-command shape (it reports,
and writes a list when given a name), that a written list records nothing
about how it was picked, the refusal to overwrite without `--force`, one list
per run across every sequence, the workspace and `.sift` version 2
requirement, SIFT matching through a KD-forest as the measure of motion, the
detection of a sequence of photographs, which stops the command, the target
step of 10% and the blur ratio of 0.6, and the walk. This draft holds what is
not settled: how the other defaults are to be set, and the smaller choices the
command made on its own.

## Settling the defaults

Every value other than the target step and the blur ratio is a starting value
([core/analysis/pick-images.md § Parameters](../core/analysis/pick-images.md#parameters)):
the band of ±30%, the inlier share of 0.5, the minimum of 30 matches, the
baseline of 200 samples, the solvability thresholds (an F-inlier share of 0.3,
an H/F of 0.97, a rotation error of 0.01 and a total motion of 100%), the z of
2, the ratio test of 0.8, the inlier bounds of 1 and 2 pixels per 960, and the
four signs of a video (a 3-frame to 1-frame ratio between 2 and 4, a 1-frame
step below 5%, neighbouring samples' steps within a factor of 2, a 1-frame
F-inlier share above 0.8, with three of the four needed). They are to be set
by running the command on the maintainer's collection of captures, covering at
least:

- walking videos, where the walk should choose steps near the target, read few
  frames, and give a list that solves in one piece;
- a video with a blurred run, where the swaps, the gaps filled in and the
  frames kept though blurred should cover the run and nothing else;
- a rotation-only panorama, which the verdict should stop;
- a video with a moving scene, which the verdict should stop;
- a capture of several sequences, one list across them;
- a folder of photographs numbered in the order they were taken, which the
  verdict should name as not a video, beside a video folder in the same run,
  which it should not;
- a rig capture, with every sensor of each chosen frame in the list.

Each value is judged by solving the lists it gives and by looking at the
solves, not by the command's own numbers alone. Where a value moves, the draft
records the captures that moved it, by kind.

## Open questions

- **The output option's name.** The command uses `--write-list NAME`. `--name`
  was the maintainer's example; it reads less clearly beside the input's
  `--named-image-list`.
- **Taking `-r` and `--named-image-list` for the input.** The command is one of
  the commands of the shared `-r/--range` option, so a list or a range can
  narrow the input. Without them the input would be `PATHS` alone.
- **The default baseline size.** 200 samples, about 1000 frames read. "A few
  hundred" could be 300; the cost grows by up to five frames per sample.
- **The band and the inlier bar.** ±30% and 0.5, and whether the band should be
  asymmetric (a short step costs only a frame more; a long one can lose the
  match).
- **The blur rule's spread.** A robust z of 2 against the MAD, one-sided. The
  MAD of a capture with very even blur readings is small and makes the rule
  strict; the walk floors a MAD of 0 at the smallest positive deviation
  ([core/analysis/pick-images.md § Blur](../core/analysis/pick-images.md#blur)),
  and a floor on a small but positive MAD may be needed too. A frame judged
  blurred by a z-score alone can have a sharpness above its median, which the
  report then shows as kept though blurred at more than 1 of the median.
- **Medians per sequence or per capture.** The walk takes the sharpness and
  tensor medians per sequence, since separate sequences can differ in light and
  scene. For several sequences of one camera in one place, one median over the
  capture may be better.
- **A failing sequence among several.** The verdict stops the whole run when
  any sequence fails. The command could instead write a list of the sequences
  that pass.
- **Rigs.** Motion from the reference sensor only, sharpness per sensor against
  its own median, and a frame blurred when any sensor is. Whether a sensor
  pointing at a featureless view should be allowed to make every frame look
  blurred, and whether `.camrig` files, as well as `rig_config.json`, should
  define the rigs.
- **Rig list entries.** One entry per sensor with the same range keeps each
  pattern literal but orders the list by sensor and then frame, so a slice of
  the list ([named-image-lists.md § Slices](../workspace/named-image-lists.md#slices))
  takes the first sensor's images before any other's, and the command
  suggests no slices for a rig list. One entry with a `*` segment would keep
  the sensors of a frame together but can match folders that are not sensors.
- **The exit status** when the verdict stops a run that was asked to write a
  list: the command exits 1. A flag to choose frames anyway, for a person who
  means to solve a panorama with a rotation-only tool, is not proposed.
- **The command's category.** *Image Feature*, beside `sift` and `match`, since
  it reads images and extracts features. *Workspace* is the other candidate,
  since it writes a workspace file.
- **Telling photographs from a video.** The four signs, their thresholds and
  the rule that three of four make a video are starting values. Whether a
  sequence with a few photograph-like stretches (a video paused and resumed,
  say) should be split rather than judged whole. A camera held still moves by
  sub-pixel jitter, whose 3-frame step is not three times its 1-frame step and
  whose steps vary by more than a factor of 2, so it fails two signs and is
  judged not a video rather than a camera that barely moves.
- **A way to choose among photographs.** How a set of photographs should be
  chosen from, for example by how much of the scene each one's matches cover
  and by sharpness alone, without steps, is a design of its own and would be
  proposed in its own draft. The command only detects such a sequence and
  stops.
- **The JSON schema** of `--json`, which is the binding's report with the
  command's keys added and has no version of its own.
- **The version 1 folder check per extraction.** `image_files_to_sift_files`
  checks, on every call that has a file to write, every `.sift` file already in
  the target folder for version 1 (`_check_versions_never_mix`), which opens
  each file. The command's source calls it once per walk step, so that check
  grows with the files the run has written, although `pick_images` has made
  the folder check once before the first extraction. `image_files_to_sift_files`
  could remember, for the life of the process, a folder it has found to hold no
  version 1 file and has since written only version 2 files into.
