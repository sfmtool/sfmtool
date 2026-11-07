# Picking Images: Open Questions

**Status:** Draft

Amends [cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md)
and [core/analysis/pick-images.md](../core/analysis/pick-images.md). Decided
with the maintainer: the command's name and its one-command shape (it reports,
and writes a list when given a name), that a written list records nothing
about how it was picked, the refusal to overwrite without `--force`, one list
per run across every sequence, the workspace and `.sift` version 2
requirement, SIFT matching through a KD-forest as the measure of motion, the
target step of 10%, the walk, which starts at the first frame with no sample
of the capture ahead of it and always takes the step to the next frame, the
match floor as its one count of matches, an absolute count of mutual matches
that every step beyond the next frame keeps, at 600, with no check of whether
a sequence is a video, that blur is judged from the sharpness alone, and only
against the neighbouring frames, with a window of 2 positions and a ratio of
0.85.

The defaults were run on a collection of captures: handheld walking videos,
slow orbits of an object, shaky videos with long blurred runs, a video of a
moving animal, rotation-only panoramas, videos from a tripod or a hand held
still, a numbered set of photographs of an object beside a video in one run,
and a two-sensor fisheye rig, with the lists of the pinhole videos judged by
solving them. The target step and the band held. The command makes no
judgement of whether a capture can be solved: it fits no geometry to the
matches, so it does not judge a moving scene or a camera that only turns, and
it does not refuse a capture whose camera moves little, which refused a
26-photograph capture that solves. This draft holds what is not settled.

## Open questions

- **The output option's name.** The command uses `--write-list NAME`. `--name`
  was the maintainer's example; it reads less clearly beside the input's
  `--named-image-list`.
- **Taking `-r` and `--named-image-list` for the input.** The command is one of
  the commands of the shared `-r/--range` option, so a list or a range can
  narrow the input. Without them the input would be `PATHS` alone.
- **An asymmetric band.** ±30% held on every capture, but a short step costs
  only a frame more and a long one can lose the match, so a band wider below
  the target than above it is still a choice.
- **Rigs.** Motion from the reference sensor only, sharpness per sensor against
  that sensor's images of the neighbouring frames, and a frame blurred against
  its neighbours when any sensor is. Whether a sensor pointing at a
  featureless view should be allowed to make a frame look blurred against its
  neighbours, and whether `.camrig` files, as well as `rig_config.json`,
  should define the rigs.
- **Rig list entries.** One entry per sensor with the same range keeps each
  pattern literal but orders the list by sensor and then frame, so a slice of
  the list ([named-image-lists.md § Slices](../workspace/named-image-lists.md#slices))
  takes the first sensor's images before any other's, and the command
  suggests no slices for a rig list. One entry with a `*` segment would keep
  the sensors of a frame together but can match folders that are not sensors.
- **The command's category.** *Image Feature*, beside `sift` and `match`, since
  it reads images and extracts features. *Workspace* is the other candidate,
  since it writes a workspace file.
- **The match floor and the feature count.** The floor is an absolute count,
  so it binds more often in a workspace that extracts fewer features per
  frame, or on smaller images, and less often in one that extracts more.
  Whether the floor should follow the features a frame has, or the option
  stay the only way to change it, is open.
- **A way to choose among photographs.** Numbered photographs are walked like
  a video's frames under the match floor: a set whose consecutive photographs
  match with fewer matches than the floor keeps every one, and a set whose
  matches stay above it is thinned like a video. How a set of photographs
  should otherwise be chosen from, for example by how much of the scene each
  one's matches cover and by sharpness alone, without steps, is a design of
  its own and would be proposed in its own draft.
- **The JSON schema** of `--json`, which is the binding's report with the
  command's keys added and has no version of its own.
