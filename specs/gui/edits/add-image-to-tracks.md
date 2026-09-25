# Add Image to Tracks

An action of the image menu that looks for every point an image does not
observe in that image's photograph, adds the observations whose appearance
agrees with the point's other observations, and installs the answer as the
reconstruction's next **version**, which an undo steps back out of. Nothing else
moves: no point, frame, bitmap or camera, and every index still means what it
meant.

It is the step that follows [Resect Image](resect-image.md). A resection gives
an image a pose that agrees with the rest of the reconstruction; this gives it
the tracks that pose now lets it see. The mechanism is
[../../core/reconstruction/add-image-to-tracks.md](../../core/reconstruction/add-image-to-tracks.md),
which owns what happens to each point and why the default rule is what it is.

Related specs: [../scene-graph.md](../scene-graph.md) § "Image menu",
[../background-tasks.md](../background-tasks.md) (the worker it runs on),
[../document-model.md](../document-model.md) and
[../edit-history.md](../edit-history.md) (the version it pushes),
[README.md](README.md) (the other edit families).

## Invocation

`Add Image to Tracks` is the second entry of the **image menu**, directly below
`Resect Image`, on an image row of the Scene tree and on a thumbnail of the
Image Browser strip. Over MCP the same step is `add_camera_image_to_tracks`
([../mcp-server.md](../mcp-server.md)).

The entry is greyed out, with the reason as its hover text, for the first of
these that holds:

1. the image is not posed;
2. an operation is running on the node (the background task's busy sentence);
3. the node has no points;
4. the node's observations are `.sift` feature indexes (an added observation
   has no feature to name; *Convert to Embedded Patches* first);
5. the node's points carry no patch frame;
6. the image's photograph is neither decoded in the viewer's cache nor a file
   at the path the reconstruction names. This one reason looks at the file
   system, and only for the image whose menu is open.

The step and the MCP tool refuse with the same sentence
(`AppState::add_image_to_tracks_refusal` in
[add_image_to_tracks.rs](../../../crates/sfm-explorer/src/add_image_to_tracks.rs),
which reads `ImageMenu::add_to_tracks_refusal` in
[image_menu.rs](../../../crates/sfm-explorer/src/image_menu.rs)).

## What runs

The step is a **background operation**, `Add image to tracks`
(`Operation::ADD_IMAGE_TO_TRACKS`), so the window stays live while it runs and
the Background panel shows its progress and offers Cancel. What crosses to the
worker is a clone of the value at the cursor and, per image, either the pyramid
the viewer's full-resolution cache already holds or the path to read the
photograph from (`AppState::view_sources_for`). On the worker the photographs
are decoded with the cancel flag polled between them; one that cannot be read is
left out of every point's references rather than refusing the step
(`ViewSources::decode_available`). The overlay is folded in when there is one,
and the core operation runs with its defaults.

A cancelled run pushes no version and logs `... cancelled`.

## The version and the log

An answer that adds at least one observation lands as one version labelled
`Added <image> to tracks`, under the identity point map (every point keeps its
index), and the Action Log row names how many tracks the image joined and why
the other candidates were refused, most first:

> Added frame_23.jpg to 12 tracks (361 candidates refused: 306 not in frame,
> 32 peak at edge, 21 below bar, 1 no peak, 1 unlocalizable)

An answer that adds nothing pushes no version, and its row says `Added <image>
to 0 tracks (...)`. The frame that installs a version drops the panels' caches
for the node, as for every background bulk edit. `Ctrl+Z` takes the added
observations back out.

## Testing

Lib tests in
[add_image_to_tracks/tests.rs](../../../crates/sfm-explorer/src/add_image_to_tracks/tests.rs),
over the plane capture of the track-at-pixel tests with one image's
observations taken out: the step runs on the worker, lands as one version that
keeps every point and rejoins the image to its tracks, writes the Action Log
sentence, and undo takes the observations out again; an image already in every
track pushes no version; the refusals are the greyed entry's sentences (unposed
image, no patch frames, no photograph, no points, busy node); the outcome text
names the refusals most first. `image_menu/tests.rs` checks the entry sits
between `Resect Image` and `Move Camera` and is greyed with the step's reason in
both places; `background/tests.rs` holds the operation to its `cancellable`
claim; the MCP catalog and parse tests cover the tool.

## Non-goals

It does not re-triangulate or adjust anything; a retriangulation or a bundle
adjustment afterwards is a separate step. It does not offer the core
operation's rule or gates as settings: the entry runs the defaults, and the
binding is where they are varied.
