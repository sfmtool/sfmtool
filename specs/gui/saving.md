# Saving a reconstruction from the viewer

A reconstruction loaded into the viewer can be changed, and what has been
changed has to be able to reach the disk again. Saving writes the value the
viewer is showing back out as a reconstruction file: over the file the node came
from, or to a path the user picks. Because the viewer holds a node's value as a
sequence of versions with a cursor on one of them, a save is also a statement
about that sequence -- it says which version the file on disk now holds, and
everything the window shows about unsaved work is read off the distance between
that version and the cursor.

This spec describes the two save commands, what a save does to the node's
history, the marks that say a node has changes the disk does not, the prompt
that stands between unsaved changes and a close, and what the Action Log
records. What a version is, and how the cursor walks, are
[document-model.md](document-model.md) and
[edit-history.md](edit-history.md).

## The commands

The saving code is
[state/save.rs](../../crates/sfm-explorer/src/state/save.rs), the menu items are
in [app.rs](../../crates/sfm-explorer/src/app.rs), and the close prompt is
[close_prompt.rs](../../crates/sfm-explorer/src/close_prompt.rs).

| Item | Shortcut | What it does |
|------|----------|--------------|
| `File > Save` | `Ctrl/Cmd+S` | Writes the selected node's value over the node's own path. |
| `File > Save As...` | `Ctrl/Cmd+Shift+S` | Asks for a path, writes there, and re-points the node at it. |

**Save is disabled for a node with no path** -- demo data -- with the hover
text `The selected reconstruction came from no file
-- use Save As`. There is nothing to write over, and the alternative is one menu
item away, so the item says which one rather than opening a dialog the user did
not ask for.

**Save As asks for the path through the platform's native save dialog**, the
counterpart of the native open dialog `File > Open` uses, filtered to `.sfmr`.
On a chosen path it writes, then sets the node's `path` to that path and the node's
`label` to the new file's stem, so the tree row, the window title and every
label-addressed operation name the file the node is now attached to. **A
dismissed dialog writes nothing and logs nothing**: cancelling a file picker is
not an action, and an Action Log row saying a save was refused would report the
user's own change of mind as a failure.

**Both shortcuts are live only while no text field holds the keyboard**, which
is the arbitration the Edit menu's shortcuts already use: a field being typed
into keeps its own editing keys, and the menu items stay reachable whatever has
focus.

## Materialise on save

A version is a base plus the point edits made on it, and a file is always a
plain reconstruction. So **a save of a value whose overlay is not empty
materialises it first** -- anything deleted or anything added -- and what
reaches the disk is that materialisation.

The materialised base is **stamped with provenance before it becomes a
version**: `operation` `edit`, `tool` `sfm-explorer`, and `tool_version` the
crate's version. **The stamping precedes the hashing**, and that order is the
whole of why it is described here. A reconstruction's metadata is inside the
metadata section's digest and so inside its content hash, so a base stamped
after its hash was taken would leave the session holding a hash the file on disk
does not have -- and that hash is what a point id minted against the base
carries.

The stamped base is then **pushed onto the node's history as an ordinary
version**, labelled `Saved <label> to <file name>`, carrying the
materialisation's row map as its point map, and the selection follows that map
like any other step. So the value that reached the disk is a version the cursor
sits on, and it can be undone, jumped away from and returned to like every other
version.

**A value whose overlay is empty is written exactly as it stands**, provenance
and all, and mints no version. It is already the content its hash names: a
second stamping would change the content the hash was taken over, and a version
identical to the one before it would say that something happened when nothing
did.

**A save never leaves a partial file at the node's path.** The write goes to a
temporary file in the target's directory and is renamed over the target only once
the whole archive is on disk, so the path holds either the previous
reconstruction or the new one and never a prefix of the new one
([archive-container.md](../formats/archive-container.md) § "Rust API"). Save over
the node's own path is exactly the case that needs it: the file being written is
the only copy of the one being replaced.

After the write, `History::set_disk_serial` names the version that reached the
disk, and the Edit History panel's disk mark follows it
([edit-history.md](edit-history.md) § "The Edit History panel").

## The dirty marker

`SceneNode::is_dirty()` is true when the **document half** of the version at the
node's cursor is not the one `History::disk_serial()` names. It answers from
state the node already keeps rather than from a flag a write has to remember to
clear.

The document half, because a version is a pair: the reconstruction and the
node's bench ([bench.md](bench.md)). A run of bench steps over a saved value
leaves the cursor off the disk version and the node clean all the same, because
a save of any of them would write the same bytes and the bench is not written at
all; each version records which version it shares its document half with, and
that is what the two are compared by. A commit, the one bench step that writes
the reconstruction, makes the node dirty like any other edit.

A node that came from no file is **not** dirty until something is done to it:
its disk serial starts on its first version like every other node's, and demo
data nobody has touched is not unsaved work. The first edit moves the cursor off
that version and the marker appears; a save, which for such a node is a Save As,
moves the disk serial to meet it.

**The Scene tree's node row** shows a leading `*` on the label while the node is
dirty ([scene-graph.md](scene-graph.md) § "Tree rows").

**The window title puts the `*` on the file name** rather than at the front:
`SfM Explorer - *foo.sfmr`, and `SfM Explorer - *foo.sfmr (+2)` with more files
loaded. The leading base title is what an attaching process matches on -- the
same reason the MCP endpoint's mention is a suffix -- so a marker in front of it
would break the match on exactly the sessions where something had been edited. A
window showing no file name keeps the bare base title.

## The close prompt

Closing a node whose cursor is not at its file, closing all with any dirty node
among them, or quitting with any dirty node, puts up **one modal titled
"Unsaved changes"** naming the dirty reconstructions, with **Save**, **Don't
Save** and **Cancel**.

- **Save** writes each dirty node first, a node with no file going through Save
  As. A failed write, or a dismissed dialog, **stops the close**: the nodes stay
  loaded and the user is where they were, because a close that proceeded past a
  refused write would discard exactly the work the prompt exists to protect.
- **Don't Save** closes without writing.
- **Cancel**, and Escape, do neither.

One modal for the whole gesture rather than one per node: closing all with three
dirty nodes is one thing the user asked for, and three prompts in a row is a
sequence they answer without reading.

## The Action Log

One entry per save, kind `File` ([action-log.md](action-log.md)):

```
Saved run_a at v7 to C:\data\run_a.sfmr
```

The serial is the version that reached the disk, which is the version the
materialisation minted when there was one and the version at the cursor when
there was not, so the row says exactly what `set_disk_serial` was given.

**A refusal is a failed entry carrying its reason**: the node is not loaded, the
node came from no file (Save), the version at the cursor was released to keep
the node inside the history budget, or the write itself failed.

## Testing

`crates/sfm-explorer/src/state/save/tests.rs` covers the save path headlessly:
that a loaded node starts clean and an edit makes it dirty, and that a node from
no file offers only Save As; that a save with an overlay
materialises into a version the cursor sits on and stamps the provenance it
hashed; that the file it writes records no ancestry; that a point id minted
before a save still resolves after it through the version graph, while the id
the panels *show* moves onto the file just written, and that an id a loaded
file's own recorded ancestry names keeps resolving across a save of the node;
that a save with no overlay writes the value and mints nothing; that
Save As writes elsewhere and re-points the node; that a save writes one log entry
naming the path and the version; that a failed write is refused with a reason and
changes nothing; and that the window title marks the first node while it is
dirty.

`crates/sfm-explorer/src/close_prompt/tests.rs` covers the modal itself: that
asking twice keeps the first question, that a prompt never asked draws nothing
and answers nothing, that a pending prompt stays up until it is answered, and
that Escape cancels and leaves nothing pending.

## Non-goals

- Saving several nodes with one command. Save and Save As act on the selected
  reconstruction; the close prompt is the one place that writes a set of nodes,
  and it does so because a close is about a set.
- Writing a version other than the one at the cursor. To save an earlier state,
  jump the cursor to it and save.
- Persisting the history. A file holds the value, not the versions, and an
  ancestor of that value contributes nothing to it.
- Point ids that outlive the session that minted them. See
  [what a saved file carries](#what-a-saved-file-carries).
- Autosave, and a backup of the file being overwritten.

## What a saved file carries

A saved file holds the reconstruction and the provenance stamped on it. It says
nothing about where its rows came from: no ancestor hash, and no mapping from an
earlier content onto its own rows.

That is a choice about what a point id is worth outside the session that made
it. Point identity is positional, a content hash and a row index in that content
([goto-point.md](goto-point.md)), so a save mints a new hash and renumbers every
row under it. Honouring an id taken against an earlier content therefore needs a
recorded map from that content to the new rows, and every such map has to stay
in the file for as long as the file exists. One entry per ancestor, forever: the
table grows with the length of the session rather than with the size of the
edit, and composing it walks the whole ancestry on every save.

**An id stays stable within a session**, which is where ids are quoted, clicked
and typed into Go to Point. An id minted before a save still names its point
afterwards because the session's version graph holds the version the save
materialised from, holds the hash of every point edit, and holds the
materialisation's own row map as an ordinary step. `point_ids::resolve` finds
the hash among those and walks to the cursor; a save is one more step on that
walk, not a break in it.

**An id does not carry into another session.** Quoting an id taken against a
content the reopened file is not gets the usual refusal, that the node has never
held content with that hash, because neither the file nor the fresh graph knows
where that content's rows went.

**Reading ancestry stays.** The `.sfmr` format keeps its optional `lineage` key,
described in
[the format spec's lineage section](../formats/sfmr-file-format.md#lineage-version-9),
and a loaded file that carries one is searched for a hash like any other place
an id can be found. So a file recording ancestry goes on resolving the ids that
ancestry names, in this session and in any later one. What the viewer no longer
does is write such a record into the files it saves.
