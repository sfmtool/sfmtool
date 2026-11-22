# The `.kdf` chunk layout legend

**Status:** Draft. Amends
[formats/kdf-file-format.md](../formats/kdf-file-format.md), § "How this
format meets the container rules" and § "Metadata and versioning". Decided:
the [archive container](../formats/archive-container.md#what-an-entry-holds)
asks that an entry which groups arrays have its layout described in the
format's metadata, and that a file explain itself to someone with only the
file, `unzip` and `zstd`. Open: the exact JSON spelling below.

## Purpose

A `.kdf` tree chunk entry, `trees/{t}/chunks/{c}/chunk.{M}.{P}.{scalar_type}.zst`,
holds three arrays back to back: the ten node columns as a row-major `(10, M)`
`uint32` array, the `M` split coordinates of the scalar type, and the `P`
`uint32` feature IDs. Version 3 states that layout, and the names of the ten
node columns, only in the format spec. Someone who opens a version 3 file
without the spec sees an entry name with two counts and a type, and cannot
tell what the bytes are. This draft proposes `.kdf` format version 4, which
adds a `chunk_layout` legend to `metadata.json` that describes the chunk
entry's arrays, so the file describes its own chunk entries.

## The `chunk_layout` field

`metadata.json.zst` gains one required field, `chunk_layout`: an array of three
objects, one per array of the chunk entry, in the order the arrays are stored.
Each object has `name`, `dtype` and `shape`; a shape lists each dimension as
the count letter from the entry name (`"M"`, `"P"`) or a fixed integer. The
node-column object also has `columns`, the name of each row of the `(10, M)`
array in order.

For a `uint8` index the field is:

```json
"chunk_layout": [
  {"name": "nodes", "dtype": "uint32", "shape": [10, "M"],
   "columns": ["kind", "logical_node_id", "split_dimension",
               "left_chunk", "left_node", "left_logical_node_id",
               "right_chunk", "right_node", "right_logical_node_id",
               "leaf_start"]},
  {"name": "splits", "dtype": "uint8", "shape": ["M"]},
  {"name": "feature_ids", "dtype": "uint32", "shape": ["P"]}
]
```

A `float32` index writes `"float32"` as the `splits` dtype, matching
`scalar_type`. The column names are those of the node-column table in
[kdf-file-format.md § Nodes, leaves, and identity](../formats/kdf-file-format.md#nodes-leaves-and-identity).

The field is required exactly as `node_kinds` is: a writer writes this value,
and a reader rejects a version 4 file whose `chunk_layout` is absent or differs
from it in any name, dtype, shape or column, or whose `splits` dtype differs
from `scalar_type`. The legend documents the layout; it does not let a writer
choose another one, so readers keep decoding the chunk by the fixed layout and
check the legend against it.

`chunk_layout` sits in `metadata.json`, so `metadata_xxh128`, and through it
`content_xxh128`, covers it. No other entry, digest or section changes.

## The version change

Following [format-versions.md § What a version change consists of](../formats/format-versions.md#what-a-version-change-consists-of):

1. **The constant.** `KDF_FORMAT_VERSION` in
   [`types.rs`](../../crates/sfmtool-kdf-format/src/types.rs) becomes `4`.
   Every release that writes `.kdf` files writes version 3, so version 3 is in
   the read window.
2. **The writer** writes version 4 with `chunk_layout`.
3. **The reader** accepts versions 3 and 4. A version 3 file has no
   `chunk_layout`; its layout is the fixed one above, so the reader converts it
   at the I/O boundary by supplying that legend, and no caller branches on the
   version. A version 3 file's `metadata_xxh128` and `content_xxh128` are
   verified against its own stored bytes, which do not hold the field. The
   version refusal message names version 4 as the version the reader reads and
   still says to rebuild the index.
4. **`sfm ws upgrade`** rewrites a version 3 index as version 4. Only
   `metadata.json.zst` and `content_hash.json.zst` change: the upgrade copies
   every other entry's stored bytes unchanged, writes the new metadata,
   recomputes `metadata_xxh128` and folds a new `content_xxh128` from the
   unchanged other section digests. The
   [ws upgrade spec](../cli/workspace/ws-upgrade-command.md) stops listing
   `.kdf` indexes among the files left as they are.
5. **Tests** read a version 3 file and check that it converts to the fixed
   legend; check that a version 4 file with a missing or altered
   `chunk_layout`, or a `splits` dtype other than `scalar_type`, is refused;
   check that a version 5 file is refused; and check that `sfm ws upgrade`
   turns a version 3 index into a version 4 index whose queries return the same
   results.
6. **The checked-in files.** `test-data/` holds no `.kdf` file, so none is
   rewritten.
7. **The format spec** gains the field in its metadata table, a "Version 3 →
   Version 4" subsection under Format versions, a Version History line, and in
   § "How this format meets the container rules" the statement that the chunk
   entry's layout is described in `metadata.json`. This draft is then deleted.
8. **The release notes** say that `.kdf` changed to version 4, that it adds
   the chunk layout legend, and that `sfm ws upgrade` converts version 3
   indexes.
