# Copied response recipient identity

This is identity retention for evaluated native response copies, not a geometry
export, renderer, intersection implementation, or optical evaluator. It does
not assert estimator completeness or convergence. Missing response evidence
remains unknown and must leave the existing Phong response visible.

## Opt-in and owners

Only `light_field_information: evaluated-responses-v1` permits
`prepare_photo_geometry(..., response_lineage_out=output_dict)`. Its ordinary
return tuple and native scene buffers are unchanged. The producer owns the
snapshot and binds it to the exact scene, camera, geometry revision, response
rows, and package receipt. An unrelated or stale snapshot is not usable.

At the original append/filter operation, `scene_builder.py` copies the actual
`np.flatnonzero(valid)` values. `studio_sphere_regions.py` copies its existing
representative selection, keep mask, and old-to-new map before those values
are discarded. Native BVH `tri_ids` resolves to these packed native triangle
indices; BVH leaf order is not a new identity namespace.

## Snapshot ABI

`camera_designer/studio_response_lineage.py` produces version
`studio-response-lineage-v1`. No positions, normals, or optical buffers are
included. `rows` and `spheres` are detached little-endian int32 arrays; JSON
packaging may use `.tolist()` without loss.

- `rows[native_triangle]`: `[source_namespace, source_face, primitive_kind,
  primitive_index]`.
- Namespace 0 is unmatched/native-only, 1 is the submitted MRP1 subject
  ordinal (`interior` then `glass`), and 2 is the fixture upload ordinal.
- Mesh rows have kind 0/index -1. An exact sphere representative has kind
  1/index equal to the actual sphere index.
- `spheres`: `[sphere_index, native_owner_triangle, source_start,
  source_count]`. Start/count are the original authored sphere membership,
  not a hit-position-derived facet.
- `fixture_parts`: the copied request `fixtures.response_lineage`, with
  `{version, parts:[{source_id,start,count}]}`. These contiguous upload ranges
  retain original `part.id` under canonical source prefixes (`fixture`,
  `room`, `backdrop`, `backdrop-holder`, `table`, `gobo`, `stand`, `shop`).
- `sources`, `geometry_sha256`, and `revision` retain source metadata. The
  packaging owner must additionally bind its exact scene/camera receipt.

`copied_recipient(lineage, native_hit, primitive_kind, primitive_owner)` uses
only these copied values. The first hit, kind, and owner must be the values
recorded at the native first-visible selection, not a later bounce. The
native observer separately preserves current event kind/owner and region
state. A virtual rod partition (kind 2) has no mesh recipient and returns
`(0,-1)` even though native transport borrows its host triangle's material.
Native camera housing and a missing first hit also return `(0,-1)`.

## Browser face retention and flat ID

`marble_response_lineage.js` copies original subject ordinals before the
existing bubble omission and original part/local-face ordinals at builder
callbacks. `withSceneCategory` and the existing subject transform preserve
the identity property. Geometry and the transparent EBO are unchanged.

Flat unsigned attribute 7 is `(namespace << 28) | (index + 1)`, where index
is less than `0x0fffffff`. Zero is unmatched. Namespaces 1 and 2 retain the
source face ordinal. Namespace 3 identifies an exact sphere primitive;
all explicitly authored member faces share that primitive ID, with no
claim of a native triangle hit. The Python `recipient_id` helper and JS
`recipientId` implement the same encoding.

Meshes retain CPU-only `responseSources` while the explicit option is on.
`resource.bindResponseLineage(lineage)` may allocate attribute 7 only after
the response consumer validates the package and its exact current ownership.
No request alone binds this attribute. The function does not touch existing
positions, materials, groups, or EBO ordering.

Bubble surfaces omitted by the existing Phong mesh have no raster recipient.
They must stay unmatched/alpha zero, never map to another face with the same
material. Exact sphere identity does not imply facet or projection equality.
Scene/camera matching and response sample validity remain package/consumer
responsibilities, not properties inferred from this ID.

## CPU verification

`tests/test_studio_response_lineage.py` exercises actual native degenerate
filtering and sphere compaction, output equality, namespaces, strict opt-in,
and rejection of inconsistent provenance. Turing's
`tests/meowdoku_response_lineage_check.mjs` checks source ordinals, bubble
omission, explicit sphere ranges, distinct same-material layers, transparent
sorting, and request gating. The actual shared scene-builder CPU checks
preserve the pre-existing four full-scene hashes with the option off and
confirm unchanged geometry/material arrays when it is on.

No render, GL context, exposure, screenshot, or deployment is part of these
tests.
