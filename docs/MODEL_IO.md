# Input and output contract

## Prepared source passed to `Collator`

Each source is an in-memory dictionary with:

- `rgb`: a Pillow RGB image in the same coordinate frame as all observations.
- `record.regions`: ordered dictionaries containing `region_id`, `semantic_group`, and `description`. Region IDs must be unique and must not contain reserved token strings. `person` is the human semantic group.
- `arrays.visual`: float32 array `[N, 2048]` of prepared region visual features.
- `arrays.geometry`: float32 array `[N, 14]` in the existing data-construction feature convention.
- `arrays.orientation`: float32 array `[N, 12]` of orientation evidence.
- `arrays.layout`: float32 array `[N, 3]` containing normalized mask centroid x/y and log mask-area fraction.
- `body`: float32 array `[N, 533]`: the prepared whole-body keypoint evidence (`wholebody133_xy_score_supported_plus_available`). It contains source observations, not target body labels.

Geometry slots used directly by the model are 0:2 normalized mask centroid, 2 log mask-area fraction, 3:5 box center, 5:7 box width/height, 10 depth-unknown indicator, 12 source log aspect ratio, and 13 the legacy role slot. **Set slot 13 to zero**, as in the main experiment. Remaining slots must retain the data builder's feature convention; do not invent replacement values for real-data inference. Full construction of features, orientation, depth, and body evidence is outside this core package.

`Collator` pads variable region counts and constructs `node_mask`, `semantic_match`, `person`, and `focus_hint`. An omitted focus argument means no manual focus hint. Source descriptions are observations; reference targets never enter these prompts.

## Model batch

The batch contains Qwen processor outputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `mm_token_type_ids`), token `positions`, and a nested `graph`. All tensor leaves must be moved to the model device. Token-position lists remain Python integer lists.

Graph shapes use batch size B and padded maximum region count N:

| Key | Shape | Type |
|---|---|---|
| visual | B × N × 2048 | float32 |
| geometry | B × N × 14 | float32 |
| orientation | B × N × 12 | float32 |
| body | B × N × 533 | float32 |
| node_mask | B × N | bool |
| semantic_match | B × N × N | float32 |
| person | B × N | bool |
| focus_hint | B × N | float32 |

## Layout outputs

- `log_aspect`: `[B]`, natural logarithm of target width / height.
- `boxes`: `[B,N,4]`, normalized `(left, top, right, bottom)` in the target frame.
- `centroids`: `[B,N,2]`, normalized target region mask centroid.
- `areas`: `[B,N]`, target mask-area fraction, bounded by predicted box area.
- `coarse_framing_logits`: `[B,N,2]`, class 0 full-body and class 1 half-body; meaningful for person regions.
- `emphasis`: `[B,N]`, internal allocation weights, not calibrated subject labels.
- `framing_logits`, `boundary_logits`, `visibility_logits`: compatibility outputs fixed to `unknown`, not fine-body predictions.

Ignore padded output rows using `node_mask`. Geometry is a target composition specification, not an image crop transform or a generated image.

## Layout supervision

For each source, targets contain ordered `region_ids` and a nonempty `references` list. Each reference has `reference_id`, `valid` (N booleans), `boxes` (N×4), `centroids` (N×2), `areas` (N), `log_aspect`, `body` (N null values for the main coarse model), and `coarse_body` (N values: 0 full, 1 half, or null).

The loss selects one complete reference using frame error plus person/environment-balanced geometry error over regions valid in all references. It then supervises the selected reference's valid regions; coarse human framing also comes from that same reference. Coarse cross-entropy has weight 0.1. This does not choose a different reference independently for each region.

## POR supervision

`JointCollator.por` accepts `(source, target)` pairs and one reason attribute index per source (0–6). POR sources use zero body evidence in the main data preparation. Its target schema is checked by `nextshot.auxiliary.validate_por_target`; it includes the bound source ID/hash, seven scores on a 0–10 scale and masks, reason text and masks, and provenance-status fields. Score queries precede the selected reason answer. All prefix tokens have label -100, so reason CE is answer-only.

POR forward returns `scores` `[B,7]`, `attribute_losses` `[B]`, `reason_losses` `[B]`, and `reason_tokens`. It supplies no reference geometry target. POR readouts are training-only and can be omitted for layout inference.
