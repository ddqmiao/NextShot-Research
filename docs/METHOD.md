# Method and implementation

NextShot learns a source-grounded composition target from observed regions. This guide follows Sections 3 and 4 of *NextShot: Learning Source-Grounded Structured Composition Targets for Portrait Photography*.

## Prediction target

For each source region, the model predicts a normalized visible-region box, mask centroid, and visible-area fraction. A global output represents the log target aspect ratio. Person regions additionally receive two coarse framing logits: full body and half body. Source IDs remain attached to all predicted rows.

| Stage | Code | Role |
| --- | --- | --- |
| Source relations | `nextshot/backbone.py`: `pair_features`, `RelationAttention`, `SourceEncoder` | Encode regional evidence and a frame anchor; add learned pair-feature biases to attention |
| Multimodal fusion | `nextshot/backbone.py`: `StructuredLayoutModel`, `attach_language_lora` | Project graph states into multimodal tokens and read out region/global query states |
| Geometry decoding | `nextshot/framing.py`: `decode_geometry`; `nextshot/decoder.py`: `CoarseDecoder` | Predict bounded boxes, centroids, areas, and full/half framing |
| Source prompts | `nextshot/source.py`: `source_text`, `Collator` | Bind descriptions to source identities without reference targets |
| Auxiliary learning | `nextshot/model.py`: `Model`, `reason_losses`; `nextshot/auxiliary.py`: `JointCollator` | Attribute-score regression and answer-only rationale supervision |
| Layout supervision | `nextshot/geometry_loss.py`: `loss`; `nextshot/losses.py`: `loss` | Select a whole reference, supervise its valid geometry and coarse framing |

## Region–frame encoding and fusion

Each region combines appearance features with geometry and orientation evidence. A learned frame token supplies a common canvas anchor. Directed pair features include relative centroid and log-area differences, distance, box overlap, overlap ratios, and semantic compatibility. The legacy role slots are zero in the main configuration.

Relation-biased attention produces source-region states and a frame state. A shared projector maps them into Qwen3-VL's hidden space. The image, textual region observations, graph tokens, and regional/global queries enter a frozen backbone with trainable language query/value LoRA adapters. The main paper configuration uses a 256-dimensional relational representation, 4 attention heads, and LoRA rank 8 with alpha 16.

## Framing-conditioned geometry

The decoder predicts a global target frame and coarse person framing. Their embeddings then condition regional updates. Residual geometry is bounded: boxes stay inside the target canvas, centroids stay inside boxes, and visible areas do not exceed the enclosing box areas.

`backbone.py → framing.py → online.py → decoder.py → aes_core.py → model.py` preserves the existing constructor chain and registered parameter names. Some inherited allocations are replaced by the active decoder. They have been retained to avoid changing initialization order. The public model is exported as `nextshot.NextShot`.

## Complete-reference supervision

For a source with multiple references, reference selection compares predictions on the intersection of valid source regions, using frame error and person/environment-balanced geometry error. The selected reference then supervises all of its valid regions. Its coarse framing labels come from that same reference. The implementation does not select one reference independently for each region.

Geometry uses smooth-L1 errors with box and centroid scales of 0.1, log-area scale of 0.7, and log-aspect scale of 0.3. Coarse framing cross-entropy has weight 0.1. The paired loss combines geometry, frame, and available coarse framing supervision.

## Auxiliary supervision

PortraitCraft-Aux supplies seven attribute scores and available rationales, not target geometry. Attribute regression and rationale losses have weights 0.25 and 0.05 in the paper. Rationale cross-entropy supervises answer tokens only; prefix labels are masked. Score queries precede the rationale answer in the causal sequence.

The auxiliary branch updates the shared representation and language adapters. Only paired geometric supervision trains the composition decoder. `layout_state()` omits attribute query/head tensors; layout inference constructs the model with `with_auxiliary=False`.

## Configuration and boundaries

`configs/nextshot_4b.json` supplies model-construction settings with a replaceable relative backbone path. Dataset membership, training schedule, optimizer state, selected weights, and complete feature preparation are not included. For the field-level contract, see [MODEL_IO.md](MODEL_IO.md); for the selected correspondence implementation, see [DATA.md](DATA.md).
