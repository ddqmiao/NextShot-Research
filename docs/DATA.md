# Data construction and released algorithms

The paper derives AesRecon-CT from AesRecon photographic demonstrations and uses PortraitCraft-Aux for auxiliary aesthetic supervision. Original photographs and full derived datasets are not distributed in this package.

## Research pipeline

1. Curate source–reference groups for portrait relevance and usable photographic content; update dependent coordinates after border removal.
2. Extract regional semantics, localization, masks, visual features, pose/orientation, and depth evidence using pretrained models.
3. Match source and reference regions using appearance, local context, and semantic compatibility. Position and scale are excluded from matching evidence because composition changes alter them.
4. Transfer reference boxes, centroids, and visible areas to matched source-region identities. Unmatched or ambiguous source regions remain inputs and prediction variables without geometric supervision from that reference.
5. Retain each reference as a complete alternative. Build validity masks and weak full/half-body labels.
6. Keep source groups intact across splits and isolate known duplicate components. Auxiliary aesthetic data supplies scores and rationales, not target layouts.

## Released matching code

| File | Responsibility |
| --- | --- |
| `data_processing/matching.py` | Combined similarity, dummy-node partial assignment, acceptance margins, rejected/ambiguous matches |
| `data_processing/semantics.py` | Transparent category/content compatibility and person-granularity checks |
| `data_processing/ids.py` | Stable correspondence identifiers |
| `data_processing/regions.py` | Mask crop bounds, aspect-preserving resize, and padding masks |

`match_regions(pair, source, target, visual, context, method, cfg)` takes:

- `pair`: `pair_id`, `worse_image_id` for the source, and `better_image_id` for the reference.
- `source` / `target`: ordered region dictionaries with `region_id`, `semantic` text, `semantic_group`, and optional `primary`.
- `visual` / `context`: finite precomputed similarity arrays of shape source-count × reference-count.
- `method`: a caller-defined version label.

It returns correspondence records and diagnostic evidence. Status values are `matched`, `ambiguous`, `source_only`, and `target_only`. Region IDs are scoped to their image; matching never assumes that equal region labels identify the same content across images.

The combined score uses appearance/context/semantic weights 0.75/0.15/0.10 and a small primary-region prior. The partial assignment permits unmatched nodes. Accepted edges must pass appearance, combined-score, assignment-margin, and granularity checks. Exact thresholds remain in `MatchConfig`.

Descriptor extraction, Qwen/GroundingDINO/SAM execution, database publication, human-review decisions, and final dataset export are not included. The pure crop utilities do not substitute for the trained feature extractor.

## Paper cohorts

| Use | Source count | Additional detail |
| --- | ---: | --- |
| Paired training | 2,915 | 4,268 valid references |
| Auxiliary training pool | 25,164 | Attribute scores and rationales |
| Development | 450 | 647 valid references; checkpoint selection |
| Test reference agreement | 781 | Eligible reference geometry |
| Test layout preference | 200 | Fixed subset shared by both model evaluators |
| Guided photography | 100 pairs | 500 human judgments |

The 25,164-image auxiliary pool is distinct from actual visits by the selected epoch-4 checkpoint: the paper reports 16,776 auxiliary visits at that checkpoint. The broader retained-image inventories are also distinct from these experiment cohorts.

The development set was selected to emphasize composition changes. Earlier models had been run on the retained test assignment; the current checkpoints and evaluation protocol were fixed before scoring. Do not describe it as a never-accessed blind test. Known duplicate-component accounting and grouped statistics are described in the manuscript.

## Upstream data

- [AesFormer / AesRecon](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026)
- [PortraitCraft project](https://github.com/yytang25/PortraitCraft) and [dataset](https://huggingface.co/datasets/zijielou/PortraitCraft)

Obtain upstream data directly and follow its current access and redistribution terms. No new redistribution permission is granted by this code package. See [third-party acknowledgments](../THIRD_PARTY_NOTICES.md).
