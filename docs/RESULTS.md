# Paper results

Values below are transcribed from the supplied NextShot manuscript. They were not recomputed for this release. The same numbers and evaluation units are available in [machine-readable JSON](../results/paper_results.json).

## Main test results (Table 1)

Reference agreement uses **781 sources**. Layout preference uses the same **200 test sources** for both evaluators. Preference columns report **NextShot over each row method**, as percentages of all planned judgments. Missing fields are not predicted by that baseline.

| Method | Box IoU | Person IoU | Environment IoU | Centroid MAE | Area MAE | Frame RE (%) | Qwen win (%) | Flash win (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoChange | 0.5639 | 0.4914 | 0.6079 | 0.0745 | 0.0781 | 9.848 | 89.5 | 77.0 |
| AR-8B | — | — | — | 0.0680 | 0.0727 | — | 79.5 | 81.0 |
| NoVLM-AttrAux | 0.6319 | 0.5902 | 0.6571 | 0.0677 | 0.0675 | 3.937 | 45.0 | 72.5 |
| w/o AesAux | 0.6384 | 0.6025 | 0.6632 | 0.0660 | 0.0666 | 3.747 | 75.5 | 84.5 |
| w/o RelBias | 0.6358 | 0.5975 | 0.6611 | 0.0664 | 0.0678 | 3.740 | 79.5 | 79.5 |
| w/o Framing Conditioning | 0.6307 | 0.5972 | 0.6519 | 0.0672 | 0.0653 | 3.460 | 73.0 | 66.5 |
| NextShot | 0.6373 | 0.5973 | 0.6630 | 0.0659 | 0.0670 | 4.068 | — | — |

Higher IoU and preference are better; lower MAE and frame RE are better. AR-8B predicts only centroids and visible areas. NoVLM-AttrAux uses attribute supervision without the rationale objective.

NextShot improves geometry over NoChange and shared centroid/area fields over AR-8B. Reference fit and layout preference measure different properties: some ablations have better individual reference-agreement metrics. Qwen prefers NoVLM-AttrAux over NextShot more often in their test comparison (NextShot wins 45.0%), while Flash favors NextShot.

## Development comparisons (Table 2)

Qwen uses 200 development sources; Flash uses a 100-source subset. These are development comparisons, separate from Table 1 test results. Rates report preference for A over B.

| A | B | Qwen (%) | Flash (%) |
| --- | --- | ---: | ---: |
| NextShot | w/o AesAux | 75.5 | 86.0 |
| NoVLM-AttrAux | NoVLM-Paired | 78.5 | 72.0 |
| w/o AesAux | NoVLM-Paired | 43.0 | 61.0 |
| NextShot | NoVLM-Paired | 78.0 | 78.0 |
| NextShot | AR-8B | 84.5 | 76.0 |
| NextShot | ZS-32B | 83.0 | 69.0 |

## Guided photography (Table 3)

The study contains 100 initial–recaptured pairs, four photographers, two subjects, and five human evaluators. Judgments are repeated observations on the same pairs; 500 judgments do not mean 500 independent photographic cases.

### Target-layout preference on real inputs

| A | B | Qwen (%) | Flash (%) |
| --- | --- | ---: | ---: |
| NextShot | NoVLM-Paired | 68.0 | 72.0 |
| NextShot | w/o RelBias | 67.0 | 86.0 |
| NoVLM-Paired | NoChange | 100.0 | 91.0 |
| w/o RelBias | NoChange | 97.0 | 94.0 |

### Human evaluation

| Unit | N | Guided | Initial | No preference / no majority | Guided rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Individual judgments | 500 | 390 | 36 | 74 | 78.0% |
| Pair-level majority (at least 3/5) | 100 | 89 | 0 | 11 | 89.0% |

The 78.0% rate is a preference share, not classification accuracy. For model evaluation of the actual photographs, Qwen and Flash favor guided recaptures on 83.0% and 90.0% of the 100 pairs, respectively.

## Resource measurements (Table 14)

Prediction service was measured on one A800, batch size one, on 50 prepared inputs after three separate warm-up inputs. Timing includes input-package loading/validation, RGB decoding and preprocessing, transfer, prediction/generation, and numerical output conversion. It excludes upstream region perception, feature extraction, model loading, and guidance rendering.

| Method | Median (s) | P95 (s) | Forward/generation (s) | Peak allocated memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| NoVLM-Paired | 0.087 | 0.124 | 0.009 | 0.051 |
| NoVLM-AttrAux | 0.059 | 0.080 | 0.009 | 0.051 |
| w/o AesAux | 0.206 | 0.306 | 0.144 | 8.729 |
| NextShot | 0.206 | 0.268 | 0.144 | 8.729 |
| NextShot-8B | 0.235 | 0.310 | 0.167 | 15.692 |
| AR-8B | 16.522 | 26.073 | 16.455 | 19.506 |

The 0.206-second NextShot value is not end-to-end processing time from an arbitrary raw photograph. The comparison with AR-8B retains each implemented output format and generation setting.

Training measurements include loop-internal validation and checkpoint I/O, and exclude upstream perception/data preparation.

| Method | Trainable parameters (M) | GPUs | Peak memory per GPU (GiB) | GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| NoVLM-Paired | 4.833 | 1 | 0.43 | 0.028 |
| NoVLM-AttrAux | 4.835 | 1 | 2.40 | 0.042 |
| w/o AesAux | 15.924 | 2 | 27.99 | 1.886 |
| NextShot | 15.949 | 2 | 43.71 | 4.660 |
| NextShot-8B | 28.265 | 2 | 43.03 | 6.049 |
| AR-8B | 3.834 | 2 | 54.56 | 19.225 |

## Data and evaluation context

The development set is used for checkpoint selection and emphasizes composition changes. Earlier models had accessed the retained test assignment; the current models and protocol were fixed before scoring. Source groups and known duplicate components are accounted for in the manuscript. See [data membership](DATA.md) and [release scope](RELEASE.md).
