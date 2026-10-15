# Release scope

## Included

- Fourteen core model/loss/collator modules from the existing 4B research implementation.
- A portable model-construction configuration, including the configuration required by the existing CPU tests.
- Selected content-based region-matching and crop/padding algorithms.
- Existing core and matching checks, with package imports adjusted where needed.
- Paper Figures 2 and 3 rendered as PNG, result tables, code navigation, and input/output documentation.

## Not included

Trained adapters, base-model weights, original photographs, full dataset inventories or annotations, training orchestration, pretrained perception/feature extraction, complete experiment evaluation drivers, and interactive examples are outside this release.

The repository supports method inspection and synthetic algorithm checks. It does not by itself reproduce the reported paper results. No future release date is promised.

## Paper source

Title: **NextShot: Learning Source-Grounded Structured Composition Targets for Portrait Photography**.

The supplied 22-page manuscript is labeled “Under review as a conference paper at ICLR 2027” and lists anonymous authors. This release does not claim acceptance or assign an author order. A permanent paper URL and complete bibliographic citation can be added when available. The full manuscript is not bundled; this package presents selected figures, method descriptions, and tables.

The figure assets preserve the manuscript's data/method illustrations. Numerical tables are transcribed from Tables 1–3 and 14, with their evaluation units retained. Source hashes and asset transformations are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Code changes for publication

Core model modules and the original five core checks are copied without changes. A missing configuration is supplied so the existing checks can find their architecture settings. The matching algorithm and semantic rules retain their definitions; matching imports are package-relative. Stable ID generation and pure crop/padding definitions are extracted without their database or pretrained-model runtime dependencies. Eight existing matching checks are retained with adjusted imports.

Published verification reports cover this package, not historical training. See [VERIFICATION.md](VERIFICATION.md). The file manifest can be checked using:

```bash
shasum -a 256 -c MANIFEST.sha256
```
