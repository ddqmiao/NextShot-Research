# Verification of this package

Checked on 2026-10-14. This record concerns the partial code release and does not claim a rerun of model training or paper evaluation.

## Passed checks

- **Five original core CPU checks:** zero-residual geometry, bounded outputs, padding masks and conditioning gradients, complete-reference selection, and chunked/empty-answer rationale loss.
- **Eight existing matching checks:** pose-change correspondence, rejection of unsupported primary matches, person granularity, semantic compatibility, partial assignment, crop/padding geometry, and invalid inputs.
- **Fourteen core modules unchanged:** their bytes match the existing research core package.
- **Matching/extraction equivalence:** the relocated matching and semantic definitions, stable ID function, and crop/padding definitions match their source syntax trees. Only imports and module boundaries change.
- **Architecture configuration:** every published architecture field matches the main 4B configuration except the portable backbone path.
- **Package build:** an offline wheel build under Python 3.12 succeeded and contains all 19 modules across `nextshot` and `data_processing`. No dependencies were downloaded or installed for this build.
- Python source syntax, package-relative imports exercised by the checks, and a limited scan for private filesystem paths and credential-like strings.

The missing `configs/nextshot_4b.json` used by the original core checks is now included. The original checks themselves are unchanged.

## CPU environment actually used

| Component | Version |
| --- | --- |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| Pillow | 11.3.0 |

The 13 checks passed directly from the repository source on this existing CPU environment. Python 3.10+ is the supported packaged environment declared in `pyproject.toml`; source-level CPU checks on Python 3.9 do not imply support for installing the complete backbone stack there.

## Recorded main-experiment environment

The original research package records Python 3.10, PyTorch 2.11.0 with CUDA 13.0, Transformers 5.9.0, and NumPy 2.2.6. `requirements-backbone.txt` preserves those reported model-library versions. They have not been installed or revalidated on a GPU during this release task.

The current checks use no photographs, Transformers installation, pretrained Qwen weights, trained NextShot adapter, GPU, or paid model API. They verify selected algorithm behavior and extraction integrity, not trained-model quality, full-checkpoint compatibility, or end-to-end numerical reproduction.

## Commands

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
shasum -a 256 -c MANIFEST.sha256
```

No test instantiates the pretrained model. Full inference additionally needs prepared input assets and separately acquired model weights; the trained adapter is outside this release.
