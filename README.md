# MABSplit Boosting (FastForest Extension)

This repository is an extension of FastForest focused on MABSplit-based split selection and a new histogram-based gradient boosting pipeline.

Upstream codebase we extended:
- FastForest: `https://github.com/ThrunGroup/FastForest`

## What We Added

The main extension in this repo is a Gradient Boosted Decision Tree (GBDT) implementation that integrates MABSplit into split search.

Core extension files:
- `gbdt_trainer.py`: GBDT training loop and model interface (`GBDTTrainer`, `GBDTParams`)
- `tree_builder.py`: tree growth logic used by the boosting pipeline
- `mabsplit_split_search.py`: MABSplit split search and exact baseline split search
- `grad_hess.py`: gradient/hessian provider used by second-order boosting
- `binning.py`: feature bin construction and application

In this extension, the comparison is:
- `split_search="mab"`: GBDT + MABSplit
- `split_search="exact"`: GBDT with exact histogram split search baseline

## Installation

From repo root:

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Run The Gradient Boosting Extension

### 1) Quick sanity run (small datasets)

```bash
python experiments/quick_mabsplit_gbdt_eval.py --datasets synthetic_reg,synthetic_clf --exact
```

This runs the extension path only and prints runtime/metric comparisons between MABSplit and exact split search.

### 2) Large benchmark (synthetic defaults)

```bash
python experiments/mabsplit_gbdt_large_benchmark.py --n-runs 5 --output result_gbdt_extension.csv
```

### 3) Large benchmark on your own real dataset

```bash
python experiments/mabsplit_gbdt_large_benchmark.py \
  --real-dataset mydata /path/to/data.csv classification target \
  --n-runs 3 \
  --output result_real_gbdt_extension.csv
```

`--real-dataset` arguments are: `NAME PATH TASK TARGET_COL`.
`TASK` must be `classification` or `regression`.

## Validate The Extension

Run targeted tests for the new path:

```bash
pytest tests/test_mabsplit_gbdt.py
```

## Reproduce Original Tables/Figures

To reproduce the existing paper tables and figure scripts in this repo:

```bash
bash repro_script.sh
```

This may take many hours.

## Codebase Notes

- `data_structures/`: base tree/forest structures and wrappers
- `experiments/`: scripts for runtime, budget, scaling, feature-stability, and boosting evaluations
- `utils/solvers.py`: core MABSplit logic used by tree/forest code paths

## Large Dataset Files

Some derived datasets are generated locally during experiments and can exceed GitHub limits.
Do not commit large generated files such as:
- `experiments/dataset/derived/covtype_1v2_binary.csv`
- `experiments/dataset/derived/paper_covtype_class2_vs_rest_581k.csv`
