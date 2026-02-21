# Attribution

This project is based on the [FastForest](https://github.com/ThrunGroup/FastForest) repository by the Thrun Group. We build upon their foundational work on efficient decision tree and forest implementations, extending their approach with MABSplit and additional experimental features.

# Description of Files

The files are organized as follows:

- `data_structures` contains all of the data structures used in our experiments, e.g., 
forests, trees, nodes, and histograms.
  - the `wrappers` subdirectory contains convenience classes to instantiate models of various types,
  e.g., a `RandomForestClassifier` is a forest classifier with `bootstrap=True` 
  (indicating to draw a bootstrap sample of the `n` datapoints for each tree), `feature_subsampling=SQRT`
  (indicating to consider only `sqrt(F)` features of the original `F` features at each node split), etc.
- the `experiments` subdirectory contains all the code for our core experiments
  - `experiments/runtime_exps` contains the script (`compare_runtimes.py`) to reproduce the results of Tables 1 and 2, as well as the results of running that script (the files ending in `_profile` or `_dict`)
  - `experiments/budget_exps` contains the script (`compare_budgets.py`) to reproduce the results of Tables 3 and 4, as well as the results of running that script (the files ending in `_dict`)
  - `experiments/sklearn_exps` contains the script (`compare_baseline_implementations.py`) to reproduce the results of Table 6 in Appendix 4
  - `experiments/scaling_exps` contains the scripts (`investigate_scaling.py` and `make_scaling_plot.py`) to reproduce Appendix Figure 1 in Appendix 2 
- the `tests` subdirectory contains verification tests for the tree/forest implementations.
- feature-stability experiments for Table 5 are in `experiments/feature_stability/feature_importance_tests.py`.
  Results are written to `experiments/feature_stability/stat_test_stability_log/reproduce_stability.csv`.
- the `utils` directory contains helper code for training forest-based models
  - `utils/solvers.py` includes the core implementation of MABSplit in the `solve_mab()` function
  
# Reproduce the tables
- To reproduce the results in all the tables, and to reproduce the figure in Appendix 2, run:
  - `bash repro_script.sh`
  This may take many hours.

# Installation and Large Dataset Files
- Install dependencies:
  - `pip install -e .`
  - or `pip install -r requirements.txt`
- Some derived datasets are generated locally during experiments and can be very large.
  GitHub rejects files above 100 MB, so large generated CSVs should not be committed.
- In particular, keep these files local only:
  - `experiments/dataset/derived/covtype_1v2_binary.csv`
  - `experiments/dataset/derived/paper_covtype_class2_vs_rest_581k.csv`
- If needed, regenerate derived datasets by rerunning the benchmark/data-prep scripts used in this repo instead of committing the raw generated CSVs.
