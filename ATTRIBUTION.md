# Attribution and Credits

## Base Repository

This project is built upon the [FastForest](https://github.com/ThrunGroup/FastForest) repository developed by the Thrun Group at Stanford University.

**FastForest GitHub:** https://github.com/ThrunGroup/FastForest

## FastForest Foundation

The FastForest project provides efficient implementations of decision tree and random forest algorithms. Our project extends and builds upon the following core components from FastForest:

- **Data Structures** (`data_structures/`): Tree and forest base classes, histogram implementations, and node structures
- **Base Classifiers and Regressors**: Foundational implementations for binary classifiers, regressors, and ensemble methods
- **Training Infrastructure**: Core training and fitting mechanisms

## MABSPLIT Extensions

This project extends FastForest with:

- **MABSplit Algorithm**: A novel split search strategy using multi-armed bandit techniques (see `utils/solvers.py`)
- **Gradient Boosting Extensions**: Enhanced implementations for gradient boosted ensembles
- **Experimental Framework**: Comprehensive experiments and benchmarking utilities (`experiments/` directory)

## Licensing

Please refer to the FastForest repository for original licensing information. Any modifications and extensions in this project follow the principles and practices of the original FastForest project.

## How to Cite

If you use this work, please cite:

1. The original FastForest paper/repository from the Thrun Group
2. This extension project (with appropriate reference details)

## Acknowledgments

We gratefully acknowledge the Thrun Group's work on FastForest, which provided the essential foundation for this research.
