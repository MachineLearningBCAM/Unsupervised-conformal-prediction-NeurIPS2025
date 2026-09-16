# Split conformal classification with unsupervised calibration

**Train with labeled data; calibrate prediction sets with new, unlabeled examples.** This repository provides **Python and MATLAB implementations** of [*Split Conformal Classification with Unsupervised Calibration*](https://proceedings.neurips.cc/paper_files/paper/2025/file/0b276510ec2d3f6613a8b60c41ff0438-Paper-Conference.pdf) (NeurIPS 2025).

The method assigns weights to every possible label of each unlabeled calibration example. It chooses the weights by matching the weighted calibration examples to labeled training examples with a Gaussian kernel, then uses a weighted conformal quantile. It is useful when obtaining additional labels solely for calibration is costly or impossible. Training still requires labeled examples.

## Choose an implementation

| | Python | MATLAB |
| --- | --- | --- |
| Start here | [Python guide](python/README.md) | [MATLAB guide](matlab/README.md) |
| Main code | [`python/src/unsupervised_conformal/`](python/src/unsupervised_conformal/) | [`matlab/`](matlab/) |
| Runnable example | `python python/examples/quickstart.py` | Run `main` from `matlab/` |
| Uses | Any classifier that returns class probabilities | Included USPS neural-network experiment |
| Main dependencies | NumPy, SciPy, OSQP | MATLAB Statistics and Machine Learning Toolbox; Optimization Toolbox |

### Python: two-minute example

From the repository root with Python 3.10 or newer:

```bash
python -m pip install -e ".[examples]"
python python/examples/quickstart.py
```

The quickstart uses bundled Iris data and downloads nothing. To compare supervised calibration, the proposed unlabeled method, and naive predicted-label calibration on the included USPS data, run `python python/examples/usps_comparison.py`. Both examples print held-out coverage and mean prediction-set size for **one** split, so their numbers are demonstrations rather than the paper's multi-split benchmark.

### MATLAB: original USPS experiment

In MATLAB, change to the repository's `matlab/` directory and run:

```matlab
main
```

The script loads [`data/usps.mat`](data/usps.mat), trains a classifier, and compares the same three calibration approaches. Its default optimization uses `quadprog`; CVX and MOSEK are optional. See the [MATLAB guide](matlab/README.md) for dependencies and settings.

## Which data do you need?

| Data | Role |
| --- | --- |
| Labeled training examples | Train the classifier and supply examples for kernel matching |
| Unlabeled calibration examples | Estimate label weights and the conformal threshold |
| Class probabilities | Compute scores for every possible label of calibration and test examples |

Training and calibration examples should come from the same distribution. Both language implementations support the randomized adaptive score used in the paper; Python also offers the deterministic score `1 - class probability`. The Python package accepts any pre-trained classifier's probabilities; it does not retrain the classifier.



## Repository layout

```text
README.md                 Overview and language choice
python/README.md          Python installation and API
python/src/               Importable Python package
python/examples/          Iris quickstart and USPS comparison
python/tests/             Python tests
matlab/README.md          MATLAB usage and function map
matlab/*.m                MATLAB method and USPS experiment
data/usps.mat             Included example data
```


## Citation and license

If you use this work in research, please cite:

```bibtex
@inproceedings{mazuelas2025split,
  title     = {Split Conformal Classification with Unsupervised Calibration},
  author    = {Mazuelas, Santiago},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

Released under the [MIT license](LICENSE). Questions: Santiago Mazuelas, smazuelas@bcamath.org.
