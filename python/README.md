# Python implementation

The importable `unsupervised_conformal` package implements the kernel label-weight optimization and weighted conformal quantile in Algorithms 2–3 of the [paper](https://arxiv.org/abs/2510.07185). It accepts probabilities from any already trained classifier.

## Install and run

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -e ".[examples]"
python python/examples/quickstart.py
python python/examples/usps_comparison.py
```

The first example uses bundled Iris data and needs no download. The second uses `data/usps.mat` and compares supervised, proposed unlabeled, and naive predicted-label calibration on one split. Its logistic classifier and small default sample sizes make it a demonstration, not a reproduction of the paper's neural-network tables.

## Calibrate your classifier

```python
from unsupervised_conformal import UnsupervisedConformalCalibrator

calibrator = UnsupervisedConformalCalibrator(
    alpha=0.1,
    loss_upper_bound=loss_bound,
    random_state=0,
).fit(
    x_train_for_kernel, y_train,
    x_calibration_for_kernel, classifier.predict_proba(x_calibration),
    classes=classifier.classes_,
)

mask = calibrator.predict_set(classifier.predict_proba(x_test))
label_sets = [calibrator.classes_[row] for row in mask]
```

All feature arrays have one example per row. `x_train_for_kernel` consists of labeled examples already used to train the classifier, or a subset of them. `x_calibration_for_kernel` is **unlabeled**. Its probability array has one column per class in `classes` order. Apply the same feature transform to training and calibration features, and use the same classifier for calibration and test probabilities.

`mask[i, j]` indicates whether class `classes[j]` belongs to example `i`'s prediction set. `predict_labels` returns the original class labels directly. The default adaptive score is randomized, and each prediction call draws fresh randomness; reuse one mask when you need the same sets. Set `score="probability"` for the deterministic score `1 - probability`.

`loss_upper_bound` is an optional estimated upper bound on expected cross-entropy. The examples estimate it with out-of-fold probabilities. The paper's stronger guarantee assumes that the true calibration labels satisfy this constraint; the code cannot verify that assumption. Omitting the bound also omits the constraint used in the submitted experiment.

The fitted object exposes `weights_`, `threshold_`, `naive_threshold_`, `bandwidth_`, and `solver_status_`. `bandwidths` accepts custom positive Gaussian scales. By default, Python uses the ten candidates in `matlab/find_quant.m`.

## Tests

```bash
python -m unittest discover -s python/tests -v
```

The optional MATLAB/Python comparison is described in [`validation/README.md`](../validation/README.md).
