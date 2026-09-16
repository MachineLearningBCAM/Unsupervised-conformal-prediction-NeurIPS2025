"""Compare three calibration strategies on the included USPS data.

Defaults are deliberately small for a quick demonstration. This is not a
reproduction of the paper's MATLAB neural-network experiment.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unsupervised_conformal import UnsupervisedConformalCalibrator, weighted_quantile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-calibration", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = loadmat(Path(__file__).resolve().parents[2] / "data" / "usps.mat")
    x = data["x"].T
    y = data["y"].ravel()
    n_total = args.n_train + args.n_calibration + args.n_test
    if min(args.n_train, args.n_calibration, args.n_test) < 1 or n_total > len(y):
        parser.error("sample counts must be positive and fit in the USPS dataset")
    if args.n_train < 50:
        parser.error("use at least 50 training examples for five-fold cross-validation")
    order = np.random.default_rng(args.seed).permutation(len(y))[:n_total]
    train_ids = order[: args.n_train]
    cal_ids = order[args.n_train : args.n_train + args.n_calibration]
    test_ids = order[args.n_train + args.n_calibration :]
    x_train, y_train = x[train_ids], y[train_ids]
    x_cal, y_cal = x[cal_ids], y[cal_ids]  # y_cal is used only for the supervised comparison.
    x_test, y_test = x[test_ids], y[test_ids]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    out_of_fold = cross_val_predict(model, x_train, y_train, cv=5, method="predict_proba")
    classes = np.unique(y_train)
    train_columns = np.searchsorted(classes, y_train)
    losses = -np.log(np.maximum(out_of_fold[np.arange(len(y_train)), train_columns], 1e-40))
    bound = float(losses.mean() + losses.std(ddof=1) / np.sqrt(len(losses)))
    model.fit(x_train, y_train)
    cal_probs = model.predict_proba(x_cal)
    test_probs = model.predict_proba(x_test)
    scaler = model.named_steps["standardscaler"]

    calibrator = UnsupervisedConformalCalibrator(
        alpha=0.1, score="probability", loss_upper_bound=bound, random_state=args.seed
    ).fit(
        scaler.transform(x_train), y_train, scaler.transform(x_cal),
        cal_probs, classes=model.classes_,
    )
    unsupervised_sets = calibrator.predict_set(test_probs)

    n = len(x_cal)
    level = 0.9 * (1 + 1 / n)
    cal_columns = np.searchsorted(model.classes_, y_cal)
    q_supervised = weighted_quantile(
        1 - cal_probs[np.arange(n), cal_columns], np.ones(n), level
    )
    q_naive = weighted_quantile(1 - np.max(cal_probs, axis=1), np.ones(n), level)
    test_columns = np.searchsorted(model.classes_, y_test)
    print(f"USPS split: {len(x_train)} training, {n} calibration, {len(x_test)} test")
    print("Method                  Coverage   Mean set size")
    for name, sets in [
        ("Supervised reference", 1 - test_probs <= q_supervised),
        ("Unsupervised kernel", unsupervised_sets),
        ("Naive predicted labels", 1 - test_probs <= q_naive),
    ]:
        coverage = sets[np.arange(len(x_test)), test_columns].mean()
        print(f"{name:24s}{coverage:8.3f}{sets.sum(axis=1).mean():16.3f}")
    print("The supervised row uses calibration labels only as a reference.")


if __name__ == "__main__":
    main()
