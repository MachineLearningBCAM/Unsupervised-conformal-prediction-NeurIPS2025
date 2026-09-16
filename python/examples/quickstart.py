"""Run a small, download-free example: python examples/quickstart.py."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unsupervised_conformal import UnsupervisedConformalCalibrator


def main() -> None:
    x, y = load_iris(return_X_y=True)
    x_train, x_other, y_train, y_other = train_test_split(
        x, y, train_size=90, stratify=y, random_state=7
    )
    x_cal, x_test, _, y_test = train_test_split(
        x_other, y_other, train_size=30, stratify=y_other, random_state=8
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # Cross-validated probabilities give a useful, though not guaranteed,
    # estimate of the cross-entropy constraint used in the paper.
    out_of_fold = cross_val_predict(model, x_train, y_train, cv=5, method="predict_proba")
    losses = -np.log(np.maximum(out_of_fold[np.arange(len(y_train)), y_train], 1e-40))
    loss_upper_bound = float(losses.mean() + losses.std(ddof=1) / np.sqrt(len(losses)))

    model.fit(x_train, y_train)
    scaler = model.named_steps["standardscaler"]
    calibrator = UnsupervisedConformalCalibrator(
        alpha=0.1, loss_upper_bound=loss_upper_bound, random_state=7
    ).fit(
        scaler.transform(x_train), y_train, scaler.transform(x_cal),
        model.predict_proba(x_cal), classes=model.classes_,
    )
    sets = calibrator.predict_set(model.predict_proba(x_test))
    print(f"Chosen Gaussian bandwidth: {calibrator.bandwidth_:.3g}")
    print(f"Conformal threshold: {calibrator.threshold_:.3f}")
    print(f"Held-out coverage: {sets[np.arange(len(y_test)), y_test].mean():.3f}")
    print(f"Mean set size: {sets.sum(axis=1).mean():.3f}")
    print("First prediction set:", calibrator.classes_[sets[0]].tolist())


if __name__ == "__main__":
    main()
