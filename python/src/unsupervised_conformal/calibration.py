"""Python implementation of Algorithms 2 and 3 in arXiv:2510.07185.

Inputs have rows for examples and columns for classes, unlike the MATLAB
experiment's feature-by-example arrays. The classifier itself is supplied by
the caller; this module only calibrates its probability estimates.
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty (examples, classes) array")
    if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    if not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError(f"each row of {name} must sum to 1")
    return result


def _features(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite (examples, features) array")
    return result


def adaptive_scores(probabilities: np.ndarray, random_state=None) -> np.ndarray:
    """Randomized adaptive scores for every possible class.

    Each example/class pair receives its own uniform draw, as in the MATLAB
    ``compute_score`` function. Lower scores indicate more plausible labels.
    """
    probs = _probabilities(probabilities, "probabilities")
    rng = np.random.default_rng(random_state)
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    sorted_scores = np.cumsum(sorted_probs, axis=1) - sorted_probs * rng.random(probs.shape)
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Smallest value whose cumulative nonnegative weight reaches ``level``.

    A level greater than one returns infinity, the usual conservative
    conformal convention for a calibration sample that is too small.
    """
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("values and weights must be nonempty arrays of equal size")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative and have positive sum")
    if not np.isfinite(level) or level < 0:
        raise ValueError("level must be finite and nonnegative")
    if level > 1:
        return float("inf")
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[order]
    if level == 0:
        return float(values[order[sorted_weights > 0][0]])
    cdf = np.cumsum(sorted_weights) / sorted_weights.sum()
    index = np.searchsorted(cdf, level - 1e-12, side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _rbf_from_squared_distances(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-distances / (2 * bandwidth**2))


class UnsupervisedConformalCalibrator:
    """Calibrate a trained classifier using unlabeled examples.

    Call ``fit`` with training features and labels, unlabeled calibration
    features, and the classifier's calibration probabilities. Call
    ``predict_set`` with probabilities from the same classifier. Features
    must use the same preprocessing in both groups.

    ``loss_upper_bound`` is an externally estimated upper bound on expected
    cross-entropy. It is optional, but the paper's stronger result assumes
    that the feasible set contains the true calibration labels. A bound
    estimated on data used to fit the classifier may be optimistic.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        score: str = "adaptive",
        bandwidths: np.ndarray | None = None,
        loss_upper_bound: float | None = None,
        random_state: int | None = 0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if score not in {"adaptive", "probability"}:
            raise ValueError("score must be 'adaptive' or 'probability'")
        if loss_upper_bound is not None and (
            not np.isfinite(loss_upper_bound) or loss_upper_bound < 0
        ):
            raise ValueError("loss_upper_bound must be finite and nonnegative")
        self.alpha = float(alpha)
        self.score = score
        self.bandwidths = bandwidths
        self.loss_upper_bound = loss_upper_bound
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _scores(self, probabilities: np.ndarray) -> np.ndarray:
        if self.score == "adaptive":
            return adaptive_scores(probabilities, self._rng)
        return 1 - probabilities

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calibration: np.ndarray,
        calibration_probabilities: np.ndarray,
        *,
        classes: np.ndarray | None = None,
    ) -> "UnsupervisedConformalCalibrator":
        x_train = _features(x_train, "x_train")
        x_calibration = _features(x_calibration, "x_calibration")
        probabilities = _probabilities(calibration_probabilities, "calibration_probabilities")
        n, n_classes = probabilities.shape
        m = x_train.shape[0]
        if x_train.shape[1] != x_calibration.shape[1] or x_calibration.shape[0] != n:
            raise ValueError("feature dimensions and calibration row counts must match")
        if n_classes < 2:
            raise ValueError("at least two classes are required")
        labels = np.asarray(y_train).ravel()
        if len(labels) != m:
            raise ValueError("y_train must have one label per training row")
        classes = np.arange(n_classes) if classes is None else np.asarray(classes).ravel()
        if len(classes) != n_classes or len(np.unique(classes)) != n_classes:
            raise ValueError("classes must contain one unique label per probability column")
        label_indices = np.searchsorted(np.sort(classes), labels)
        sorted_classes = np.sort(classes)
        if np.any(label_indices >= n_classes) or not np.array_equal(sorted_classes[label_indices], labels):
            raise ValueError("every training label must appear in classes")
        # Map training labels to the caller's probability-column order.
        label_indices = np.array([np.flatnonzero(classes == label)[0] for label in labels])
        if len(np.unique(label_indices)) != n_classes:
            raise ValueError("y_train must contain at least one example of every class")

        scores = self._scores(probabilities)
        predicted = np.argmax(probabilities, axis=1)
        level = (1 - self.alpha) * (1 + 1 / n)
        naive = weighted_quantile(scores[np.arange(n), predicted], np.ones(n), level)
        indicators = (scores <= naive).astype(float)

        d = x_train.shape[1]
        bandwidths = (
            np.sqrt(d / 2) * 10 ** np.linspace(-2, 1, 10)
            if self.bandwidths is None
            else np.asarray(self.bandwidths, dtype=float).ravel()
        )
        if bandwidths.size == 0 or not np.all(np.isfinite(bandwidths)) or np.any(bandwidths <= 0):
            raise ValueError("bandwidths must contain positive finite values")
        cal_distances = cdist(x_calibration, x_calibration, "sqeuclidean")
        train_cal_distances = cdist(x_train, x_calibration, "sqeuclidean")
        norms = []
        for bandwidth in bandwidths:
            kernel = _rbf_from_squared_distances(cal_distances, bandwidth)
            # MATLAB uses 1e-10 on the diagonal for kernel selection.
            kernel.flat[:: n + 1] += 1e-10
            try:
                factor = cho_factor(kernel, check_finite=False)
                coefficients = cho_solve(factor, indicators, check_finite=False)
                norms.append(float(np.sum(indicators * coefficients)))
            except np.linalg.LinAlgError:
                norms.append(float("inf"))
        if not np.isfinite(norms).any():
            raise RuntimeError("all candidate kernel systems were singular; try narrower bandwidths")
        chosen = int(np.argmin(norms))
        bandwidth = float(bandwidths[chosen])
        kernel = _rbf_from_squared_distances(cal_distances, bandwidth)
        kernel.flat[:: n + 1] += 1e-8
        cross_kernel = _rbf_from_squared_distances(train_cal_distances, bandwidth)
        v = np.stack([cross_kernel[label_indices == j].sum(axis=0) for j in range(n_classes)])

        # Class-major flattening matches kron(I_c, K) and MATLAB's scores(:).
        objective = sparse.kron(sparse.eye(n_classes), sparse.csc_matrix(2 * kernel / n), format="csc")
        linear = (-2 / m) * v.ravel()
        rows = [sparse.kron(sparse.csr_matrix(np.ones((1, n_classes))), sparse.eye(n), format="csc")]
        lower = [np.ones(n)]
        upper = [np.ones(n)]
        if self.loss_upper_bound is not None:
            losses = -np.log(np.maximum(probabilities, 1e-40))
            if np.mean(np.min(losses, axis=1)) > self.loss_upper_bound + 1e-9:
                raise ValueError("loss_upper_bound is infeasible even for the best class at each example")
            rows.append(sparse.csc_matrix(losses.T.reshape(1, -1) / n))
            lower.append(np.array([-np.inf]))
            upper.append(np.array([self.loss_upper_bound]))
        rows.append(sparse.eye(n * n_classes, format="csc"))
        lower.append(np.zeros(n * n_classes))
        upper.append(np.ones(n * n_classes))
        constraints = sparse.vstack(rows, format="csc")

        solver = osqp.OSQP()
        solver.setup(
            P=objective, q=linear, A=constraints,
            l=np.concatenate(lower), u=np.concatenate(upper),
            verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50_000,
            polishing=True,
        )
        solution = solver.solve(raise_error=False)
        if solution.info.status != "solved" or solution.x is None:
            raise RuntimeError(f"weight optimization failed: {solution.info.status}")
        weights = solution.x.reshape(n_classes, n).T
        if (
            np.min(weights) < -1e-4
            or np.max(np.abs(weights.sum(axis=1) - 1)) > 1e-4
            or (
                self.loss_upper_bound is not None
                and np.sum(weights * losses) / n > self.loss_upper_bound + 1e-4
            )
        ):
            raise RuntimeError("weight optimization returned an infeasible solution")
        weights = np.maximum(weights, 0)
        weights /= weights.sum(axis=1, keepdims=True)
        if self.loss_upper_bound is not None and np.sum(weights * losses) / n > self.loss_upper_bound + 1e-4:
            raise RuntimeError("normalizing weights violated the loss constraint")
        threshold = weighted_quantile(scores.T.ravel(), weights.T.ravel(), level)

        self.classes_ = classes.copy()
        self.weights_ = weights
        self.threshold_ = threshold
        self.naive_threshold_ = naive
        self.bandwidth_ = bandwidth
        self.bandwidth_norms_ = np.asarray(norms)
        self.solver_status_ = solution.info.status
        return self

    def predict_set(self, probabilities: np.ndarray) -> np.ndarray:
        """Return a boolean (examples, classes) prediction-set mask."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("call fit before predict_set")
        probabilities = _probabilities(probabilities, "probabilities")
        if probabilities.shape[1] != len(self.classes_):
            raise ValueError("probability columns must match the fitted classes")
        return self._scores(probabilities) <= self.threshold_

    def predict_labels(self, probabilities: np.ndarray) -> list[np.ndarray]:
        """Return prediction sets as arrays of original class labels."""
        return [self.classes_[row] for row in self.predict_set(probabilities)]
