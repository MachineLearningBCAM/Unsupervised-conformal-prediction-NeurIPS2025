import unittest

import numpy as np

from unsupervised_conformal import (
    UnsupervisedConformalCalibrator,
    adaptive_scores,
    weighted_quantile,
)


class CalibrationTests(unittest.TestCase):
    def test_weighted_quantile_and_tiny_sample_convention(self):
        self.assertEqual(weighted_quantile([3, 1, 2], [0.2, 0.6, 0.2], 0.6), 1)
        self.assertEqual(weighted_quantile([3, 1, 2], [0.2, 0.6, 0.2], 0.8), 2)
        self.assertEqual(weighted_quantile([3, 1], [1, 1], 1.1), float("inf"))

    def test_adaptive_scores_reproducible_and_ordered(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        scores = adaptive_scores(probs, random_state=5)
        np.testing.assert_array_equal(scores, adaptive_scores(probs, random_state=5))
        self.assertLess(scores[0, 0], scores[0, 1])
        self.assertLess(scores[1, 1], scores[1, 0])

    def test_fit_respects_simplex_and_loss_constraint(self):
        rng = np.random.default_rng(10)
        x_train = np.r_[rng.normal(-1, 0.4, (20, 2)), rng.normal(1, 0.4, (20, 2))]
        y_train = np.array(["left"] * 20 + ["right"] * 20)
        x_cal = np.r_[rng.normal(-1, 0.4, (12, 2)), rng.normal(1, 0.4, (12, 2))]
        probabilities = np.r_[np.tile([0.8, 0.2], (12, 1)), np.tile([0.2, 0.8], (12, 1))]
        calibrator = UnsupervisedConformalCalibrator(
            score="probability", bandwidths=[0.5, 1.0, 2.0],
            loss_upper_bound=0.3, random_state=4,
        ).fit(x_train, y_train, x_cal, probabilities, classes=["left", "right"])
        np.testing.assert_allclose(calibrator.weights_.sum(axis=1), 1, atol=1e-5)
        self.assertGreaterEqual(calibrator.weights_.min(), 0)
        self.assertLessEqual(np.sum(calibrator.weights_ * -np.log(probabilities)) / len(probabilities), 0.3001)
        self.assertEqual(calibrator.predict_set(probabilities[:3]).shape, (3, 2))
        self.assertEqual(calibrator.predict_labels(probabilities[:1])[0].dtype.kind, "U")

    def test_infeasible_loss_bound_is_reported(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        probs = np.tile([0.6, 0.4], (4, 1))
        with self.assertRaisesRegex(ValueError, "infeasible"):
            UnsupervisedConformalCalibrator(
                loss_upper_bound=0.1, bandwidths=[1.0]
            ).fit(x, [0, 1, 0, 1], x, probs)


if __name__ == "__main__":
    unittest.main()
