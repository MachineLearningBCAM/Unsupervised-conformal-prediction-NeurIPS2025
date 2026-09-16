# MATLAB implementation

This directory contains the original USPS experiment and the MATLAB functions for the [unsupervised calibration method](https://arxiv.org/abs/2510.07185). The experiment uses MATLAB's neural network classifier and compares supervised, proposed unlabeled, and naive predicted-label calibration.

## Run the USPS experiment

In MATLAB, change to this directory and run:

```matlab
main
```

`main.m` loads `../data/usps.mat` relative to its own file location. It uses 3,000 training, 1,000 test, and 1,000 unlabeled calibration examples and reports `ave_cove_*` and `ave_size_*` variables for one random partition. The split and adaptive scores are randomized, so results vary across runs. Set `rng(0)` before `main` if you want a repeatable MATLAB run.

The default `mosek=0` branch uses `quadprog` from Optimization Toolbox. The classifier uses `fitcnet` from Statistics and Machine Learning Toolbox. Set `mosek=1` only if CVX and MOSEK are installed. The `quadprog` objective has been rescaled by the calibration size, matching the mathematically equivalent CVX objective, with tighter optimization tolerances. This change improves agreement with the Python optimizer.

## Function map

| File | Purpose |
| --- | --- |
| `main.m` | USPS train/calibration/test experiment |
| `find_quant.m` | Select kernel, optimize label weights, and return conformal quantile |
| `select_sigma.m` | Select Gaussian bandwidth via interpolation norm |
| `find_p.m` | Solve the constrained quadratic program |
| `weighted_quantile.m` | Weighted empirical quantile |
| `compute_score.m` | Randomized adaptive score; optional third argument supplies a fixed uniform draw for validation |

MATLAB uses feature-by-example arrays (`d × n`) and class labels `1,…,c`. The Python package uses example-by-feature arrays (`n × d`) and accepts arbitrary class labels.

The `figures/` scripts in the local submission depend on results files that are not present in this repository. They are not needed to run `main.m` or the Python examples.
