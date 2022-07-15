# Stable Invariant Models via Koopman Spectra

This repository contains source code for Stable Invariant Models via Koopman Spectra.

## Dependencies

We tested the implementation with the following versions.

- Python 3.8.8
- Numpy 1.19.5
- Pytorch 1.8.1
- Ray 1.3.0
- Imageio 2.9.0

## Usage

The code is located in `src`. For example, `run_imgcl.py` runs the single-tier stable invariant model in the image classification task:

    python3 run_imgcl.py --sfix --tune --dataset mnist sim-single --basis cnn

- `--sfix` option sets a specific seed to control randomness.
- `--tune` option runs the code together with hyperparameter tuning via Ray Tune.
- `--dataset` option specifies an evaluated dataset.
- `sim-single` is replaced with `sim-two` if you want to run the two-tier stable invariant model.
- `--basis` option determines the basis function (for now, the only convolutional network (`cnn`) is implemented for the image classification task).

If you want to reduce the size of log files for Ray Tune, set `TUNE_MAX_PENDING_TRIALS_PG` to 1. Ray Tune also uses multiple GPUs by default. Setting `CUDA_VISIBLE_DEVICES` can reduce available GPUs. For example,

    TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgcl.py --sfix --tune --dataset mnist sim-single --basis cnn

Please see `run_imgcl.py` for more details. The following scripts in `src` describe running examples involving the other two tasks in the paper:

- `cpmem.sh`: copy memory task
- `imgcl.sh`: image classification task
- `imgreg.sh`: image regression task
