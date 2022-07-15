#!/bin/bash
set -e

mkdir -p ../data
mkdir -p ../ray_results
mkdir -p ../result/cpmem/recent

# Run single-tier stable invariant model
TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_cpmem.py \
  --sfix \
  --tune \
  --T 500 \
  sim-single \
  --basis tcn

# Run two-tier stable invariant model
TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_cpmem.py \
  --sfix \
  --tune \
  --T 500 \
  sim-two \
  --basis tcn
