#!/bin/bash
set -e

mkdir -p ../data
mkdir -p ../ray_results
mkdir -p ../result/imgcl/recent

# Run single-tier stable invariant model
for dataset in mnist cifar svhn
do
  TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgcl.py \
    --sfix \
    --tune \
    --dataset ${dataset} \
    sim-single \
    --basis cnn
done

# Run two-tier stable invariant model
for dataset in mnist cifar svhn
do
  TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgcl.py \
    --sfix \
    --tune \
    --dataset ${dataset} \
    sim-two \
    --basis cnn
done
