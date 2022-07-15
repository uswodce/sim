#!/bin/bash
set -e

mkdir -p ../data
mkdir -p ../ray_results
mkdir -p ../result/imgreg//recent/imgs

# Or use gdown if the following downloading fails
wget -O ../data/data_div2k.npz https://drive.google.com/uc?id=1TtwlEDArhOMoH18aUyjIMSZ3WODFmUab
wget -O ../data/data_2d_text.npz https://drive.google.com/uc?id=1V-RQJcMuk9GD4JCUn70o7nwQE0hEzHoT

# Run single-tier stable invariant model
for dataset in natural text
do
  for imgid in {0..15}
  do
    TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgreg.py \
      --sfix \
      --tune \
      --dataset ${dataset} \
      --imgid ${imgid} \
      sim-single \
      --basis fcn
  done
done

# Run two-tier stable invariant model
for dataset in natural text
do
  for imgid in {0..15}
  do
    TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgreg.py \
      --sfix \
      --tune \
      --dataset ${dataset} \
      --imgid ${imgid} \
      sim-two \
      --basis fcn
  done
done

# Run RFF only stable invariant model
# Note that we implement it as a single-tier model where basis is RFF
for dataset in natural text
do
  for imgid in {0..15}
  do
    TUNE_MAX_PENDING_TRIALS_PG=1 CUDA_VISIBLE_DEVICES=0 python3 run_imgreg.py \
      --sfix \
      --tune \
      --dataset ${dataset} \
      --imgid ${imgid} \
      sim-single \
      --basis rff
  done
done
