#!/bin/bash

python subject_level_cross_validation.py \
  --manifest_csv manifests/adni_all.csv \
  --data_root data \
  --atlas_path data/atlas/AAL90_MNI152.nii.gz \
  --output_dir outputs/subject_level_cross_validation \
  --num_folds 5 \
  --epochs 300 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --gpu_id 0
