#!/bin/bash

python train_amr_hnr_net.py \
  --train_csv manifests/adni_train.csv \
  --val_csv manifests/adni_val.csv \
  --data_root data \
  --atlas_path data/atlas/AAL90_MNI152.nii.gz \
  --output_dir outputs/amr_hnr_training \
  --input_size 96,96,96 \
  --num_classes 3 \
  --num_rois 90 \
  --class_counts 320,380,250 \
  --feature_channels 32 \
  --graph_hidden 128 \
  --graph_out 128 \
  --epochs 300 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --gamma 2.0 \
  --lambda_graph 0.01 \
  --eta_l2 0.00001 \
  --gpu_id 0 \
  --amp
