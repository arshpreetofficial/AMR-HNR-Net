
# Adaptive Multi-Resolution Neuroanatomical Representation Learning through biological topology-guided Hierarchical Neuroanatomical Reasoning for Brain Abnormality Analysis


## Overview

**AMR-HNR-Net** is a biologically-inspired, decoupled deep learning framework designed for scale-invariant and topologically aware analysis of brain abnormalities in structural MRI (sMRI). Traditional feed-forward architectures often dilute critical high-frequency neuroanatomical micro-textures and fail to model the non-Euclidean spread of neurodegeneration along biological networks. 

To resolve these structural bottlenecks, our framework explicitly decouples visual representation learning from topological disease inference using a two-stage approach:
1. **Adaptive Multi-Resolution Learning (MRL) Module:** Synthesizes low-resolution (macro-structural) and high-resolution (micro-textural) features using a parameter-efficient **Cross-Resolution Attention Fusion (CRAF)** mechanism to achieve scale invariance.
2. **Hierarchical Neuroanatomical Reasoning (HNR) Module:** Projects image features out of Euclidean voxel spaces into a discrete biological network via slice-level attention, atlas-guided region aggregation (e.g., AAL atlas), and **Graph Convolutional Networks (GCNs)** to track multi-scale brain atrophy propagation.

### Key Features
* **Cross-Cohort Generalization:** Robust performance against scanner-induced domain shifts (1.5T vs. 3T) across multiple clinical datasets (ADNI, OASIS-3, AIBL).
* **Intrinsic Clinical Interpretability:** Layer-by-layer transparency mapping computational attention weights directly to established neuroanatomical markers (e.g., Hippocampus, Amygdala).
* **Memory-Efficient Architecture:** Optimized attention and tensor operations designed to work natively with 3D MRI grids without feature exhaustion.

## Pipeline Architecture

<img width="6144" height="4096" alt="archi_new" src="https://github.com/user-attachments/assets/0c37dd85-420b-4914-8a97-2b2f66d3e81a" />

## Requirements

The framework has been tested on a Linux platform. Please ensure your environment meets the following dependencies (CUDA 11.3 is used as an example):

```bash
# Core dependencies
Python == 3.8.0
torch == 1.12.1+cu113 
torchvision == 0.13.1+cu113
numpy == 1.22.3
```

## Data Preparation

Extensive experiments in our paper were conducted using one internal cohort and two external cohorts. Please ensure your data is properly preprocessed and organized before training:

* **ADNI** (Alzheimer's Disease Neuroimaging Initiative)
* **OASIS** (Open Access Series of Imaging Studies )
* **AIBL** (Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing)





