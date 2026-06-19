
# E2AD: Enhanced and Explainable Alzheimer’s Disease Detection Framework

[![Paper](https://img.shields.io/badge/Paper-Medical%20Image%20Analysis-blue)](#) <!-- 替换为实际的论文链接 -->

This is the official code implementation of **E<sup>2</sup>AD** proposed in the manuscript: "**E<sup>2</sup>AD: Enhanced and Explainable Alzheimer’s Disease Detection Framework via Anatomy- and Relation-aware Cross-modal Knowledge Distillation**".

## 📖 Overview

We introduce **E<sup>2</sup>AD**, an Enhanced and Explainable Alzheimer's disease detection framework that trains on paired MRI-PET data but requires **only MRI at inference**. To improve upon traditional knowledge distillation, E<sup>2</sup>AD transfers multimodal knowledge via:

* **Anatomy-aware Distillation:** Captures and transfers within-subject anatomical dependencies.
* **Relation-aware Distillation:** Aligns between-subject structural relations for better cross-cohort generalization.

Additionally, E<sup>2</sup>AD includes a tailored multi-agent workflow as an add-on to translate the model's anatomical attention into structured, clinician-oriented diagnostic reports.

<div align="center">
 <img src="./readme_files/img.jpg" alt="overall_framework" width="1000">
</div>

## 🚀 Requirements

The framework has been tested on a Linux platform. Please ensure your environment meets the following dependencies (CUDA 11.3 is used as an example):

```bash
# Core dependencies
Python == 3.8.0
torch == 1.12.1+cu113 
torchvision == 0.13.1+cu113
numpy == 1.22.3
```

## 📁 Data Preparation

Extensive experiments in our paper were conducted using one internal cohort and two external cohorts. Please ensure your data is properly preprocessed and organized before training:

* **ADNI** (Alzheimer's Disease Neuroimaging Initiative)
* **NACC** (National Alzheimer's Coordinating Center)
* **AIBL** (Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing)

Please refer to our **[Unified 3D Cross-Modality Synthesis Codebase](https://github.com/thibault-wch/A-Unified-3D-Cross-Modality-Synthesis-Codebase)** for:  **[Multi-thread preprocessing](https://github.com/thibault-wch/A-Unified-3D-Cross-Modality-Synthesis-Codebase/tree/main/preprocess)** codes for 3D MRI and PET brain images. 

## ⚙️ Training & Inference Pipeline

Our training pipeline is structurally organized into three main stages. Execute the corresponding shell scripts to reproduce the framework:

**Step 1: Single-modal Pre-training (Optional)**
Train the baseline MRI model on all available MRI subjects. This step provides a solid initialization for the subsequent stages.

```bash
    bash single_train.sh

```

**Step 2: Multi-modal Teacher Training**
Train the paired MRI-PET teacher model exclusively on subjects that have both MRI and PET scans available.

```bash
    bash multi_train.sh

```

**Step 3: Cross-modal Knowledge Distillation**
Execute the core KD process to transfer rich anatomy- and relation-aware knowledge from the multi-modal teacher to the MRI-only student model.

```bash
    bash distill_train.sh

```

> **📌 Inference Note:** Once the distillation training (Step 3) is complete, the resulting student model requires **only MRI** inputs for AD detection.

## 📝 Citation

If you find this code or our paper useful for your research, please star 🌟 this repository and cite our work:

```bibtex
@article{wang2026e2ad,
  title={E2AD: Enhanced and explainable {Alzheimer's} disease detection framework via anatomy-and relation-aware cross-modal knowledge distillation},
  author={Wang, Chenhui and Piao, Sirong and Chen, Zhihao and Chen, Tao and Li, Zhaoyang and Zhang, Tongrui and Li, Yuxin and Zhao, Xing-Ming and Shan, Hongming},
  journal={Medical Image Analysis},
  pages={104099},
  year={2026},
  publisher={Elsevier}
}

```
