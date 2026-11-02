# KAR-STGCN: Kinematic-Aware Spatio-Temporal Graph Convolutional Network for 3D Hand Skeletal Restoration and Gesture Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation, multi-seed evaluation protocols, and **KAR-Bench Studio** software suite for the manuscript:

> **"Kinematic-Aware Spatio-Temporal Graph Convolutional Network with Visibility-Guarded L2 Bone Normalization for 3D Hand Skeletal Restoration and Gesture Recognition under Optical Tracking Dropouts"**
> **Author:** Siriporn Noiumkar
> *Department of Creative Media, Faculty of Informatics, Mahasarakham University, Kantarawichai District, Maha Sarakham 44150, Thailand* (`siriporn.p@msu.ac.th`)
> *Submitted to Multimedia Tools and Applications (Springer)*

---

## Core Scientific Contributions

1. **Visibility-Guarded Wrist-Centered L2 Palm-Bone Normalization (Offline & Causal Regimes):**
   - Unifies heterogeneous 3D hand benchmarks (**SHREC'17**, **FPHA**, **LMDHG**, and **Dual-LMC2**) into a canonical 21-joint hand topology (21 joints, 20 bone connections).
   - Formulates both **System A (Offline Sequence-Median)** for post-capture animation cleaning and **System B (Causal Running-Median)** for zero-lookahead real-time streaming, eliminating camera-distance shortcuts and preventing division-by-zero explosions during reference-joint dropouts.
2. **Dilated KAR-STGCN Architecture:**
   - Combines a two-stage dilated depthwise temporal bridge (kernel size 9, dilation 1 and 2, receptive field = 25 frames) with anatomical graph convolutions, bone-length consistency regularization, and dual-stream position-velocity encoding (0.42M parameters, 0.18 GFLOPs, 0.42 ms GPU forward pass, 0.88 ms total causal end-to-end streaming latency).
3. **KAR-Bench Studio & 3D Character Animation Retargeting Suite:**
   - Provides real-time palm-bone oscilloscopy, per-finger biomechanical error analysis, TorchScript deployment, and automated Biovision Hierarchy (`.bvh`) export compatible with **Blender**, **Unity**, and **Unreal Engine**, reducing anatomical Joint-Angle Violation Rates (JAVR) from 31.40% (1-Euro Filter) to **1.12%**.

---

## Repository Structure

```text
KAR-STGCN-3D-Hand-Restoration/
|-- run_lmc2_pipeline.py                 # Canonical 21-joint alignment & 6-type optical dropout simulator
|-- run_lmc2_benchmark_training.py       # Subject-independent training, baselines & statistical tests
|-- run_sim_to_real_lmdhg.py             # Cross-dataset transfer onto real Leap Motion streams (LMDHG)
|-- run_dual_lmc2_real_benchmark.py      # Orthogonal Dual-LMC2 hardware cross-view ground-truth evaluation
|-- kar_bench_studio.py                  # KAR-Bench Studio v2.0 dashboard, 1-Euro Filter & .BVH exporter
|-- build_overleaf_project.py            # Automated LaTeX table & 300-DPI figure builder
|-- LMC2_Benchmark_Dataset/
|   |-- 02_preprocessed_l2norm/          # Canonical 21-joint L2-normalized tensors (.npz)
|   |-- 03_lmc2_degraded/                # Degraded levels & real sensor benchmarks (.npz)
|   |-- 04_splits_metadata/              # Quantitative CSV logs (Restoration, Sim2Real, Dual-LMC2, Finger JAVR)
|   |-- 05_latex_outputs/                # Auto-generated booktabs LaTeX tables (.tex)
|   |-- 06_bvh_exports/                  # Standard 3D character animation files (.bvh) for Blender/Unity
|   `-- 07_q1_figures/                   # 300-DPI publication figures & KAR-Bench Studio dashboard (.pdf/.png)
|-- requirements.txt
|-- CITATION.cff
|-- LICENSE
`-- README.md
```

---

## Reproducibility & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main Benchmark & Subject-Level Statistical Evaluation
```bash
python run_lmc2_pipeline.py --skip-download
python run_lmc2_benchmark_training.py --epochs 25
```

### 3. Run Real Leap Motion Transfer (LMDHG) & Orthogonal Dual-LMC2 Hardware Evaluation
```bash
python run_sim_to_real_lmdhg.py
python run_dual_lmc2_real_benchmark.py
```

### 4. Launch KAR-Bench Studio & Export .BVH 3D Character Animations
```bash
python kar_bench_studio.py
```

---

## Summary of Key Benchmark Results

| Dataset & Scenario (Unit of Analysis N) | Method / System | MPJPE_all (mm) | MPJPE_miss (mm) | BLCE (mm) | JAVR (%) | Macro-F1 (%) | End-to-End Lat. (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SHREC'17 L3 Burst** (N=28) | 1-Euro Filter (VR Std.) | 26.10 +/- 1.28 | 44.20 +/- 2.50 | 18.10 +/- 0.92 | 31.40% | -- | **0.05** (0f buf) |
| **SHREC'17 L3 Burst** (N=28) | BlockGCN (CVPR 2024) | 12.35 +/- 0.49 | 18.60 +/- 0.88 | 5.80 +/- 0.25 | 11.65% | 86.90 +/- 1.05 | 100.94 (Offline) |
| **SHREC'17 L3 Burst** (N=28) | SAITS + CTR-GCN | 10.45 +/- 0.42 | 14.85 +/- 0.68 | 5.62 +/- 0.24 | 7.80% | 89.50 +/- 0.92 | 251.32 (Offline) |
| **SHREC'17 L3 Burst** (N=28) | **Sys. B: KAR-STGCN (Causal)** | 9.18 +/- 0.35 | 12.35 +/- 0.53 | 4.38 +/- 0.18 | 1.85% | 92.60 +/- 0.78 | **0.88** (**0f buf**) |
| **SHREC'17 L3 Burst** (N=28) | **Sys. A: KAR-STGCN (Offline)** | **8.64 +/- 0.31** | **11.42 +/- 0.48** | **4.12 +/- 0.16** | **1.12%** | **93.15 +/- 0.74** | 100.88 (12f buf) |
| **SHREC'17 L4 Compound** (N=28) | **Sys. A: KAR-STGCN (Offline)** | **11.38 +/- 0.39** | **15.64 +/- 0.58** | **5.46 +/- 0.22** | **1.12%** | **91.84 +/- 0.85** | 100.90 (12f buf) |
| **FPHA L4 Compound** (N=6, 45-cls) | **Sys. A: KAR-STGCN (Offline)** | **16.42 +/- 0.52** | **19.88 +/- 0.71** | **7.85 +/- 0.34** | **1.45%** | **82.76 +/- 1.12** | 100.90 (12f buf) |
| **LMDHG Real LMC1** (N=21, 13-cls) | **Sys. A: KAR-STGCN (Fine-Tuned)** | -- | **9.62 +/- 0.41** | **4.45 +/- 0.22** | **1.28%** | **89.45 +/- 1.38** | 100.88 (12f buf) |
| **Dual-LMC2 Real GT** (N=25, 15-cls) | **Sys. B: KAR-STGCN (Causal)** | 7.62 +/- 0.34 | 10.85 +/- 0.49 | 3.82 +/- 0.17 | 1.64% | 92.85 +/- 0.78 | **0.88** (**0f buf**) |
| **Dual-LMC2 Real GT** (N=25, 15-cls) | **Sys. A: KAR-STGCN (Offline)** | **7.15 +/- 0.31** | **10.12 +/- 0.45** | **3.54 +/- 0.15** | **1.05%** | **93.60 +/- 0.72** | 100.88 (12f buf) |

---

## Funding Acknowledgment
This research project was financially supported by the **Faculty of Informatics, Mahasarakham University**.
