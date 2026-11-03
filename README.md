# KAR-STGCN: Kinematic-Aware Spatio-Temporal Graph Convolutional Network for 3D Hand Skeletal Restoration and Gesture Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Benchmarks: 6 (N=130)](https://img.shields.io/badge/Benchmarks-6%20Datasets%20(N%3D130)-8e44ad.svg)](#summary-of-key-benchmark-results-6-datasets-n--130-subjects)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation, multi-seed subject-independent evaluation protocols across **6 public benchmarks ($N = 130$ independent subjects)**, and the **KAR-Bench Studio** 3D animation retargeting suite for the manuscript:

> **"Kinematic-Aware Spatio-Temporal Graph Convolutional Network with Visibility-Guarded L2 Bone Normalization for 3D Hand Skeletal Restoration and Gesture Recognition under Optical Tracking Dropouts"**
> **Author:** Siriporn Noiumkar
> *Department of Creative Media, Faculty of Informatics, Mahasarakham University, Kantarawichai District, Maha Sarakham 44150, Thailand* (`siriporn.p@msu.ac.th`)
> *Submitted to Multimedia Tools and Applications (Springer)*

---

## Core Scientific Contributions (Three Pillars)

1. **Visibility-Guarded Wrist-Centered L2 Palm-Bone Normalization (System A Offline & System B Causal Regimes):**
   - Unifies **6 heterogeneous 3D hand benchmarks** (**SHREC'17**, **FPHA**, **LMDHG**, **Orthogonal Dual-LMC2**, **Briareo Automotive LMC**, and **DexYCB Hand-Object Grasping**; $N = 130$ total independent subjects) into a canonical 21-joint hand graph ($|\mathcal{V}|=21, |\mathcal{E}|=20$).
   - Formulates both **System A (Offline Sequence-Median)** for post-capture animation cleaning and **System B (Causal Running-Median)** for zero-lookahead real-time streaming, eliminating camera-distance shortcuts and preventing division-by-zero explosions ($\sim 10^7\text{--}10^8\text{ mm}$) during reference-joint dropouts.
2. **Dilated KAR-STGCN Architecture & Complementary Module Synergy:**
   - Couples a two-stage dilated depthwise temporal bridge ($K_b=9, d \in \{1, 2\}$, receptive field = 25 frames) with anatomical graph message passing ($\hat{\mathbf{A}}$), bone-length consistency regularization ($\mathcal{L}_{\text{bone}}$), and dual-stream position-velocity encoding ($0.42\text{M}$ parameters, $0.18\text{ GFLOPs}$, $0.42\text{ ms}$ GPU forward pass, **$0.88\text{ ms}$ total causal end-to-end streaming latency**).
   - Isolates and proves theoretical synergy: $\mathcal{B}_{\text{temp}}$ serves as the primary driver under whole-hand infrared glare bursts (**Briareo**, $N=40$), whereas $\hat{\mathbf{A}} + \mathcal{L}_{\text{bone}}$ serves as the primary driver under prolonged solid-object finger occlusions (**DexYCB**, $N=10$).
3. **KAR-Bench Studio & 3D Character Animation Retargeting Suite:**
   - Provides real-time palm-bone oscilloscopy, per-finger biomechanical error breakdown, TorchScript deployment ($1.72\text{ MB}$), and automated Biovision Hierarchy (`.bvh`) export compatible with **Blender**, **Unity**, and **Unreal Engine**, reducing anatomical Joint-Angle Violation Rates (JAVR) from $31.40\%$ ($1\text{€}$ Filter) to **$1.12\%$**.

---

## Clean Repository Structure

```text
KAR-STGCN-3D-Hand-Restoration/
|-- main_mtap_manuscript.tex             # Complete LaTeX manuscript (9 Sections, 9 Tables, 3 Vector Figures)
|-- run_lmc2_pipeline.py                 # Canonical 21-joint alignment & 6-type optical dropout simulator
|-- run_lmc2_benchmark_training.py       # Subject-independent training, baselines & statistical tests
|-- run_sim_to_real_lmdhg.py             # Cross-dataset transfer onto real Leap Motion streams (LMDHG, N=21)
|-- run_dual_lmc2_real_benchmark.py      # Orthogonal Dual-LMC2 hardware cross-view GT evaluation (N=25)
|-- run_cross_domain_briareo_dexycb.py   # Theoretical synergy on Briareo (N=40) & DexYCB (N=10)
|-- kar_bench_studio.py                  # KAR-Bench Studio v2.0 dashboard, 1-Euro Filter & .BVH exporter
|-- LMC2_Benchmark_Dataset/
|   |-- 02_preprocessed_l2norm/          # Canonical 21-joint L2-normalized tensors (.npz)
|   |-- 03_lmc2_degraded/                # Degraded levels & real sensor benchmarks (.npz)
|   |-- 04_splits_metadata/              # Quantitative CSV logs across all 6 benchmarks (N=130 subjects)
|   |-- 05_latex_outputs/                # Auto-generated booktabs LaTeX tables (.tex)
|   |-- 06_bvh_exports/                  # Standard 3D character animation files (.bvh) for Blender/Unity
|   `-- 07_q1_figures/                   # 300-DPI publication figures & KAR-Bench Studio dashboard (.pdf/.png)
|-- requirements.txt
|-- CITATION.cff
|-- LICENSE
`-- README.md
```

---

## Summary of Key Benchmark Results (6 Datasets, N = 130 Subjects)

| Benchmark & Scenario (Independent Subjects N, df) | Method / System | MPJPE_all (mm) | MPJPE_miss (mm) | BLCE (mm) | JAVR (%) | Macro-F1 (%) | End-to-End Lat. (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | 1-Euro Filter (VR Std.) | 26.10 +/- 1.28 | 44.20 +/- 2.50 | 18.10 +/- 0.92 | 31.40% | -- | **0.05** (0f buf) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | BlockGCN (CVPR 2024) | 12.35 +/- 0.49 | 18.60 +/- 0.88 | 5.80 +/- 0.25 | 11.65% | 86.90 +/- 1.05 | 100.94 (Offline) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | **Sys. B: KAR-STGCN (Causal)** | 9.18 +/- 0.35 | 12.35 +/- 0.53 | 4.38 +/- 0.18 | 1.85% | 92.60 +/- 0.78 | **0.88** (**0f buf**) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | **Sys. A: KAR-STGCN (Offline)** | **8.64 +/- 0.31** | **11.42 +/- 0.48** | **4.12 +/- 0.16** | **1.12%** | **93.15 +/- 0.74** | 100.88 (12f buf) |
| **1. SHREC'17 L4 Compound** (N=28, df=27) | **Sys. A: KAR-STGCN (Offline)** | **11.38 +/- 0.39** | **15.64 +/- 0.58** | **5.46 +/- 0.22** | **1.12%** | **91.84 +/- 0.85** | 100.90 (12f buf) |
| **2. FPHA L4 Compound** (N=6, df=5, 45-cls) | **Sys. A: KAR-STGCN (Offline)** | **16.42 +/- 0.52** | **19.88 +/- 0.71** | **7.85 +/- 0.34** | **1.45%** | **82.76 +/- 1.12** | 100.90 (12f buf) |
| **3. LMDHG Real LMC1** (N=21, df=20, 13-cls) | **Sys. A: KAR-STGCN (Fine-Tuned)** | -- | **9.62 +/- 0.41** | **4.45 +/- 0.22** | **1.28%** | **89.45 +/- 1.38** | 100.88 (12f buf) |
| **4. Dual-LMC2 Real GT** (N=25, df=24, 15-cls) | **Sys. B: KAR-STGCN (Causal)** | 7.62 +/- 0.34 | 10.85 +/- 0.49 | 3.82 +/- 0.17 | 1.64% | 92.85 +/- 0.78 | **0.88** (**0f buf**) |
| **4. Dual-LMC2 Real GT** (N=25, df=24, 15-cls) | **Sys. A: KAR-STGCN (Offline)** | **7.15 +/- 0.31** | **10.12 +/- 0.45** | **3.54 +/- 0.15** | **1.05%** | **93.60 +/- 0.72** | 100.88 (12f buf) |
| **5. Briareo Auto LMC** (N=40, df=39, 12-cls) | **Sys. A: KAR-STGCN (Offline)** | -- | **8.84 +/- 0.39** | **3.68 +/- 0.16** | **1.08%** | **92.90 +/- 0.74** | 100.88 (12f buf) |
| **6. DexYCB Object Grasp** (N=10, df=9, 20-cls) | **Sys. A: KAR-STGCN (Offline)** | -- | **11.15 +/- 0.49** | **4.18 +/- 0.19** | **1.34%** | **89.35 +/- 0.96** | 100.88 (12f buf) |

---

## Ethical Approval & Funding Acknowledgment
- **Ethical Compliance:** This study utilizes exclusively publicly available, fully anonymized 3D skeletal benchmark datasets (**SHREC'17**, **FPHA**, **LMDHG**, **Dual-LMC2**, **Briareo**, and **DexYCB**). No new human subjects were recruited by the author.
- **Funding:** This research project was financially supported by the **Faculty of Informatics, Mahasarakham University**.
