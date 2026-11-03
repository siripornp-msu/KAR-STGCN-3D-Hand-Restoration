# KAR-STGCN: Kinematic-Aware Spatio-Temporal Graph Convolutional Network for 3D Hand Skeletal Restoration and Gesture Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Benchmarks: 4 Core](https://img.shields.io/badge/Benchmarks-SHREC'17%20%7C%20FPHA%20%7C%20LMDHG%20%7C%20Dual--LMC2-8e44ad.svg)](#summary-of-subject-independent-empirical-results-4-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation, subject-independent cross-validation protocols across **4 complementary 3D hand benchmarks**, canonical 21-joint preprocessed tensors (`.npz`), per-subject evaluation logs (`.csv`), and the **KAR-Bench Studio** 3D animation retargeting suite for the manuscript:

> **"Kinematic-Aware Spatio-Temporal Graph Convolutional Network with Visibility-Guarded L2 Bone Normalization for 3D Hand Skeletal Restoration and Gesture Recognition under Optical Tracking Dropouts"**
> **Author:** Siriporn Noiumkar
> *Department of Creative Media, Faculty of Informatics, Mahasarakham University, Kantarawichai District, Maha Sarakham 44150, Thailand* (`siriporn.p@msu.ac.th`)
> *Submitted to Multimedia Tools and Applications (Springer)*

---

## Core Scientific Contributions (Three Pillars)

1. **Visibility-Guarded Wrist-Centered L2 Palm-Bone Normalization (System A Offline & System B Causal Sliding-Window Regimes):**
   - Aligns **4 heterogeneous 3D hand benchmarks** (**SHREC'17** $N=28$, **FPHA** $N=6$, **LMDHG** $N=21$, and **Orthogonal Dual-LMC2 ML2HP** $N=21$, 17 hand pose classes) into a canonical 21-joint hand graph ($|\mathcal{V}|=21, |\mathcal{E}|=20$).
   - Formulates both **System A (Offline Sequence-Median)** for segmented post-capture animation cleaning and **System B (Causal Sliding-Window Running-Median)** operating on a fixed-length FIFO sliding window ($W=60$ raw frames) at native sensor frame rates without sequence-level resampling, preventing division-by-zero explosions ($\sim 10^7\text{--}10^8\text{ mm}$) and eliminating scale shortcuts.
2. **Dilated KAR-STGCN Architecture:**
   - Couples a two-stage dilated depthwise temporal bridge ($K_b=9, d \in \{1, 2\}$, receptive field = 25 frames) with anatomical graph message passing ($\hat{\mathbf{A}}$), bone-length consistency regularization ($\mathcal{L}_{\text{bone}}$), and dual-stream position-velocity encoding ($0.42\text{M}$ parameters, $0.18\text{ GFLOPs}$, **$0.48\text{ ms}$ GPU forward pass**, **$0.93\text{ ms}$ causal sliding-window latency** with $0$ lookahead frames).
3. **KAR-Bench Studio & 3D Character Animation Retargeting Suite:**
   - Provides real-time palm-bone oscilloscopy, per-finger biomechanical error breakdown, TorchScript deployment ($1.72\text{ MB}$), and automated Biovision Hierarchy (`.bvh`) export compatible with **Blender**, **Unity**, and **Unreal Engine**, reducing Bone-Length Consistency Error (BLCE) to **$2.04\text{--}3.85\text{ mm}$** and biomechanical Joint-Angle Violation Rates (JAVR) to **$1.15\%\text{--}1.68\%$**.

---

## Summary of Subject-Independent Empirical Results (4 Benchmarks)

| Benchmark & Scenario (Independent Subjects N, df) | Method / System | MPJPE_all (mm) | MPJPE_miss (mm) | PA-MPJPE (mm) | BLCE (mm) | JAVR (%) | Macro-F1 (%) | End-to-End Lat. (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | 1-Euro Filter (VR Std.) | 14.62 +/- 0.74 | 42.18 +/- 2.15 | 13.84 +/- 0.68 | 9.82 +/- 0.49 | 28.40% | -- | **0.05** (0f buf) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | CTR-GCN (ICCV 2021) | 7.12 +/- 0.35 | 19.08 +/- 0.89 | 7.56 +/- 0.37 | 4.38 +/- 0.21 | 10.40% | 84.10% | 100.97 (Offline) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | BlockGCN (CVPR 2024) | 6.54 +/- 0.32 | 17.64 +/- 0.82 | 6.92 +/- 0.34 | 4.02 +/- 0.19 | 8.95% | 85.60% | 100.94 (Offline) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | SAITS + CTR-GCN (2-Stage) | 5.68 +/- 0.28 | 14.12 +/- 0.66 | 5.45 +/- 0.27 | 3.45 +/- 0.17 | 5.10% | 88.45% | 251.36 (Offline) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | **Sys. B: KAR-STGCN (Causal)** | 4.52 +/- 0.23 | 10.28 +/- 0.48 | 4.41 +/- 0.22 | 2.34 +/- 0.12 | 1.52% | 92.85% | **0.93** (**0f buf**) |
| **1. SHREC'17 L3 Burst** (N=28, df=27) | **Sys. A: KAR-STGCN (Offline)** | **4.18 +/- 0.21** | **9.42 +/- 0.44 (-50.6%)** | **4.08 +/- 0.20** | **2.18 +/- 0.11** | **1.08%** | **93.60%** | 100.93 (12f buf) |
| **1. SHREC'17 L4 Compound** (N=28, df=27) | CTR-GCN (ICCV 2021) | 10.38 +/- 0.51 | 23.58 +/- 1.12 | 10.08 +/- 0.49 | 6.72 +/- 0.33 | 11.40% | 81.40% | 100.97 (Offline) |
| **1. SHREC'17 L4 Compound** (N=28, df=27) | SAITS + CTR-GCN (2-Stage) | 7.62 +/- 0.37 | 16.85 +/- 0.79 | 7.41 +/- 0.36 | 4.85 +/- 0.24 | 5.60% | 86.90% | 251.36 (Offline) |
| **1. SHREC'17 L4 Compound** (N=28, df=27) | **Sys. B: KAR-STGCN (Causal)** | 5.88 +/- 0.29 | 12.54 +/- 0.58 | 5.72 +/- 0.28 | 2.89 +/- 0.15 | 1.78% | 91.82% | **0.93** (**0f buf**) |
| **1. SHREC'17 L4 Compound** (N=28, df=27) | **Sys. A: KAR-STGCN (Offline)** | **5.42 +/- 0.27** | **11.68 +/- 0.54 (-50.5%)** | **5.28 +/- 0.26** | **2.74 +/- 0.14** | **1.15%** | **92.64%** | 100.93 (12f buf) |
| **2. FPHA L4 Compound** (N=6, df=5, 45-cls) | CTR-GCN (ICCV 2021) | 14.92 +/- 0.74 | 29.24 +/- 1.38 | 14.20 +/- 0.70 | 8.65 +/- 0.42 | 12.80% | 70.85% | 100.97 (Offline) |
| **2. FPHA L4 Compound** (N=6, df=5, 45-cls) | **Sys. B: KAR-STGCN (Causal)** | 9.54 +/- 0.45 | 16.92 +/- 0.74 | 9.21 +/- 0.43 | 4.12 +/- 0.21 | 2.15% | 81.40% | **0.93** (**0f buf**) |
| **2. FPHA L4 Compound** (N=6, df=5, 45-cls) | **Sys. A: KAR-STGCN (Offline)** | **8.92 +/- 0.41** | **15.84 +/- 0.68 (-45.8%)** | **8.58 +/- 0.39** | **3.85 +/- 0.19** | **1.68%** | **82.35%** | 100.93 (12f buf) |
| **3. LMDHG Real LMC1** (N=21, df=20, 13-cls) | CTR-GCN (Fine-Tuned) | 6.95 +/- 0.34 | 17.62 +/- 0.82 | 6.72 +/- 0.33 | 4.15 +/- 0.20 | 10.15% | 80.65% | 100.97 (Offline) |
| **3. LMDHG Real LMC1** (N=21, df=20, 13-cls) | **Sys. A: KAR-STGCN (Fine-Tuned)** | **4.35 +/- 0.21** | **9.86 +/- 0.45 (-44.0%)** | **4.22 +/- 0.20** | **2.28 +/- 0.12** | **1.24%** | **89.12%** | 100.93 (12f buf) |
| **4. Dual-LMC2 ML2HP** (N=21, df=20, 17-cls) | CTR-GCN (ICCV 2021) | 6.28 +/- 0.31 | 16.78 +/- 0.79 | 6.02 +/- 0.29 | 3.72 +/- 0.18 | 9.80% | 85.60% | 100.97 (Offline) |
| **4. Dual-LMC2 ML2HP** (N=21, df=20, 17-cls) | **Sys. B: KAR-STGCN (Causal)** | 3.98 +/- 0.19 | 9.92 +/- 0.46 | 3.82 +/- 0.18 | 2.12 +/- 0.11 | 1.62% | 93.10% | **0.93** (**0f buf**) |
| **4. Dual-LMC2 ML2HP** (N=21, df=20, 17-cls) | **Sys. A: KAR-STGCN (Offline)** | **3.68 +/- 0.18** | **9.18 +/- 0.42 (-45.3%)** | **3.54 +/- 0.17** | **2.04 +/- 0.10** | **1.15%** | **93.85%** | 100.93 (12f buf) |

---

## Public Benchmark Data Provenance & Official Download Links

1. **SHREC'17 Track (N=28, 14 & 28 classes):** [http://www-rech.telecom-lille.fr/shrec2017-hand/](http://www-rech.telecom-lille.fr/shrec2017-hand/)
2. **FPHA Benchmark (N=6, 45 action classes):** [https://github.com/guiggh/hand_pose_action](https://github.com/guiggh/hand_pose_action)
3. **LMDHG Real LMC1 (N=21, 13 classes):** [https://www-intuidoc.irisa.fr/en/english-leap-motion-dynamic-hand-gesture-lmdhg-database/](https://www-intuidoc.irisa.fr/en/english-leap-motion-dynamic-hand-gesture-lmdhg-database/)
4. **Orthogonal Dual-LMC2 (ML2HP, N=21, 17 hand pose classes, 714,000 instances):** Gil-Martin, M., Marini, M., San-Segundo, R., & Cinque, L. (2024). *Dual Leap Motion Controller 2: A robust dataset for multi-view hand pose recognition*. Scientific Data, 11, 1102. [https://doi.org/10.1038/s41597-024-03939-0](https://doi.org/10.1038/s41597-024-03939-0)

---

## Ethical Approval & Funding Acknowledgment
- **Ethical Compliance:** This study utilizes exclusively publicly available, fully anonymized 3D skeletal benchmark datasets (**SHREC'17**, **FPHA**, **LMDHG**, and **Dual-LMC2 ML2HP**). No new human subjects were recruited by the author.
- **Funding:** This research project was financially supported by the **Faculty of Informatics, Mahasarakham University**.
