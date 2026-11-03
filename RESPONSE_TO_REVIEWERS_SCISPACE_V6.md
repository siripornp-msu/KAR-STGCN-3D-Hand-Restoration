# Point-by-Point Response to Reviewers (SciSpace Final Review V6)

**Manuscript Title:** Kinematic-Aware Spatio-Temporal Graph Convolutional Network with Visibility-Guarded $L_2$ Bone Normalization for 3D Hand Skeletal Restoration and Gesture Recognition under Optical Tracking Dropouts  
**Author:** Siriporn Noiumkar (Department of Creative Media, Faculty of Informatics, Mahasarakham University, Thailand)  
**Target Journal:** *Multimedia Tools and Applications* (Springer)  
**Public Code & Audit Repository:** https://github.com/siripornp-msu/KAR-STGCN-3D-Hand-Restoration (Release Tag `v2.0.0`)

---

## Response to Meta-Review & Critical Requirements

1. **Component-Wise Ablation Study & Sensitivity Sweeps (Required #1 — R1, R2, R3):**
   - **Location in Manuscript:** **Section 3.3 (Page 4)**, **Section 6.1–6.2 (Pages 7–8)**, **Table 5 (Page 8)**, and **Table 6 (Page 8)**.
   - **Details:** Table 6 explicitly ablates: (a) w/o Bone Loss ($\lambda_{\text{bone}}=0$, $\text{MPJPE}_{\text{all}}=7.84$ mm, $\text{BLCE}=4.76$ mm), (b) w/o Velocity Stream ($\Delta\hat{X}$, Macro-F1 drops from $92.64\%$ to $87.10\%$), (c) w/o Dilated Bridge ($\mathcal{B}_{\text{temp}}$, $\text{MPJPE}_{\text{miss}}=18.12$ mm) and Narrow Bridge ($\text{RF}=5$, $\text{MPJPE}_{\text{all}}=6.94$ mm), and (d) No Visibility Mask ($V=1$, $\text{MPJPE}_{\text{all}}=8.92$ mm). Section 6.2 further ablates our kinematic-aware adjacency ($\hat{A}$) against an unconstrained end-to-end learned adjacency matrix ($A_{\text{learn}} \in \mathbb{R}^{21\times 21}$, which degrades $\text{MPJPE}_{\text{all}}$ from $5.42$ mm to $6.48 \pm 0.32$ mm and $\text{MPJPE}_{\text{miss}}$ to $14.25 \pm 0.66$ mm). Section 3.3 reports full validation grid sweeps across $\lambda_{\text{bone}} \in \{0.0, 0.1, 0.3, 0.6, 1.0, 2.0\}$ and receptive fields $\text{RF} \in \{5, 13, 25, 37\}$ frames. Table 5 ablates 8 normalization regimes across 5 hand scale factors ($\times 0.70$ to $\times 1.40$).

2. **Expansion of Dedicated Imputation & Motion Infilling Baselines (Required #2 — R1, R2, R3):**
   - **Location in Manuscript:** **Section 2.2 (Pages 2–3)**, **Section 5.1 (Pages 5–6)**, and **Tables 1, 2, 3, 4, 8 (Pages 6–10)**.
   - **Details:** Both **BRITS [10]** and **SAITS [16]** (as well as the two-stage **SAITS + CTR-GCN** hybrid) are included across the main quantitative tables. Furthermore, Section 5.1 benchmarks KAR-STGCN against dedicated skeleton motion completion models on SHREC'17 Level 4, including a **Spatial-Temporal Graph U-Net Infiller** ($\text{MPJPE}_{\text{miss}} = 13.84 \pm 0.64$ mm, $\text{BLCE} = 3.38$ mm, $1.12$ ms GPU latency) and a **50-step Generative Diffusion baseline [21]** ($\text{MPJPE}_{\text{miss}} = 11.95 \pm 0.56$ mm, $38.40$ ms GPU forward latency).

3. **Public Code Repository URL & Reproducibility Pipeline (Required #3 — R1, R2, R3):**
   - **Location in Manuscript:** **Section 4.4 (Page 5)** and **Declarations (Page 11)**.
   - **Details:** The repository is publicly accessible at `https://github.com/siripornp-msu/KAR-STGCN-3D-Hand-Restoration` (Release Tag `v2.0.0`), containing the complete 7-Stage Scientific Evidence Validation Pipeline (`dataset_manifest.csv`, `subject_split_manifest.csv`, `reference_quality_report.csv`, `causality_test.py`, `seed_results.csv`, `verify_tables.py`, `statistical_validation.ipynb`, `robustness_matrix.csv`, `deployment_benchmark.csv`, and `REPRODUCTION_CERTIFICATE.md`).

4. **Primary Gesture Recognition Results (Required #4 — R2, R3):**
   - **Location in Manuscript:** **Abstract (Page 1)**, **Section 5.2–5.3 (Pages 6–7)**, and **Tables 2, 3, 4 (Page 8)**.
   - **Details:** Full subject-independent gesture recognition results (Macro-F1) are reported in Table 2 (SHREC'17 14-class: $92.64\%$, 28-class: $86.42\%$; FPHA 45-class: $82.35\%$), Table 3 (LMDHG 13-class real LMC1 transfer: $89.12\%$), and Table 4 (Dual-LMC2 17-class: $93.85\%$).

5. **Mathematical Specification of $\mathcal{L}_{\text{bone}}$, Adjacency $\mathbf{A}$, Startup Padding, and Hardware Latency (Required #5 — R1, R2):**
   - **Location in Manuscript:** **Section 3.1 (Page 3)**, **Section 3.3 (Page 4, Eqs. 5–6)**, and **Section 7.1 (Page 9)**.
   - **Details:** Section 3.1 defines all 20 anatomical bones and 4 inter-metacarpal palm-arch edges ($1\leftrightarrow 5\leftrightarrow 9\leftrightarrow 13\leftrightarrow 17$) forming the symmetric binary adjacency $\mathbf{A} \in \{0,1\}^{21\times 21}$. Section 3.3 specifies first-frame replication ($\tilde{X}_{t\le 0} \leftarrow \tilde{X}_1$) during sequence startup ($t < 25$), defines $\mathcal{L}_{\text{bone}}$ in the normalized wrist-centered frame, and verifies zero future-frame leakage via `causality_test.py`. Section 7.1 specifies the exact hardware testbed (NVIDIA GeForce RTX 4090 GPU, 24 GB VRAM, Intel Core i9-13900K CPU, PyTorch 2.2, CUDA 12.1, FP32, batch size $= 1$).

6. **Affiliation, Small-$N$ Statistical Power, Limitations, and Biometric Privacy (Recommended — R3):**
   - **Location in Manuscript:** **Title Page (Page 1)**, **Section 6.3 (Page 7)**, **Section 8 (Pages 9–10)**, and **Declarations (Page 11)**.
   - **Details:** The affiliation is correctly stated as *Mahasarakham University*. Section 6.3 and Table 7 supplement FPHA ($N=6, df=5$) paired $t$-tests with exact Wilcoxon signed-rank tests ($p=0.03125$), a hierarchical Linear Mixed-Effects Model ($F_{1,23}=74.8, p<10^{-6}$), and non-central $t$ statistical power analysis ($>99.1\%$ power). Section 8 details sensor, model, and application limitations, and Declarations (Page 11) addresses biometric hand-scale privacy de-identification via unit-palm $L_2$ normalization.
