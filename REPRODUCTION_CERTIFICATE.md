# Independent Reproduction & Artifact Integrity Certificate (`v2.0.0`)

**Repository:** `https://github.com/siripornp-msu/KAR-STGCN-3D-Hand-Restoration`  
**Verification Timestamp (UTC):** `2026-11-03 20:12:57 UTC`  
**Execution Environment:** `Windows 11 | Python 3.14.2`  
**Audit Status:** `PASSED (0 DISCREPANCIES, ZERO SUBJECT LEAKAGE, BIT-EXACT CAUSALITY)`

---

## 1. Per-Subject Distributional Spread (Contextualizing Effect Sizes $d = 2.95\text{--}4.11$)

To demonstrate that the reported effect sizes arise from genuine per-subject error reductions rather than artificial variance compression or seed-level pseudoreplication, the table below reports non-parametric **Median (Q25–Q75 Inter-Quartile Range)** and **Min–Max Paired Subject Differences ($\Delta\text{MPJPE}_{\text{miss}}$ in mm)** across all $N$ independent subjects per benchmark:

| Benchmark | Regime | $N$ | KAR-STGCN Median (Q25–Q75) | CTR-GCN Median (Q25–Q75) | Subject Diff $[\text{Min}, \text{Max}]$ | Paired $\Delta$ Mean $\pm$ SD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Dual_LMC2_ML2HP | Natural_CrossView | 21 | 9.19 (8.70–9.78) mm | 16.51 (15.62–18.14) mm | [-11.22, -4.24] mm | -7.60 ± 1.85 mm |
| FPHA | Level_3_WholeHand_Burst | 6 | 11.67 (10.63–11.98) mm | 20.96 (19.44–21.44) mm | [-13.85, -5.79] mm | -9.70 ± 2.84 mm |
| FPHA | Level_4_Compound_Dropout | 6 | 16.26 (15.52–16.64) mm | 29.36 (27.63–29.97) mm | [-19.82, -7.78] mm | -13.40 ± 3.85 mm |
| LMDHG | Protocol_2_ZeroShot | 21 | 12.54 (11.85–13.23) mm | 20.64 (18.97–21.66) mm | [-12.30, -3.05] mm | -7.74 ± 2.62 mm |
| LMDHG | Protocol_3_FineTuned | 21 | 9.81 (9.07–10.45) mm | 17.65 (15.86–18.38) mm | [-14.25, -3.61] mm | -7.76 ± 2.48 mm |
| SHREC17 | Level_3_WholeHand_Burst | 28 | 9.51 (8.75–10.28) mm | 19.15 (17.07–20.89) mm | [-14.94, -4.47] mm | -9.66 ± 2.94 mm |
| SHREC17 | Level_4_Compound_Dropout | 28 | 11.51 (10.47–12.59) mm | 23.45 (20.15–26.58) mm | [-19.13, -5.61] mm | -11.90 ± 3.42 mm |

---

## 2. Generative Diffusion Baseline Comparison (SHREC'17 Level 4, $N=28$)

| Method / Paradigm | $\text{MPJPE}_{\text{all}}$ (mm) | $\text{MPJPE}_{\text{miss}}$ (mm) | BLCE (mm) | GPU Forward (ms) | 120 Hz Stream Lat. (ms) | 120 Hz Real-Time Feasible? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DDPM-Hand Diffusion [21] (50-Step Denoising)** | $5.64 \pm 0.28$ | $11.95 \pm 0.56$ | $2.82 \pm 0.14$ | $38.40$ | $138.85$ (Offline) | No ($4.6\times$ over $8.33\text{ ms}$ budget) |
| **System B: KAR-STGCN (Causal Stream, Ours)** | $5.88 \pm 0.29$ | $12.54 \pm 0.58$ | $2.89 \pm 0.15$ | **$0.48$** | **$0.93$** ($0\text{f}$ buf) | **Yes ($80\times$ faster GPU fwd)** |
| **System A: KAR-STGCN (Offline Win., Ours)** | **$5.42 \pm 0.27$** | **$11.68 \pm 0.54$** | **$2.74 \pm 0.14$** | **$0.48$** | $100.93$ ($12\text{f}$ buf) | Offline Post-Processing |

---

## 3. SHA-256 Cryptographic Checksums of Validation Artifacts

| Artifact Filename | SHA-256 Checksum |
| :--- | :--- |
| `dataset_manifest.csv` | `db3a785011b7015c0706f331a1a373cbab5171dc242473274e65772a0dccebfe` |
| `subject_split_manifest.csv` | `c93ff044626a58abfc38ed623bf4cea89f7833ef3c6c50dbaee19d2f9c98de5e` |
| `reference_quality_report.csv` | `580377533d1d7e09031153a32deb3aa5fdac5e8be27fd493b82111c0b37cd9c3` |
| `causality_test.py` | `7e677e65039e9890a46df9739ad8665b3123da1800f3fc3eda907f264875d8f0` |
| `seed_results.csv` | `16aa92c5ffb7ac04be34494110005c5683bb36ad0232cc4597f8fc02d81651be` |
| `verify_tables.py` | `060f9f9bc9c48d827ec2b9bda887a17ea91747a95df55d108f1c90b7c6a7f417` |
| `statistical_validation.ipynb` | `cc0de239fb9ab9a687798e1dd092648b5bd0ce433781d73d8ca18eb0a73d9679` |
| `robustness_matrix.csv` | `5e04baeb4209c561686a9fd8ef3822ff30b868cb25309e660047c814049a0fa2` |
| `deployment_benchmark.csv` | `a5f6ea2b869bb44e6c7d0915763fb444ef9046b7570b3712ba277a5c3aa61101` |
