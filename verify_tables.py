#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAR-STGCN v2.0.0 — Automated Manuscript-to-Log Synchronization & Audit Script (verify_tables.py)
Author: Siriporn Noiumkar (Mahasarakham University)
Repository: https://github.com/siripornp-msu/KAR-STGCN-3D-Hand-Restoration
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad


def assert_close(name: str, actual: float, expected: float, tol: float = 0.02):
    if abs(actual - expected) > tol:
        raise AssertionError(f"[FAIL] {name}: actual={actual:.4f} vs expected={expected:.4f}")
    print(f"[PASS] {name:<58} | Log: {actual:8.2f} == Manuscript: {expected:8.2f}")


def exact_noncentral_t_power(n: int, d: float, alpha: float = 0.05 / 7.0) -> float:
    """
    Computes exact two-sided statistical power for a paired t-test using the exact
    Gaussian-ChiSquare conditioning integral (immune to SciPy nct.cdf NaN underflow on Windows).
    """
    df = n - 1
    ncp = d * np.sqrt(float(n))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    def integrand(v: float) -> float:
        scale = np.sqrt(v / df)
        prob_reject_given_v = stats.norm.sf(t_crit * scale - ncp) + stats.norm.cdf(-t_crit * scale - ncp)
        return float(prob_reject_given_v * stats.chi2.pdf(v, df))

    power_val, _ = quad(integrand, 0.0, np.inf, limit=200)
    return float(power_val * 100.0)


def main():
    print("=" * 88)
    print("KAR-STGCN v2.0.0 — Automated Manuscript Table & Statistical Synchronization Audit")
    print("=" * 88)

    # 1. Verify Subject-Independent Split Integrity
    df_splits = pd.read_csv("subject_split_manifest.csv")
    for _, r in df_splits.iterrows():
        tr = set(r["train_subject_ids"].split(";"))
        va = set(r["val_subject_ids"].split(";"))
        te = set(r["test_subject_ids"].split(";"))
        assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0
    print("[PASS] Subject-Independent Split Integrity (20 Folds across 4 Benchmarks) | ZERO LEAKAGE")

    for b_id, exp_tr, exp_te in [
        ("SHREC17", 2240.0, 560.0),
        ("FPHA", 940.0, 235.0),
        ("LMDHG", 486.4, 121.6),
        ("Dual_LMC2_ML2HP", 564.0, 141.0),
    ]:
        sub = df_splits[df_splits["benchmark"] == b_id]
        assert_close(f"{b_id} Mean Train+Val Sequences per Fold", sub["train_plus_val_sequences"].mean(), exp_tr, 0.1)
        assert_close(f"{b_id} Mean Test Sequences per Fold", sub["test_sequences"].mean(), exp_te, 0.1)

    # 2. Verify Dual-LMC2 Kabsch SE(3) Reference Calibration
    df_ref = pd.read_csv("reference_quality_report.csv")
    assert_close("Dual-LMC2 Kabsch SE(3) Residual Mean (mm)", df_ref["kabsch_se3_residual_mean_mm"].mean(), 1.84)
    assert_close("Dual-LMC2 Kabsch SE(3) Residual SD (mm)", df_ref["kabsch_se3_residual_mean_mm"].std(ddof=1), 0.42)
    assert_close("Dual-LMC2 Kabsch SE(3) Residual Median p50 (mm)", df_ref["kabsch_se3_residual_mean_mm"].median(), 1.76)
    assert_close("Dual-LMC2 Valid Cross-View Windows Retained", float(df_ref["valid_cross_view_windows"].sum()), 705.0)

    # 3. Verify Subject-Level Metrics & Paired Hypothesis Tests (Tables 1-4 & Table 7)
    df_subj = (
        pd.read_csv("seed_results.csv")
        .groupby(["benchmark", "regime", "subject_id"], as_index=False)
        .mean(numeric_only=True)
    )
    checks = [
        ("SHREC17", "Level_3_WholeHand_Burst", 28, 9.42, 19.08, -9.66, 2.94),
        ("SHREC17", "Level_4_Compound_Dropout", 28, 11.68, 23.58, -11.90, 3.42),
        ("FPHA", "Level_3_WholeHand_Burst", 6, 11.45, 21.15, -9.70, 2.84),
        ("FPHA", "Level_4_Compound_Dropout", 6, 15.84, 29.24, -13.40, 3.85),
        ("LMDHG", "Protocol_2_ZeroShot", 21, 12.74, 20.48, -7.74, 2.62),
        ("LMDHG", "Protocol_3_FineTuned", 21, 9.86, 17.62, -7.76, 2.48),
        ("Dual_LMC2_ML2HP", "Natural_CrossView", 21, 9.18, 16.78, -7.60, 1.85),
    ]
    for b_name, reg, n_exp, k_exp, c_exp, d_exp, sd_exp in checks:
        sub = df_subj[(df_subj["benchmark"] == b_name) & (df_subj["regime"] == reg)]
        assert len(sub) == n_exp
        k_v = sub["KAR_STGCN_SysA_MPJPE_miss"].to_numpy()
        c_v = sub["CTR_GCN_MPJPE_miss"].to_numpy()
        d_v = k_v - c_v
        assert_close(f"{b_name} ({reg}) KAR-STGCN MPJPE_miss", k_v.mean(), k_exp)
        assert_close(f"{b_name} ({reg}) CTR-GCN MPJPE_miss", c_v.mean(), c_exp)
        assert_close(f"{b_name} ({reg}) Paired Diff Mean", d_v.mean(), d_exp)
        assert_close(f"{b_name} ({reg}) Paired Diff SD", d_v.std(ddof=1), sd_exp)

    # 4. Verify Statistical Power Analysis on Small-N Benchmark (FPHA N=6, df=5)
    for label, d_obs in [("FPHA Level 3 (d=3.42)", 3.42), ("FPHA Level 4 (d=3.48)", 3.48)]:
        power = exact_noncentral_t_power(n=6, d=d_obs, alpha=0.05 / 7.0)
        assert power >= 99.0, f"Expected statistical power >= 99%, got {power:.2f}%"
        print(f"[PASS] Statistical Power ({label}, N=6, alpha_Holm=0.0071) | Power: {power:6.2f}% >= 99.00%")

    print("=" * 88)
    print("ALL MANUSCRIPT TABLES, SPLITS, AND STATISTICAL TESTS VERIFIED SUCCESSFULLY (0 DISCREPANCIES).")
    print("=" * 88)


if __name__ == "__main__":
    main()
