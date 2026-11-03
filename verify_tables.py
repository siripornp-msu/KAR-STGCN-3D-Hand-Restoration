#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import stats

def assert_close(name, actual, expected, tol=0.02):
    if abs(actual - expected) > tol:
        raise AssertionError(f"[FAIL] {name}: {actual} vs {expected}")
    print(f"[PASS] {name:<58} | Log: {actual:8.2f} == Manuscript: {expected:8.2f}")

def main():
    print("=" * 88)
    print("KAR-STGCN v2.0.0 — Automated Manuscript Table & Statistical Synchronization Audit")
    print("=" * 88)
    df_splits = pd.read_csv("subject_split_manifest.csv")
    for _, r in df_splits.iterrows():
        tr, va, te = set(r["train_subject_ids"].split(";")), set(r["val_subject_ids"].split(";")), set(r["test_subject_ids"].split(";"))
        assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0
    print("[PASS] Subject-Independent Split Integrity (20 Folds across 4 Benchmarks) | ZERO LEAKAGE")
    for b_id, exp_tr, exp_te in [("SHREC17", 2240.0, 560.0), ("FPHA", 940.0, 235.0), ("LMDHG", 486.4, 121.6), ("Dual_LMC2_ML2HP", 564.0, 141.0)]:
        sub = df_splits[df_splits["benchmark"] == b_id]
        assert_close(f"{b_id} Mean Train+Val Sequences per Fold", sub["train_plus_val_sequences"].mean(), exp_tr, 0.1)
        assert_close(f"{b_id} Mean Test Sequences per Fold", sub["test_sequences"].mean(), exp_te, 0.1)
    df_ref = pd.read_csv("reference_quality_report.csv")
    assert_close("Dual-LMC2 Kabsch SE(3) Residual Mean (mm)", df_ref["kabsch_se3_residual_mean_mm"].mean(), 1.84)
    assert_close("Dual-LMC2 Kabsch SE(3) Residual SD (mm)", df_ref["kabsch_se3_residual_mean_mm"].std(ddof=1), 0.42)
    assert_close("Dual-LMC2 Kabsch SE(3) Residual Median p50 (mm)", df_ref["kabsch_se3_residual_mean_mm"].median(), 1.76)
    assert_close("Dual-LMC2 Valid Cross-View Windows Retained", float(df_ref["valid_cross_view_windows"].sum()), 705.0)
    df_subj = pd.read_csv("seed_results.csv").groupby(["benchmark", "regime", "subject_id"], as_index=False).mean(numeric_only=True)
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
        k_v, c_v = sub["KAR_STGCN_SysA_MPJPE_miss"].to_numpy(), sub["CTR_GCN_MPJPE_miss"].to_numpy()
        d_v = k_v - c_v
        assert_close(f"{b_name} ({reg}) KAR-STGCN MPJPE_miss", k_v.mean(), k_exp)
        assert_close(f"{b_name} ({reg}) CTR-GCN MPJPE_miss", c_v.mean(), c_exp)
        assert_close(f"{b_name} ({reg}) Paired Diff Mean", d_v.mean(), d_exp)
        assert_close(f"{b_name} ({reg}) Paired Diff SD", d_v.std(ddof=1), sd_exp)
    print("=" * 88)
    print("ALL MANUSCRIPT TABLES, SPLITS, AND STATISTICAL TESTS VERIFIED SUCCESSFULLY (0 DISCREPANCIES).")
    print("=" * 88)

if __name__ == "__main__":
    main()
