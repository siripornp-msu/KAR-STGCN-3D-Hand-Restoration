# -*- coding: utf-8 -*-
"""Automated Causality & Zero-Lookahead Verification for System B (KAR-STGCN)"""
import numpy as np

def causal_running_median_norm(X_raw, V_mask, tau_min=15.0, d_default=80.0):
    T, J, C = X_raw.shape
    X_norm = np.zeros_like(X_raw)
    d_scale = np.zeros(T, dtype=np.float32)
    valid_palms = []
    last_valid_wrist = np.zeros(3, dtype=np.float32)
    has_wrist = False
    for t in range(T):
        if V_mask[t, 0] == 1:
            last_valid_wrist = X_raw[t, 0].copy()
            has_wrist = True
        wrist_anchor = last_valid_wrist if has_wrist else np.zeros(3, dtype=np.float32)
        d_inst = np.linalg.norm(X_raw[t, 9] - wrist_anchor)
        if V_mask[t, 0] == 1 and V_mask[t, 9] == 1 and d_inst > tau_min:
            valid_palms.append(d_inst)
            d_t = d_inst
        else:
            d_t = float(np.median(valid_palms)) if len(valid_palms) > 0 else d_default
        d_scale[t] = d_t
        X_norm[t] = (V_mask[t, :, None]) * ((X_raw[t] - wrist_anchor[None, :]) / d_t)
    return X_norm, d_scale

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_stream = rng.normal(50.0, 20.0, size=(60, 21, 3)).astype(np.float32)
    V_stream = (rng.uniform(0, 1, size=(60, 21)) > 0.2).astype(np.float32)
    X_norm_A, d_A = causal_running_median_norm(X_stream, V_stream)
    # Perturb future frames t > 30 with extreme values (+10,000 mm)
    X_perturbed = X_stream.copy()
    X_perturbed[31:, :, :] += 10000.0
    X_norm_B, d_B = causal_running_median_norm(X_perturbed, V_stream)
    max_diff_past = np.max(np.abs(X_norm_A[:31] - X_norm_B[:31]))
    assert max_diff_past == 0.0, f"Causality violation detected: {max_diff_past}"
    print(f"[PASSED] System B Causality Audit: max diff on past/current frames (t <= 30) = {max_diff_past:.8f} mm")
