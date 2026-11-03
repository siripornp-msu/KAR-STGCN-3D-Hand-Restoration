#!/usr/bin/env python3
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalVisibilityDetector:
    def __init__(self, zero_tol=1e-3, freeze_tol=1e-5, jump_thresh=60.0):
        self.zero_tol, self.freeze_tol, self.jump_thresh = zero_tol, freeze_tol, jump_thresh
    def __call__(self, p_seq):
        T, J, _ = p_seq.shape
        v_hat = np.ones((T, J), dtype=np.float32)
        freeze_counter = np.zeros(J, dtype=np.int32)
        for t in range(T):
            zero_collapse = np.linalg.norm(p_seq[t], axis=-1) < self.zero_tol
            if t > 0:
                delta = np.linalg.norm(p_seq[t] - p_seq[t - 1], axis=-1)
                unphysical_jump = delta > self.jump_thresh
                freeze_counter = np.where(delta < self.freeze_tol, freeze_counter + 1, 0)
                sdk_freeze = freeze_counter >= 2
            else:
                unphysical_jump = np.zeros(J, dtype=bool)
                sdk_freeze = np.zeros(J, dtype=bool)
            v_hat[t, zero_collapse | unphysical_jump | sdk_freeze] = 0.0
        return v_hat

class VisibilityGuardedNormalizer:
    def __init__(self, tau_min=15.0, d_default=80.0):
        self.tau_min, self.d_default = tau_min, d_default
    def normalize_causal_system_b(self, p_seq, v_seq):
        T, J, _ = p_seq.shape
        p_tilde = np.zeros_like(p_seq, dtype=np.float64)
        p_valid_wrist = np.zeros((T, 3), dtype=np.float64)
        d_palm_seq = np.zeros(T, dtype=np.float64)
        last_wrist, valid_palm_history = np.zeros(3, dtype=np.float64), []
        for t in range(T):
            if v_seq[t, 0] == 1.0:
                last_wrist = p_seq[t, 0].astype(np.float64).copy()
            p_valid_wrist[t] = last_wrist
            inst_dist = float(np.linalg.norm(p_seq[t, 9].astype(np.float64) - last_wrist))
            if (v_seq[t, 0] == 1.0) and (v_seq[t, 9] == 1.0) and (inst_dist > self.tau_min):
                valid_palm_history.append(inst_dist)
                d_t = inst_dist
            else:
                d_t = float(np.median(valid_palm_history)) if len(valid_palm_history) > 0 else self.d_default
            d_palm_seq[t] = d_t
            for j in range(J):
                if v_seq[t, j] == 1.0:
                    p_tilde[t, j] = (p_seq[t, j].astype(np.float64) - last_wrist) / d_t
        return p_tilde.astype(np.float32), p_valid_wrist.astype(np.float32), d_palm_seq.astype(np.float32)

class CausalKARSTGCNStream(nn.Module):
    def __init__(self, num_joints=21, hidden_dim=64, alpha_vis=0.15):
        super().__init__()
        self.alpha_vis = alpha_vis
        c = num_joints * 3
        self.conv1 = nn.Conv1d(c, c, kernel_size=9, dilation=1, groups=c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=9, dilation=2, groups=c)
        self.in_proj = nn.Linear(4, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim * num_joints)
        self.t_conv = nn.Conv1d(hidden_dim * num_joints, hidden_dim * num_joints, kernel_size=3, groups=num_joints)
        self.dec = nn.Linear(hidden_dim, 3)
    def forward(self, x_tilde, v_mask, d_palm, p_wrist):
        B, T, J, _ = x_tilde.shape
        x_flat = x_tilde.reshape(B, T, J * 3).transpose(1, 2)
        h = F.gelu(self.conv1(F.pad(x_flat, (8, 0), mode="replicate")))
        x_bridge = self.conv2(F.pad(h, (16, 0), mode="replicate")).transpose(1, 2).reshape(B, T, J, 3)
        v_exp = v_mask.unsqueeze(-1)
        x_imp = v_exp * x_tilde + (1.0 - v_exp) * x_bridge
        h_proj = self.in_proj(torch.cat([x_imp, v_exp], dim=-1))
        h_bn = self.bn(h_proj.reshape(B, T, -1).transpose(1, 2))
        h_tc = F.relu(self.t_conv(F.pad(h_bn, (2, 0)))).transpose(1, 2).reshape(B, T, J, -1)
        gate = self.alpha_vis * v_exp + (1.0 - self.alpha_vis) * (1.0 - v_exp)
        x_hat_norm = x_imp + gate * self.dec(h_tc)
        vel = torch.zeros_like(x_hat_norm)
        vel[:, 1:] = x_hat_norm[:, 1:] - x_hat_norm[:, :-1]
        return x_hat_norm * d_palm.unsqueeze(-1).unsqueeze(-1) + p_wrist.unsqueeze(2), vel

class TestSystemBCausalityAndLeakage(unittest.TestCase):
    def setUp(self):
        np.random.seed(42); torch.manual_seed(42)
        self.detector = CausalVisibilityDetector()
        self.normalizer = VisibilityGuardedNormalizer()
        self.model = CausalKARSTGCNStream().eval()
        self.stream = np.random.uniform(-40.0, 40.0, size=(60, 21, 3)).astype(np.float32)
        self.stream[:, 0, :] = [0.0, 0.0, 250.0]
        self.stream[:, 9, :] = [0.0, 82.5, 250.0]
        self.stream[20:31] = 0.0
    def test_01_visibility_detector_causality(self):
        for t_cut in [10, 25, 42]:
            s_b = self.stream.copy(); s_b[t_cut+1:] = 5000.0
            self.assertEqual(np.max(np.abs(self.detector(self.stream)[:t_cut+1] - self.detector(s_b)[:t_cut+1])), 0.0)
    def test_02_running_median_normalization_causality(self):
        v_a = self.detector(self.stream)
        for t_cut in [12, 24, 45]:
            s_b = self.stream.copy(); s_b[t_cut+1:, 9, :] = 99999.0
            v_b = self.detector(s_b)
            na, wa, da = self.normalizer.normalize_causal_system_b(self.stream, v_a)
            nb, wb, db = self.normalizer.normalize_causal_system_b(s_b, v_b)
            self.assertEqual(np.max(np.abs(na[:t_cut+1] - nb[:t_cut+1])), 0.0)
    def test_03_end_to_end_neural_stream_causality(self):
        v_a = self.detector(self.stream)
        na, wa, da = self.normalizer.normalize_causal_system_b(self.stream, v_a)
        with torch.no_grad():
            oa, va = self.model(torch.from_numpy(na)[None], torch.from_numpy(v_a)[None], torch.from_numpy(da)[None], torch.from_numpy(wa)[None])
        for t_cut in [15, 30, 48]:
            s_b = self.stream.copy(); s_b[t_cut+1:] = -9999.0
            v_b = self.detector(s_b)
            nb, wb, db = self.normalizer.normalize_causal_system_b(s_b, v_b)
            with torch.no_grad():
                ob, vb = self.model(torch.from_numpy(nb)[None], torch.from_numpy(v_b)[None], torch.from_numpy(db)[None], torch.from_numpy(wb)[None])
            self.assertEqual(torch.max(torch.abs(oa[:, :t_cut+1] - ob[:, :t_cut+1])).item(), 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
