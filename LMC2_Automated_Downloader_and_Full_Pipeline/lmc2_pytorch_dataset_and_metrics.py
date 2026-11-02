"""
PyTorch Dataset Loader, Kinematic Bone-Length Loss & LaTeX Benchmark Exporter
=============================================================================
Designed for Windows / CUDA Workstation (e.g., RTX 2050 / Anaconda PyTorch).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


class LMC2SkeletalDataset(Dataset):
    """PyTorch Dataset for loading degraded LMC2 3D hand sequences."""

    def __init__(self, npz_path: str, use_l2norm: bool = True):
        data = np.load(npz_path)
        key_in = "X_corrupted_l2norm" if use_l2norm else "X_corrupted_mm"
        key_gt = "X_gt_l2norm" if use_l2norm else "X_gt_mm"

        self.X_in = torch.from_numpy(data[key_in]).float()          # (N, T, J, 3)
        self.X_gt = torch.from_numpy(data[key_gt]).float()          # (N, T, J, 3)
        self.X_gt_mm = torch.from_numpy(data["X_gt_mm"]).float()    # (N, T, J, 3)
        self.mask = torch.from_numpy(data["visibility_mask"]).float() # (N, T, J)
        self.y = torch.from_numpy(data["y"]).long()                 # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x_corrupted": self.X_in[idx],
            "x_gt": self.X_gt[idx],
            "x_gt_mm": self.X_gt_mm[idx],
            "visibility_mask": self.mask[idx],
            "label": self.y[idx]
        }


class KinematicBoneConsistencyLoss(nn.Module):
    """
    Hybrid Reconstruction + Bone-Length Consistency Loss to prevent over-smoothing
    and anatomical shrinkage during severe self-occlusion.
    """

    def __init__(self, lambda_bone: float = 0.25, lambda_vel: float = 0.10):
        super().__init__()
        self.lambda_bone = lambda_bone
        self.lambda_vel = lambda_vel
        self.parent_idx = [p for p, _ in BONE_PAIRS]
        self.child_idx = [c for _, c in BONE_PAIRS]

    def forward(self, pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> torch.Tensor:
        # 1. MPJPE (L2 joint position loss)
        joint_loss = torch.mean(torch.norm(pred_pos - gt_pos, dim=-1))

        # 2. Anatomical Bone-Length Consistency Loss (BLCE)
        pred_bones = torch.norm(pred_pos[..., self.child_idx, :] - pred_pos[..., self.parent_idx, :], dim=-1)
        gt_bones = torch.norm(gt_pos[..., self.child_idx, :] - gt_pos[..., self.parent_idx, :], dim=-1)
        bone_loss = torch.mean(torch.abs(pred_bones - gt_bones))

        # 3. Temporal Velocity Smoothness Loss
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        vel_loss = torch.mean(torch.norm(pred_vel - gt_vel, dim=-1))

        return joint_loss + self.lambda_bone * bone_loss + self.lambda_vel * vel_loss


def compute_metrics_and_export_latex(results_dict: dict, out_tex_path: str = "lmc2_benchmark_table.tex"):
    """Exports Multi-Seed evaluation metrics (Mean +/- SD) directly to an Overleaf-ready LaTeX table."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Multi-Seed Performance Comparison on LMC2 Synthetic Occlusion \& Long-Tail Benchmark (Mean $\pm$ SD across 5 Seeds)}",
        r"\label{tab:lmc2_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method / Architecture} & \textbf{MPJPE ($\downarrow$ mm)} & \textbf{PA-MPJPE ($\downarrow$ mm)} & \textbf{BLCE ($\downarrow$ mm)} & \textbf{Macro F1 ($\uparrow$ \%)} \\",
        r"\midrule"
    ]
    for model_name, m in results_dict.items():
        row = (
            f"{model_name} & "
            f"${m['mpjpe_mean']:.2f} \\pm {m['mpjpe_sd']:.2f}$ & "
            f"${m['pampjpe_mean']:.2f} \\pm {m['pampjpe_sd']:.2f}$ & "
            f"${m['blce_mean']:.2f} \\pm {m['blce_sd']:.2f}$ & "
            f"${m['f1_mean']:.2f} \\pm {m['f1_sd']:.2f}$ \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    print("PyTorch LMC2 Dataset & Kinematic Loss module loaded successfully.")
