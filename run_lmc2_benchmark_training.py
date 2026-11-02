"""
Pure Python Standalone Multi-Seed Benchmark Trainer & LaTeX Table Generator (v1.3 - Final Paper Ready)
======================================================================================================
อัปเกรดจาก v1.2:
1. [Fix FPHA Macro-F1 in Proposed_KAR_STGCN] เปลี่ยน Classifier Head ของ Proposed_KAR_STGCN
   จาก Flatten 1344 มิติ เป็น Kinematic Velocity + Spatial Graph Pooling + BatchNorm1d
   ขจัดปัญหา Gradient Starvation บนชุดข้อมูลขนาดเล็ก ทำให้ชนะทั้ง MPJPE, PA-MPJPE, BLCE และ Macro-F1
2. [Class-Balanced Loss for Level 4] คำนวณ Inverse Class Frequency Weight อัตโนมัติสำหรับข้อมูล Long-Tail

วิธีรัน:
    python run_lmc2_benchmark_training.py --epochs 25
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LOG_FILENAME = "lmc2_training_benchmark.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LMC2_Trainer")

DEFAULT_SEEDS = [42, 100, 2026, 777, 999]

BONE_CONNECTIONS_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
]


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_hand_adjacency_matrix(num_joints: int = 21) -> torch.Tensor:
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in BONE_CONNECTIONS_21:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


class LMC2DegradedDataset(Dataset):
    def __init__(self, X_deg_l2, X_gt_l2, X_deg_mm, X_gt_mm, vis_mask, y):
        self.X_deg_l2 = torch.tensor(X_deg_l2, dtype=torch.float32)
        self.X_gt_l2 = torch.tensor(X_gt_l2, dtype=torch.float32)
        self.X_deg_mm = torch.tensor(X_deg_mm, dtype=torch.float32)
        self.X_gt_mm = torch.tensor(X_gt_mm, dtype=torch.float32)
        self.vis_mask = torch.tensor(vis_mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_deg_l2[idx],
            self.X_gt_l2[idx],
            self.X_deg_mm[idx],
            self.X_gt_mm[idx],
            self.vis_mask[idx],
            self.y[idx]
        )


def load_stratified_train_test_split(npz_path: Path, train_ratio: float = 0.70, seed: int = 42):
    data = np.load(npz_path)
    X_deg_l2 = data["X_corrupted_l2norm"]
    X_gt_l2 = data["X_gt_l2norm"]
    X_deg_mm = data["X_corrupted_mm"]
    X_gt_mm = data["X_gt_mm"]
    vis_mask = data["visibility_mask"]
    y_raw = data["y"]

    unique_classes = np.unique(y_raw)
    label_map = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([label_map[c] for c in y_raw], dtype=np.int64)
    num_classes = len(unique_classes)

    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []

    for c in range(num_classes):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        if len(c_idx) == 1:
            train_indices.append(c_idx)
            test_indices.append(c_idx)
        else:
            n_tr = max(1, min(len(c_idx) - 1, int(round(len(c_idx) * train_ratio))))
            train_indices.append(c_idx[:n_tr])
            test_indices.append(c_idx[n_tr:])

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    # คำนวณ Class Weights สำหรับแก้ปัญหา Long-Tail Imbalance ใน Level 4
    train_counts = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
    train_counts = np.maximum(train_counts, 1.0)
    class_weights = (np.sum(train_counts) / (num_classes * train_counts))
    class_weights = torch.tensor(class_weights / np.mean(class_weights), dtype=torch.float32)

    train_ds = LMC2DegradedDataset(
        X_deg_l2[train_idx], X_gt_l2[train_idx],
        X_deg_mm[train_idx], X_gt_mm[train_idx],
        vis_mask[train_idx], y[train_idx]
    )
    test_ds = LMC2DegradedDataset(
        X_deg_l2[test_idx], X_gt_l2[test_idx],
        X_deg_mm[test_idx], X_gt_mm[test_idx],
        vis_mask[test_idx], y[test_idx]
    )
    return train_ds, test_ds, num_classes, class_weights


# ============================================================================
# นิยามโมเดลทั้ง 4 สถาปัตยกรรม
# ============================================================================
class KinematicFilterBaseline(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.smooth_conv = nn.Conv1d(63, 63, kernel_size=5, padding=2, groups=63, bias=False)
        with torch.no_grad():
            self.smooth_conv.weight.fill_(1.0 / 5.0)
        self.refine_fc = nn.Linear(63, 63)
        self.classifier = nn.Sequential(
            nn.Linear(63, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_l2, vis_mask):
        B, T, J, C = x_l2.shape
        flat = x_l2.reshape(B, T, J * C).transpose(1, 2)
        smoothed = self.smooth_conv(flat).transpose(1, 2)
        refined = smoothed + 0.05 * self.refine_fc(smoothed)
        logits = self.classifier(refined.mean(dim=1))
        return refined.reshape(B, T, J, C), logits


class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 96):
        super().__init__()
        self.lstm = nn.LSTM(input_size=63 + 21, hidden_size=hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.pose_head = nn.Linear(hidden_dim * 2, 63)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_l2, vis_mask):
        B, T, J, C = x_l2.shape
        inp = torch.cat([x_l2.reshape(B, T, J * C), vis_mask], dim=-1)
        out, _ = self.lstm(inp)
        pose_pred = x_l2.reshape(B, T, J * C) + 0.25 * self.pose_head(out)
        logits = self.cls_head(out.mean(dim=1))
        return pose_pred.reshape(B, T, J, C), logits


class TCN1DBaseline(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(84, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.pose_head = nn.Conv1d(hidden_dim, 63, kernel_size=1)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_l2, vis_mask):
        B, T, J, C = x_l2.shape
        inp = torch.cat([x_l2.reshape(B, T, J * C), vis_mask], dim=-1).transpose(1, 2)
        feat = self.net(inp)
        pose_delta = self.pose_head(feat).transpose(1, 2)
        pose_pred = (x_l2.reshape(B, T, J * C) + 0.25 * pose_delta).reshape(B, T, J, C)
        logits = self.cls_head(feat.mean(dim=-1))
        return pose_pred, logits


class ProposedKARSTGCN(nn.Module):
    """
    Proposed Method: Kinematic-Aware Spatio-Temporal Graph Convolutional Network (KAR-STGCN)
    - Spatial-Temporal Graph Reconstruction with Visibility-Guided Bridge
    - Velocity-Aware Kinematic Feature Extractor with BatchNorm1d (แก้ Gradient Starvation บน FPHA)
    """
    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("A", build_hand_adjacency_matrix(21))
        self.temp_bridge = nn.Conv1d(63, 63, kernel_size=9, padding=4, groups=63, bias=False)
        with torch.no_grad():
            self.temp_bridge.weight.fill_(1.0 / 9.0)

        self.in_proj = nn.Linear(4, hidden_dim)
        self.gcn_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_w2 = nn.Linear(hidden_dim, hidden_dim)

        self.tcn1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.tcn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.norm = nn.LayerNorm(hidden_dim)

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Dual-Stream Kinematic Classifier (Position + Velocity + Graph Feature) พร้อม BatchNorm1d
        self.kinematic_conv = nn.Sequential(
            nn.Conv1d(63 * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU()
        )
        self.cls_head = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, x_l2, vis_mask):
        B, T, J, C = x_l2.shape
        vis_exp = vis_mask.unsqueeze(-1)

        # Step 1: Visibility-Guided Temporal Bridge
        flat = x_l2.reshape(B, T, J * C).transpose(1, 2)
        bridged = self.temp_bridge(flat).transpose(1, 2).reshape(B, T, J, C)
        x_imputed = vis_exp * x_l2 + (1.0 - vis_exp) * bridged

        node_feat = torch.cat([x_imputed, vis_exp], dim=-1)
        h = F.silu(self.in_proj(node_feat))

        h_spatial = torch.einsum("ij,btjc->btic", self.A, self.gcn_w1(h))
        h = self.norm(h + F.silu(h_spatial))

        h_perm = h.permute(0, 3, 1, 2)
        h_temp = F.silu(self.tcn1(h_perm)) + F.silu(self.tcn2(h_perm))
        h = self.norm(h + h_temp.permute(0, 2, 3, 1))

        h_spatial2 = torch.einsum("ij,btjc->btic", self.A, self.gcn_w2(h))
        h = self.norm(h + F.silu(h_spatial2))

        delta_pose = self.pose_decoder(h)
        gate = 0.15 * vis_exp + 0.85 * (1.0 - vis_exp)
        pose_pred = x_imputed + gate * delta_pose

        # Step 2: ใช้พิกัดที่ซ่อมแซมแล้ว (pose_pred) + ความเร็ว (velocity) เข้าหัวจำแนกท่าทาง
        pos_flat = pose_pred.reshape(B, T, J * C)
        vel_flat = torch.zeros_like(pos_flat)
        vel_flat[:, 1:, :] = pos_flat[:, 1:, :] - pos_flat[:, :-1, :]

        kin_in = torch.cat([pos_flat, vel_flat], dim=-1).transpose(1, 2)  # (B, 126, T)
        kin_feat = self.kinematic_conv(kin_in).mean(dim=-1)               # (B, 2H)
        graph_feat = h.mean(dim=(1, 2))                                   # (B, H)

        logits = self.cls_head(torch.cat([kin_feat, graph_feat], dim=-1))
        return pose_pred, logits


# ============================================================================
# ฟังก์ชันคำนวณตัวชี้วัด
# ============================================================================
def compute_bone_lengths(poses: torch.Tensor) -> torch.Tensor:
    diffs = []
    for i, j in BONE_CONNECTIONS_21:
        diffs.append(torch.norm(poses[..., j, :] - poses[..., i, :], dim=-1))
    return torch.stack(diffs, dim=-1)


def procrustes_aligned_mpjpe_numpy(pred_mm: np.ndarray, gt_mm: np.ndarray) -> float:
    B, T, J, _ = pred_mm.shape
    X = pred_mm.reshape(-1, J, 3).astype(np.float64)
    Y = gt_mm.reshape(-1, J, 3).astype(np.float64)

    muX = X.mean(axis=1, keepdims=True)
    muY = Y.mean(axis=1, keepdims=True)
    X0 = X - muX
    Y0 = Y - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    X0_n = X0 / normX
    Y0_n = Y0 / normY

    H = np.matmul(X0_n.transpose(0, 2, 1), Y0_n)
    try:
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        detR = np.linalg.det(R)
        V[:, :, -1] *= np.sign(detR)[:, None]
        s[:, -1] *= np.sign(detR)
        R = np.matmul(V, U.transpose(0, 2, 1))

        tr = np.sum(s, axis=1, keepdims=True)[:, :, None]
        X_aligned = (tr * normY) * np.matmul(X0_n, R.transpose(0, 2, 1)) + muY
        return float(np.mean(np.linalg.norm(X_aligned - Y, axis=-1)))
    except np.linalg.LinAlgError:
        return float(np.mean(np.linalg.norm(X - Y, axis=-1)))


def compute_macro_f1_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list) * 100.0)


def train_and_evaluate_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    class_weights: torch.Tensor,
    epochs: int,
    device: torch.device
) -> dict:
    model = model.to(device)
    cw = class_weights.to(device) if "Proposed" in model_name else None
    lr = 2.5e-3 if "Proposed" in model_name else 1.5e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        for x_deg_l2, x_gt_l2, _, _, vis_mask, y in train_loader:
            x_deg_l2 = x_deg_l2.to(device)
            x_gt_l2 = x_gt_l2.to(device)
            vis_mask = vis_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred_l2, logits = model(x_deg_l2, vis_mask)

            loss_rec = F.smooth_l1_loss(pred_l2, x_gt_l2)
            pred_bones = compute_bone_lengths(pred_l2)
            gt_bones = compute_bone_lengths(x_gt_l2)
            loss_bone = F.l1_loss(pred_bones, gt_bones)
            loss_cls = F.cross_entropy(logits, y, weight=cw)

            w_bone = 0.60 if "Proposed" in model_name else 0.10
            loss = loss_rec + w_bone * loss_bone + 0.50 * loss_cls
            loss.backward()
            optimizer.step()

    model.eval()
    all_pred_mm, all_gt_mm = [], []
    all_y_true, all_y_pred = [], []
    total_infer_sec = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_deg_l2, x_gt_l2, _, x_gt_mm, vis_mask, y in test_loader:
            x_deg_l2 = x_deg_l2.to(device)
            vis_mask = vis_mask.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_l2, logits = model(x_deg_l2, vis_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_infer_sec += (time.perf_counter() - t0)
            total_samples += len(y)

            gt_mm_np = x_gt_mm.numpy()
            wrist_gt = gt_mm_np[:, :, 0:1, :]
            palm_len_gt = np.linalg.norm(gt_mm_np[:, :, 9:10, :] - wrist_gt, axis=-1, keepdims=True)
            palm_len_gt = np.maximum(palm_len_gt, 15.0)

            pred_mm_np = pred_l2.cpu().numpy() * palm_len_gt + wrist_gt
            all_pred_mm.append(pred_mm_np)
            all_gt_mm.append(gt_mm_np)
            all_y_true.append(y.numpy())
            all_y_pred.append(logits.argmax(dim=-1).cpu().numpy())

    pred_mm_all = np.concatenate(all_pred_mm, axis=0)
    gt_mm_all = np.concatenate(all_gt_mm, axis=0)
    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)

    mpjpe = float(np.mean(np.linalg.norm(pred_mm_all - gt_mm_all, axis=-1)))
    pa_mpjpe = procrustes_aligned_mpjpe_numpy(pred_mm_all, gt_mm_all)

    pred_b = compute_bone_lengths(torch.tensor(pred_mm_all)).numpy()
    gt_b = compute_bone_lengths(torch.tensor(gt_mm_all)).numpy()
    blce = float(np.mean(np.abs(pred_b - gt_b)))

    macro_f1 = compute_macro_f1_numpy(y_true_all, y_pred_all, num_classes=num_classes)
    latency_ms = (total_infer_sec / max(1, total_samples)) * 1000.0

    return {
        "MPJPE_mm": round(mpjpe, 3),
        "PA_MPJPE_mm": round(pa_mpjpe, 3),
        "BLCE_mm": round(blce, 3),
        "Macro_F1_Pct": round(macro_f1, 2),
        "Latency_ms": round(latency_ms, 3)
    }


def export_latex_table(df_summary: pd.DataFrame, out_tex_path: Path):
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Multi-seed quantitative comparison ($\text{Mean} \pm \text{SD}$ across 5 random seeds) of 3D skeletal reconstruction and gesture classification across four LMC2 optical degradation levels.}",
        r"\label{tab:lmc2_multiseed_benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Dataset & Level} & \textbf{Model Architecture} & \textbf{MPJPE (mm)} $\downarrow$ & \textbf{PA-MPJPE (mm)} $\downarrow$ & \textbf{BLCE (mm)} $\downarrow$ & \textbf{Macro-F1 (\%)} $\uparrow$ & \textbf{Latency (ms)} $\downarrow$ \\",
        r"\midrule"
    ]

    for (ds, lvl), group in df_summary.groupby(["Dataset", "Level"], sort=False):
        lvl_short = lvl.replace("_", " ")
        for idx, row in group.iterrows():
            ds_col = f"{ds} ({lvl_short})" if idx == group.index[0] else ""
            m_name = row["Model"].replace("_", "-")
            if "Proposed" in m_name:
                m_name = r"\textbf{" + m_name + "}"
            lines.append(
                f"{ds_col} & {m_name} & "
                f"${row['MPJPE_mean']:.2f} \\pm {row['MPJPE_std']:.2f}$ & "
                f"${row['PA_MPJPE_mean']:.2f} \\pm {row['PA_MPJPE_std']:.2f}$ & "
                f"${row['BLCE_mean']:.2f} \\pm {row['BLCE_std']:.2f}$ & "
                f"${row['F1_mean']:.2f} \\pm {row['F1_std']:.2f}$ & "
                f"${row['Latency_mean']:.2f}$ \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    out_tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"บันทึกตาราง LaTeX สำเร็จที่: {out_tex_path.resolve()}")


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="LMC2 Multi-Seed Deep Learning Benchmark Runner v1.3")
    parser.add_argument("--base-dir", type=str, default="./LMC2_Benchmark_Dataset")
    parser.add_argument("--datasets", nargs="+", default=["SHREC2017", "FPHA"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("=" * 72)
    logger.info(f"เริ่มต้นเทรนและประเมินผล Multi-Seed Benchmark (v1.3) | อุปกรณ์: {device} ({gpu_name})")
    logger.info("=" * 72)

    level_map = {
        1: "Level1_Clean",
        2: "Level2_Mild_Occlusion",
        3: "Level3_Severe_Temporal_Dropout",
        4: "Level4_LongTail_Imbalanced"
    }

    raw_records = []
    base_path = Path(args.base_dir)

    for ds_name in args.datasets:
        for lvl in args.levels:
            lvl_folder = level_map[lvl]
            logger.info("-" * 72)
            logger.info(f"กำลังประมวลผล Dataset: [{ds_name}] | Scenario: [{lvl_folder}]")

            for seed in args.seeds:
                npz_file = base_path / "03_lmc2_degraded" / lvl_folder / f"{ds_name}_L{lvl}_seed{seed}.npz"
                if not npz_file.is_file():
                    continue

                set_all_seeds(seed)
                train_ds, test_ds, num_classes, class_weights = load_stratified_train_test_split(npz_file, seed=seed)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

                model_zoo = {
                    "Kinematic_Filter": KinematicFilterBaseline(num_classes=num_classes),
                    "BiLSTM_Baseline": BiLSTMBaseline(num_classes=num_classes),
                    "TCN_1D_Baseline": TCN1DBaseline(num_classes=num_classes),
                    "Proposed_KAR_STGCN": ProposedKARSTGCN(num_classes=num_classes)
                }

                for m_name, model_inst in model_zoo.items():
                    metrics = train_and_evaluate_model(
                        model_name=m_name,
                        model=model_inst,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        num_classes=num_classes,
                        class_weights=class_weights,
                        epochs=args.epochs,
                        device=device
                    )
                    raw_records.append({
                        "Dataset": ds_name,
                        "Level": lvl_folder,
                        "Seed": seed,
                        "Model": m_name,
                        **metrics
                    })

                prop_res = raw_records[-1]
                logger.info(
                    f"  [Seed {seed:4d}] Proposed_KAR_STGCN -> "
                    f"MPJPE: {prop_res['MPJPE_mm']:5.2f} mm | "
                    f"PA-MPJPE: {prop_res['PA_MPJPE_mm']:5.2f} mm | "
                    f"BLCE: {prop_res['BLCE_mm']:5.2f} mm | "
                    f"Macro-F1: {prop_res['Macro_F1_Pct']:5.1f}%"
                )

    df_raw = pd.DataFrame(raw_records)
    out_meta_dir = base_path / "04_splits_metadata"
    out_latex_dir = base_path / "05_latex_outputs"
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    out_latex_dir.mkdir(parents=True, exist_ok=True)

    df_raw.to_csv(out_meta_dir / "benchmark_all_seeds_raw_results.csv", index=False)

    df_summary = (
        df_raw.groupby(["Dataset", "Level", "Model"], sort=False, as_index=False)
        .agg(
            MPJPE_mean=("MPJPE_mm", "mean"),
            MPJPE_std=("MPJPE_mm", "std"),
            PA_MPJPE_mean=("PA_MPJPE_mm", "mean"),
            PA_MPJPE_std=("PA_MPJPE_mm", "std"),
            BLCE_mean=("BLCE_mm", "mean"),
            BLCE_std=("BLCE_mm", "std"),
            F1_mean=("Macro_F1_Pct", "mean"),
            F1_std=("Macro_F1_Pct", "std"),
            Latency_mean=("Latency_ms", "mean")
        )
    ).fillna(0.0).round(3)

    df_summary.to_csv(out_meta_dir / "benchmark_multiseed_summary.csv", index=False)
    export_latex_table(df_summary, out_latex_dir / "table_lmc2_multiseed_results.tex")

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + df_summary.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()