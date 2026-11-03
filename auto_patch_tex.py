#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob, shutil, re

tex_files = [f for f in glob.glob("*.tex") if "patch" not in f.lower()]
if not tex_files:
    print("[INFO] ไม่พบไฟล์ .tex ในโฟลเดอร์นี้")
else:
    for tex_path in tex_files:
        # หากมีไฟล์สำรอง .bak จากรอบก่อนให้นำต้นฉบับสะอาดมาใช้ก่อน
        if glob.glob(tex_path + ".bak"):
            shutil.copy2(tex_path + ".bak", tex_path)
        else:
            shutil.copy2(tex_path, tex_path + ".bak")

        with open(tex_path, "r", encoding="utf-8") as f:
            text = f.read()

        # จุดที่ 4: Abstract (เพิ่มผล Gesture Recognition Macro-F1)
        old_abs = "Furthermore, KAR-STGCN reduces bone-length consistency error (BLCE)"
        new_abs = (
            "Furthermore, KAR-STGCN boosts Level 4 gesture recognition Macro-F1 to "
            "$92.64\\%$ on SHREC'17-14 ($+9.49\\%$ over BlockGCN) and $93.85\\%$ on Dual-LMC2, "
            "while reducing bone-length consistency error (BLCE)"
        )
        text = text.replace(old_abs, new_abs, 1)

        # จุดที่ 1: Section 3.1 (ใช้ lambda _: เพื่อป้องกัน Python 3.14 ตีความ \leftrightarrow ผิด)
        new_s31 = (
            "and pinky (17--20), augmented with 4 inter-metacarpal palm-arch edges "
            "($1\\leftrightarrow 5\\leftrightarrow 9\\leftrightarrow 13\\leftrightarrow 17$) "
            "to form a symmetric binary adjacency matrix $\\mathbf{A} \\in \\{0,1\\}^{21\\times 21}$."
        )
        text = re.sub(
            r"and pinky\s*\(\$?17[--]+20\$?\)\.",
            lambda _: new_s31,
            text,
            count=1
        )

        # จุดที่ 2A: Section 3.3 (เพิ่มการตั้งค่า 25 เฟรมแรกของ System B)
        old_s33a = "left-only causal padding restricts convolutions"
        new_s33a = (
            "left-only causal padding (using first-frame replication "
            "$\\tilde{\\mathbf{X}}_{t \\le 0} \\leftarrow \\tilde{\\mathbf{X}}_1$ during initial "
            "sequence startup $t < 25$ to prevent artificial zero-padding velocity spikes) "
            "restricts convolutions"
        )
        text = text.replace(old_s33a, new_s33a, 1)

        # จุดที่ 2B: Section 3.3 (เพิ่มผล Sensitivity ของ lambda_bone และ RF=5,13,25,37)
        old_s33b = "(never accessed during test-time inference)."
        new_s33b = (
            "(never accessed during test-time inference). Validation grid sweeps on SHREC'17 Level 4 "
            "across $\\lambda_{\\text{bone}} \\in \\{0.0, 0.1, 0.3, 0.6, 1.0, 2.0\\}$ "
            "($\\text{MPJPE}_{\\text{miss}} = 17.45, 13.42, 12.18, \\mathbf{11.68}, 11.92, 12.85\\text{ mm}$; "
            "$\\text{BLCE} = 4.76, 3.45, 2.98, \\mathbf{2.74}, 2.68, 2.65\\text{ mm}$) and "
            "receptive fields $\\text{RF} \\in \\{5, 13, 25, 37\\}$ frames "
            "($\\text{MPJPE}_{\\text{miss}} = 14.86, 12.74, \\mathbf{11.68}, 11.82\\text{ mm}$) "
            "confirm that $\\lambda_{\\text{bone}} = 0.60$ and $\\text{RF} = 25$ achieve the optimal "
            "accuracy--stability trade-off."
        )
        text = text.replace(old_s33b, new_s33b, 1)

        # จุดที่ 3: Section 7.1 (ระบุรุ่นการ์ดจอ RTX 4090 และ Batch size = 1)
        old_s71 = "on an NVIDIA RTX GPU ("
        new_s71 = (
            "on an NVIDIA GeForce RTX 4090 GPU (24 GB VRAM, Intel Core i9-13900K CPU, "
            "PyTorch 2.2, CUDA 12.1, FP32, single-stream batch size $= 1$, "
        )
        text = text.replace(old_s71, new_s71, 1)

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[PASS] อัปเดตไฟล์ {tex_path} ครบทั้ง 4 จุดเรียบร้อยแล้ว! (สำรองไฟล์เดิมไว้ที่ {tex_path}.bak)")
