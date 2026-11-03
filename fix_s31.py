#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob, re

for tex_path in [f for f in glob.glob("*.tex") if "patch" not in f.lower()]:
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "inter-metacarpal palm-arch edges" not in text:
        addon = (
            ", augmented with 4 inter-metacarpal palm-arch edges "
            "($1\\leftrightarrow 5\\leftrightarrow 9\\leftrightarrow 13\\leftrightarrow 17$) "
            "to form a symmetric binary adjacency matrix $\\mathbf{A} \\in \\{0,1\\}^{21\\times 21}$."
        )
        # รองรับทั้ง (17--20)., ($17$--$20$). และ ($17\text{--}20$).
        text_new, n = re.subn(
            r"(pinky\s*\([^\)]*20[^\)]*\))\.",
            lambda m: m.group(1) + addon,
            text,
            count=1
        )
        if n > 0:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(text_new)
            print(f"[PASS] อัปเดต Section 3.1 ในไฟล์ {tex_path} สำเร็จครบ 100% แล้ว!")
        else:
            print(f"[WARN] ไม่พบคำว่า pinky (...20). ใน {tex_path}")
    else:
        print(f"[PASS] ไฟล์ {tex_path} มีข้อมูล Section 3.1 ครบอยู่แล้ว!")
