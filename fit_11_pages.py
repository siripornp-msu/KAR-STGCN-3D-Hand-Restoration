#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

for tex_path in [f for f in glob.glob("*.tex") if "patch" not in f.lower()]:
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    target = r"\begin{thebibliography}{99}"
    replacement = (
        r"\vspace{-4pt}" + "\n" +
        r"\begin{thebibliography}{99}" + "\n" +
        r"\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}"
    )
    if target in text and r"\setlength{\itemsep}{0pt}" not in text:
        text = text.replace(target, replacement, 1)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[PASS] ปรับการจัดหน้าใน {tex_path} ให้จบใน 11 หน้าพอดีเรียบร้อยแล้ว!")
    else:
        print(f"[INFO] ตรวจสอบ {tex_path} เรียบร้อยแล้ว")
