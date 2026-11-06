#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

target_files = [
    "README.md",
    "statistical_validation.ipynb",
    "REPRODUCTION_CERTIFICATE.md",
    "RESPONSE_TO_REVIEWERS_SCISPACE_V6.md",
    "generate_certificate.py",
    "build_validation_pipeline.py",
    "apply_scispace_v6_final.py"
]

replacements = [
    ("*Multimedia Tools and Applications* (Springer)", "*Journal of Visual Communication and Image Representation* (Elsevier)"),
    ("Multimedia Tools and Applications (Springer)", "Journal of Visual Communication and Image Representation (Elsevier)"),
    ("Multimedia Tools and Applications", "Journal of Visual Communication and Image Representation")
]

for fname in target_files:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
        orig = content
        for old_str, new_str in replacements:
            content = content.replace(old_str, new_str)
        if content != orig:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[UPDATED] {fname} -> Journal of Visual Communication and Image Representation (Elsevier)")
        else:
            print(f"[OK] {fname} already up to date.")

# Ensure Highlights file for Elsevier submission is also generated locally
highlights_text = """Highlights

• Guarded L2 palm normalization prevents scale shortcuts and division singularities.
• KAR-STGCN couples a 25-frame dilated bridge with kinematic graph convolutions.
• Cuts occluded 3D hand joint error by 44.0%–50.6% across four benchmark datasets.
• Causal streaming achieves 0.93 ms latency with zero future-frame lookahead delay.
• KAR-Bench Studio exports .bvh rigs with 1.08%–1.15% joint-angle violation rates.
"""
with open("HIGHLIGHTS_ELSEVIER.txt", "w", encoding="utf-8") as f:
    f.write(highlights_text)
print("[GENERATED] HIGHLIGHTS_ELSEVIER.txt (All 5 bullet points <= 85 characters)")
