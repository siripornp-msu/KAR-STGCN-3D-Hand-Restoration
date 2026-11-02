@echo off
echo [INFO] Creating directory structure for LMC2 Deep Learning Research...
mkdir "LMC2_Benchmark_Dataset\01_raw_public\SHREC2017" 2>nul
mkdir "LMC2_Benchmark_Dataset\01_raw_public\FPHA" 2>nul
mkdir "LMC2_Benchmark_Dataset\01_raw_public\LeapGestureDB" 2>nul
mkdir "LMC2_Benchmark_Dataset\02_preprocessed_l2norm" 2>nul
mkdir "LMC2_Benchmark_Dataset\03_lmc2_degraded\Level1_Clean" 2>nul
mkdir "LMC2_Benchmark_Dataset\03_lmc2_degraded\Level2_Mild_Occlusion" 2>nul
mkdir "LMC2_Benchmark_Dataset\03_lmc2_degraded\Level3_Severe_Temporal_Dropout" 2>nul
mkdir "LMC2_Benchmark_Dataset\03_lmc2_degraded\Level4_LongTail_Imbalanced" 2>nul
mkdir "LMC2_Benchmark_Dataset\04_splits_metadata" 2>nul
mkdir "LMC2_Benchmark_Dataset\05_latex_outputs" 2>nul
echo [SUCCESS] Workspace directories created!
