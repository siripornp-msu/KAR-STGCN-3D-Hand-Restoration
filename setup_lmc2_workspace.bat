@echo off
REM ============================================================================
REM Project: LMC2-Synthetic-Occlusion-Benchmark (IRB-Free Deep Learning Study)
REM Description: Automated Workspace & Directory Structure Generator
REM ============================================================================

echo [INFO] Creating directory structure for LMC2 Deep Learning Research...

mkdir "data\01_raw_public\SHREC2017" 2>nul
mkdir "data\01_raw_public\FPHA" 2>nul
mkdir "data\01_raw_public\LeapGestureDB" 2>nul

mkdir "data\02_preprocessed_l2norm\cross_subject_train" 2>nul
mkdir "data\02_preprocessed_l2norm\cross_subject_val" 2>nul
mkdir "data\02_preprocessed_l2norm\cross_subject_test" 2>nul

mkdir "data\03_lmc2_degraded\Level1_Clean" 2>nul
mkdir "data\03_lmc2_degraded\Level2_Mild_Occlusion" 2>nul
mkdir "data\03_lmc2_degraded\Level3_Severe_Temporal_Dropout" 2>nul
mkdir "data\03_lmc2_degraded\Level4_LongTail_Imbalanced" 2>nul

mkdir "data\04_splits_metadata" 2>nul

mkdir "src\simulator" 2>nul
mkdir "src\models" 2>nul
mkdir "src\losses" 2>nul
mkdir "src\evaluation" 2>nul

mkdir "experiments\logs" 2>nul
mkdir "experiments\checkpoints" 2>nul
mkdir "experiments\results_csv" 2>nul
mkdir "experiments\latex_tables" 2>nul

echo [SUCCESS] Workspace directories created successfully!
pause
