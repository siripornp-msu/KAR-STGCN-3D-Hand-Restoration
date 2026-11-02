@echo off
REM ============================================================================
REM One-Click Automated Downloader & Canonical Converter for LMC2 Research
REM ============================================================================
echo [STEP 1] Creating workspace directories...
call setup_lmc2_workspace.bat

echo.
echo [STEP 2] Running Automated Public Benchmark Downloader and Extractor...
python lmc2_automated_downloader.py --base-dir "./LMC2_Benchmark_Dataset" --download ALL --seq-len 60

echo.
echo [DONE] All available datasets have been verified, L2-normalized, and exported to .npz!
pause
