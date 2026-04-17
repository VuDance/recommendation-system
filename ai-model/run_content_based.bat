@echo off
REM Content-Based Filtering Pipeline Runner (Windows)
REM This script runs the complete content-based filtering pipeline:
REM 1. Generate test set from products_clean.csv
REM 2. Evaluate content-based filtering

echo ============================================
echo   Content-Based Filtering Pipeline
echo ============================================

REM Configuration
set CSV_PATH=data/dataset/products_clean.csv
set OUTPUT_DIR=data/processed_data
set N_QUERIES=1000

echo.
echo Configuration:
echo   CSV Path: %CSV_PATH%
echo   Output Dir: %OUTPUT_DIR%
echo   Test Queries: %N_QUERIES%
echo.

REM Step 1: Generate test set
echo ============================================
echo   Step 1: Generate Test Set
echo ============================================
python data\generate_content_test_set.py --csv-path "%CSV_PATH%" --output-dir "%OUTPUT_DIR%" --n-queries %N_QUERIES%

echo.

REM Step 2: Evaluate content-based filtering
echo ============================================
echo   Step 2: Evaluate Content-Based Filtering
echo ============================================
python main\evaluate_content_based.py --data-dir "%OUTPUT_DIR%" --k-values 5 10 20 50

echo.
echo ============================================
echo   Pipeline Complete!
echo ============================================
pause
