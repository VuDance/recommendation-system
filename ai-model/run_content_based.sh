#!/bin/bash
# Content-Based Filtering Pipeline Runner
# This script runs the complete content-based filtering pipeline:
# 1. Generate test set from products_clean.csv
# 2. Evaluate content-based filtering

set -e

echo "============================================"
echo "  Content-Based Filtering Pipeline"
echo "============================================"

# Configuration
CSV_PATH="${CSV_PATH:-data/dataset/products_clean.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed_data}"
N_QUERIES="${N_QUERIES:-1000}"
K_VALUES="${K_VALUES:-5 10 20 50}"

echo ""
echo "Configuration:"
echo "  CSV Path: $CSV_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Test Queries: $N_QUERIES"
echo "  K Values: $K_VALUES"
echo ""

# Step 1: Generate test set
echo "============================================"
echo "  Step 1: Generate Test Set"
echo "============================================"
python data/generate_content_test_set.py \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-queries "$N_QUERIES"

echo ""

# Step 2: Evaluate content-based filtering
echo "============================================"
echo "  Step 2: Evaluate Content-Based Filtering"
echo "============================================"
python main/evaluate_content_based.py \
    --data-dir "$OUTPUT_DIR" \
    --k-values $K_VALUES

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"