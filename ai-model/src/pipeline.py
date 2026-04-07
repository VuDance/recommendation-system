"""Main entry point for content-based filtering pipeline.

Usage:
    python -m src.main --csv-path data/dataset/products_clean.csv
    python -m src.main --mode preprocess
    python -m src.main --mode evaluate
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from src.evaluate import print_evaluation, evaluate_content_model
from src.model import cosine_similarity_sparse, retrieve_similar

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
