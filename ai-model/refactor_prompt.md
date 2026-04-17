Role: You are an expert Machine Learning Engineer specializing in Recommender Systems and MLOps.

Context: I am working on a recommendation system. Currently, the code in the /ai-model directory is cluttered and needs a complete refactor to meet production standards. We are using a Content-Based Filtering approach.

    Dataset: product_clean.csv (already cleaned).

    Goal: Clean up the codebase, ensure correct ML logic, and implement evaluation/testing pipelines.

Tasks:

    Code Cleanup: Remove any redundant, commented-out, or legacy code in the /ai-model folder.

    Implementation (Content-Based Filtering):

        Refactor the model to use product_clean.csv.

        Implement feature engineering (e.g., TF-IDF or Embedding) on product metadata.

        Ensure the similarity computation is efficient for production.

    Evaluation & Splitting:

        Create a proper logic for splitting data (Train/Test set).

        Implement evaluation metrics relevant to Content-Based systems (e.g., Precision@K, Recall@K, or Cosine Similarity distribution).

    Production Standards:

        Refactor code into modular Python scripts (e.g., preprocess.py, model.py, evaluate.py).

        Use Type Hinting and Docstrings for all functions.

        Implement logging and basic error handling.

        Create a main.py or a dedicated class to orchestrate the pipeline.
Project Exploration:
   * First, explore the current /ai-model directory structure using available tools (ls, tree, cat files).
   * Analyze existing files before refactoring.
   * Only modify files after understanding the current implementation.
Output Requirement:
Please provide the refactored code structure and the full content of the key files. Explain your reasoning for the chosen architecture.
