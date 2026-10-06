#!/usr/bin/env python3
"""
Model Training & Evaluation Pipeline for ReviewLens
Trains Calibrated Linear SVM on 10,000 Balanced IMDB Dataset,
Evaluates with Classification Report & Confusion Matrix,
and Serializes Artifacts using Python Pickle.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.model_trainer import train_and_save_model

if __name__ == "__main__":
    print("=" * 60)
    print("[*] REVIEWLENS - MOVIE REVIEW SENTIMENT ANALYSIS")
    print("Training Calibrated Linear SVM on 10,000 Dataset...")
    print("=" * 60)
    metrics = train_and_save_model(
        data_path="data/movie_reviews_10k.csv",
        model_output_dir="models"
    )
    print("=" * 60)
    print("[+] Training Pipeline Complete! Evaluation Results:")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 60)
