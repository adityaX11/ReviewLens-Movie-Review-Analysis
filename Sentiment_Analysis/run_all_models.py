#!/usr/bin/env python3
"""
ReviewLens - Main execution script to train models and evaluate performance.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from src.model_trainer import train_and_save_model

def main():
    print("=" * 60)
    print("[*] REVIEWLENS - MOVIE REVIEW SENTIMENT ANALYSIS")
    print("=" * 60)
    data_path = "data/movie_reviews_10k.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "data", "movie_reviews_10k.csv")
    
    metrics = train_and_save_model(data_path=data_path, model_output_dir="models")
    print("\n[+] Training Complete:")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\nRun Streamlit dashboard with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()