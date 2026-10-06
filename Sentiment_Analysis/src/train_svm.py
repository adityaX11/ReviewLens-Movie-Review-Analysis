import os
import sys

# Add src and parent to path if needed
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model_trainer import train_and_save_model

if __name__ == '__main__':
    train_and_save_model(
        data_path="data/movie_reviews_10k.csv",
        model_output_dir="models"
    )
