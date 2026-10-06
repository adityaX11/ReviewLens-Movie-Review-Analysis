# 🎬 ReviewLens: Movie Review Sentiment Analysis Engine
### Production-Grade Multi-Algorithm Machine Learning & NLP Dashboard

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Champion Accuracy](https://img.shields.io/badge/Champion%20Accuracy-89.80%25-brightgreen.svg)]()
[![Champion R2](https://img.shields.io/badge/Champion%20R%C2%B2-0.6866-success.svg)]()
[![Serialization](https://img.shields.io/badge/Model%20Format-Pickle%20(.pkl)-blueviolet.svg)]()

> **ReviewLens** is an end-to-end NLP and multi-algorithm Machine Learning suite for movie audience sentiment classification, consensus forecasting, and batch analytics.

---

## 🌟 Executive Overview

**ReviewLens** translates thousands of unstructured audience movie reviews into clear sentiment insights. Built on a balanced **10,000 movie review dataset**, the system trains and benchmarks **4 distinct Machine Learning algorithms**:
1. **Linear SVM (Calibrated)** *(Champion Model)*
2. **Logistic Regression**
3. **Multinomial Naive Bayes**
4. **Random Forest**

The system automatically identifies the **Champion Model** based on top **Accuracy** and **R² Score**, saving all models and vectorizers using Python's standard **`pickle` (`.pkl`)** format.

The project features a production-ready **Streamlit Dashboard** allowing real-time single-review inference, audience sentiment forecasting by movie title, custom CSV batch ingestion, and side-by-side multi-algorithm diagnostic comparisons.

---

## 🏆 Multi-Model Benchmark Leaderboard

Trained and evaluated on the balanced 10,000 review benchmark (8,000 train samples, 2,000 holdout test samples):

| Rank | Machine Learning Model | Test Accuracy | R² Score | F1-Score | Precision | Recall | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Linear SVM (Calibrated)** | **89.80%** | **0.6866** | **0.8980** | **0.8980** | **0.8980** | **🏆 Champion (Selected)** |
| 🥈 | **Logistic Regression** | 89.30% | 0.6043 | 0.8930 | 0.8932 | 0.8930 | Evaluated & Available |
| 🥉 | **Multinomial Naive Bayes** | 88.05% | 0.5426 | 0.8805 | 0.8805 | 0.8805 | Evaluated & Available |
| 4️⃣ | **Random Forest** | 85.40% | 0.3735 | 0.8540 | 0.8545 | 0.8540 | Evaluated & Available |

> **Automated Champion Selection**: Linear SVM achieved the highest classification accuracy (**89.80%**) and the strongest coefficient of determination (**R² = 0.6866**), making it the primary model for audience consensus forecasting. All 4 models remain fully serialized and selectable on-the-fly in the Streamlit UI.

---

## 🚀 Key Features

### 1. 🎬 Movie Audience Sentiment Analyzer
- Search or select from popular indexed movie titles (e.g., *Inception*, *The Dark Knight*, *Interstellar*, *Titanic*, *Avatar*).
- Instant **Audience Verdict Banner** (*Universal Acclaim 🌟*, *Generally Positive 👍*, *Mixed ⚖️*, *Critical Backlash 🛑*).
- Positive vs. negative percentage splits, review counts, and average confidence levels.
- Interactive Plotly donut charts and confidence distribution histograms.
- Filterable audience review table with sentiment badges.

### 2. ⚡ Real-Time Review Tester
- Instant classification for single or multi-line movie reviews with your chosen ML algorithm.
- Calibrated posterior probability meter (0% to 100%).
- Extraction of high-weight positive and negative indicator keywords present in the text.
- One-click sample test buttons (Positive, Negative, Nuanced).

### 3. 📁 Custom CSV Batch Analyzer & Audience Reporting
- Upload any movie reviews CSV file for automated mass sentiment prediction.
- Auto-detection of review columns (`review`, `reviews`, `text`, `content`).
- Optional filtering by movie title within the uploaded file.
- Automated **Confusion Matrix & Classification Report** when true sentiment labels are present.
- Downloadable enriched dataset with predicted labels, confidence scores, and probabilities.
- Built-in **Download Sample CSV Template** button for instant user onboarding.

### 4. 📊 Multi-Model Comparison & Diagnostics
- Side-by-side grouped bar chart comparing all 4 models on Accuracy, R² Score, and F1-Score.
- Interactive Confusion Matrix heatmaps and classification reports for each of the 4 models.
- Most influential positive and negative words (TF-IDF feature weights).

### 5. 💾 Pickle Model Serialization
- Standard Python `pickle` serialization for all 4 trained classifiers and the TF-IDF vectorizer:
  - `models/sentiment_model.pkl` (Champion Linear SVM)
  - `models/model_Linear_SVM.pkl`
  - `models/model_Logistic_Regression.pkl`
  - `models/model_Naive_Bayes.pkl`
  - `models/model_Random_Forest.pkl`
  - `models/tfidf_vectorizer.pkl`

---

## 🏗️ Project Architecture

```
ReviewLens-Movie-Review-Analysis/
├── app.py                         # Production Streamlit Web Dashboard
├── train.py                       # Multi-Model Training & Evaluation Pipeline
├── requirements.txt               # Production Python Dependencies
├── README.md                      # Complete Project Documentation
├── .gitignore                     # Ignores bytecode and cache files
├── data/
│   ├── movie_reviews_10k.csv      # 10,000 Balanced IMDB Dataset
│   └── sample_movie_reviews.csv   # Downloadable 25-Row Sample Test CSV
├── models/
│   ├── sentiment_model.pkl        # Champion Pickled Model (Linear SVM)
│   ├── model_Linear_SVM.pkl       # Pickled Linear SVM
│   ├── model_Logistic_Regression.pkl # Pickled Logistic Regression
│   ├── model_Naive_Bayes.pkl      # Pickled Naive Bayes
│   ├── model_Random_Forest.pkl    # Pickled Random Forest
│   ├── tfidf_vectorizer.pkl       # Pickled TF-IDF Vectorizer
│   ├── base_svm_features.pkl      # Pickled LinearSVC weights for token attribution
│   └── model_metrics.json         # Complete Leaderboard & Diagnostics Data
└── src/
    ├── __init__.py
    ├── preprocess.py              # Robust NLP Text Normalizer (HTML, Lemmatizer, Negation)
    ├── model_trainer.py           # Multi-Model Training logic & pickle exporter
    └── predictor.py               # Multi-Model inference engine with confidence scoring
```

---

## 📋 CSV Dataset Format Specification

When uploading your own dataset in the Streamlit app:

| Column Name | Required | Type | Example / Description |
| :--- | :---: | :--- | :--- |
| **`review`** | **Mandatory** | String | The audience review text. *(Aliases: `reviews`, `text`, `content`)* |
| **`movie_title`** | Optional | String | Name of the movie *(e.g. `Inception`)* for filtering & consensus analysis. |
| **`sentiment`** | Optional | String | Ground truth label (`positive` or `negative`). Enables automated evaluation & confusion matrix! |

### Example CSV:
```csv
movie_title,review,sentiment
Inception,"A visual and intellectual masterpiece that keeps you engaged.",positive
Titanic,"Heartbreaking story with unforgettable performances and soundtrack.",positive
The Room,"Horrible acting, nonsensical dialogue, and bizarre editing.",negative
Gladiator,"Russell Crowe delivers an iconic performance with epic battle scenes.",positive
```

---

## ⚡ Quick Start & Deployment Guide

### 1. Clone the Repository
```bash
git clone https://github.com/adityaX11/ReviewLens-Movie-Review-Analysis.git
cd ReviewLens-Movie-Review-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain All 4 Models
All 4 models are pre-trained and serialized as `.pkl`. To retrain and benchmark from scratch:
```bash
python train.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack & Methodologies

- **Language**: Python 3.9+
- **Frontend / UI**: Streamlit, Plotly Express & Graph Objects, Custom Modern Cinema CSS
- **Machine Learning**: scikit-learn (`LinearSVC`, `LogisticRegression`, `MultinomialNB`, `RandomForestClassifier`, `CalibratedClassifierCV`, `TfidfVectorizer`)
- **Natural Language Processing**: NLTK (`stopwords`, `WordNetLemmatizer`, `word_tokenize`, regex)
- **Serialization**: Python standard `pickle` (`.pkl`)
- **Data Engineering**: Pandas, NumPy
