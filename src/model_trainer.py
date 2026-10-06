import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    confusion_matrix,
    classification_report
)

try:
    from src.preprocess import preprocess
except ImportError:
    from preprocess import preprocess

def train_and_evaluate_all_models(
    data_path: str = "data/movie_reviews_10k.csv",
    model_output_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train multiple ML algorithms on the 10,000 dataset:
    - Linear SVM (Calibrated)
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    
    Evaluates Accuracy, R2 Score, F1, Precision, Recall, and Confusion Matrix.
    Selects the champion model and saves all models via PICKLE.
    """
    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        alt_path = os.path.join("Sentiment_Analysis", data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples. Distribution:\n{df['sentiment'].value_counts()}")

    print("Preprocessing review texts...")
    df['clean_review'] = df['review'].astype(str).apply(preprocess)
    
    valid_mask = df['clean_review'].str.strip() != ""
    df = df[valid_mask].copy()

    X = df['clean_review']
    y = df['sentiment']

    # Stratified Train/Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Fitting TF-IDF Vectorizer (n-grams (1,2), max_features=10,000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Numerical binary mapping for R2 Score calculation
    label_map = {'negative': 0, 'positive': 1}
    y_test_num = y_test.map(label_map).to_numpy()

    # Define the 4 ML Algorithms
    models_config = {
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=random_state, max_iter=2000),
            cv=3
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        "Naive Bayes": MultinomialNB(
            alpha=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=random_state,
            n_jobs=-1
        )
    }

    # Also fit a standalone LinearSVC to extract interpretable word coefficients
    standalone_svm = LinearSVC(C=1.0, random_state=random_state, max_iter=2000)
    standalone_svm.fit(X_train, y_train)

    trained_models = {}
    model_results = {}
    leaderboard = []

    for name, clf in models_config.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        trained_models[name] = clf

        # Predictions
        y_pred = clf.predict(X_test)
        
        # Probabilities for R2 score computation
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            classes = list(clf.classes_)
            pos_idx = classes.index("positive") if "positive" in classes else 1
            y_pred_prob = probs[:, pos_idx]
        else:
            y_pred_prob = np.where(y_pred == "positive", 1.0, 0.0)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # R2 score between true binary labels and predicted probabilities
        r2 = r2_score(y_test_num, y_pred_prob)

        cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"[{name}] Accuracy: {acc*100:.2f}% | R2 Score: {r2:.4f} | F1: {f1:.4f}")

        model_results[name] = {
            "accuracy": round(float(acc), 4),
            "r2_score": round(float(r2), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "confusion_matrix": {
                "labels": ["positive", "negative"],
                "matrix": cm.tolist()
            },
            "classification_report": report
        }

        leaderboard.append({
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "r2_score": round(float(r2), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4)
        })

    # Sort leaderboard by Accuracy descending, then R2 score
    leaderboard = sorted(leaderboard, key=lambda x: (x["accuracy"], x["r2_score"]), reverse=True)
    champion_name = leaderboard[0]["model_name"]
    champion_model = trained_models[champion_name]
    print(f"\n========================================================")
    print(f"CHAMPION MODEL SELECTED: {champion_name}")
    print(f"Accuracy: {leaderboard[0]['accuracy']*100:.2f}% | R2: {leaderboard[0]['r2_score']:.4f}")
    print(f"========================================================")

    # Feature Importance / Keyword Extraction from standalone SVM
    feature_names = vectorizer.get_feature_names_out()
    coefs = standalone_svm.coef_[0]
    top_pos_idx = np.argsort(coefs)[-25:][::-1]
    top_neg_idx = np.argsort(coefs)[:25]

    top_pos_features = [
        {"word": str(feature_names[i]), "weight": float(coefs[i])} for i in top_pos_idx
    ]
    top_neg_features = [
        {"word": str(feature_names[i]), "weight": float(coefs[i])} for i in top_neg_idx
    ]

    metrics_payload = {
        "dataset_size": int(len(df)),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "champion_model": champion_name,
        "leaderboard": leaderboard,
        "all_models_metrics": model_results,
        # Default top-level metrics point to champion
        "accuracy": leaderboard[0]["accuracy"],
        "r2_score": leaderboard[0]["r2_score"],
        "f1_score": leaderboard[0]["f1_score"],
        "precision": leaderboard[0]["precision"],
        "recall": leaderboard[0]["recall"],
        "confusion_matrix": model_results[champion_name]["confusion_matrix"],
        "classification_report": model_results[champion_name]["classification_report"],
        "top_positive_features": top_pos_features,
        "top_negative_features": top_neg_features
    }

    # Save artifacts using PICKLE mechanism
    save_dirs = [model_output_dir, os.path.join("Sentiment_Analysis", model_output_dir)]
    for target_dir in save_dirs:
        os.makedirs(target_dir, exist_ok=True)

        # 1. Champion Model as default sentiment_model.pkl
        with open(os.path.join(target_dir, "sentiment_model.pkl"), "wb") as f:
            pickle.dump(champion_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. Vectorizer
        with open(os.path.join(target_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 3. Base SVM for word weights
        with open(os.path.join(target_dir, "base_svm_features.pkl"), "wb") as f:
            pickle.dump(standalone_svm, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Save all individual models
        for name, mdl in trained_models.items():
            safe_name = name.replace(" ", "_")
            fname = os.path.join(target_dir, f"model_{safe_name}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(mdl, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 5. Save comprehensive metrics JSON
        with open(os.path.join(target_dir, "model_metrics.json"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

    print("Successfully trained, evaluated, and pickled all 4 models!")
    return metrics_payload

# Alias for backward compatibility
train_and_save_model = train_and_evaluate_all_models

if __name__ == "__main__":
    train_and_evaluate_all_models()
