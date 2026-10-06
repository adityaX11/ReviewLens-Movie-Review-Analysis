import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Suppress cross-version scikit-learn unpickling warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

try:
    from src.preprocess import preprocess
except ImportError:
    from preprocess import preprocess

class SentimentPredictor:
    """
    Production-grade predictor for movie review sentiment classification.
    Supports Champion model and switching between all trained models:
    - Linear SVM
    - Logistic Regression
    - Naive Bayes
    - Random Forest
    """
    AVAILABLE_MODELS = [
        "Champion (Best Selected)",
        "Linear SVM",
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest"
    ]

    def __init__(self, model_dir: Optional[str] = None, model_name: str = "Champion (Best Selected)"):
        self.model_dir = model_dir
        self.current_model_name = model_name
        self.model = None
        self.vectorizer = None
        self.base_svm = None
        self.metrics = None
        self.load_artifacts(model_dir, model_name)

    def _resolve_path(self, filename: str, custom_dir: Optional[str] = None) -> str:
        candidates = []
        if custom_dir:
            candidates.append(os.path.join(custom_dir, filename))
        candidates.extend([
            os.path.join("models", filename),
            os.path.join("..", "models", filename),
            filename
        ])
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not locate artifact '{filename}' in candidate paths: {candidates}")

    def load_artifacts(self, model_dir: Optional[str] = None, model_name: str = "Champion (Best Selected)"):
        """Load pickled model and TF-IDF vectorizer"""
        self.current_model_name = model_name
        vec_path = self._resolve_path("tfidf_vectorizer.pkl", model_dir)

        if model_name in ["Champion (Best Selected)", "Champion"]:
            model_file = "sentiment_model.pkl"
        else:
            safe_name = model_name.replace(" ", "_")
            model_file = f"model_{safe_name}.pkl"

        model_path = self._resolve_path(model_file, model_dir)

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Base SVM for word weights
        try:
            base_svm_path = self._resolve_path("base_svm_features.pkl", model_dir)
            with open(base_svm_path, "rb") as f:
                self.base_svm = pickle.load(f)
        except Exception:
            self.base_svm = None

    def set_model(self, model_name: str):
        """Switch active model dynamically"""
        if model_name != self.current_model_name:
            self.load_artifacts(self.model_dir, model_name)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single review with confidence metrics."""
        cleaned = preprocess(text)
        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 50.0,
                "positive_prob": 0.50,
                "negative_prob": 0.50,
                "clean_text": "",
                "sentiment_words": [],
                "model_used": self.current_model_name
            }

        X = self.vectorizer.transform([cleaned])
        pred = self.model.predict(X)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            pos_idx = classes.index("positive") if "positive" in classes else 1
            neg_idx = classes.index("negative") if "negative" in classes else 0
            pos_prob = float(probs[pos_idx])
            neg_prob = float(probs[neg_idx])
        else:
            score = self.model.decision_function(X)[0]
            pos_prob = 1 / (1 + np.exp(-score))
            neg_prob = 1.0 - pos_prob

        confidence = pos_prob if pred == "positive" else neg_prob

        # Extract influential keywords present in this specific review
        sentiment_words = []
        if self.base_svm is not None and hasattr(self.base_svm, "coef_"):
            words = cleaned.split()
            feature_names = self.vectorizer.get_feature_names_out()
            feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
            coefs = self.base_svm.coef_[0]

            for w in set(words):
                if w in feature_idx_map:
                    idx = feature_idx_map[w]
                    weight = float(coefs[idx])
                    sentiment_words.append({
                        "word": w,
                        "polarity": "positive" if weight > 0 else "negative",
                        "weight": round(weight, 3)
                    })
            sentiment_words = sorted(sentiment_words, key=lambda x: abs(x["weight"]), reverse=True)[:8]

        return {
            "sentiment": str(pred),
            "confidence": round(float(confidence) * 100, 2),
            "positive_prob": round(pos_prob * 100, 2),
            "negative_prob": round(neg_prob * 100, 2),
            "clean_text": cleaned,
            "sentiment_words": sentiment_words,
            "model_used": self.current_model_name
        }

    def predict_batch(self, df: pd.DataFrame, review_col: str = "review", movie_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Predict sentiment for a batch DataFrame, optionally filtered by movie.
        Returns the annotated DataFrame and comprehensive audience summary metrics.
        """
        data = df.copy()
        
        # Standardize column if needed
        if review_col not in data.columns:
            found = False
            for c in data.columns:
                if c.strip().lower() in ['review', 'reviews', 'text', 'comment', 'content']:
                    data['review'] = data[c]
                    found = True
                    break
            if not found:
                raise ValueError(f"CSV must contain a review column (found: {list(data.columns)})")
        else:
            data['review'] = data[review_col]

        # Optional movie filter
        movie_title_col = None
        for c in data.columns:
            if c.strip().lower() in ['movie_title', 'movie', 'title', 'film']:
                movie_title_col = c
                break

        if movie_name and movie_name.strip() and movie_title_col in data.columns:
            data = data[data[movie_title_col].astype(str).str.contains(movie_name.strip(), case=False, na=False)].copy()

        if len(data) == 0:
            return pd.DataFrame(), {
                "total": 0, "positive": 0, "negative": 0,
                "positive_pct": 0, "negative_pct": 0,
                "verdict": "No matching reviews found",
                "model_used": self.current_model_name
            }

        data['clean_review'] = data['review'].astype(str).apply(preprocess)
        X = self.vectorizer.transform(data['clean_review'])

        preds = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            classes = list(self.model.classes_)
            pos_idx = classes.index("positive") if "positive" in classes else 1
            pos_probs = probs[:, pos_idx]
            confs = [p if pred == 'positive' else (1.0 - p) for pred, p in zip(preds, pos_probs)]
        else:
            confs = [1.0] * len(preds)
            pos_probs = [1.0 if p == 'positive' else 0.0 for p in preds]

        data['predicted_sentiment'] = preds
        data['confidence_score'] = [round(float(c) * 100, 2) for c in confs]
        data['pos_probability'] = [round(float(p) * 100, 2) for p in pos_probs]

        total = len(data)
        pos_count = int((data['predicted_sentiment'] == 'positive').sum())
        neg_count = int((data['predicted_sentiment'] == 'negative').sum())
        pos_pct = round((pos_count / total) * 100, 2) if total > 0 else 0
        neg_pct = round((neg_count / total) * 100, 2) if total > 0 else 0
        avg_conf = round(float(np.mean(data['confidence_score'])), 2) if total > 0 else 0

        # Audience Verdict determination
        if pos_pct >= 85:
            verdict = "Universal Acclaim / Blockbuster Hit 🌟🌟🌟"
            status = "positive"
        elif pos_pct >= 65:
            verdict = "Generally Positive Audience Reception 👍"
            status = "positive"
        elif pos_pct >= 45:
            verdict = "Mixed / Polarized Audience Opinions ⚖️"
            status = "mixed"
        elif pos_pct >= 25:
            verdict = "Generally Negative / Critical Reception 👎"
            status = "negative"
        else:
            verdict = "Overwhelming Audience Backlash 🛑"
            status = "negative"

        summary = {
            "movie_name": movie_name if movie_name else "All Reviews / Dataset",
            "total_reviews": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "avg_confidence": avg_conf,
            "verdict": verdict,
            "verdict_status": status,
            "model_used": self.current_model_name
        }

        return data, summary
