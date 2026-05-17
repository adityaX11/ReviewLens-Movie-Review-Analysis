from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_tfidf_features(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def save_svm_model_and_vectorizer(model, vectorizer, model_path='svm_sentiment_model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_svm_model_and_vectorizer(model_path='svm_sentiment_model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer