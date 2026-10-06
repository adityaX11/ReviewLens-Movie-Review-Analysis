import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are available
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Initialize stopwords and lemmatizer safely
try:
    stop_words = set(stopwords.words('english'))
    # Keep negation words as they are critical for sentiment analysis
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'barely', 'hardly', 'scarcely', 'rarely', "don't", "didn't", "wasn't", "weren't", "won't", "isn't", "aren't"}
    stop_words = stop_words - negation_words
    lemmatizer = WordNetLemmatizer()
except Exception:
    stop_words = set()
    lemmatizer = None

def preprocess(text: str) -> str:
    """
    Clean and normalize raw movie review text for NLP feature extraction.
    1. Removes HTML markup (e.g. <br />, <b>).
    2. Strips URLs.
    3. Converts to lowercase.
    4. Removes punctuation and numeric digits.
    5. Tokenizes and lemmatizes words while filtering non-negation stopwords.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove HTML tags (common in IMDB reviews)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove punctuation & digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Lemmatize and remove stopwords
    if lemmatizer:
        cleaned_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if len(token) > 1 and token not in stop_words
        ]
    else:
        cleaned_tokens = [
            token
            for token in tokens
            if len(token) > 1 and token not in stop_words
        ]

    return ' '.join(cleaned_tokens)
