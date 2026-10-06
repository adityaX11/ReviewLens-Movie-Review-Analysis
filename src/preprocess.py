import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configure private, secure NLTK data directory to prevent world-writable warnings
NLTK_DATA_DIR = os.path.expanduser('~/nltk_data')
try:
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if hasattr(os, 'chmod'):
        os.chmod(NLTK_DATA_DIR, 0o700)
except Exception:
    pass

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

REQUIRED_NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]

for check_path, pkg_name in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(check_path)
    except LookupError:
        try:
            nltk.download(pkg_name, download_dir=NLTK_DATA_DIR, quiet=True)
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
