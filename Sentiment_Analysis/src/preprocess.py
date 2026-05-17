import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download all required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # Add the missing resource
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet
except Exception as e:
    print(f"Warning: Some NLTK resources could not be downloaded: {e}")

# Initialize stopwords and lemmatizer
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"Warning: Could not initialize NLTK components: {e}")
    # Fallback to basic preprocessing
    stop_words = set()
    lemmatizer = None

def preprocess(text):
    # Ensure input is a string
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Simple tokenization as fallback if NLTK fails
    try:
        tokens = word_tokenize(text)
    except:
        # Fallback to simple split if NLTK tokenization fails
        tokens = text.split()

    # Lemmatize and remove stopwords
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    else:
        # Fallback without lemmatization
        tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)
