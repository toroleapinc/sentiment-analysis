"""Text preprocessing for sentiment data."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text, stem=True):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)
