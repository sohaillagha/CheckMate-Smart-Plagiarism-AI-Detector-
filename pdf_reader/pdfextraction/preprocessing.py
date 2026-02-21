import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def clean_extracted_text(text):
    text = re.split(r'\nreferences\n', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_tfidf(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    doc = nlp(" ".join(tokens))
    return " ".join([t.lemma_ for t in doc])

def preprocess_sbert(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]

def preprocess_ai_detection(text):
    return sent_tokenize(re.sub(r'\s+', ' ', text))
