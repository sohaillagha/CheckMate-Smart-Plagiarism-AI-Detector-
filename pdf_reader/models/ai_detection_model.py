import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "ai_detection", "ai_detection.csv")

vectorizer = TfidfVectorizer()
model = LogisticRegression()
_trained = False

def train_ai_detector():
    global _trained
    if _trained:
        return  # Already trained, skip
    print("  ⏳ First run: training AI detector (will be cached)...")
    df = pd.read_csv(DATASET_PATH)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    model.fit(X, y)
    _trained = True
    print("  ✅ AI detector cached.")

def predict_ai_probability(sentences):
    if not sentences:
        return 0.0, []
        
    # Vectorize all sentences
    X = vectorizer.transform(sentences)
    probs = model.predict_proba(X)[:, 1] # Probability of class 1 (AI)
    
    matched_indices = []
    threshold = 0.5 # or higher if sentence-level is noisy
    for i, p in enumerate(probs):
        if p > threshold:
            matched_indices.append(i)
            
    score = float(len(matched_indices) / len(sentences))
    return score, matched_indices
