import os
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

# Ensure we can import from pdfextraction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdfextraction.preprocessing import preprocess_tfidf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "tfidf")
CACHE_FILE = os.path.join(DATASET_DIR, "tfidf_cache.pkl")

# Cache: load corpus, vectorizer, and matrix only once
_cached_corpus = None
_cached_vectorizer = None
_cached_dataset_matrix = None

def load_tfidf_dataset():
    global _cached_corpus, _cached_vectorizer, _cached_dataset_matrix
    if _cached_corpus is not None:
        return _cached_corpus

    if os.path.exists(CACHE_FILE):
        print("  ⚡ Loading TF-IDF cache from disk (super fast)...")
        try:
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                _cached_corpus = cache_data['corpus']
                _cached_vectorizer = cache_data['vectorizer']
                _cached_dataset_matrix = cache_data['matrix']
            print("  ✅ TF-IDF loaded from disk cache.")
            return _cached_corpus
        except Exception as e:
            print(f"  ⚠️ Cache corrupted, rebuilding... ({e})")

    print("  ⏳ First run: loading and fitting TF-IDF corpus (this may take a few minutes)...")
    corpus = []
    
    if not os.path.exists(DATASET_DIR):
        _cached_corpus = []
        _cached_vectorizer = TfidfVectorizer()
        _cached_dataset_matrix = _cached_vectorizer.fit_transform([""])
        return _cached_corpus
        
    txt_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")]
    total_files = len(txt_files)
    
    for i, file in enumerate(txt_files):
        if i % 10 == 0 or i == total_files - 1:
            print(f"    processing paper {i+1}/{total_files}...")
            
        file_path = os.path.join(DATASET_DIR, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    sentences = sent_tokenize(content)
                    for s in sentences:
                        processed = preprocess_tfidf(s)
                        if processed.strip():
                            corpus.append(processed)
        except Exception as e:
            print(f"Error reading dataset file {file}: {e}")
    
    _cached_corpus = corpus
    _cached_vectorizer = TfidfVectorizer()
    if _cached_corpus:
        _cached_dataset_matrix = _cached_vectorizer.fit_transform(_cached_corpus)
    else:
        _cached_dataset_matrix = _cached_vectorizer.fit_transform([""])
        
    # Save to disk
    try:
        print("  💾 Saving TF-IDF cache to disk for future runs...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                'corpus': _cached_corpus,
                'vectorizer': _cached_vectorizer,
                'matrix': _cached_dataset_matrix
            }, f)
    except Exception as e:
        print(f"  ⚠️ Could not save cache: {e}")
        
    print("  ✅ TF-IDF corpus and vectorizer cached.")
    return _cached_corpus

def check_tfidf_similarity(sentences):
    """
    Checks similarity for a list of sentences against the TF-IDF dataset.
    Returns (score, matched_indices)
    """
    if not sentences:
        return 0.0, []

    processed_sentences = [preprocess_tfidf(s) for s in sentences]
    valid_indices = [i for i, s in enumerate(processed_sentences) if s.strip()]
    if not valid_indices:
        return 0.0, []
        
    valid_processed = [processed_sentences[i] for i in valid_indices]

    dataset_sentences = load_tfidf_dataset()
    if not dataset_sentences:
         return 0.0, []

    # Transform input using pre-fitted vectorizer
    input_matrix = _cached_vectorizer.transform(valid_processed)
    
    # Compute similarities against cached dataset matrix
    similarities = cosine_similarity(input_matrix, _cached_dataset_matrix)
    
    threshold = 0.75 
    matched_indices = []
    
    for idx_in_valid, row_scores in enumerate(similarities):
        max_score = max(row_scores) if len(row_scores) > 0 else 0
        if max_score > threshold:
            original_idx = valid_indices[idx_in_valid]
            matched_indices.append(original_idx)
            
    score = len(matched_indices) / len(valid_indices) if valid_indices else 0.0
    return score, matched_indices
