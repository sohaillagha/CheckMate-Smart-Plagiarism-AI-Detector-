import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

# Ensure we can import from pdfextraction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdfextraction.preprocessing import preprocess_tfidf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "tfidf")

# Cache: load corpus only once
_cached_corpus = None

def load_tfidf_dataset():
    global _cached_corpus
    if _cached_corpus is not None:
        return _cached_corpus  # Return cached corpus

    print("  ⏳ First run: loading TF-IDF corpus (will be cached)...")
    corpus = []
    # Split dataset documents into sentences to allow granular matching
    if not os.path.exists(DATASET_DIR):
        _cached_corpus = []
        return _cached_corpus
        
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(DATASET_DIR, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        # Split into sentences first
                        sentences = sent_tokenize(content)
                        # Preprocess each sentence
                        for s in sentences:
                            processed = preprocess_tfidf(s)
                            if processed.strip(): # Only add non-empty
                                corpus.append(processed)
            except Exception as e:
                print(f"Error reading dataset file {file}: {e}")
    
    _cached_corpus = corpus
    print("  ✅ TF-IDF corpus cached.")
    return _cached_corpus

def check_tfidf_similarity(sentences):
    """
    Checks similarity for a list of sentences against the TF-IDF dataset.
    Returns (score, matched_indices)
    """
    if not sentences:
        return 0.0, []

    # sentences is expected to be a list of raw text strings of sentences
    # We must preprocess them individually
    processed_sentences = [preprocess_tfidf(s) for s in sentences]
    # Filter out empty processed sentences (e.g. only stopwords)
    valid_indices = [i for i, s in enumerate(processed_sentences) if s.strip()]
    if not valid_indices:
        return 0.0, []
        
    valid_processed = [processed_sentences[i] for i in valid_indices]

    dataset_sentences = load_tfidf_dataset()
    if not dataset_sentences:
         return 0.0, []

    # Fit on dataset + input to ensure vocabulary coverage
    all_texts = dataset_sentences + valid_processed
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split back into dataset content and input sentences
    n_dataset = len(dataset_sentences)
    dataset_matrix = tfidf_matrix[:n_dataset]
    input_matrix = tfidf_matrix[n_dataset:]
    
    # helper for computing similarities
    # outcome: (n_input_sentences, n_dataset_sentences)
    similarities = cosine_similarity(input_matrix, dataset_matrix)
    
    # Find sentences that match ANY sentence in the dataset with high similarity
    threshold = 0.75 # Threshold for "Copied" (lower because preprocessing is aggressive)
    matched_indices = []
    
    for idx_in_valid, row_scores in enumerate(similarities):
        max_score = max(row_scores) if len(row_scores) > 0 else 0
        if max_score > threshold:
            original_idx = valid_indices[idx_in_valid]
            matched_indices.append(original_idx)
            
    # Use valid sentences as denominator (not ALL sentences — junk like TOC/headers shouldn't dilute score)
    score = len(matched_indices) / len(valid_indices) if valid_indices else 0.0
    return score, matched_indices
