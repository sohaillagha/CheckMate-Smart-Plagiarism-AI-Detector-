import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "sbert", "sbert_pairs.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Cache: encode dataset embeddings only once
_cached_dataset_embeddings = None

def load_sbert_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        # Ensure we have the required columns
        if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
            # Fallback for headerless or differently named columns if repair failed silently
            if df.shape[1] >= 2:
                 # If columns are missing/renamed, assume first 2 cols
                cols = df.columns.tolist()
                df.rename(columns={cols[0]: 'sentence1', cols[1]: 'sentence2'}, inplace=True)

        if 'sentence1' in df.columns and 'sentence2' in df.columns:
             sentences = pd.concat([df['sentence1'], df['sentence2']]).astype(str).unique().tolist()
             return sentences
        else:
             print("Error: SBERT CSV missing required columns.")
             return []
    except Exception as e:
        print(f"Error loading SBERT dataset: {e}")
        return []

def check_sbert_similarity(sentences):
    global _cached_dataset_embeddings

    if not sentences:
        return 0.0, []
        
    # Encode dataset only on first call, then reuse cached embeddings
    if _cached_dataset_embeddings is None:
        print("  ⏳ First run: encoding SBERT dataset (will be cached)...")
        dataset_sentences = load_sbert_dataset()
        if not dataset_sentences:
            return 0.0, []
        _cached_dataset_embeddings = model.encode(dataset_sentences, convert_to_tensor=True)
        print("  ✅ SBERT dataset cached.")

    emb1 = model.encode(sentences, convert_to_tensor=True)
    
    scores = util.cos_sim(emb1, _cached_dataset_embeddings)
    
    # Calculate percentage of sentences that match the dataset
    # matching_threshold = 0.95 means 95% similarity or higher counts as a "match"
    best_matches_per_sentence = scores.max(dim=1).values
    threshold = 0.85  # Lowered from 0.95 — PDF re-extraction can introduce minor text differences
    
    matched_indices = []
    for i, score in enumerate(best_matches_per_sentence):
        if score > threshold:
            matched_indices.append(i)
            
    final_score = float(len(matched_indices) / len(sentences))
    return final_score, matched_indices
