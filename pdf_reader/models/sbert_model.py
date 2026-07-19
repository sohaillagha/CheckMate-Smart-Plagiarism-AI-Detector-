import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "sbert", "sbert_pairs.csv")
CACHE_FILE = os.path.join(BASE_DIR, "datasets", "sbert", "sbert_cache.pt")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Cache: encode dataset embeddings only once
_cached_dataset_embeddings = None

def load_sbert_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        # Ensure we have the required columns
        if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
            if df.shape[1] >= 2:
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
        if os.path.exists(CACHE_FILE):
            print("  ⚡ Loading SBERT cache from disk (super fast)...")
            try:
                _cached_dataset_embeddings = torch.load(CACHE_FILE, weights_only=False)
                print("  ✅ SBERT loaded from disk cache.")
            except Exception as e:
                print(f"  ⚠️ Cache corrupted, rebuilding... ({e})")
                _cached_dataset_embeddings = None

        if _cached_dataset_embeddings is None:
            print(f"  ⏳ First run: encoding SBERT dataset (this may take up to 20 minutes for 100k+ sentences)...")
            dataset_sentences = load_sbert_dataset()
            if not dataset_sentences:
                return 0.0, []
            
            # Show a progress bar by encoding in batches and concatenating
            print(f"    Encoding {len(dataset_sentences)} sentences using neural network...")
            _cached_dataset_embeddings = model.encode(dataset_sentences, convert_to_tensor=True, show_progress_bar=True)
            
            try:
                print("  💾 Saving SBERT cache to disk for future runs...")
                torch.save(_cached_dataset_embeddings, CACHE_FILE)
            except Exception as e:
                print(f"  ⚠️ Could not save SBERT cache: {e}")
                
            print("  ✅ SBERT dataset cached.")

    # Encode the new user uploaded document
    emb1 = model.encode(sentences, convert_to_tensor=True)
    
    scores = util.cos_sim(emb1, _cached_dataset_embeddings)
    
    # Calculate percentage of sentences that match the dataset
    best_matches_per_sentence = scores.max(dim=1).values
    threshold = 0.85 
    
    matched_indices = []
    for i, score in enumerate(best_matches_per_sentence):
        if score > threshold:
            matched_indices.append(i)
            
    final_score = float(len(matched_indices) / len(sentences))
    return final_score, matched_indices
