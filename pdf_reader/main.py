from pdfextraction.reader import extract_pdf_text
from pdfextraction.preprocessing import (
    clean_extracted_text,
    preprocess_tfidf,
    preprocess_sbert,
    preprocess_ai_detection
)

from models.tfidf_model import check_tfidf_similarity
from models.sbert_model import check_sbert_similarity
from models.ai_detection_model import train_ai_detector, predict_ai_probability

import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the PDF file
PDF_PATH = os.path.join(
    script_dir, 
    "userinput", 
    "paper_001.pdf"
)

def main():
    print("📄 Reading PDF...")
    try:
        raw_text = extract_pdf_text(PDF_PATH)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    cleaned_text = clean_extracted_text(raw_text)

    print("🔍 TF-IDF plagiarism check (Copied Detection)...")
    tfidf_text = preprocess_tfidf(cleaned_text)
    tfidf_score = check_tfidf_similarity(tfidf_text)

    print("🧠 SBERT semantic similarity (Paraphrased Detection)...")
    sbert_sentences = preprocess_sbert(cleaned_text)
    sbert_score = check_sbert_similarity(sbert_sentences)

    print("🤖 AI content detection...")
    train_ai_detector()
    ai_sentences = preprocess_ai_detection(cleaned_text)
    try:
        ai_prob = predict_ai_probability(ai_sentences)
    except Exception as e:
        print(f"AI Detection Error: {e}")
        ai_prob = 0.0

    print("✍️  Originality calculation...")
    
    # Weighted Scoring Formula
    # Weights: Copied=0.4, Paraphrased=0.3, AI=0.3
    
    w_copied = 0.4
    w_paraphrased = 0.3
    w_ai = 0.3

    weighted_penalty = (
        (w_copied * tfidf_score) + 
        (w_paraphrased * sbert_score) + 
        (w_ai * ai_prob)
    )

    originality_score = max(0.0, 1.0 - weighted_penalty)

    print("\n====== RESULTS ======")
    print(f"1. Copied Detection (TF-IDF)      : {tfidf_score:.2f} (Weight: {w_copied})")
    print(f"2. Paraphrased Detection (SBERT)  : {sbert_score:.2f} (Weight: {w_paraphrased})")
    print(f"3. AI Detection (Probability)     : {ai_prob:.2f} (Weight: {w_ai})")
    print("-" * 30)
    print(f"FINAL ORIGINALITY SCORE           : {originality_score:.2%}")

if __name__ == "__main__":
    main()
