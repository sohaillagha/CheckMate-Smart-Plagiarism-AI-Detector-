from pdfextraction.reader import extract_pdf_text
from pdfextraction.preprocessing import clean_extracted_text
from nltk.tokenize import sent_tokenize

from models.tfidf_model import check_tfidf_similarity
from models.sbert_model import check_sbert_similarity
from models.ai_detection_model import train_ai_detector, predict_ai_probability

import os

def analyze_pdf(pdf_path):
    """
    Analyzes the PDF and returns a dictionary of scores and matched content for highlighting.
    """
    results = {
        "error": None,
        "tfidf_score": 0.0,
        "sbert_score": 0.0,
        "ai_prob": 0.0,
        "originality_score": 0.0,
        "raw_text_snippet": "",
        "content_analysis": [] 
    }

    print(f"📄 Reading PDF: {pdf_path}")
    try:
        raw_text = extract_pdf_text(pdf_path)
        results["raw_text_snippet"] = raw_text[:500] + "..."
    except Exception as e:
        results["error"] = f"Error reading PDF: {e}"
        return results

    cleaned_text = clean_extracted_text(raw_text)
    
    # Split into sentences for granular analysis
    sentences = [s.strip() for s in sent_tokenize(cleaned_text) if len(s.strip()) > 0]
    
    if not sentences:
         results["error"] = "No text found in PDF."
         return results

    # 1. Copied Detection
    print("🔍 TF-IDF plagiarism check...")
    copied_indices = []
    try:
        # Now returns (score, indices)
        results["tfidf_score"], copied_indices = check_tfidf_similarity(sentences)
    except Exception as e:
        print(f"TF-IDF Error: {e}")

    # 2. Paraphrased Detection
    print("🧠 SBERT semantic similarity...")
    paraphrased_indices = []
    try:
        # Now returns (score, indices)
        results["sbert_score"], paraphrased_indices = check_sbert_similarity(sentences)
    except Exception as e:
         print(f"SBERT Error: {e}")

    # 3. AI Detection
    print("🤖 AI content detection...")
    ai_indices = []
    try:
        train_ai_detector()
        # Now returns (score, indices)
        results["ai_prob"], ai_indices = predict_ai_probability(sentences)
    except Exception as e:
        print(f"AI Detection Error: {e}")


    # Calculate Final Score
    w_copied = 0.4
    w_paraphrased = 0.3
    w_ai = 0.3

    weighted_penalty = (
        (w_copied * results["tfidf_score"]) + 
        (w_paraphrased * results["sbert_score"]) + 
        (w_ai * results["ai_prob"])
    )

    results["originality_score"] = max(0.0, 1.0 - weighted_penalty)
    
    # Construct Content Analysis for Highlighting
    # We iterate over sentences and attach labels
    
    # Convert index lists to sets for O(1) lookup
    copied_set = set(copied_indices)
    para_set = set(paraphrased_indices)
    ai_set = set(ai_indices)
    
    analysis = []
    for i, sentence in enumerate(sentences):
        labels = []
        if i in copied_set:
            labels.append("copied")
        if i in para_set:
            labels.append("paraphrased")
        if i in ai_set:
            labels.append("ai")
            
        analysis.append({
            "text": sentence,
            "labels": labels
        })
        
    results["content_analysis"] = analysis
    
    return results
