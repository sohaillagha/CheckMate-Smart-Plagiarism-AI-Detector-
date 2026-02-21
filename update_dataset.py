
import os
import sys
import pandas as pd
import nltk

# Ensure nltk data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Fix path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pdf_reader.pdfextraction.reader import extract_pdf_text
from pdf_reader.pdfextraction.preprocessing import clean_extracted_text

def update_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    userinput_dir = os.path.join(base_dir, "userinput")
    tfidf_dir = os.path.join(base_dir, "datasets", "tfidf")
    sbert_csv_path = os.path.join(base_dir, "datasets", "sbert", "sbert_pairs.csv")
    
    print("--- Syncing PDF content to Datasets ---")
    
    # 1. Iterate over all PDFs in userinput
    for filename in os.listdir(userinput_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(userinput_dir, filename)
            
            # Construct expected txt filename (e.g. paper_001.pdf -> paper_001.txt)
            input_name = os.path.splitext(filename)[0]
            txt_filename = input_name + ".txt"
            txt_path = os.path.join(tfidf_dir, txt_filename)
            
            # Process all PDFs — auto-create dataset entry for new papers
            action = "Updating" if os.path.exists(txt_path) else "Adding NEW"
            print(f"{action} {txt_filename} from {filename}...")
            try:
                full_text = extract_pdf_text(pdf_path)
                full_text = clean_extracted_text(full_text)
                
                if len(full_text) < 100:
                     print(f"  Warning: Extracted text for {filename} is very short ({len(full_text)} chars). Skipping.")
                     continue
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"  Success! Size: {len(full_text)} chars")
            except Exception as e:
                print(f"  Failed: {e}")

    # 2. Rebuild SBERT CSV from ALL txt files
    all_sentences = []
    
    print("\n--- Rebuilding SBERT Dataset ---")
    for filename in os.listdir(tfidf_dir):
        if filename.endswith(".txt"):
            path = os.path.join(tfidf_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Tokenize into sentences
                sentences = sent_tokenize(text)
                all_sentences.extend(sentences)
                print(f"  Processed {filename}: {len(sentences)} sentences")
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
                
    # Create DataFrame
    df = pd.DataFrame({
        'sentence1': all_sentences,
        'sentence2': [''] * len(all_sentences)
    })
    
    # Remove short garbage sentences
    df = df[df['sentence1'].str.len() > 10]
    
    print(f"Total valid sentences: {len(df)}")
    df.to_csv(sbert_csv_path, index=False)
    print(f"Saved to {sbert_csv_path}")

if __name__ == "__main__":
    update_dataset()
