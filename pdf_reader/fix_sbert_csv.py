
import os
import pandas as pd
import csv

def fix_csv():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "datasets", "sbert", "sbert_pairs.csv")
    
    print(f"Reading corrupted file: {input_path}")
    
    # Read raw lines to avoid pandas parsing errors with bad delimiters
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Read {len(lines)} lines.")
    
    cleaned_rows = []
    # Header logic: if the first line is valid, keep it. 
    # But looking at previous output, even the header might have excessive commas.
    # We know the columns should be "sentence1,sentence2" or similar.
    
    valid_data_found = 0
    
    for i, line in enumerate(lines):
        # Remove trailing whitespace/newlines
        line = line.strip()
        
        # If line ends with many commas, strip them.
        # But simply stripping commas might merge data if commas are delimiters.
        # Strategy: The file seems to have valid data in the first few "cells" and then thousands of empty cells.
        # We will split by comma and take the first few non-empty items.
        
        parts = line.split(',')
        # Filter out empty strings
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 2:
            # Assuming first two valid parts are sentence1 and sentence2
            # CAUTION: If sentences contain commas, this split is dangerous.
            # Ideally we'd use a csv reader, but the file is malformed.
            # Let's try to infer if it was quoted.
            pass
            
    # BETTER APPROACH: Use pandas with 'error_bad_lines=False' or limited columns doesn't work well if rows vary.
    # Let's try to just read the first 2 columns.
    
    try:
        # Re-read with pandas, only first 2 columns, assuming the data is there just followed by garbage
        df = pd.read_csv(input_path, usecols=[0, 1], header=0, names=["sentence1", "sentence2"])
        print("Pandas read successful with usecols=[0,1]")
        print(df.head())
        print(f"Shape: {df.shape}")
        
        # Save it back cleanly
        print("Saving cleaned CSV...")
        df.to_csv(input_path, index=False)
        print("File repaired.")
        
    except Exception as e:
        print(f"Pandas failed: {e}")
        # Fallback: manual parsing
        print("Attempting manual clean...")
        cleaned_data = []
        for line in lines:
            line = line.rstrip(',\n') # remove trailing commas and newline
            if not line: continue
            # This relies on the fact that the garbage is trailing commas
            cleaned_data.append(line)
            
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_data))
        print("Manual strip completed.")

if __name__ == "__main__":
    fix_csv()
