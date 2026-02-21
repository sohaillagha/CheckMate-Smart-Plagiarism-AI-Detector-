"""
import_arxiv.py — Download full research papers from arXiv and add them to the dataset.

Usage:
    python import_arxiv.py                          # Uses defaults (10 CS/AI papers)
    python import_arxiv.py "deep learning" 20       # Custom query + count
"""

import os
import sys
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

# Allow imports from pdf_reader
sys.path.append(os.path.join(os.path.dirname(__file__), 'pdf_reader'))
from pdfextraction.reader import extract_pdf_text
from pdfextraction.preprocessing import clean_extracted_text

# ─── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_DIR = os.path.join(BASE_DIR, "datasets", "tfidf")
TEMP_PDF_DIR = os.path.join(BASE_DIR, "temp_downloads")
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_papers")

# Defaults (can be overridden via command-line args)
DEFAULT_QUERY = "machine learning"
DEFAULT_COUNT = 10


def sanitize_filename(title, max_length=50):
    """Convert a paper title into a safe filename."""
    # Keep only alphanumeric, spaces, and hyphens
    clean = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    clean = re.sub(r'\s+', '_', clean.strip())
    # Truncate
    return clean[:max_length].rstrip('_')


def get_next_paper_number():
    """Find the next available paper number by scanning existing .txt files."""
    os.makedirs(TFIDF_DIR, exist_ok=True)
    existing = [f for f in os.listdir(TFIDF_DIR) if f.startswith("paper_") and f.endswith(".txt")]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split("_")[1].split(".")[0]))
        except ValueError:
            continue
    return max(nums) + 1 if nums else 1


def search_arxiv(query, max_results):
    """Query arXiv API and return list of (title, pdf_url) tuples."""
    encoded_query = query.replace(" ", "+")
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )

    print(f"🔍 Searching arXiv for '{query}' (max {max_results} results)...")
    response = urllib.request.urlopen(url)
    xml_data = response.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    papers = []
    for entry in entries:
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        if pdf_url:
            papers.append((title, pdf_url))

    print(f"   Found {len(papers)} papers with PDF links.\n")
    return papers


def download_and_extract(papers, start_number, query):
    """Download PDFs, extract text, save to datasets/tfidf/ and reference_papers/."""
    os.makedirs(TEMP_PDF_DIR, exist_ok=True)
    os.makedirs(TFIDF_DIR, exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    success_count = 0
    index_entries = []  # Track paper info for index file

    for i, (title, pdf_url) in enumerate(papers):
        paper_num = start_number + i
        pdf_filename = f"paper_{paper_num:03d}.pdf"
        txt_filename = f"paper_{paper_num:03d}.txt"
        pdf_path = os.path.join(TEMP_PDF_DIR, pdf_filename)
        txt_path = os.path.join(TFIDF_DIR, txt_filename)

        # Reference PDF with descriptive name: paper_006_Privacy_Preserving_ML.pdf
        safe_title = sanitize_filename(title)
        ref_pdf_name = f"paper_{paper_num:03d}_{safe_title}.pdf"
        ref_pdf_path = os.path.join(REFERENCE_DIR, ref_pdf_name)

        print(f"  [{i+1}/{len(papers)}] {title[:60]}...")

        # Download PDF
        try:
            print(f"    📥 Downloading PDF...")
            urllib.request.urlretrieve(pdf_url, pdf_path)
        except Exception as e:
            print(f"    ❌ Download failed: {e}")
            time.sleep(3)
            continue

        # Copy PDF to reference_papers/ with descriptive name
        try:
            import shutil
            shutil.copy2(pdf_path, ref_pdf_path)
            print(f"    📂 Reference PDF: {ref_pdf_name}")
        except Exception as e:
            print(f"    ⚠️  Could not copy reference PDF: {e}")

        # Extract text using existing reader
        try:
            print(f"    📄 Extracting text...")
            full_text = extract_pdf_text(pdf_path)
            full_text = clean_extracted_text(full_text)

            if len(full_text) < 200:
                print(f"    ⚠️  Too little text extracted ({len(full_text)} chars), skipping.")
                time.sleep(3)
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            success_count += 1
            index_entries.append((paper_num, title, query))
            print(f"    ✅ Saved {txt_filename} ({len(full_text):,} chars)")

        except Exception as e:
            print(f"    ❌ Extraction failed: {e}")

        # arXiv rate limit: 3 seconds between requests
        if i < len(papers) - 1:
            print(f"    ⏳ Waiting 3s (arXiv rate limit)...")
            time.sleep(3)

    # Update index file
    if index_entries:
        update_index(index_entries)

    return success_count


def update_index(new_entries):
    """Append new paper entries to the reference index file."""
    index_path = os.path.join(REFERENCE_DIR, "index.txt")

    # Read existing entries if file exists
    existing_lines = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    with open(index_path, "w", encoding="utf-8") as f:
        # Write header if new file
        if not existing_lines:
            f.write("=" * 70 + "\n")
            f.write("  CHECKMATE — Reference Papers Index\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Paper #':<12}{'Topic':<25}{'Title'}\n")
            f.write("-" * 70 + "\n")
        else:
            f.writelines(existing_lines)

        for paper_num, title, query in new_entries:
            f.write(f"paper_{paper_num:03d}    {query:<25}{title}\n")

    print(f"\n  📋 Updated index: {index_path}")


def main():
    # Parse command-line arguments
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    count = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_COUNT

    print("=" * 50)
    print("  arXiv Paper Importer for CheckMate")
    print("=" * 50)
    print(f"  Query:  {query}")
    print(f"  Count:  {count}")
    print("=" * 50 + "\n")

    # Find next available paper number
    start_num = get_next_paper_number()
    print(f"📁 Next paper number: paper_{start_num:03d}\n")

    # Search arXiv
    papers = search_arxiv(query, count)
    if not papers:
        print("No papers found. Try a different query.")
        return

    # Download and extract
    success = download_and_extract(papers, start_num, query)

    # Summary
    print("\n" + "=" * 50)
    print(f"  ✅ Successfully imported {success}/{len(papers)} papers")
    print(f"  📂 Text saved to:  {TFIDF_DIR}")
    print(f"  📂 PDFs saved to:  {REFERENCE_DIR}")
    print(f"  📋 Index file:     {os.path.join(REFERENCE_DIR, 'index.txt')}")
    print(f"\n  👉 Now run: python update_dataset.py")
    print(f"     to rebuild the SBERT dataset.")
    print("=" * 50)


if __name__ == "__main__":
    main()
