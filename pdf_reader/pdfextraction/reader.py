#import re
#from pdfminer.high_level import extract_pages,extract_text

#text= extract_text("Improving_Semiconductor_Device_Modeling_for_Electronic_Design_Automation_by_Machine_Learning_Techniques.pdf")
#print(text)
    
#import fitz #pymupdf
#import PIL.Image
#import io #to work with image byte data

#pdf =fitz.open("Improving_Semiconductor_Device_Modeling_for_Electronic_Design_Automation_by_Machine_Learning_Techniques.pdf")
#counter=1
#for i in range(len(pdf)):
#    page =pdf[i]
#    images=page.get_images()
#    for image in images:
#        base_img=pdf.extract_image(image[0])
#        image_data=base_img["image"]
#        img=PIL.Image.open(io.BytesIO(image_data))
#        extension=base_img["ext"]
#        img.save(open(f"image{counter}.{extension}","wb"))
#        counter+=1

#import pdfplumber

#with pdfplumber.open("Improving_Semiconductor_Device_Modeling_for_Electronic_Design_Automation_by_Machine_Learning_Techniques.pdf") as pdf:
#    for page_num, page in enumerate(pdf.pages, start=1):
#        tables = page.extract_tables()
#        for table in tables:
#            print(f"\n--- Page {page_num} ---")
#            for row in table:
#                print(" | ".join(cell if cell else "" for cell in row))#


# ===============================
# reader.py
# PDF READING MODULE
# ===============================

import re
import fitz              # PyMuPDF
import pdfplumber
import io
from PIL import Image
from pdfminer.high_level import extract_text


# ===============================
# 1️⃣ TEXT EXTRACTION
# ===============================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract complete raw text from a PDF using PyMuPDF (fitz).
    This is generally faster and handles layout better than pdfminer.
    """
    try:
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        
        full_text = "\n".join(text)
        
        # Normalize whitespace (replaces multiple spaces/newlines with single space)
        full_text = re.sub(r'\s+', ' ', full_text)

        return full_text.strip()

    except Exception as e:
        raise RuntimeError(f"Text extraction failed: {e}")


# ===============================
# 2️⃣ IMAGE EXTRACTION (OPTIONAL)
# ===============================

def extract_images(pdf_path: str, output_dir: str = "images"):
    """
    Extract images from PDF using PyMuPDF
    (Optional – not required for plagiarism)
    """
    pdf = fitz.open(pdf_path)
    img_count = 1

    for page in pdf:
        for img in page.get_images():
            base_img = pdf.extract_image(img[0])
            image_bytes = base_img["image"]
            ext = base_img["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"{output_dir}/image_{img_count}.{ext}")
            img_count += 1


# ===============================
# 3️⃣ TABLE EXTRACTION (OPTIONAL)
# ===============================

def extract_tables(pdf_path: str):
    """
    Extract tables using pdfplumber
    Returns list of tables with page numbers
    """
    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append({
                    "page": page_no,
                    "table": table
                })

    return extracted_tables
