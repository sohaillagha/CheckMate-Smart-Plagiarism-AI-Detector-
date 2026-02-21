import os
from pdfextraction.reader import extract_pdf_text

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the PDF file
pdf_path = os.path.join(
    script_dir, 
    "userinput", 
    "Improving_Semiconductor_Device_Modeling_for_Electronic_Design_Automation_by_Machine_Learning_Techniques.pdf"
)

# Check if file exists before processing
if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
else:
    text = extract_pdf_text(pdf_path)
    print(text)
