import os
import json
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_folder, output_json):
    data = {}
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            data[pdf_file] = text
    with open(output_json, "w") as f:
        json.dump(data, f)
    print(f"Extracted text saved to {output_json}")
