import os
import json
from PyPDF2 import PdfReader

os.makedirs(r"C:\Users\shrut\Documents\Fall24\Advance NLP\rag_tutor\data\processed", exist_ok=True)
def pdf_to_text(pdf_folder, output_file):
    data = {}
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_folder, file_name))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            data[file_name] = text
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

pdf_to_text(r"C:\Users\shrut\Documents\Fall24\Advance NLP\rag_tutor\data", r"C:\Users\shrut\Documents\Fall24\Advance NLP\rag_tutor\data\processed\lecture_notes.json")

print("Processed lecture notes saved to data/processed/lecture_notes.json")
