import os
import json
from PyPDF2 import PdfReader

# Extract text from PDFs
def extract_text_from_pdfs(pdf_folder, output_json):
    extracted_data = {}
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
                extracted_data[pdf_file] = text

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(extracted_data, f)

# Example usage
if __name__ == "__main__":
    pdf_folder = "data/pdfs"
    output_json = "data/processed/lectureNotes.json"
    extract_text_from_pdfs(pdf_folder, output_json)
    print(f"Extracted text saved to {output_json}")
