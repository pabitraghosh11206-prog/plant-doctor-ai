import json
from pathlib import Path

from pypdf import PdfReader


# Find the main project folder from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_FILE = PROJECT_ROOT / "rag" / "metadata" / "documents.json"
PDF_FOLDER = PROJECT_ROOT / "rag" / "documents"
OUTPUT_FILE = PROJECT_ROOT / "rag" / "data" / "extracted_pages.jsonl"


def main():
    with REGISTRY_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    pages = []

    for document in documents:
        pdf_path = PDF_FOLDER / document["relative_path"]

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Could not find this PDF: {pdf_path}\n"
                "Check its filename and relative_path in documents.json."
            )

        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        for page_number, pdf_page in enumerate(reader.pages, start=1):
            text = pdf_page.extract_text() or ""

            if not text.strip():
                print(f"Warning: no text found on page {page_number}")
                continue

            pages.append({
                "document_id": document["document_id"],
                "source": pdf_path.name,
                "source_url": document["source_url"],
                "title": document["title"],
                "organization": document["organization"],
                "year": document.get("year"),
                "crop": document["crop"],
                "disease": document["disease"],
                "page": page_number,
                "text": text.strip()
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for page in pages:
            file.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"\nPages with text extracted: {len(pages)}")
    print(f"Saved the extracted text to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()