from pathlib import Path
import pymupdf


DOCUMENTS_DIR = Path("data/documents")


def load_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number + 1,
            "source": pdf_path.name
        })

    document.close()

    return pages


def load_all_pdfs():
    all_pages = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    pages = load_all_pdfs()

    print(f"Total pages loaded: {len(pages)}")

    for page in pages[:3]:
        print("\n--------------------")
        print(f"Source: {page['source']}")
        print(f"Page: {page['page']}")
        print(page["text"][:500])