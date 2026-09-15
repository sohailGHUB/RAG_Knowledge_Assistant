from document_loader import load_all_pdfs

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


DOCUMENT_TYPES = {
    "spark.pdf": "Apache Spark",
    "azureDatabricks.pdf": "Azure Databricks",
    "dataEngineering.pdf": "Data Engineering",
    "azureDataFactory.pdf": "Azure Data Factory",
    "azureSynapse.pdf": "Azure Synapse Analytics",
    "sqlDatabase.pdf": "Azure SQL Database"
}


KNOWN_HEADINGS = {
    "Core ideas",
    "Lakehouse model",
    "Azure integration",
    "Control and compute planes",
    "Unity Catalog hierarchy",
    "Metadata vs data",
    "Delta Lake Fundamentals",
    "Streaming",
    "Data engineering",
    "Architecture",
    "Security",
    "Monitoring",
    "Best practices"
}


def is_major_heading(text):
    """
    Detect headings such as:

    1. Azure Databricks Overview
    2. Azure Databricks Architecture
    """

    stripped = text.strip()

    if len(stripped) < 3:
        return False

    if stripped[0].isdigit() and ". " in stripped:
        return True

    return False


def is_subheading(text):
    """
    Detect known subsection headings.
    """

    return text.strip() in KNOWN_HEADINGS


def split_large_text(text):
    """
    Split large text into chunks while trying
    to preserve complete words.
    """

    pieces = []
    start = 0

    while start < len(text):

        end = min(start + CHUNK_SIZE, len(text))

        # Move the end backward to a word boundary
        if end < len(text):

            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        # Stop when we reach the end
        if end >= len(text):
            break

        # Create overlap
        overlap_start = max(start, end - CHUNK_OVERLAP)

        # Move overlap start forward to a word boundary
        next_space = text.find(" ", overlap_start, end)

        if next_space != -1:
            start = next_space + 1
        else:
            start = end

    return pieces


def add_chunk(
    chunks,
    text,
    source,
    page,
    document_type,
    section
):
    """
    Add a chunk together with its metadata.
    """

    if not text.strip():
        return

    chunks.append({
        "text": text.strip(),
        "source": source,
        "document_type": document_type,
        "page": page,
        "section": section
    })


def create_chunks(pages):

    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        source = page["source"]

        document_type = DOCUMENT_TYPES.get(
            source,
            "Unknown"
        )

        page_number = page["page"]

        # Start with the page's major heading if available
        current_section = "Unknown"

        # Split the page into lines
        lines = text.splitlines()

        current_text = ""
        current_chunk_section = "Unknown"

        for line in lines:

            line = line.strip()

            # Blank line
            if not line:

                if current_text.strip():

                    # Handle text collected so far
                    if len(current_text) > CHUNK_SIZE:

                        pieces = split_large_text(
                            current_text
                        )

                        for piece in pieces:
                            add_chunk(
                                chunks,
                                piece,
                                source,
                                page_number,
                                document_type,
                                current_chunk_section
                            )

                    else:

                        add_chunk(
                            chunks,
                            current_text,
                            source,
                            page_number,
                            document_type,
                            current_chunk_section
                        )

                    current_text = ""

                continue

            # Check whether this line is a heading
            if (
                is_major_heading(line)
                or is_subheading(line)
            ):

                # Save text before the heading
                if current_text.strip():

                    if len(current_text) > CHUNK_SIZE:

                        pieces = split_large_text(
                            current_text
                        )

                        for piece in pieces:
                            add_chunk(
                                chunks,
                                piece,
                                source,
                                page_number,
                                document_type,
                                current_chunk_section
                            )

                    else:

                        add_chunk(
                            chunks,
                            current_text,
                            source,
                            page_number,
                            document_type,
                            current_chunk_section
                        )

                    current_text = ""

                # The heading becomes the new section
                current_section = line

                # Future text belongs to this section
                current_chunk_section = current_section

                continue

            # Normal text
            if current_text:

                current_text += " " + line

            else:

                current_text = line

                # If this is the first text after
                # a heading, assign that heading
                current_chunk_section = current_section

        # Save remaining text at end of page
        if current_text.strip():

            if len(current_text) > CHUNK_SIZE:

                pieces = split_large_text(
                    current_text
                )

                for piece in pieces:
                    add_chunk(
                        chunks,
                        piece,
                        source,
                        page_number,
                        document_type,
                        current_chunk_section
                    )

            else:

                add_chunk(
                    chunks,
                    current_text,
                    source,
                    page_number,
                    document_type,
                    current_chunk_section
                )

    return chunks


if __name__ == "__main__":

    pages = load_all_pdfs()

    chunks = create_chunks(pages)

    print(f"Loaded {len(pages)} pages.")
    print(f"Created {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks[:5], start=1):

        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print("=" * 60)

        print(f"Source: {chunk['source']}")
        print(f"Document Type: {chunk['document_type']}")
        print(f"Page: {chunk['page']}")
        print(f"Section: {chunk['section']}")
        print(f"Characters: {len(chunk['text'])}")

        print("\n" + chunk["text"])