"""
STEP 1 of the pipeline: turn PDF files into small, labeled chunks of text.

How you organize your files controls the labeling — put files into
folders named after the topic, like this:

    data/materials/algorithms/textbook_ch4.pdf
    data/materials/algorithms/past_paper_2021.pdf
    data/materials/networks/textbook_ch7.pdf

The folder name becomes the "topic" tag on every chunk from that file.
No filename tricks needed — just organize your PDFs into the right folders.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

MATERIALS_DIR = Path("./data/materials")


def load_all_documents(materials_dir: Path = MATERIALS_DIR):
    """
    Walk every topic folder inside data/materials, load each PDF found,
    and tag every page with which topic folder it came from.

    Returns a flat list of "documents" (one per PDF page), each with
    metadata like {"topic": "algorithms", "source": "textbook_ch4.pdf"}.
    """
    documents = []
    if not materials_dir.exists():
        return documents

    for topic_folder in sorted(materials_dir.iterdir()):
        if not topic_folder.is_dir():
            continue
        topic = topic_folder.name  # e.g. "algorithms"

        for pdf_path in topic_folder.glob("*.pdf"):
            is_past_paper = "paper" in pdf_path.stem.lower()
            loader = PyPDFLoader(pdf_path.as_posix())
            for page_doc in loader.load():
                page_doc.metadata["topic"] = topic
                page_doc.metadata["source_file"] = pdf_path.name
                page_doc.metadata["is_past_paper"] = is_past_paper
                documents.append(page_doc)

    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    STEP 1b: cut each page into smaller overlapping chunks (~800 characters,
    roughly a paragraph or two). This matters because embedding a whole
    textbook chapter as one block loses precision — a question about one
    small definition would get buried in a huge chunk. Smaller chunks =
    the search in Step 3 can find the exact relevant bit.

    metadata (topic, source_file, is_past_paper) is automatically copied
    onto every chunk that comes from a tagged page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
