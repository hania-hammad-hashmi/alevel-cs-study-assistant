"""
First for pipeline the  PDF files are turned into small, labeled chunks of text.

The files are organized in topic wise manner such as : 

    data/materials/algorithms/textbook_ch4.pdf
    data/materials/algorithms/past_paper_2021.pdf
    data/materials/networks/textbook_ch7.pdf
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

MATERIALS_DIR = Path("./data/materials")


def load_all_documents(materials_dir: Path = MATERIALS_DIR):
    
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
    moreover , cut each page into smaller overlapping chunks . This matters because embedding a whole
    textbook chapter as one block loses precision 
    metadata (topic, source_file, is_past_paper) is automatically copied
    onto every chunk that comes from a tagged page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
