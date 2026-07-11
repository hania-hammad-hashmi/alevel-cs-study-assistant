"""
STEP 2: turn text chunks into numbers (embeddings) and store them.
STEP 3: given a question, find the closest-meaning chunks.

Both steps use Chroma, a simple local database built for exactly this
kind of "find similar text" search.
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma

VECTOR_STORE_DIR = Path("./data/vector_store")


def build_vectorstore(chunks, embeddings):
    """
    STEP 2: for every chunk, call the embedding model to turn its text
    into a list of numbers, then save those numbers (plus the original
    text and metadata) to disk in ./data/vector_store.

    You only need to run this once per set of documents — after that,
    use load_vectorstore() instead, which is much faster since it
    skips re-embedding everything.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR.as_posix(),
    )


def load_vectorstore(embeddings):
    """Reopen the already-built store without re-embedding."""
    return Chroma(
        persist_directory=VECTOR_STORE_DIR.as_posix(),
        embedding_function=embeddings,
    )


def get_retriever(vectorstore, topic: str | None = None, past_papers_only: bool = False, k: int = 4):
    """
    STEP 3: build a "retriever" — something you can hand a question to,
    and it returns the k closest-matching chunks.

    If `topic` is set (e.g. "algorithms"), only chunks tagged with that
    topic are searched — this is what lets you say "only quiz me on
    networks" instead of searching your whole syllabus every time.

    If `past_papers_only` is True, it only searches chunks that came
    from a file with "paper" in its name — useful for the quiz feature.
    """
    filters = {}
    if topic:
        filters["topic"] = topic
    if past_papers_only:
        filters["is_past_paper"] = True

    search_kwargs = {"k": k}
    if filters:
        # Chroma needs a specific format once there's more than one filter condition
        if len(filters) == 1:
            search_kwargs["filter"] = filters
        else:
            search_kwargs["filter"] = {"$and": [{k: v} for k, v in filters.items()]}

    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def list_topics(vectorstore):
    """Return every distinct topic currently stored, for a dropdown menu."""
    try:
        metadatas = vectorstore.get()["metadatas"]
        return sorted({m.get("topic") for m in metadatas if m.get("topic")})
    except Exception:
        return []
