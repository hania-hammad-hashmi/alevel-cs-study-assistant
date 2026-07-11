"""
The UI: pick a topic, then either ask questions (Explain mode) or get
quizzed with real past-paper-style questions (Quiz mode).
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from document_loader import load_all_documents, split_documents
from vectorstore import build_vectorstore, load_vectorstore, get_retriever, list_topics, VECTOR_STORE_DIR
from chatbot import build_llm, build_explain_chain, generate_quiz_question

load_dotenv()
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")


def get_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def get_or_build_vectorstore(embeddings):
    if VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()):
        return load_vectorstore(embeddings)
    documents = load_all_documents()
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError(
            "No PDFs found in data/materials/<topic>/. Add your textbook "
            "chapters and past papers into topic folders first."
        )
    return build_vectorstore(chunks, embeddings)


def main():
    embeddings = get_embeddings()
    vectorstore = get_or_build_vectorstore(embeddings)
    llm = build_llm(HF_API_KEY)
    topics = list_topics(vectorstore) or ["all"]

    def explain(question, topic):
        t = None if topic == "all" else topic
        retriever = get_retriever(vectorstore, topic=t)
        chain = build_explain_chain(llm, retriever)
        result = chain.invoke({"query": question})
        sources = result.get("source_documents", [])
        source_note = ""
        if sources:
            files = sorted({d.metadata.get("source_file", "unknown") for d in sources})
            source_note = f"\n\n_Sources: {', '.join(files)}_"
        return result["result"] + source_note

    def quiz_me(topic):
        t = None if topic == "all" else topic
        retriever = get_retriever(vectorstore, topic=t, past_papers_only=True, k=6)
        return generate_quiz_question(llm, retriever, topic=t or "the syllabus")

    with gr.Blocks(theme=gr.themes.Soft(), title="A-level CS Study Assistant") as iface:
        gr.Markdown("## A-level CS Study Assistant")
        topic_dropdown = gr.Dropdown(choices=["all"] + topics, value="all", label="Topic")

        with gr.Tab("Explain"):
            question_box = gr.Textbox(label="Ask a question")
            explain_btn = gr.Button("Explain")
            explain_output = gr.Markdown()
            explain_btn.click(explain, inputs=[question_box, topic_dropdown], outputs=explain_output)

        with gr.Tab("Quiz me"):
            quiz_btn = gr.Button("Give me a practice question")
            quiz_output = gr.Markdown()
            quiz_btn.click(quiz_me, inputs=[topic_dropdown], outputs=quiz_output)

    iface.launch(share=True)


if __name__ == "__main__":
    main()
