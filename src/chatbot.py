"""
STEP 4: hand the retrieved chunks + the question to an AI model, and get
back an answer that's grounded in your actual textbook/past papers
instead of the model just guessing from general knowledge.

Two modes:
- "explain"
- "quiz"
"""

import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

EXPLAIN_TEMPLATE = """
You are a study assistant for an A-level Computer Science student.
Use only the context below to answer. If the context doesn't contain
the answer, say so honestly instead of guessing.

Context: {context}

Question: {question}

Answer clearly, in a way suitable for A-level revision:
"""

QUIZ_TEMPLATE = """
You are helping an A-level Computer Science student practice for exams.
Below is real content from their past papers and textbook on this topic.

Context: {context}

Based only on this context, write ONE exam-style practice question
(do not answer it yet — just ask it, the student will answer next).

Question:
"""


def build_prompt(mode: str = "explain") -> PromptTemplate:
    template = QUIZ_TEMPLATE if mode == "quiz" else EXPLAIN_TEMPLATE
    input_vars = ["context"] if mode == "quiz" else ["context", "question"]
    return PromptTemplate(input_variables=input_vars, template=template)


def build_llm(api_key: str | None = None, temperature: float = 0.5):
    """
    NOTE: free Hugging Face Inference API endpoints occasionally get
    deprecated or rate-limited. If this model stops responding, check
    https://huggingface.co/models for a currently supported instruct
    model and swap the repo_id below.
    """
    api_key = api_key or os.getenv("HUGGING_FACE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Hugging Face API key found. Set HUGGING_FACE_API_KEY in .env "
            "(see .env.example). Never hardcode it in code."
        )
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        max_new_tokens=512,
    )


def build_explain_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": build_prompt("explain")},
    )


def generate_quiz_question(llm, retriever, topic: str) -> str:
    """
    Pull real past-paper/textbook chunks for a topic and ask the model
    to turn them into one practice question. Returns just the question
    text (a string), not a full chain — quiz mode is single-shot, not
    conversational.
    """
    docs = retriever.invoke(f"exam question about {topic}")
    context = "\n\n".join(d.page_content for d in docs)
    prompt = build_prompt("quiz").format(context=context)
    response = llm.invoke(prompt)
    return response
