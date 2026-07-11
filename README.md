# A-level CS Study Assistant

A retrieval-based study tool for A-level Computer Science: ask questions
answered directly from my own textbook and past papers (not generic
internet knowledge), or get quizzed with real past-paper-style questions
by topic.

## How it works (in plain English)
1. **Read & chunk** (`document_loader.py`) — PDFs placed in
   `data/materials/<topic>/` (e.g. `algorithms/`, `networks/`) are loaded
   and cut into small overlapping pieces of text (~800 characters each).
   Small pieces make later search far more precise than searching whole
   chapters at once.
2. **Turn into numbers** (`vectorstore.py`, `build_vectorstore`) — each
   chunk is converted into a list of numbers (an embedding) that
   represents its meaning, using a small embedding model via the
   Hugging Face Inference API, and stored locally with Chroma.
3. **Search** (`vectorstore.py`, `get_retriever`) — a question gets
   converted the same way, and the closest-matching chunks are found.
   Search can be scoped to one topic, and further scoped to
   past-papers-only for quiz mode.
4. **Answer or quiz** (`chatbot.py`) — the matching chunks plus the
   question are sent to an instruction-following model
   (`mistralai/Mistral-7B-Instruct-v0.2`) which either answers directly
   (Explain mode) or turns real past-paper content into a fresh practice
   question (Quiz mode) without answering it for me.
5. **UI** (`app.py`) — a simple Gradio interface with a topic dropdown
   and two tabs: Explain and Quiz me.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add your own Hugging Face token
```
Add your PDFs into `data/materials/<topic>/`, e.g.:
```
data/materials/algorithms/textbook_ch4.pdf
data/materials/algorithms/past_paper_2021.pdf
data/materials/networks/textbook_ch7.pdf
```
Then:
```bash
cd src && python app.py
```

## Notes on scope and honesty
- This project uses my own study materials for my own A-level CS course
  — no other students' data, no impersonation of any real teacher.
- Textbook/past-paper PDFs are **not** committed to this repo (see
  `.gitignore`) since I don't hold copyright on them — only my own code is.
- Built with AI assistance (Claude) for the retrieval/RAG architecture,
  while I worked through and tested each part to understand what it does
  and why (see the plain-English comments throughout the code).
- Known limitation: retrieval quality hasn't been formally benchmarked;
  a natural next step is comparing answers against actual mark schemes.

## Known limitation on the AI model
Hugging Face's free Inference API sometimes deprecates or rate-limits
specific models. If generation stops responding, swap the `repo_id` in
`chatbot.py` for a currently supported instruct model.
