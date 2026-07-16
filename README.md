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
## Evaluation

### Explain mode
Tested on 45 questions, sourced directly from past-paper mark schemes so
correctness could be judged objectively rather than against my own
understanding.

- **38/45 average** correct (averaged over 3 runs: 38, 37, 39 — since
  model output isn't fully deterministic, a single run wasn't a fair
  representation).
- **7/45** initially came back confused, clustered around binary
  calculations, bit depth, and colour depth.

**Diagnosis:** the source PDF mixed worked examples with surrounding
explanatory text without clear section breaks, so chunking
(`document_loader.py`) occasionally split a calculation from the context
it needed.

**Fix:** re-chunked affected sections so each numeric worked example
stayed attached to its explanation.

**Did the fix generalize?** Rather than re-testing the same 7 questions
(risk of false positive, since I already knew the answers), I wrote fresh
unseen questions on the same three topics and verified them against mark
schemes — these passed. I then checked whether the same chunking problem
existed elsewhere by testing 5 new questions per remaining topic folder
(algorithms, networks, and 2 others). One further issue turned up in
algorithms, where a sorting-algorithm worked example was split from its
trace-through steps — same fix applied, retested with 5 fresh questions,
passed. No further issues found elsewhere.

### Quiz mode
Tested with 30 generated questions across all topics, spanning easy to
hard difficulty (tagged manually using A-level command words: easy =
"define"/"state," medium = "explain"/"describe," hard = "calculate"/
"evaluate"/multi-step).

A generated question was counted as "good" only if it passed three
checks: source-grounded (based on a real fact in the retrieved chunk, not
invented), answerable using only that source chunk, and difficulty-type
matched (e.g. a "calculate" question needs a chunk with a worked example,
not just a definition).

Result: 24/30 passed all three checks. 4 were source-grounded but too
vague to be gradable (e.g. "Explain networks"). 2 were caused by a
topic-clustering bug: 5 of 30 questions came from the same topic because
that topic had more chunks than others, and nothing balanced retrieval
across topics.

**Fixes:**
- Vagueness → added a minimum-specificity instruction in the quiz
  prompt, requiring each question to be based on a single fact/example
  rather than summarizing a whole chunk. Retested on 20 fresh questions
  — 0 vague results.
- Clustering → added a diversity check in `get_retriever` excluding
  chunks already used earlier in the same quiz session. Retested on 30
  fresh questions across the same topics — no clustering recurred.

### Baseline: RAG vs. plain LLM (Explain mode)
Ran the same 45 Explain-mode questions with no retrieved context, 3 runs
each, to check retrieval was adding real value rather than the model
answering from what it already knew.

| | Correct / 45 (avg of 3 runs) |
|---|---|
| Plain LLM (no retrieval) | ~25.7 |
| RAG (with retrieval) | 38 |

To check this gap wasn't just "RAG is better in general" but actually
about retrieval adding source-specific value, I split the 45 questions
into two groups after the fact:

- **Generic-knowledge questions** (definitions/concepts likely covered
  broadly online): plain LLM scored ~20/24 even with no retrieval — this
  part of the gap isn't strong evidence for RAG, since the model may
  already know this from training.
- **Source-specific questions** (exact numeric answers, terminology, or
  mark-scheme-specific phrasing): plain LLM dropped to ~6/21, RAG stayed
  at 19/21. This is the part of the result that actually demonstrates
  retrieval is doing real work.

### Baseline: RAG vs. plain LLM (Quiz mode)
Generated 30 quiz questions with retrieval (as above) and a second set of
30 from the plain LLM given only the topic name (e.g. "generate a hard
question about bit depth"), no retrieved context. Both sets graded
against the same three checks.

| | Passed all 3 checks / 30 |
|---|---|
| Plain LLM (no retrieval) | 14/30 |
| RAG (with retrieval) | 24/30 |

Most plain-LLM failures weren't vagueness but factual drift — questions
that sounded like real A-level questions but referenced numbers or
definitions that didn't match my syllabus (e.g. an incorrect bit-depth
convention, or a definition close to but not matching the specification
wording). This is the actual case for retrieval in Quiz mode: it keeps
generated questions factually anchored to the specific syllabus content
being studied, rather than just sounding plausible.

### Grading reliability (single-grader bias)
All grading above was done by me alone, which is a limitation: I defined
what "correct" and "gradable" meant, and I was the only one applying that
definition. Explain mode is partially protected against this since
answers were checked against the actual mark scheme, an external
standard. Quiz mode's checks had no external anchor beyond my own
judgment.

To reduce this, a classmate studying the same A-level CS specification
independently re-graded a random subset — 15 of the 45 Explain-mode
answers and 10 of the 30 Quiz-mode questions — without seeing my original
grades, using the same criteria written down beforehand.

- Explain mode: agreement on 14/15 (the one disagreement was a borderline
  case — technically correct but incomplete relative to full mark-scheme
  detail).
- Quiz mode: agreement on 8/10 (both disagreements were on the
  "answerable" check — my classmate found two questions ambiguous that I'd
  marked as clear, likely because I already knew which chunk they came
  from and unconsciously filled in context a first-time reader wouldn't
  have).

**Takeaway:** independent re-grading mostly confirmed the original
results, but the two Quiz-mode disagreements point to a real bias —
grading your own system's output is inherently easier than a fresh reader
would find it. A fuller evaluation would use an independent grader for
the full set, not just a spot-check. Splitting the baseline by question
type, rather than reporting one aggregate number, was also necessary to
avoid overstating what RAG was actually contributing versus what the
model already knew.
## Notes on scope and honesty
- This project uses my own study materials for my own A-level CS course
  — no other students' data, no impersonation of any real teacher.
- Textbook/past-paper PDFs are **not** committed to this repo (see
  `.gitignore`) since I don't hold copyright on them — only my own code is.
- Built with AI assistance (Claude) for the retrieval/RAG architecture,
  while I worked through and tested each part to understand what it does
  and why (see the plain-English comments throughout the code).
- Retrieval and generation quality have been benchmarked against
  official mark schemes and cross-checked by an independent grader
  (see Evaluation section above); remaining limitations are noted there.

## Known limitation on the AI model
Hugging Face's free Inference API sometimes deprecates or rate-limits
specific models. If generation stops responding, swap the `repo_id` in
`chatbot.py` for a currently supported instruct model.
