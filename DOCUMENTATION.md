# Familiarity Model — Technical Documentation

## What Is This?

**ella-word-familiarity** (or "Familiarity Model") is a FastAPI service that computes **per-token familiarity scores** for text in a target language. It helps language learners understand which words in a passage they are likely to know, based on:

1. **Word frequency** — How common the word is in the target language
2. **Cognate detection** — Whether the word has a similar counterpart in the learner’s native language
3. **Named entities** — Proper nouns (people, places, organizations) are treated as highly familiar

The API is designed for language-learning applications (e.g., Ella) that need to highlight or adapt content based on vocabulary difficulty.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application (main.py)                       │
│  POST /familiarity  │  GET /languages  │  GET /health  │  GET /              │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FamiliarityScorer (core/score_model.py)               │
│  • Document-level processing                                                  │
│  • Cognate lookup + LLM validation                                            │
│  • Token-level score computation                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │                              │                              │
         ▼                              ▼                              ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Stanza         │         │   CogNet          │         │   OpenAI         │
│   Tokenizer      │         │   (cognates)      │         │   GPT-4o         │
│   POS, Lemma,    │         │   TSV dataset     │         │   Cognate        │
│   NER            │         │   top 6 langs     │         │   validation     │
└──────────────────┘         └──────────────────┘         └──────────────────┘
         │
         ▼
┌──────────────────┐
│   wordfreq       │
│   Zipf frequency │
└──────────────────┘
```

---

## How It Works

### 1. Request Flow

A typical request looks like:

```json
{
  "content": "yo fui a la escuela",
  "learning_language": "spa",
  "native_language": "eng"
}
```

The API returns sentence-by-sentence token scores, including base familiarity and optional cognate-boosted scores.

### 2. Processing Pipeline

#### Step 1: Tokenization (Stanza)

- **Input:** Raw text and learning language code
- **Output:** Sentences with tokens, each token having:
  - `text` — Surface form
  - `lemma` — Base form (e.g., "escuelas" → "escuela")
  - `pos` — Part-of-speech (NOUN, VERB, ADJ, ADV, etc.)
  - `entity` — Named entity type (PER, LOC, ORG) if applicable

Stanza pipelines are preloaded at startup for all supported languages to avoid cold-start latency.

#### Step 2: Cognate Search (CogNet)

- **Eligible tokens:** NOUN, VERB, ADJ, ADV only
- **Search key:** Lemma for NOUN/ADJ, surface form for VERB/ADV
- **Dataset:** `cognates/CogNet-top6.tsv` — TSV with columns:
  - `concept id`, `lang 1`, `word 1`, `lang 2`, `word 2`, `translit 1`, `translit 2`
- **POS filtering:** Concept ID first letter maps to POS (`n`→NOUN, `v`→VERB, `a`→ADJ, `r`→ADV); candidates with mismatched POS are dropped
- **Concurrency:** Searches run in parallel via `ThreadPoolExecutor`

#### Step 3: LLM Validation (OpenAI GPT-4o)

- **Purpose:** Filter false cognates (similar spelling, different meaning)
- **Batching:** Groups of up to 30 search words per request
- **Input:** For each word, a list of candidate cognates and sentence context
- **Output:** One selected cognate per word (or empty string if none valid)
- **Concurrency:** Up to 4 batches processed in parallel

#### Step 4: Token Scoring

For each token:

1. **Named entities:** Score = 1.0 (assumed familiar)
2. **Frequency score:** `wordfreq.zipf_frequency()` → normalized to [0, 1] using `MIN_ZIPF` (2.3) and `MAX_ZIPF` (7.7)
3. **Cognate boost (if validated):**
   - Jaro–Winkler similarity between search word and cognate
   - Boost = `MIN_COGNATE_BOOST + similarity * (MAX_COGNATE_BOOST - MIN_COGNATE_BOOST)` (0.2–0.4)
   - `cognate_boosted_familiarity_score = min(1.0, familiarity_score + boost)`

---

## Response Model

| Field | Description |
|-------|-------------|
| `content` | Original input text |
| `learning_language` | Target language code |
| `native_language` | User’s native language code |
| `timestamp` | ISO timestamp of analysis |
| `sentences` | List of sentence objects |
| `total_tokens` | Total token count (excluding punctuation) |

**Sentence object:**

| Field | Description |
|-------|-------------|
| `text` | Sentence text |
| `index` | Sentence index in document |
| `tokens` | List of token score objects |

**Token score object:**

| Field | Description |
|-------|-------------|
| `text` | Token surface form |
| `familiarity_score` | Base score (0–1) from frequency |
| `cognate_boosted_familiarity_score` | Score with cognate boost, or `null` |
| `cognate_before_LLM` | CogNet candidates before LLM validation |
| `cognate_after_LLM` | LLM-selected cognate, or `null` |
| `cognate_similarity` | Jaro–Winkler similarity (0–1), or `null` |
| `entity` | NER type (PER, LOC, ORG), or `null` |

---

## Supported Languages

| ISO Code | Language | Stanza Model |
|----------|----------|--------------|
| `eng` | English | en |
| `ita` | Italian | it |
| `spa` | Spanish | es |
| `fra` | French | fr |
| `deu` | German | de |

---

## Configuration

| Constant | Location | Description |
|----------|----------|-------------|
| `MIN_COGNATE_BOOST` | `core/constants.py` | Minimum boost when cognate found (0.2) |
| `MAX_COGNATE_BOOST` | `core/constants.py` | Maximum boost when cognate found (0.4) |
| `MIN_ZIPF` / `MAX_ZIPF` | `core/constants.py` | Zipf range for frequency normalization |
| `COGNATE_BATCH_SIZE` | `core/constants.py` | Max search words per OpenAI batch (30) |
| `OPENAI_MODEL` | `core/constants.py` | Model for cognate validation (gpt-4o) |
| `COGNET_PATH` | `core/constants.py` | Path to CogNet TSV |

---

## Dependencies

- **FastAPI** — Web framework
- **Stanza** — Tokenization, POS, lemma, NER
- **wordfreq** — Zipf frequency data
- **Polars** — CogNet dataset handling
- **OpenAI** — Cognate validation
- **jellyfish** — Jaro–Winkler similarity
- **python-dotenv** — Environment variables (e.g. `OPENAI_API_KEY`)

---

## Running the Service

```bash
# Install
poetry install

# Set OpenAI key (required for cognate validation)
export OPENAI_API_KEY=your_key

# Run (production port 7000, dev with reload on 8000)
poetry run uvicorn main:app --host 0.0.0.0 --port 7000
```

With `DEV_MODE=true`, the server runs on port 8000 with hot reload.

---

## Project Structure

```
familiarity-model/
├── main.py              # FastAPI app, routes, lifespan
├── core/
│   ├── constants.py     # Scoring params, paths, language config
│   ├── score_model.py   # FamiliarityScorer, cognate logic, LLM validation
│   └── tokenizer.py     # Stanza tokenizer wrapper
├── cognates/
│   └── CogNet-top6.tsv  # Cognate dataset (Git LFS)
├── pyproject.toml       # Poetry config
└── README.md            # User-facing docs
```

---

## Summary

The Familiarity Model is a language-learning API that scores how familiar each word in a text is to a learner. It combines frequency data, cognate detection via CogNet, and LLM validation to produce per-token scores, with optional boosts for validated cognates. The service is built for integration into apps like Ella that adapt content based on vocabulary difficulty.
