# Word Familiarity API

A FastAPI application that computes familiarity scores for individual words in target language texts, helping language learners identify which words they're likely to know.

## Features

- Token-level analysis with POS tagging and lemmatization using Stanza
- Frequency-based scoring using the wordfreq library
- Cognate detection with CogNet dataset and OpenAI validation
- Support for many languages (canonical ISO 639-3 codes plus BCP-47 locale aliases such as `en-US`, `pt-BR`, `zh-Hans`)
- REST API with automatic documentation
- All models preloaded at startup for consistent performance

## Supported Languages

Canonical codes (also returned in API responses after normalization):

| ISO Code | Language | Stanza `lang` |
|----------|----------|---------------|
| `eng` | English | `en` |
| `ita` | Italian | `it` |
| `spa` | Spanish | `es` |
| `fra` | French | `fr` |
| `deu` | German | `de` |
| `por` | Portuguese | `pt` |
| `nld` | Dutch | `nl` |
| `pol` | Polish | `pl` |
| `rus` | Russian | `ru` |
| `jpn` | Japanese | `ja` |
| `kor` | Korean | `ko` |
| `cmn` | Mandarin (Simplified) | `zh-hans` |
| `arb` | Arabic | `ar` |
| `heb` | Hebrew | `he` |

**Locale aliases:** `GET /` and `GET /languages` include a `locale_aliases` object (e.g. `en-US` → `eng`, `es-ES` → `spa`, `pt-BR` → `por`, `zh-Hans` → `cmn`). You may send either canonical codes or these aliases in `learning_language` / `native_language`.

**Frequency notes:** Chinese Zipf scores use **jieba** (bundled dependency). Japanese and Korean `wordfreq` lookups may require optional system packages (e.g. MeCab); if missing, frequency falls back to zero without crashing.

## Installation

Requirements:
- Python 3.10+
- Poetry

Steps:
1. Install dependencies: `poetry install`
2. Create `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Place `CogNet-top6.tsv` in the `cognates/` directory

Stanza models will download automatically on first startup (may take 30-60 seconds).

## Docker (private registry)

See **[DOCKER.md](DOCKER.md)** for building the image, pushing to a **private Docker Hub** repository, and the image URL + credential fields for “Existing image” deploys.

## Usage

Start the server:
```bash
poetry run uvicorn main:app --reload
```

API documentation: `http://localhost:8000/docs`

### Example Request

```bash
curl -X POST "http://localhost:8000/familiarity" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "yo fui a la escuela",
    "learning_language": "spa",
    "native_language": "eng"
  }'
```

### Example Response

```json
{
  "content": "yo fui a la escuela",
  "learning_language": "spa",
  "native_language": "eng",
  "timestamp": "2025-10-22T16:32:00Z",
  "sentences": [
    {
      "text": "yo fui a la escuela",
      "index": 0,
      "tokens": [
        {
          "text": "yo", 
          "familiarity_score": 0.85,
          "cognate_boosted_familiarity_score": null,
          "cognate_before_LLM": null,
          "cognate_after_LLM": null,
          "cognate_similarity": null
        },
        {
          "text": "escuela", 
          "familiarity_score": 0.79,
          "cognate_boosted_familiarity_score": 1.0,
          "cognate_before_LLM": "school, école",
          "cognate_after_LLM": "school",
          "cognate_similarity": 0.743
        }
      ]
    }
  ],
  "total_tokens": 5
}
```

## Testing

Run tests with:
```bash
poetry run python tests.py
poetry run pytest tests/test_language_support.py -v
```

## Configuration

Scoring parameters can be adjusted in `core/constants.py`. To add new languages, update `SUPPORTED_LANGUAGES`, `ISO_TO_STANZA_LANG` / `ISO_TO_WORDFREQ_LANG` / processor overrides in `core/language_codes.py`, and Stanza model availability.

### CORS

Browser-based apps (e.g. Vite on `http://localhost:5173` or a static site on Hostinger) need **CORS** headers on this API. `main.py` registers `CORSMiddleware` with explicit `allow_origins`. If your frontend is served from another URL, add that origin to `allow_origins` and redeploy.

## How it Works

The API uses:
- **wordfreq** library for word frequency data
- **CogNet dataset** for cognate detection across languages
- **OpenAI GPT-4o** for cognate validation
- **Stanza** for tokenization and linguistic analysis

## Processing Pipeline

1. Text tokenization with POS tagging and lemmatization
2. Cognate lookup in CogNet dataset
3. OpenAI validation of cognate candidates
4. Similarity-based score boosting using Jaro-Winkler distance



## Development

The project is structured as:
- `main.py` - FastAPI application
- `core/` - Core logic modules
- `cognates/` - CogNet dataset

To add new languages, update `SUPPORTED_LANGUAGES` in `core/constants.py` and mappings in `core/language_codes.py`.

## Dependencies

Key libraries: FastAPI, Stanza, wordfreq, jieba (Chinese tokenization for wordfreq), OpenAI, Polars. See `pyproject.toml` for the complete list.