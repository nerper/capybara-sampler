# Word Familiarity API — Call Guide

Quick reference for calling the Word Familiarity API.

**Base URL:** `http://localhost:PORT` (replace `PORT` with your `.env` value, e.g. `8080`)

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and supported languages |
| GET | `/health` | Health check |
| GET | `/languages` | List supported languages |
| POST | `/familiarity` | Analyze text and get per-token familiarity scores |

---

## Supported languages

The API uses **canonical ISO 639-3** three-letter codes internally and in responses. For requests you may send either those codes or **locale aliases** (BCP-47–style tags such as `en-US`, `pt-BR`, `zh-Hans`); they are normalized before scoring.

### Canonical codes (use in `learning_language` / `native_language`)

| Code | Language |
|------|----------|
| `eng` | English |
| `ita` | Italian |
| `spa` | Spanish |
| `fra` | French |
| `deu` | German |
| `por` | Portuguese |
| `nld` | Dutch |
| `pol` | Polish |
| `rus` | Russian |
| `jpn` | Japanese |
| `kor` | Korean |
| `cmn` | Mandarin (Simplified) |
| `arb` | Arabic |
| `heb` | Hebrew |

### How to connect

1. **Discover codes at runtime** (recommended so your client stays in sync with the server):

   ```bash
   curl https://YOUR_HOST/languages
   ```

   The JSON includes:

   - `supported_languages` — map of canonical code → human-readable name (authoritative list).
   - `locale_aliases` — map of accepted request strings (lowercase, hyphenated) → canonical code, e.g. `en-us` → `eng`, `zh-hans` → `cmn`.

   The same payload is also returned on `GET /` under `supported_languages` and `locale_aliases`.

2. **Score text** — send a JSON body to `POST /familiarity` with:

   - `learning_language` — language of the **content** being analyzed (the learner’s target language).
   - `native_language` — the learner’s **L1** (used for cognate detection relative to that language).
   - `content` — non-empty string in the learning language.

   Use canonical codes or any key from `locale_aliases` for both language fields.

3. **Browser / SPA** — same URLs; ensure your deployment’s CORS settings allow your origin (see `main.py` for allowed origins in this repo).

4. **Interactive testing** — with the server running, open `/docs` (Swagger) or `/redoc` and try `POST /familiarity` with the fields above.

If a language string is not in `supported_languages` and does not match an alias, the API returns **400** with the list of supported canonical codes.

---

## GET `/`

Returns API metadata.

**Example:**
```bash
curl http://localhost:8080/
```

**Response:** includes `supported_languages` (canonical ISO 639-3 → display name), `locale_aliases` (request tokens such as `en-US`, `pt-BR`, `zh-Hans` → canonical code), and `endpoints`. Exact keys match the live server; use `curl` against your deployment to inspect the full map.

---

## GET `/health`

Simple health check.

**Example:**
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{"status": "healthy"}
```

---

## GET `/languages`

Returns supported language codes and names.

**Example:**
```bash
curl http://localhost:8080/languages
```

**Response:** `supported_languages` and `locale_aliases` (see GET `/`).

---

## POST `/familiarity`

Analyzes text and returns per-token familiarity scores.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `learning_language` | string | Yes | Canonical code (e.g. `eng`, `jpn`, `cmn`) or locale alias from `locale_aliases` (e.g. `en-US`, `pt-BR`, `zh-Hans`) |
| `native_language` | string | Yes | Same as `learning_language` |
| `content` | string | Yes | Text to analyze (non-empty) |

### Example Request

```bash
curl -X POST http://localhost:8080/familiarity \
  -H "Content-Type: application/json" \
  -d '{
    "learning_language": "spa",
    "native_language": "eng",
    "content": "Yo fui a la escuela ayer."
  }'
```

### Response

```json
{
  "content": "Yo fui a la escuela ayer.",
  "learning_language": "spa",
  "native_language": "eng",
  "timestamp": "2026-03-11T12:00:00.000000",
  "total_tokens": 6,
  "sentences": [
    {
      "text": "Yo fui a la escuela ayer.",
      "index": 0,
      "tokens": [
        {
          "text": "Yo",
          "familiarity_score": 0.85,
          "cognate_boosted_familiarity_score": null,
          "cognate_before_LLM": null,
          "cognate_after_LLM": null,
          "cognate_similarity": null,
          "entity": null
        },
        {
          "text": "escuela",
          "familiarity_score": 0.72,
          "cognate_boosted_familiarity_score": 0.92,
          "cognate_before_LLM": "school",
          "cognate_after_LLM": "school",
          "cognate_similarity": 0.85,
          "entity": null
        }
      ]
    }
  ]
}
```

### Token Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Token surface form |
| `familiarity_score` | float | Base score 0–1 from word frequency |
| `cognate_boosted_familiarity_score` | float \| null | Score with cognate boost, if found |
| `cognate_before_LLM` | string \| null | Cognate candidate before LLM validation |
| `cognate_after_LLM` | string \| null | Validated cognate in native language |
| `cognate_similarity` | float \| null | Jaro–Winkler similarity (0–1) |
| `entity` | string \| null | Named entity type (PER, LOC, ORG) if applicable |

---

## Error Responses

**400 Bad Request** — Invalid language or empty content:
```json
{
  "detail": "Learning language 'xyz' not supported. Supported: ['arb', 'cmn', 'deu', 'eng', ...]"
}
```

Use `GET /languages` for the authoritative `supported_languages` list.

**500 Internal Server Error** — Processing failure:
```json
{
  "detail": "Internal server error: <message>"
}
```

---

## Interactive Docs

When the API is running, open:

- **Swagger UI:** `http://localhost:PORT/docs`
- **ReDoc:** `http://localhost:PORT/redoc`

---

## Example: JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8080/familiarity', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    learning_language: 'spa',
    native_language: 'eng',
    content: 'El gato está en la mesa.'
  })
});
const data = await response.json();
console.log(data.sentences);
```

---

## Example: Python (requests)

```python
import requests

response = requests.post(
    'http://localhost:8080/familiarity',
    json={
        'learning_language': 'spa',
        'native_language': 'eng',
        'content': 'El gato está en la mesa.'
    }
)
data = response.json()
for sentence in data['sentences']:
    for token in sentence['tokens']:
        print(f"{token['text']}: {token['familiarity_score']:.2f}")
```
