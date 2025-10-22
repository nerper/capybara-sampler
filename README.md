# Word Familiarity API

A FastAPI application for computing per-token familiarity scores for phrases in target languages based on word frequency.

## Features

- **Token-level Analysis**: Processes phrases using Stanza tokenization with POS tagging and lemmatization
- **Frequency Scoring**: Uses wordfreq library for normalized frequency scores
- **Cognate Boosting**: Advanced cognate detection with CogNet dataset and OpenAI LLM validation
- **Similarity-based Scoring**: Dynamic boost calculation using Jaro-Winkler similarity
- **Multi-language Support**: English (eng), Italian (ita), Spanish (spa), French (fra), Portuguese (por), German (deu)
- **RESTful API**: Clean FastAPI interface with comprehensive documentation
- **Preloaded Models**: All Stanza pipelines and cognate dataset loaded at startup for consistent performance
- **Comprehensive Logging**: Detailed pipeline monitoring and performance tracking

## Supported Languages

| ISO Code | Language   | Stanza Model |
|----------|------------|-------------|
| `eng`    | English    | en          |
| `ita`    | Italian    | it          |
| `spa`    | Spanish    | es          |
| `fra`    | French     | fr          |
| `por`    | Portuguese | pt          |
| `deu`    | German     | de          |

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management

### Setup

1. Navigate to the project directory:
```bash
cd ella-word-familiarity
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. **Setup Environment Variables**: Create a `.env` file in the project root:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Add CogNet Dataset**: Place the `CogNet-top6.tsv` file in the `cognates/` directory for cognate detection functionality.

5. **Optional**: Pre-download Stanza models (they will be downloaded automatically on first startup):
```bash
poetry run python -c "import stanza; stanza.download('es'); stanza.download('en'); stanza.download('it'); stanza.download('fr'); stanza.download('pt'); stanza.download('de')"
```

**Note**: 
- All Stanza models and cognate dataset are preloaded at startup for consistent performance
- Initial startup may take 30-60 seconds while loading all models and processing cognate data
- OpenAI API key is required for cognate validation functionality

## Usage

### Starting the API Server

```bash
poetry run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

**Startup Process:**
- Preloads all Stanza pipelines for supported languages
- Loads and processes CogNet cognate dataset
- Initializes OpenAI client for LLM validation
- Fails fast if any models or datasets cannot be loaded
- Initial startup takes 30-60 seconds but ensures consistent performance

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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

## Running Tests

Execute the test file to verify functionality:

```bash
poetry run python tests.py
```

This will demonstrate the API's token analysis and scoring capabilities.

## Configuration

### Scoring Parameters

Modify scoring parameters in `core/constants.py`:

```python
MIN_COGNATE_BOOST = 0.2    # Minimum cognate boost (for 0 similarity)
MAX_COGNATE_BOOST = 0.4    # Maximum cognate boost (for 1.0 similarity)
MIN_ZIPF = 2.3             # Minimum Zipf score for normalization
MAX_ZIPF = 7.7             # Maximum Zipf score for normalization
COGNATE_BATCH_SIZE = 30    # Maximum cognate pairs per OpenAI API request
```

### Environment Variables

Configure in `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Language Support

Add new languages by updating `SUPPORTED_LANGUAGES` in `core/constants.py` and ensuring corresponding Stanza models are available.

## Data Sources

### Word Frequency Data

Uses the `wordfreq` library which aggregates frequency data from multiple sources including Google Books, Twitter, Wikipedia, and other corpora.

### Cognate Data

Uses the **CogNet dataset** for cross-lingual cognate detection:
- Contains orthographically similar word pairs across 6 languages
- Pre-filtered for top language pairs (English, Spanish, French, Italian, Portuguese, German)
- Enhanced with OpenAI GPT-4o validation for semantic and etymological accuracy
- POS-filtered to ensure grammatical compatibility

### LLM Validation

Integrates **OpenAI GPT-4o** for:
- Semantic validation of cognate candidates
- Context-aware cognate selection
- Etymological relationship verification

## NLP Cognate Finding Process

1. **Tokenization**: Stanza extracts tokens, POS tags, and lemmas.

2. **Search Word Selection**: Uses lemma for nouns/adjectives, surface form for verbs/adverbs. (*Note: Could benefit from regularity check on noun lemmas to avoid using 'mouse' for 'mice'*)

3. **Dataset Lookup**: Exact string matching against CogNet dataset on lowercased word pairs.

4. **POS Filtering**: Rejects candidates where Stanza POS doesn't match cognate concept POS (from concept ID first letter).

5. **LLM Selection**: GPT-4o directly selects the most appropriate cognate candidate considering phrase context, or returns empty string if none are valid.

6. **Similarity-Based Boost**: Calculates Jaro-Winkler similarity between search word and selected cognate, then applies dynamic boost from MIN_COGNATE_BOOST (0.2 for 0 similarity) to MAX_COGNATE_BOOST (0.4 for 1.0 similarity) based on orthographic similarity.

## Project Structure

```
ella-word-familiarity/
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Poetry configuration and dependencies
├── .env                    # Environment variables (API keys)
├── tests.py                # Test cases and examples
├── core/
│   ├── constants.py        # Configuration constants and language mappings
│   ├── tokenizer.py        # Stanza-based tokenization with POS and lemmatization
│   └── score_model.py      # Familiarity scoring with cognate boosting
├── cognates/
│   └── CogNet-top6.tsv     # Cognate dataset for cross-lingual detection
└── data/
    └── README.md           # Data directory documentation
```

## Development

### Adding New Features

1. **New Language Pairs**: Update constants and ensure model availability
2. **Alternative Scoring**: Modify `score_model.py` with new algorithms  
3. **Additional Endpoints**: Extend `main.py` with new FastAPI routes

### Testing

Run the test file to verify functionality:
```bash
poetry run python tests.py
```

For development testing, you can also test individual components:
```bash
# Test tokenization  
poetry run python -c "from core.tokenizer import tokenizer; print(tokenizer.tokenize('Hello world', 'eng'))"

# Test scoring model
poetry run python -c "from core.score_model import familiarity_scorer; print('Familiarity scorer ready')"
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Stanza**: Stanford NLP library for tokenization, POS tagging, and lemmatization
- **wordfreq**: Word frequency data from multiple corpora
- **Polars**: Fast DataFrame library for cognate dataset processing
- **OpenAI**: GPT-4o integration for cognate validation
- **jellyfish**: Jaro-Winkler similarity calculation
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server for FastAPI

## Performance Notes

- **Startup time**: 30-60 seconds due to model and dataset preloading
- **Request latency**: ~200-500ms after startup (includes LLM validation for cognates)
- **Memory usage**: ~2-3GB for all loaded Stanza models and cognate dataset
- **Concurrency**: Cognate searches and LLM validations run concurrently for performance
- **Caching**: Pre-computed cognate results cached per document to avoid redundant processing
- **LLM Batching**: Cognate validation batched to optimize OpenAI API usage

## License

This project is configured for academic and research use.