# Word Familiarity API

A FastAPI application for computing per-token familiarity scores for phrases in target languages based on word frequency.

## Features

- **Token-level Analysis**: Processes phrases using Stanza tokenization (without lemmatization for speed)
- **Frequency Scoring**: Uses wordfreq library for normalized frequency scores
- **Multi-language Support**: Italian (ita), English (eng), Spanish (spa), French (fra)
- **RESTful API**: Clean FastAPI interface with comprehensive documentation
- **Preloaded Models**: All Stanza pipelines loaded at startup for consistent performance
- **Comprehensive Logging**: Detailed pipeline monitoring

## Supported Languages

| ISO Code | Language | Stanza Model |
|----------|----------|-------------|
| `ita`    | Italian  | it          |
| `eng`    | English  | en          |
| `spa`    | Spanish  | es          |
| `fra`    | French   | fr          |

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

3. **Optional**: Pre-download Stanza models (they will be downloaded automatically on first startup):
```bash
poetry run python -c "import stanza; stanza.download('es'); stanza.download('en'); stanza.download('it'); stanza.download('fr')"
```

**Note**: 
- All Stanza models are preloaded at startup for consistent performance
- Initial startup may take 10-20 seconds while loading all models

## Usage

### Starting the API Server

```bash
poetry run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

**Startup Process:**
- Preloads all Stanza pipelines for supported languages
- Fails fast if any models cannot be loaded
- Initial startup takes 10-20 seconds but ensures consistent performance

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
    "learning_language": "spa"
  }'
```

### Example Response

```json
{
  "content": "yo fui a la escuela",
  "learning_language": "spa", 
  "timestamp": "2025-10-13T16:32:00Z",
  "sentences": [
    {
      "text": "yo fui a la escuela",
      "index": 0,
      "tokens": [
        {"text": "yo", "familiarity_score": 0.85},
        {"text": "fui", "familiarity_score": 0.62},
        {"text": "a", "familiarity_score": 0.98},
        {"text": "la", "familiarity_score": 0.95},
        {"text": "escuela", "familiarity_score": 0.79}
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
COGNATE_WEIGHT = 0.2   # Cognate weight constant (kept for legacy reasons)
MIN_ZIPF = 2.3         # Minimum Zipf score for normalization
MAX_ZIPF = 7.7         # Maximum Zipf score for normalization
```

### Language Support

Add new languages by updating `SUPPORTED_LANGUAGES` in `core/constants.py` and ensuring corresponding Stanza models are available.

## Data Sources

### Word Frequency Data

Uses the `wordfreq` library which aggregates frequency data from multiple sources including Google Books, Twitter, Wikipedia, and other corpora.

## Project Structure

```
ella-word-familiarity/
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Poetry configuration and dependencies
├── tests.py                # Test cases and examples
├── core/
│   ├── constants.py        # Configuration constants and language mappings
│   ├── tokenizer.py        # Stanza-based tokenization (no lemmatization)
│   └── score_model.py      # Familiarity scoring logic based on word frequency
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
- **Stanza**: Stanford NLP library for tokenization and POS tagging
- **wordfreq**: Word frequency data from multiple corpora
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server for FastAPI

## Performance Notes

- **Startup time**: 10-20 seconds due to model preloading
- **Request latency**: ~100-200ms after startup (models are cached)
- **Memory usage**: ~1-2GB for all loaded Stanza models
- **No lemmatization**: Tokenization optimized for speed over linguistic accuracy

## License

This project is configured for academic and research use.