"""
Core constants and coefficients for the familiarity scoring system.
"""

# Scoring weights
MIN_COGNATE_BOOST = 0.2
MAX_COGNATE_BOOST = 0.4

# Frequency normalization
MIN_ZIPF = 2.3
MAX_ZIPF = 7.7

# API configuration
API_VERSION = "1.0.0"

# Cognates configuration
COGNET_PATH = "cognates/CogNet-top6.tsv"
TOP_LANGS = ["eng", "spa", "fra", "ita", "por", "deu"]  # English, Spanish, French, Italian, Portuguese, German
COGNATE_BATCH_SIZE = 30  # Maximum cognate pairs per OpenAI API request

# OpenAI configuration
OPENAI_MODEL = "gpt-4o"
COGNATE_VALIDATION_PROMPT = """You are a linguistic expert specializing in cognate validation. You will receive word pairs that were identified as potential cognates through orthographic similarity search only. Your task is to determine if they are true cognates (words that share a common etymological origin and have related meanings).

Important considerations:
- Words must share etymological origin AND have related meanings in the given context to be considered cognates
- Context matters: "vivid" (English adjective meaning bright/intense) and "vivo" (Spanish adjective meaning bright/lively) ARE cognates, but "vivid" and "vivo" (Spanish verb meaning "I live") are NOT cognates despite sharing etymology

For each pair, respond with only "true" or "false" (no field names, explanations, or additional text to save tokens)."""

# Language codes
# SUPPORTED_LANGUAGES = {
#     "eng": "English",
#     "ita": "Italian", 
#     "spa": "Spanish"
# }

SUPPORTED_LANGUAGES = {
    "eng": "English",
    "ita": "Italian", 
    "spa": "Spanish",
    "fra": "French",
    "por": "Portuguese", 
    "deu": "German",
}