"""
Core constants and coefficients for the familiarity scoring system.
"""

# Scoring weights
COGNATE_BOOST = 0.2

# Frequency normalization
MIN_ZIPF = 2.3
MAX_ZIPF = 7.7

# API configuration
API_VERSION = "1.0.0"

# Cognates configuration
COGNET_PATH = "cognates/CogNet-top6.tsv"
TOP_LANGS = ["eng", "spa", "fra", "ita", "por", "deu"]  # English, Spanish, French, Italian, Portuguese, German

# Language codes
SUPPORTED_LANGUAGES = {
    "eng": "English",
    "ita": "Italian", 
    "spa": "Spanish"
}

# SUPPORTED_LANGUAGES = {
#     "eng": "English",
#     "ita": "Italian", 
#     "spa": "Spanish",
#     "fra": "French",
#     "por": "Portuguese", 
#     "deu": "German",
# }