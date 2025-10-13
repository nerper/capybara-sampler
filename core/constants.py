"""
Core constants and coefficients for the familiarity scoring system.
"""

# Scoring weights
FREQ_WEIGHT = 0.8
COGNATE_WEIGHT = 0.2

# Frequency normalization
MAX_ZIPF = 7.0

# API configuration
API_VERSION = "1.0.0"

# Language codes
SUPPORTED_LANGUAGES = {
    "ita": "Italian",
    "eng": "English", 
    "spa": "Spanish",
    "fra": "French",
}

# Default values
DEFAULT_NATIVE_LANGUAGE = "eng"