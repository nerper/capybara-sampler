"""
Core constants and coefficients for the familiarity scoring system.
"""

import logging
import time

# Configure logging for constants
logger = logging.getLogger(__name__)
constants_load_start = time.time()
logger.debug("Loading constants module...")

# Scoring weights
COGNATE_WEIGHT = 0.2

# Frequency normalization
MIN_ZIPF = 2.3
MAX_ZIPF = 7.7

# API configuration
API_VERSION = "1.0.0"

# Language codes
SUPPORTED_LANGUAGES = {
    "eng": "English",
    "ita": "Italian", 
    "spa": "Spanish",
}

constants_load_time = time.time() - constants_load_start
logger.debug("Constants module loaded in %.3f seconds", constants_load_time)