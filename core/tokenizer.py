"""
Stanza-based tokenizer for processing text in various languages.
"""

import logging
from typing import List
import stanza
from .constants import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


# Mapping from 3-letter ISO codes to Stanza 2-letter codes
ISO_TO_STANZA_MAPPING = {
    "ita": "it",
    "eng": "en",
    "spa": "es", 
    "fra": "fr"
}


class StanzaTokenizer:
    """Tokenizer using Stanza for multilingual text processing."""
    
    def __init__(self):
        self._pipelines = {}
        
    def preload_all_pipelines(self):
        """Preload all Stanza pipelines for supported languages."""
        logger.info("Preloading all Stanza pipelines...")
        
        for iso_code, stanza_code in ISO_TO_STANZA_MAPPING.items():
            if iso_code in SUPPORTED_LANGUAGES:
                try:
                    logger.info("Preloading Stanza pipeline for %s (%s)", iso_code, stanza_code)
                    self._get_pipeline(iso_code)
                except Exception as e:
                    logger.error("Failed to preload pipeline for %s: %s", iso_code, str(e))
                    # Continue with other languages even if one fails
        
        logger.info("Stanza pipeline preloading complete. Loaded %d pipelines", len(self._pipelines))
    
    def _get_pipeline(self, language: str) -> stanza.Pipeline:
        """Get or create a Stanza pipeline for the specified language."""
        # Convert 3-letter ISO code to 2-letter Stanza code
        stanza_lang = ISO_TO_STANZA_MAPPING.get(language, language)
        logger.info("Loading Stanza pipeline for language: %s", stanza_lang)
        
        if stanza_lang not in self._pipelines:
            try:
                logger.info("Creating new Stanza pipeline for %s", stanza_lang)
                # Try to load the pipeline, download if necessary
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors='tokenize,pos',
                    verbose=False
                )
                logger.info("Successfully loaded Stanza pipeline for %s", stanza_lang)
            except Exception as e:
                logger.warning("Stanza model not found for %s, downloading: %s", stanza_lang, e)
                # If model is not available, download it
                stanza.download(stanza_lang, verbose=False)
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors='tokenize,pos',
                    verbose=False
                )
                logger.info("Successfully downloaded and loaded Stanza model for %s", stanza_lang)
        
        return self._pipelines[stanza_lang]
    
    def tokenize(self, text: str, language: str) -> List[dict]:
        """
        Tokenize text using Stanza and return token information.
        
        Args:
            text: Input text to tokenize
            language: Language code (e.g., 'spa', 'eng')
            
        Returns:
            List of dictionaries with token information including:
            - text: the token text
            - lemma: lemmatized form
            - pos: part of speech tag
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.error("Unsupported language for tokenization: %s", language)
            raise ValueError(f"Language '{language}' not supported. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
        
        logger.info("Tokenizing text in language: %s", language)
        pipeline = self._get_pipeline(language)
        doc = pipeline(text)
        
        tokens = []
        for sentence in doc.sentences:
            for word in sentence.words:
                tokens.append({
                    'text': word.text,
                    'pos': word.pos
                })
        
        logger.info("Tokenized into %d tokens", len(tokens))
        return tokens


# Global tokenizer instance
tokenizer = StanzaTokenizer()