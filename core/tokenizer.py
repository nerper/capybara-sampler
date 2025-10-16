"""
Stanza-based tokenizer for processing text in various languages.
"""

import logging
import time
from typing import List
import stanza
from .constants import SUPPORTED_LANGUAGES

# Configure detailed logging
logger = logging.getLogger(__name__)

# Add module-level timing
MODULE_LOAD_START = time.time()
logger.debug("Loading tokenizer module...")

# Time the stanza import
stanza_import_start = time.time()
logger.debug("Importing stanza library...")
logger.debug("Stanza library imported in %.3f seconds", time.time() - stanza_import_start)


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
        preload_total_start = time.time()
        logger.info("=== STANZA PIPELINE PRELOADING STARTED ===")
        logger.info("Preloading all Stanza pipelines for %d languages...", len(SUPPORTED_LANGUAGES))
        
        pipeline_count = 0
        success_count = 0
        
        for iso_code, language_name in SUPPORTED_LANGUAGES.items():
            if iso_code in ISO_TO_STANZA_MAPPING:
                pipeline_count += 1
                lang_start = time.time()
                stanza_code = ISO_TO_STANZA_MAPPING[iso_code]
                
                logger.info("Preloading pipeline %d/%d: %s (%s -> %s)", 
                           pipeline_count, len(ISO_TO_STANZA_MAPPING), 
                           language_name, iso_code, stanza_code)
                
                try:
                    self._get_pipeline(iso_code)
                    success_count += 1
                    lang_duration = time.time() - lang_start
                    logger.info("✓ Pipeline for %s loaded successfully in %.3f seconds", 
                               language_name, lang_duration)
                    
                except Exception as e:
                    lang_duration = time.time() - lang_start
                    logger.error("✗ Failed to preload pipeline for %s after %.3f seconds: %s", 
                                language_name, lang_duration, str(e))
                    # Continue with other languages even if one fails
        
        total_preload_time = time.time() - preload_total_start
        logger.info("=== STANZA PIPELINE PRELOADING COMPLETE ===")
        logger.info("Success: %d/%d pipelines loaded in %.3f seconds", 
                   success_count, pipeline_count, total_preload_time)
        
        if success_count == 0:
            raise RuntimeError("Failed to load any Stanza pipelines")
        elif success_count < pipeline_count:
            logger.warning("Some pipelines failed to load - API may have limited functionality")
    
    def _get_pipeline(self, language: str) -> stanza.Pipeline:
        """Get or create a Stanza pipeline for the specified language."""
        # Convert 3-letter ISO code to 2-letter Stanza code
        stanza_lang = ISO_TO_STANZA_MAPPING.get(language, language)
        
        pipeline_load_start = time.time()
        logger.debug("Loading Stanza pipeline for language: %s (mapped to %s)", language, stanza_lang)
        
        if stanza_lang not in self._pipelines:
            try:
                creation_start = time.time()
                logger.debug("Creating new Stanza pipeline for %s", stanza_lang)
                
                # Try to load the pipeline, download if necessary
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors='tokenize,pos',
                    verbose=False
                )
                
                creation_time = time.time() - creation_start
                logger.info("✓ Successfully created Stanza pipeline for %s in %.3f seconds", 
                           stanza_lang, creation_time)
                
            except FileNotFoundError as e:
                download_start = time.time()
                logger.warning("Stanza model not found for %s, downloading: %s", stanza_lang, e)
                
                # If model is not available, download it
                logger.info("Downloading Stanza model for %s...", stanza_lang)
                stanza.download(stanza_lang, verbose=False)
                
                download_time = time.time() - download_start
                logger.info("Model download completed for %s in %.3f seconds", stanza_lang, download_time)
                
                # Now try to create the pipeline again
                pipeline_creation_start = time.time()
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors='tokenize,pos',
                    verbose=False
                )
                
                pipeline_creation_time = time.time() - pipeline_creation_start
                total_time = time.time() - pipeline_load_start
                logger.info("✓ Successfully downloaded and loaded Stanza model for %s in %.3f seconds total (%.3f download + %.3f creation)", 
                           stanza_lang, total_time, download_time, pipeline_creation_time)
        else:
            logger.debug("Using cached Stanza pipeline for %s", stanza_lang)
        
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
        
        tokenize_start = time.time()
        logger.debug("Tokenizing text in language: %s (text length: %d chars)", language, len(text))
        
        pipeline = self._get_pipeline(language)
        
        doc_processing_start = time.time()
        doc = pipeline(text)  # This returns a stanza.Document
        doc_processing_time = time.time() - doc_processing_start
        
        logger.debug("Stanza processing completed in %.3f seconds", doc_processing_time)
        
        tokens = []
        token_extraction_start = time.time()
        
        # doc.sentences is a list of stanza.Sentence objects
        for sentence in doc.sentences:  # type: ignore
            for word in sentence.words:  # type: ignore
                tokens.append({
                    'text': word.text,  # type: ignore
                    'pos': word.pos  # type: ignore
                })
        
        token_extraction_time = time.time() - token_extraction_start
        total_tokenize_time = time.time() - tokenize_start
        
        logger.debug("Token extraction completed in %.3f seconds", token_extraction_time)
        logger.info("Tokenized into %d tokens in %.3f seconds total", len(tokens), total_tokenize_time)
        
        return tokens


# Global tokenizer instance
logger.debug("Creating global tokenizer instance...")
tokenizer_creation_start = time.time()
tokenizer = StanzaTokenizer()
tokenizer_creation_time = time.time() - tokenizer_creation_start

total_module_time = time.time() - MODULE_LOAD_START
logger.debug("Tokenizer module loaded completely in %.3f seconds (%.3f for tokenizer creation)", 
            total_module_time, tokenizer_creation_time)