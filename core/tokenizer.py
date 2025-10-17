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
        # Choose processors once; some langs expect mwt (multi-word tokenizer)
        processors = 'tokenize,mwt,pos,lemma'

        if stanza_lang not in self._pipelines:
            try:
                logger.info("Creating new Stanza pipeline for %s", stanza_lang)
                # Try to load the pipeline, download if necessary
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors=processors,
                    verbose=False
                )
                logger.info("Successfully loaded Stanza pipeline for %s", stanza_lang)
            except FileNotFoundError as e:
                logger.warning("Stanza model not found for %s, downloading: %s", stanza_lang, e)
                # If model is not available, download it
                stanza.download(stanza_lang, verbose=False)
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors=processors,
                    verbose=False
                )
                logger.info("Successfully downloaded and loaded Stanza model for %s", stanza_lang)
        
        return self._pipelines[stanza_lang]
    
    def tokenize_document(self, text: str, language: str) -> List[dict]:
        """
        Tokenize a document by first segmenting into sentences, then tokenizing each sentence.
        
        Args:
            text: Input document text to tokenize
            language: Language code (e.g., 'spa', 'eng')
            
        Returns:
            List of dictionaries with sentence information including:
            - text: the original sentence text
            - index: index of the sentence in the document
            - tokens: list of token dictionaries with text, lemma, pos
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.error("Unsupported language for tokenization: %s", language)
            raise ValueError(f"Language '{language}' not supported. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
        
        logger.info("Tokenizing document in language: %s", language)
        pipeline = self._get_pipeline(language)
        doc = pipeline(text)  # This returns a stanza.Document
        
        sentences_data = []
        # doc.sentences is a list of stanza.Sentence objects
        for sent_idx, sentence in enumerate(doc.sentences):  # type: ignore
            tokens = []
            for word in sentence.words:  # type: ignore
                tokens.append({
                    'text': word.text,  # type: ignore
                    'pos': word.pos,  # type: ignore
                    'lemma': word.lemma  # type: ignore
                })
            
            sentences_data.append({
                'text': sentence.text,  # type: ignore
                'index': sent_idx,
                'tokens': tokens
            })
        
        total_tokens = sum(len(sent['tokens']) for sent in sentences_data)
        logger.info("Tokenized document into %d sentences with %d total tokens", len(sentences_data), total_tokens)
        return sentences_data

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
        doc = pipeline(text)  # This returns a stanza.Document
        
        tokens = []
        # doc.sentences is a list of stanza.Sentence objects
        for sentence in doc.sentences:  # type: ignore
            for word in sentence.words:  # type: ignore
                tokens.append({
                    'text': word.text,  # type: ignore
                    'pos': word.pos,  # type: ignore
                    'lemma': word.lemma  # type: ignore
                })
        
        logger.info("Tokenized into %d tokens", len(tokens))
        return tokens


# Global tokenizer instance
tokenizer = StanzaTokenizer()
