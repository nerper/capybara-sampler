"""
Stanza-based tokenizer for processing text in various languages.
"""

import logging
from typing import Any

import stanza  # type: ignore

from .constants import SUPPORTED_LANGUAGES
from .language_codes import (
    DEFAULT_STANZA_PROCESSORS,
    ISO_TO_STANZA_LANG,
    STANZA_PROCESSORS_BY_STANZA_LANG,
)

logger = logging.getLogger(__name__)

# Backward-compatible name; maps canonical ISO 639-3 to Stanza `lang=` code.
ISO_TO_STANZA_MAPPING = ISO_TO_STANZA_LANG


class StanzaTokenizer:
    """Tokenizer using Stanza for multilingual text processing."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}

    def preload_all_pipelines(self) -> None:
        """Preload all Stanza pipelines for supported languages."""
        self.preload_pipelines(list(SUPPORTED_LANGUAGES.keys()))

    def preload_pipelines(self, languages: list[str]) -> None:
        """Preload Stanza pipelines for the given language codes (e.g. ['spa', 'eng'])."""
        logger.info("Preloading Stanza pipelines for %s...", languages)

        for iso_code in languages:
            if iso_code in SUPPORTED_LANGUAGES:
                try:
                    stanza_code = ISO_TO_STANZA_LANG.get(iso_code, iso_code)
                    logger.info("Preloading Stanza pipeline for %s (%s)", iso_code, stanza_code)
                    self._get_pipeline(iso_code)
                except Exception as e:
                    logger.error("Failed to preload pipeline for %s: %s", iso_code, str(e))
                    # Continue with other languages even if one fails

        logger.info("Stanza pipeline preloading complete. Loaded %d pipelines", len(self._pipelines))

    def _get_pipeline(self, language: str) -> stanza.Pipeline:
        """Get or create a Stanza pipeline for the specified language."""
        stanza_lang = ISO_TO_STANZA_LANG.get(language, language)
        processors = STANZA_PROCESSORS_BY_STANZA_LANG.get(stanza_lang, DEFAULT_STANZA_PROCESSORS)

        if stanza_lang not in self._pipelines:
            logger.info("Loading Stanza pipeline for language: %s (%s)", stanza_lang, processors)
            try:
                logger.info("Creating new Stanza pipeline for %s", stanza_lang)
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors=processors,
                    verbose=False,
                    download_method=stanza.DownloadMethod.REUSE_RESOURCES
                )
                logger.info("Successfully loaded Stanza pipeline for %s", stanza_lang)
            except FileNotFoundError as e:
                logger.warning("Stanza model not found for %s, downloading: %s", stanza_lang, e)
                stanza.download(stanza_lang, verbose=False)
                self._pipelines[stanza_lang] = stanza.Pipeline(
                    lang=stanza_lang,
                    processors=processors,
                    verbose=False,
                    download_method=stanza.DownloadMethod.REUSE_RESOURCES
                )
                logger.info("Successfully downloaded and loaded Stanza model for %s", stanza_lang)
        else:
            logger.debug("Using cached Stanza pipeline for %s", stanza_lang)

        return self._pipelines[stanza_lang]

    def tokenize_document(self, text: str, language: str) -> list[dict]:
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
        ner_entities = []  # Track named entities for logging
        for sent_idx, sentence in enumerate(doc.sentences):
            token_to_ner = {}
            if hasattr(sentence, 'ents') and sentence.ents:
                for entity in sentence.ents:
                    ner_entities.append(f"{entity.text}:{entity.type}")
                    logger.debug("NER entity identified: '%s' (type: %s) in sentence %d",
                               entity.text, entity.type, sent_idx)

                    if hasattr(entity, 'tokens'):
                        for token in entity.tokens:
                            if hasattr(token, 'id') and len(token.id) > 0:
                                token_id = token.id[0]
                                token_to_ner[token_id] = entity.type

            tokens = []
            for word in sentence.words:
                word_id = word.id if hasattr(word, 'id') else None

                token_data = {
                    'text': word.text,
                    'pos': word.pos,
                    'lemma': word.lemma,
                    'entity': token_to_ner.get(word_id, None)
                }
                tokens.append(token_data)

            sentences_data.append({
                'text': sentence.text,
                'index': sent_idx,
                'tokens': tokens
            })

        total_tokens = sum(len(sent['tokens']) for sent in sentences_data)
        logger.info("Tokenized document into %d sentences with %d total tokens", len(sentences_data), total_tokens)

        if ner_entities:
            logger.info("Found %d NER entities: %s", len(ner_entities), ', '.join(ner_entities))
        else:
            logger.info("No NER entities found in document")

        return sentences_data


# Global tokenizer instance
tokenizer = StanzaTokenizer()
