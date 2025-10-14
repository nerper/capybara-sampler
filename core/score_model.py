"""
Core scoring model for computing familiarity scores with frequency and cognate boosting.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import wordfreq
from .constants import COGNATE_WEIGHT, MIN_ZIPF, MAX_ZIPF
from .tokenizer import tokenizer, ISO_TO_STANZA_MAPPING
from .cognate_lookup import cognate_loader

logger = logging.getLogger(__name__)


class FamiliarityScorer:
    """Computes familiarity scores for tokens using frequency and cognate data."""
    
    def __init__(self):
        pass
    
    def normalize_frequency(self, zipf_score: float) -> float:
        """
        Normalize Zipf frequency score to [0,1] range.
        
        Args:
            zipf_score: Raw Zipf frequency score from wordfreq
            
        Returns:
            Normalized frequency score between 0 and 1
        """
        if zipf_score <= 0:
            return 0.0
        
        # Clamp to MIN_ZIPF and MAX_ZIPF range and normalize
        clamped_score = max(min(zipf_score, MAX_ZIPF), MIN_ZIPF)
        return (clamped_score - MIN_ZIPF) / (MAX_ZIPF - MIN_ZIPF)
    
    def get_frequency_score(self, word: str, language: str) -> tuple[float, float]:
        """
        Get normalized frequency score for a word.
        
        Args:
            word: The word to score
            language: 3-letter ISO language code
            
        Returns:
            Tuple of (normalized frequency score, raw zipf score)
        """
        # Convert 3-letter ISO code to 2-letter for wordfreq
        wordfreq_lang = ISO_TO_STANZA_MAPPING.get(language, language)
        zipf_score = wordfreq.zipf_frequency(word, wordfreq_lang)
        normalized = self.normalize_frequency(zipf_score)
        logger.debug("Word '%s' (%s): zipf=%.3f, normalized=%.3f", word, wordfreq_lang, zipf_score, normalized)
        return normalized, zipf_score
    
    def has_cognate(self, word: str, learning_lang: str, native_lang: str) -> bool:
        """
        Check if a word has cognates in the native language.
        
        Args:
            word: Word to check for cognates
            learning_lang: Learning language code
            native_lang: Native language code
            
        Returns:
            True if cognates exist, False otherwise
        """
        cognates = cognate_loader.find_cognates_for_word(word, learning_lang, native_lang)
        return len(cognates) > 0
    
    def compute_token_scores(self, token_info: dict, learning_lang: str, native_lang: str) -> dict:
        """
        Compute familiarity scores for a single token.
        
        Args:
            token_info: Dictionary with token information (text, lemma, pos)
            learning_lang: Learning language code
            native_lang: Native language code
            
        Returns:
            Dictionary with token and computed scores
        """
        original_text = token_info['text']
        word = original_text.lower()
        
        logger.debug("Processing token: '%s' (pos: %s) -> word: '%s'", 
                    original_text, token_info.get('pos', 'unknown'), word)
        
        # Get frequency score using the word directly
        freq_score, zipf_score = self.get_frequency_score(word, learning_lang)
        
        # Compute base familiarity score (use full frequency score, no 0.8 weighting)
        base_familiarity = freq_score
        
        # Prepare result
        result = {
            'text': original_text,  # Use original text to preserve case
            'familiarity_score': round(base_familiarity, 3)
        }
        
        # Check for cognates and add boosted score if found
        stanza_pos = token_info.get('pos')  # Get POS from Stanza tokenization
        cognates = cognate_loader.find_cognates_for_word(word, learning_lang, native_lang, stanza_pos)
        if len(cognates) > 0:
            cognate_familiarity = min(base_familiarity + COGNATE_WEIGHT, 1.0)
            result['cognate_familiarity_score'] = round(cognate_familiarity, 3)
            result['cognate'] = cognates[0]  # Use the first (typically only) cognate found
            logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=%.3f (cognate: %s)", 
                        original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'], result['cognate_familiarity_score'], cognates[0])
            logger.info("Token '%s' has cognate '%s' - boosted score: %.3f -> %.3f", 
                       original_text, cognates[0], result['familiarity_score'], result['cognate_familiarity_score'])
        else:
            logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (no cognate)", 
                        original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
            logger.debug("Token '%s' - no cognate found, base score: %.3f", 
                        original_text, result['familiarity_score'])
        
        return result
    
    def compute_familiarity(self, phrase: str, learning_language: str, 
                          native_language: str) -> dict:
        """
        Compute familiarity scores for all tokens in a phrase.
        
        Args:
            phrase: Input phrase to analyze
            learning_language: Target language code
            native_language: Native language code for cognate boosting
            
        Returns:
            Dictionary with phrase analysis and token scores
        """
        logger.info("Starting familiarity computation for %s -> %s", learning_language, native_language)
        
        # Tokenize the phrase
        tokens = tokenizer.tokenize(phrase, learning_language)
        logger.info("Tokenization complete - processing %d tokens", len(tokens))
        
        # Compute scores for each token
        scored_tokens = []
        cognate_count = 0
        
        for i, token_info in enumerate(tokens):
            logger.debug("Processing token %d/%d", i + 1, len(tokens))
            token_scores = self.compute_token_scores(token_info, learning_language, native_language)
            scored_tokens.append(token_scores)
            
            if 'cognate_familiarity_score' in token_scores:
                cognate_count += 1
        
        logger.info("Scoring complete - %d tokens processed, %d cognates found", len(tokens), cognate_count)
        
        # Prepare response
        result = {
            'phrase': phrase,
            'learning_language': learning_language,
            'native_language': native_language,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'tokens': scored_tokens
        }
        
        return result


# Global scorer instance
familiarity_scorer = FamiliarityScorer()