"""
Core scoring model for computing familiarity scores with frequency and cognate boosting.
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

# Configure detailed logging
logger = logging.getLogger(__name__)

# Track module loading time
MODULE_LOAD_START = time.time()
logger.debug("Loading score model module...")

# Time each import
logger.debug("Importing wordfreq...")
wordfreq_import_start = time.time()
import wordfreq
logger.debug("wordfreq imported in %.3f seconds", time.time() - wordfreq_import_start)

logger.debug("Importing constants...")
constants_import_start = time.time()
from .constants import COGNATE_WEIGHT, MIN_ZIPF, MAX_ZIPF
logger.debug("constants imported in %.3f seconds", time.time() - constants_import_start)

logger.debug("Importing tokenizer...")
tokenizer_import_start = time.time()
from .tokenizer import tokenizer, ISO_TO_STANZA_MAPPING
logger.debug("tokenizer imported in %.3f seconds", time.time() - tokenizer_import_start)

logger.debug("Importing OpenAI cognate detector...")
cognate_import_start = time.time()
from .openai_cognate_detector import openai_cognate_detector
logger.debug("OpenAI cognate detector imported in %.3f seconds", time.time() - cognate_import_start)


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
    
    def compute_token_scores(self, token_info: dict, learning_lang: str, native_lang: str, 
                           phrase: Optional[str] = None, cognate_info: Optional[dict] = None) -> dict:
        """
        Compute familiarity scores for a single token.
        
        Args:
            token_info: Dictionary with token information (text, lemma, pos)
            learning_lang: Learning language code
            native_lang: Native language code
            phrase: Full phrase for context (needed for OpenAI cognate detection)
            cognate_info: Pre-computed cognate information for this token
            
        Returns:
            Dictionary with token and computed scores
        """
        original_text = token_info['text']
        word = original_text.lower()
        stanza_pos = token_info.get('pos')
        
        logger.debug("Processing token: '%s' (pos: %s) -> word: '%s'", 
                    original_text, stanza_pos or 'unknown', word)
        
        # Get frequency score using the word directly
        freq_score, zipf_score = self.get_frequency_score(word, learning_lang)
        
        # Compute base familiarity score (use full frequency score, no 0.8 weighting)
        base_familiarity = freq_score
        
        # Prepare result
        result = {
            'text': original_text,  # Use original text to preserve case
            'familiarity_score': round(base_familiarity, 3)
        }
        
        # Check cognate information and apply appropriate scoring
        if cognate_info:
            cognate_type = cognate_info.get('cognate_type', 'none')
            cognate_word = cognate_info.get('cognate_word')
            
            if cognate_type == 'true_cognate' and cognate_word:
                # True cognate: boost the score
                cognate_familiarity = min(base_familiarity + COGNATE_WEIGHT, 1.0)
                result['cognate_familiarity_score'] = round(cognate_familiarity, 3)
                result['cognate'] = cognate_word
                result['cognate_type'] = 'true_cognate'
                logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=%.3f (TRUE cognate: %s)", 
                            original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'], result['cognate_familiarity_score'], cognate_word)
            elif cognate_type == 'false_cognate' and cognate_word:
                # False cognate: penalize the score
                cognate_familiarity = max(base_familiarity - COGNATE_WEIGHT, 0.0)
                result['cognate_familiarity_score'] = round(cognate_familiarity, 3)
                result['cognate'] = f"{cognate_word} (false friend)"
                result['cognate_type'] = 'false_cognate'
                logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=%.3f (FALSE cognate: %s)", 
                            original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'], result['cognate_familiarity_score'], cognate_word)
            else:
                # No cognate: just the base score
                logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (no cognate)", 
                            original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
        else:
            logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (no cognate)", 
                        original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
        
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
        compute_start = time.time()
        logger.info("=== STARTING FAMILIARITY COMPUTATION ===")
        logger.info("Starting familiarity computation for '%s' (%s -> %s)", 
                   phrase, learning_language, native_language)
        
        # Tokenize the phrase
        tokenize_start = time.time()
        logger.info("Starting tokenization...")
        tokens = tokenizer.tokenize(phrase, learning_language)
        tokenize_time = time.time() - tokenize_start
        logger.info("Tokenization complete - processing %d tokens in %.3f seconds", 
                   len(tokens), tokenize_time)
        
        # Get cognate information for all tokens using OpenAI
        cognate_map = {}
        cognate_start = time.time()
        
        try:
            logger.info("=== COGNATE DETECTION PHASE ===")
            logger.info("Calling OpenAI for cognate detection...")
            
            # Get cognate detection for the full phrase
            request = {
                "phrase": phrase,
                "learning_language": learning_language,
                "native_language": native_language,
                "tokens": tokens
            }
            
            openai_call_start = time.time()
            cognate_results = openai_cognate_detector.detect_cognates_batch([request])
            openai_call_time = time.time() - openai_call_start
            
            logger.info("OpenAI call completed in %.3f seconds", openai_call_time)
            
            # Process results and create cognate map
            result_processing_start = time.time()
            if cognate_results.get("results"):
                result_tokens = cognate_results["results"][0].get("tokens", [])
                for token_result in result_tokens:
                    token_text = token_result.get("text", "")
                    cognate_status = token_result.get("cognate_status")
                    cognate_word = token_result.get("cognate")
                    
                    if cognate_status == "true_cognate":
                        cognate_map[token_text.lower()] = {
                            'cognate_type': 'true_cognate',
                            'cognate_word': cognate_word
                        }
                        logger.info("OpenAI found true cognate: '%s' -> '%s'", token_text, cognate_word)
                    elif cognate_status == "false_cognate":
                        cognate_map[token_text.lower()] = {
                            'cognate_type': 'false_cognate',
                            'cognate_word': cognate_word
                        }
                        logger.info("OpenAI found false cognate: '%s' -> '%s'", token_text, cognate_word)
                    else:
                        cognate_map[token_text.lower()] = {
                            'cognate_type': 'none',
                            'cognate_word': None
                        }
            
            result_processing_time = time.time() - result_processing_start
            logger.info("Cognate result processing completed in %.3f seconds", result_processing_time)
            
        except Exception as e:
            cognate_error_time = time.time() - cognate_start
            logger.error("Error calling OpenAI for cognate detection after %.3f seconds: %s", 
                        cognate_error_time, str(e))
            cognate_map = {}
        
        cognate_total_time = time.time() - cognate_start
        logger.info("=== COGNATE DETECTION COMPLETE ===")
        logger.info("Total cognate detection time: %.3f seconds", cognate_total_time)
        
        # Compute scores for each token with cognate information
        scoring_start = time.time()
        logger.info("=== TOKEN SCORING PHASE ===")
        logger.info("Starting token scoring for %d tokens...", len(tokens))
        
        scored_tokens = []
        cognate_count = 0
        
        for i, token_info in enumerate(tokens):
            token_start = time.time()
            logger.debug("Processing token %d/%d: '%s'", i + 1, len(tokens), token_info.get('text', ''))
            
            # Get cognate info for this token
            token_text = token_info['text'].lower()
            token_pos = token_info.get('pos', 'unknown')
            
            # If token not in cognate_map, it means either:
            # 1. It had invalid POS (not NOUN/VERB/ADJ/ADV) so wasn't sent to OpenAI
            # 2. OpenAI didn't return results for it
            # In both cases, we default to no cognate
            cognate_info = cognate_map.get(token_text, {'cognate_type': 'none', 'cognate_word': None})
            
            if token_text not in cognate_map and token_pos not in {'NOUN', 'VERB', 'ADJ', 'ADV'}:
                logger.debug("Token '%s' [POS:%s] skipped for cognate detection (invalid POS)", 
                           token_info['text'], token_pos)
            
            token_scores = self.compute_token_scores(
                token_info, learning_language, native_language, phrase, cognate_info
            )
            scored_tokens.append(token_scores)
            
            if 'cognate_familiarity_score' in token_scores:
                cognate_count += 1
            
            token_time = time.time() - token_start
            logger.debug("Token '%s' scored in %.3f seconds", token_info.get('text', ''), token_time)
        
        scoring_time = time.time() - scoring_start
        logger.info("=== TOKEN SCORING COMPLETE ===")
        logger.info("Scoring complete - %d tokens processed in %.3f seconds, %d cognates found", 
                   len(tokens), scoring_time, cognate_count)
        
        # Prepare response
        response_start = time.time()
        result = {
            'phrase': phrase,
            'learning_language': learning_language,
            'native_language': native_language,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'tokens': scored_tokens
        }
        response_time = time.time() - response_start
        
        total_compute_time = time.time() - compute_start
        logger.info("=== FAMILIARITY COMPUTATION COMPLETE ===")
        logger.info("Response prepared in %.3f seconds", response_time)
        logger.info("TOTAL computation time: %.3f seconds (tokenize: %.3f, cognate: %.3f, scoring: %.3f)", 
                   total_compute_time, tokenize_time, cognate_total_time, scoring_time)
        
        return result


# Global scorer instance
logger.debug("Creating global familiarity scorer instance...")
scorer_creation_start = time.time()
familiarity_scorer = FamiliarityScorer()
scorer_creation_time = time.time() - scorer_creation_start

total_module_time = time.time() - MODULE_LOAD_START
logger.debug("Score model module loaded completely in %.3f seconds (%.3f for scorer creation)", 
            total_module_time, scorer_creation_time)