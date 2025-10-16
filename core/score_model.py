"""
Core scoring model for computing familiarity scores with frequency and cognate boosting.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import wordfreq
from .constants import COGNATE_WEIGHT, MIN_ZIPF, MAX_ZIPF
from .tokenizer import tokenizer, ISO_TO_STANZA_MAPPING
from .openai_cognate_detector import openai_cognate_detector

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
            is_invalid_pos = cognate_info.get('invalid_pos', False)
            
            if cognate_type == 'true_cognate' and cognate_word and cognate_word != "null":
                # True cognate: boost the score
                cognate_familiarity = min(base_familiarity + COGNATE_WEIGHT, 1.0)
                result['cognate_familiarity_score'] = round(cognate_familiarity, 3)
                result['cognate'] = cognate_word
                result['cognate_type'] = 'true_cognate'
                logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=%.3f (TRUE cognate: %s)", 
                            original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'], result['cognate_familiarity_score'], cognate_word)
            elif cognate_type == 'false_cognate' and cognate_word and cognate_word != "null":
                # False cognate: penalize the score
                cognate_familiarity = max(base_familiarity - COGNATE_WEIGHT, 0.0)
                result['cognate_familiarity_score'] = round(cognate_familiarity, 3)
                result['cognate'] = f"{cognate_word} (false friend)"
                result['cognate_type'] = 'false_cognate'
                logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=%.3f (FALSE cognate: %s)", 
                            original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'], result['cognate_familiarity_score'], cognate_word)
            else:
                # No cognate or invalid POS
                if is_invalid_pos:
                    logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (invalid POS)", 
                                original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
                else:
                    logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (no cognate)", 
                                original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
        else:
            logger.info("Token '%s' [POS:%s]: zipf=%.3f, base_score=%.3f, cognate_score=None (no cognate)", 
                        original_text, stanza_pos or 'unknown', zipf_score, result['familiarity_score'])
        
        return result
    
    def compute_document_familiarity(self, document: str, learning_language: str, 
                                   native_language: str) -> dict:
        """
        Compute familiarity scores for all sentences and tokens in a document.
        
        Args:
            document: Input document text to analyze
            learning_language: Target language code
            native_language: Native language code for cognate boosting
            
        Returns:
            Dictionary with document analysis and sentence/token scores
        """
        logger.info("Starting document familiarity computation for %s -> %s", learning_language, native_language)
        
        # Tokenize the document into sentences
        sentences_data = tokenizer.tokenize_document(document, learning_language)
        logger.info("Document segmentation complete - processing %d sentences", len(sentences_data))
        
        # Get cognate information for all sentences using OpenAI
        cognate_map = {}
        translated_sentences = {}  # Store translations for each sentence
        try:
            logger.info("Calling OpenAI for cognate detection on all sentences...")
            
            cognate_results = openai_cognate_detector.detect_cognates_for_sentences(
                sentences_data, learning_language, native_language
            )
            
            # Process results and create cognate map organized by sentence index
            if cognate_results.get("results"):
                for sentence_result in cognate_results["results"]:
                    index = sentence_result.get("index", 0)
                    tokens = sentence_result.get("tokens", [])
                    translated_text = sentence_result.get("translated_text", "")
                    
                    # Store the translated text for this sentence
                    if translated_text:
                        translated_sentences[index] = translated_text
                        logger.info("Sentence %d translated to: '%s'", index, translated_text)
                    
                    if index not in cognate_map:
                        cognate_map[index] = {}
                    
                    for token_result in tokens:
                        token_text = token_result.get("text", "")
                        cognate_status = token_result.get("cognate_status")
                        cognate_word = token_result.get("cognate")
                        
                        # Handle contractions like "l'hotel" -> ["l'", "hotel"]
                        # Check if token_text contains an apostrophe and might be a contraction
                        if "'" in token_text:
                            # Split on apostrophe and map the second part (main word)
                            parts = token_text.split("'", 1)
                            if len(parts) == 2:
                                main_word = parts[1].lower()  # e.g., "hotel" from "l'hotel"
                                contraction_prefix = parts[0].lower() + "'"  # e.g., "l'" from "l'hotel"
                                
                                # Map both the full form and the main word
                                if cognate_status == "true_cognate":
                                    cognate_map[index][token_text.lower()] = {
                                        'cognate_type': 'true_cognate',
                                        'cognate_word': cognate_word
                                    }
                                    cognate_map[index][main_word] = {
                                        'cognate_type': 'true_cognate',
                                        'cognate_word': cognate_word
                                    }
                                    logger.info("OpenAI found true cognate in sentence %d: '%s' -> '%s' (mapped to both '%s' and '%s')", 
                                               index, token_text, cognate_word, token_text.lower(), main_word)
                                elif cognate_status == "false_cognate":
                                    cognate_map[index][token_text.lower()] = {
                                        'cognate_type': 'false_cognate',
                                        'cognate_word': cognate_word
                                    }
                                    cognate_map[index][main_word] = {
                                        'cognate_type': 'false_cognate',
                                        'cognate_word': cognate_word
                                    }
                                    logger.info("OpenAI found false cognate in sentence %d: '%s' -> '%s' (mapped to both '%s' and '%s')", 
                                               index, token_text, cognate_word, token_text.lower(), main_word)
                                else:
                                    cognate_map[index][token_text.lower()] = {
                                        'cognate_type': 'none',
                                        'cognate_word': None
                                    }
                                    cognate_map[index][main_word] = {
                                        'cognate_type': 'none',
                                        'cognate_word': None
                                    }
                                continue
                        
                        # Regular token mapping
                        if cognate_status == "true_cognate":
                            cognate_map[index][token_text.lower()] = {
                                'cognate_type': 'true_cognate',
                                'cognate_word': cognate_word
                            }
                            logger.info("OpenAI found true cognate in sentence %d: '%s' -> '%s'", 
                                       index, token_text, cognate_word)
                        elif cognate_status == "false_cognate":
                            cognate_map[index][token_text.lower()] = {
                                'cognate_type': 'false_cognate',
                                'cognate_word': cognate_word
                            }
                            logger.info("OpenAI found false cognate in sentence %d: '%s' -> '%s'", 
                                       index, token_text, cognate_word)
                        else:
                            cognate_map[index][token_text.lower()] = {
                                'cognate_type': 'none',
                                'cognate_word': None
                            }
            
        except Exception as e:
            logger.error("Error calling OpenAI for cognate detection: %s", str(e))
            cognate_map = {}
            translated_sentences = {}
        
        # Process each sentence
        sentence_scores = []
        total_tokens = 0
        total_cognates = 0
        
        for sentence_data in sentences_data:
            index = sentence_data['index']
            text = sentence_data['text']
            tokens = sentence_data['tokens']
            
            logger.info("Processing sentence %d with %d tokens", index, len(tokens))
            
            # Get cognate map for this sentence
            sentence_cognate_map = cognate_map.get(index, {})
            
            # Compute scores for each token in this sentence
            scored_tokens = []
            sentence_cognate_count = 0
            
            for token_info in tokens:
                token_text = token_info['text'].lower()
                token_pos = token_info.get('pos', 'unknown')
                
                # Skip punctuation tokens entirely
                if token_pos == 'PUNCT':
                    logger.debug("Skipping punctuation token: '%s' [POS:%s]", token_info['text'], token_pos)
                    continue
                
                # Get cognate info for this token
                cognate_info = sentence_cognate_map.get(token_text, {'cognate_type': 'none', 'cognate_word': None})
                
                # Check if POS is invalid for cognate detection
                is_invalid_pos = token_pos not in {'NOUN', 'VERB', 'ADJ', 'ADV'}
                if token_text not in sentence_cognate_map and is_invalid_pos:
                    logger.debug("Token '%s' [POS:%s] in sentence %d skipped for cognate detection (invalid POS)", 
                               token_info['text'], token_pos, index)
                    # Mark as invalid POS for logging purposes
                    cognate_info['invalid_pos'] = True
                
                token_scores = self.compute_token_scores(
                    token_info, learning_language, native_language, text, cognate_info
                )
                scored_tokens.append(token_scores)
                
                if 'cognate_familiarity_score' in token_scores:
                    sentence_cognate_count += 1
            
            # Create sentence score object
            sentence_score = {
                'text': text,
                'index': index,
                'translated_text': translated_sentences.get(index),  # Include translated text if available
                'tokens': scored_tokens
            }
            sentence_scores.append(sentence_score)
            
            total_tokens += len(scored_tokens)  # Count only processed tokens (non-punctuation)
            total_cognates += sentence_cognate_count
            
            logger.info("Sentence %d complete - %d tokens processed, %d cognates found", 
                       index, len(scored_tokens), sentence_cognate_count)
        
        logger.info("Document scoring complete - %d sentences, %d total tokens, %d total cognates", 
                   len(sentences_data), total_tokens, total_cognates)
        
        # Prepare response
        result = {
            'content': document,
            'learning_language': learning_language,
            'native_language': native_language,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sentences': sentence_scores,
            'total_tokens': total_tokens
        }
        
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
        
        # Get cognate information for all tokens using OpenAI
        cognate_map = {}
        try:
            logger.info("Calling OpenAI for cognate detection...")
            
            # Get cognate detection for the full phrase
            request = {
                "phrase": phrase,
                "learning_language": learning_language,
                "native_language": native_language,
                "tokens": tokens
            }
            
            cognate_results = openai_cognate_detector.detect_cognates_batch([request])
            
            # Process results and create cognate map
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
            
        except Exception as e:
            logger.error("Error calling OpenAI for cognate detection: %s", str(e))
            cognate_map = {}
        
        # Compute scores for each token with cognate information
        scored_tokens = []
        cognate_count = 0
        
        for i, token_info in enumerate(tokens):
            logger.debug("Processing token %d/%d", i + 1, len(tokens))
            
            # Get cognate info for this token
            token_text = token_info['text'].lower()
            token_pos = token_info.get('pos', 'unknown')
            
            # Skip punctuation tokens entirely
            if token_pos == 'PUNCT':
                logger.debug("Skipping punctuation token: '%s' [POS:%s]", token_info['text'], token_pos)
                continue
            
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
        
        logger.info("Scoring complete - %d tokens processed, %d cognates found", len(scored_tokens), cognate_count)
        
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