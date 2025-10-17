"""
Core scoring model for computing familiarity scores based on word frequency.
"""

import logging
from typing import List, Dict
from datetime import datetime
import wordfreq
import polars as pl
from .constants import MIN_ZIPF, MAX_ZIPF, COGNATE_BOOST, COGNET_PATH, TOP_LANGS
from .tokenizer import tokenizer, ISO_TO_STANZA_MAPPING

logger = logging.getLogger(__name__)


class FamiliarityScorer:
    """Computes familiarity scores for tokens using frequency and cognate data."""
    
    def __init__(self):
        self.cognates_df = None
        self.filtered_cognates_df = None
    
    def load_cognates_dataset(self):
        """Load and filter the cognates dataset for the top languages."""
        try:
            logger.info("Loading cognates dataset from %s", COGNET_PATH)
            self.cognates_df = pl.read_csv(
                COGNET_PATH,
                separator="\t",
                has_header=True,
                low_memory=True,
            )
            
            logger.info(f"Loaded {self.cognates_df.shape[0]:,} rows with columns {self.cognates_df.columns}")
            
            # Pre-filter for top languages
            self.filtered_cognates_df = self.cognates_df.filter(
                (self.cognates_df["lang 1"].is_in(TOP_LANGS)) & (self.cognates_df["lang 2"].is_in(TOP_LANGS))
            )
            logger.info(f"Subset size after filtering: {self.filtered_cognates_df.shape[0]:,} rows")
            
        except Exception as e:
            logger.error("Failed to load cognates dataset: %s", str(e))
            raise RuntimeError(f"Could not load cognates dataset: {str(e)}") from e
    
    def find_cognates(self, word: str, learning_language: str, native_language: str) -> pl.DataFrame:
        """
        Find cognates for a given word from learning_language to native_language within the top languages subset.
        Returns a Polars DataFrame with matches.
        """
        if self.filtered_cognates_df is None:
            return pl.DataFrame()
        
        word = word.lower()
        learning_language = learning_language.lower()
        native_language = native_language.lower()

        cond1 = (
            (self.filtered_cognates_df["lang 1"].str.to_lowercase() == learning_language)
            & (self.filtered_cognates_df["word 1"].str.to_lowercase() == word)
            & (self.filtered_cognates_df["lang 2"].str.to_lowercase() == native_language)
        )

        cond2 = (
            (self.filtered_cognates_df["lang 2"].str.to_lowercase() == learning_language)
            & (self.filtered_cognates_df["word 2"].str.to_lowercase() == word)
            & (self.filtered_cognates_df["lang 1"].str.to_lowercase() == native_language)
        )

        results = self.filtered_cognates_df.filter(cond1 | cond2)
        return results.select(
            [
                "concept id",
                "lang 1",
                "word 1",
                "lang 2",
                "word 2",
                "translit 1",
                "translit 2",
            ]
        )
    
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
    
    def compute_token_scores(self, token_info: dict, learning_lang: str, native_lang: str) -> dict:
        """
        Compute familiarity scores for a single token.
        
        Args:
            token_info: Dictionary with token information (text, pos)
            learning_lang: Learning language code
            native_lang: Native language code
            
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
        
        # Compute familiarity score
        familiarity_score = freq_score
        
        # Check for cognates and apply boost if found
        # Only check cognates for nouns, verbs, adverbs, and adjectives
        cognate = None
        cognate_boosted_familiarity_score = None
        
        cognate_eligible_pos = {'NOUN', 'VERB', 'ADV', 'ADJ'}
        
        if (self.filtered_cognates_df is not None and 
            stanza_pos in cognate_eligible_pos):
            
            cognates = self.find_cognates(word, learning_lang, native_lang)
            if len(cognates) > 0:
                # Get the first cognate match
                first_cognate = cognates.row(0, named=True)
                
                # Extract the cognate word (the word in the native language)
                if first_cognate['lang 1'].lower() == native_lang.lower():
                    cognate = first_cognate['word 1']
                else:
                    cognate = first_cognate['word 2']
                
                cognate_boosted_familiarity_score = min(1.0, familiarity_score + COGNATE_BOOST)
                logger.debug("Cognate found for '%s' [%s]: '%s', applying boost %.3f -> %.3f", 
                           word, stanza_pos, cognate, familiarity_score, cognate_boosted_familiarity_score)
        else:
            if stanza_pos not in cognate_eligible_pos:
                logger.debug("Skipping cognate check for '%s' [%s]: POS not eligible", word, stanza_pos)
        
        # Prepare result
        result = {
            'text': original_text,  # Use original text to preserve case
            'familiarity_score': round(familiarity_score, 3),
            'cognate_boosted_familiarity_score': round(cognate_boosted_familiarity_score, 3) if cognate_boosted_familiarity_score is not None else None,
            'cognate': cognate
        }
        
        logger.info("Token '%s' [POS:%s]: zipf=%.3f, familiarity_score=%.3f, cognate='%s', cognate_boosted=%s", 
                    original_text, stanza_pos or 'unknown', zipf_score, 
                    result['familiarity_score'], cognate or 'None',
                    result['cognate_boosted_familiarity_score'] or 'None')
        
        return result
    
    def compute_document_familiarity(self, document: str, learning_language: str, native_language: str) -> dict:
        """
        Compute familiarity scores for all sentences and tokens in a document.
        
        Args:
            document: Input document text to analyze
            learning_language: Target language code
            native_language: Native language code
            
        Returns:
            Dictionary with document analysis and sentence/token scores
        """
        logger.info("Starting document familiarity computation for %s -> %s", learning_language, native_language)
        
        # Tokenize the document into sentences
        sentences_data = tokenizer.tokenize_document(document, learning_language)
        logger.info("Document segmentation complete - processing %d sentences", len(sentences_data))
        
        # Process each sentence
        sentence_scores = []
        total_tokens = 0
        
        for sentence_data in sentences_data:
            index = sentence_data['index']
            text = sentence_data['text']
            tokens = sentence_data['tokens']
            
            logger.info("Processing sentence %d with %d tokens", index, len(tokens))
            
            # Compute scores for each token in this sentence
            scored_tokens = []
            
            for token_info in tokens:
                token_pos = token_info.get('pos', 'unknown')
                
                # Skip punctuation tokens entirely
                if token_pos == 'PUNCT':
                    logger.debug("Skipping punctuation token: '%s' [POS:%s]", token_info['text'], token_pos)
                    continue
                
                token_scores = self.compute_token_scores(token_info, learning_language, native_language)
                scored_tokens.append(token_scores)
            
            # Create sentence score object
            sentence_score = {
                'text': text,
                'index': index,
                'tokens': scored_tokens
            }
            sentence_scores.append(sentence_score)
            
            total_tokens += len(scored_tokens)  # Count only processed tokens (non-punctuation)
            
            logger.info("Sentence %d complete - %d tokens processed", index, len(scored_tokens))
        
        logger.info("Document scoring complete - %d sentences, %d total tokens", 
                   len(sentences_data), total_tokens)
        
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

# Global scorer instance
familiarity_scorer = FamiliarityScorer()