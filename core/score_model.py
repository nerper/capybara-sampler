"""
Core scoring model for computing familiarity scores based on word frequency.
"""

import logging
import os
from typing import List, Dict
from datetime import datetime
import wordfreq
import polars as pl
from openai import OpenAI
from .constants import MIN_ZIPF, MAX_ZIPF, COGNATE_BOOST, COGNET_PATH, TOP_LANGS, OPENAI_MODEL, COGNATE_VALIDATION_PROMPT
from .tokenizer import tokenizer, ISO_TO_STANZA_MAPPING

logger = logging.getLogger(__name__)


class FamiliarityScorer:
    """Computes familiarity scores for tokens using frequency and cognate data."""
    
    def __init__(self):
        self.cognates_df = None
        self.filtered_cognates_df = None
        self.openai_client = None
        self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client if API key is available."""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", str(e))
                self.openai_client = None
        else:
            logger.warning("OPENAI_API_KEY not found - cognate validation will be skipped")
    
    def validate_cognates_batch(self, cognate_pairs: List[tuple]) -> Dict[tuple, bool]:
        """
        Validate multiple cognate pairs using OpenAI GPT-4o Mini in a single request.
        
        Args:
            cognate_pairs: List of tuples (word1, lang1, word2, lang2, phrase)
            
        Returns:
            Dictionary mapping cognate pairs to validation results
        """
        if not self.openai_client or not cognate_pairs:
            # Default to accepting all pairs when OpenAI unavailable
            return {pair: True for pair in cognate_pairs}
        
        try:
            # Create language names mapping
            lang_names = {
                'eng': 'English', 'spa': 'Spanish', 'fra': 'French', 
                'ita': 'Italian', 'por': 'Portuguese', 'deu': 'German'
            }
            
            # Build the batch prompt
            pair_prompts = []
            for i, (word1, lang1, word2, lang2, phrase) in enumerate(cognate_pairs, 1):
                lang1_name = lang_names.get(lang1, lang1)
                lang2_name = lang_names.get(lang2, lang2)
                pair_prompts.append(f'{i}. "{word1}" ({lang1_name}) and "{word2}" ({lang2_name}) - Context: "{phrase}"')
            
            user_prompt = '\n'.join(pair_prompts)
            
            # Log the input prompt
            logger.info("OpenAI cognate validation batch input:\n%s", user_prompt)
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": COGNATE_VALIDATION_PROMPT + "\n\nRespond with one 'true' or 'false' per line, matching the numbered order."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=len(cognate_pairs) * 10,
                temperature=0
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Log the raw response
            logger.info("OpenAI cognate validation batch raw response:\n%s", raw_response)
            
            # Parse the response
            response_lines = [line.strip().lower() for line in raw_response.split('\n') if line.strip()]
            results = {}
            
            for i, pair in enumerate(cognate_pairs):
                if i < len(response_lines):
                    is_valid = response_lines[i] == "true"
                    results[pair] = is_valid
                    logger.info("Cognate validation: %s(%s)-%s(%s) in '%s' -> %s", 
                               pair[0], pair[1], pair[2], pair[3], pair[4], is_valid)
                else:
                    # Default to True if response is shorter than expected
                    results[pair] = True
                    logger.warning("Missing response for cognate pair %s(%s)-%s(%s), defaulting to True", 
                                 pair[0], pair[1], pair[2], pair[3])
            
            return results
            
        except Exception as e:
            logger.error("OpenAI batch validation failed: %s", str(e))
            # Default to accepting all pairs on API errors
            return {pair: True for pair in cognate_pairs}
    
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
    
    def compute_token_scores(self, token_info: dict, learning_lang: str, native_lang: str, cognate_validation_results: Dict = None) -> dict:
        """
        Compute familiarity scores for a single token.
        
        Args:
            token_info: Dictionary with token information (text, pos)
            learning_lang: Learning language code
            native_lang: Native language code
            cognate_validation_results: Dictionary with validation results from OpenAI
            
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
        cognate_before_LLM = None
        cognate_after_LLM = None
        cognate_boosted_familiarity_score = None
        
        cognate_eligible_pos = {'NOUN', 'VERB', 'ADV', 'ADJ'}
        
        if (self.filtered_cognates_df is not None and 
            stanza_pos in cognate_eligible_pos):
            
            # Use lemma (base form) for cognate search only for nouns, otherwise use the word
            if stanza_pos == 'NOUN' and token_info.get('lemma'):
                search_word = token_info.get('lemma').lower()
                logger.info("Using lemma '%s' for noun cognate search (original: '%s')", search_word, word)
            else:
                search_word = word
            
            cognates = self.find_cognates(search_word, learning_lang, native_lang)
            if len(cognates) > 0:
                # Get the first cognate match
                first_cognate = cognates.row(0, named=True)
                
                # Extract the cognate word (the word in the native language)
                if first_cognate['lang 1'].lower() == native_lang.lower():
                    candidate_cognate = first_cognate['word 1']
                else:
                    candidate_cognate = first_cognate['word 2']
                
                # Store the cognate found before LLM validation
                cognate_before_LLM = candidate_cognate
                
                # Look for this cognate pair in validation results
                is_valid_cognate = True  # Default to true if no validation results
                if cognate_validation_results is not None:
                    is_valid_cognate = False
                    for (val_word, val_lang1, val_cognate, val_lang2, val_sentence), is_valid in cognate_validation_results.items():
                        if (val_word == search_word and val_lang1 == learning_lang and 
                            val_cognate == candidate_cognate and val_lang2 == native_lang):
                            is_valid_cognate = is_valid
                            break
                
                if is_valid_cognate:
                    cognate_after_LLM = candidate_cognate  # Cognate after LLM validation
                    cognate_boosted_familiarity_score = min(1.0, familiarity_score + COGNATE_BOOST)
                    logger.debug("Valid cognate confirmed for '%s' [%s]: '%s', applying boost %.3f -> %.3f", 
                               word, stanza_pos, cognate_after_LLM, familiarity_score, cognate_boosted_familiarity_score)
                else:
                    logger.debug("OpenAI rejected cognate pair: '%s'(%s) - '%s'(%s)", 
                               word, learning_lang, candidate_cognate, native_lang)
        else:
            if stanza_pos not in cognate_eligible_pos:
                logger.debug("Skipping cognate check for '%s' [%s]: POS not eligible", word, stanza_pos)
        
        # Prepare result
        result = {
            'text': original_text,  # Use original text to preserve case
            'familiarity_score': round(familiarity_score, 3),
            'cognate_boosted_familiarity_score': round(cognate_boosted_familiarity_score, 3) if cognate_boosted_familiarity_score is not None else None,
            'cognate_before_LLM': cognate_before_LLM,
            'cognate_after_LLM': cognate_after_LLM
        }
        
        logger.info("Token '%s' [POS:%s]: zipf=%.3f, familiarity_score=%.3f, cognate_before='%s', cognate_after='%s', cognate_boosted=%s", 
                    original_text, stanza_pos or 'unknown', zipf_score, 
                    result['familiarity_score'], cognate_before_LLM or 'None', cognate_after_LLM or 'None',
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
        
        # First pass: collect all potential cognate candidates
        cognate_candidates = []
        cognate_pairs_seen = set()  # Track unique pairs to avoid duplicates
        cognate_eligible_pos = {'NOUN', 'VERB', 'ADV', 'ADJ'}
        
        if self.filtered_cognates_df is not None:
            for sentence_data in sentences_data:
                sentence_text = sentence_data['text']
                for token_info in sentence_data['tokens']:
                    stanza_pos = token_info.get('pos', 'unknown')
                    
                    if stanza_pos in cognate_eligible_pos and stanza_pos != 'PUNCT':
                        word = token_info['text'].lower()
                        # Use lemma (base form) for cognate search only for nouns
                        if stanza_pos == 'NOUN' and token_info.get('lemma'):
                            search_word = token_info.get('lemma').lower()
                        else:
                            search_word = word
                        cognates = self.find_cognates(search_word, learning_language, native_language)
                        
                        if len(cognates) > 0:
                            first_cognate = cognates.row(0, named=True)
                            
                            if first_cognate['lang 1'].lower() == native_language.lower():
                                candidate_cognate = first_cognate['word 1']
                            else:
                                candidate_cognate = first_cognate['word 2']
                            
                            # Create simple 4-tuple key for deduplication using search_word (lemma)
                            cognate_key = (search_word, learning_language, candidate_cognate, native_language)
                            
                            if cognate_key not in cognate_pairs_seen:
                                cognate_pairs_seen.add(cognate_key)
                                # Add 5-tuple with sentence context for OpenAI validation
                                cognate_pair = (search_word, learning_language, candidate_cognate, native_language, sentence_text)
                                cognate_candidates.append(cognate_pair)
        
        logger.info("Found %d unique cognate candidates for batch validation", len(cognate_candidates))
        
        # Validate all cognates in batch
        cognate_validation_results = self.validate_cognates_batch(cognate_candidates)
        
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
                
                token_scores = self.compute_token_scores(token_info, learning_language, native_language, cognate_validation_results)
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