"""
Core scoring model for computing familiarity scores based on word frequency.
"""

import logging
import os
from typing import List, Dict, Optional
from datetime import datetime
import wordfreq
import polars as pl
from openai import OpenAI
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
from difflib import SequenceMatcher
from .constants import MIN_ZIPF, MAX_ZIPF, COGNATE_BOOST, COGNET_PATH, TOP_LANGS, OPENAI_MODEL, COGNATE_VALIDATION_PROMPT, COGNATE_BATCH_SIZE
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
    
    def validate_cognates_batch(self, flat_cognate_list: List[tuple], group_indices: Dict, cognate_groups: Dict) -> Dict[tuple, bool]:
        """
        Validate cognate groups using OpenAI with nested array responses.
        
        Args:
            flat_cognate_list: Flattened list of all cognate candidates
            group_indices: Mapping of search_key to (start_idx, end_idx) in flat list
            cognate_groups: Original grouped cognate data
            
        Returns:
            Dictionary mapping individual cognate pairs to validation results after LLM + similarity filtering
        """
        if not self.openai_client or not flat_cognate_list:
            # Default to accepting all pairs when OpenAI unavailable
            return {pair: True for pair in flat_cognate_list}
        
        # First, get LLM validation for all candidates in groups
        grouped_validation_results = self._validate_cognate_groups_with_llm(flat_cognate_list, group_indices, cognate_groups)
        
        # Then apply similarity-based disambiguation for groups with multiple valid cognates
        final_results = {}
        
        for search_key, candidates in cognate_groups.items():
            search_word = search_key[0]
            native_language = search_key[2]
            total_candidates = len(candidates)
            
            # Step 1: Apply LLM validation results to filter candidates
            validated_candidates = []
            for candidate in candidates:
                if candidate in grouped_validation_results and grouped_validation_results[candidate]:
                    validated_candidates.append(candidate)
            
            # Log LLM filtering results
            logger.info("LLM validation for '%s': %d/%d candidates validated as true", 
                       search_word, len(validated_candidates), total_candidates)
            
            if len(validated_candidates) == 0:
                # No valid cognates after LLM filtering
                logger.info("No valid cognates for '%s' after LLM validation", search_word)
                for candidate in candidates:
                    final_results[candidate] = False
            elif len(validated_candidates) == 1:
                # Only one valid cognate after LLM filtering - no disambiguation needed
                logger.info("Single valid cognate for '%s' after LLM validation: '%s'", 
                           search_word, validated_candidates[0][2])
                for candidate in candidates:
                    final_results[candidate] = (candidate in validated_candidates)
            else:
                # Multiple valid cognates after LLM filtering - similarity disambiguation needed
                valid_cognate_words = [c[2] for c in validated_candidates]
                logger.info("Multiple valid cognates for '%s' after LLM validation: %s - disambiguating with similarity", 
                           search_word, valid_cognate_words)
                
                # Use the post-LLM similarity method
                best_candidate = self._select_best_cognate_post_llm(validated_candidates, search_word)
                
                # Mark only the best candidate as true, all others as false
                for candidate in candidates:
                    final_results[candidate] = (candidate == best_candidate)
        
        return final_results
    
    def _validate_cognate_groups_with_llm(self, flat_cognate_list: List[tuple], group_indices: Dict, cognate_groups: Dict) -> Dict[tuple, bool]:
        """
        Send cognate groups to LLM and get nested array responses.
        Batches by groups (search words), not individual cognate pairs.
        """
        # Split into batches by groups (search words), not individual pairs
        batch_size = COGNATE_BATCH_SIZE  # Max groups per batch
        group_keys = list(cognate_groups.keys())
        group_batches = [group_keys[i:i + batch_size] for i in range(0, len(group_keys), batch_size)]
        
        # Convert group batches back to flat cognate lists for processing
        batches = []
        for group_batch in group_batches:
            batch_candidates = []
            for group_key in group_batch:
                batch_candidates.extend(cognate_groups[group_key])
            batches.append(batch_candidates)
        
        logger.info("Processing %d cognate groups (%d total candidates) in %d batches of max %d groups each", 
                   len(group_keys), len(flat_cognate_list), len(batches), batch_size)
        
        all_results = {}
        start_time = datetime.now()
        max_concurrent = min(len(batches), 4)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_batch = {
                executor.submit(self._validate_single_batch_grouped, batch, batch_idx, group_indices, cognate_groups): batch 
                for batch_idx, batch in enumerate(batches)
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.update(batch_results)
                    logger.info("Completed grouped batch with %d pairs", len(batch))
                except Exception as e:
                    logger.error("Grouped batch validation failed: %s", str(e))
                    for pair in batch:
                        all_results[pair] = True
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info("Completed grouped LLM validation of %d cognate candidates in %.2f seconds", 
                   len(all_results), elapsed)
        return all_results
    
    def _validate_single_batch_grouped(self, cognate_batch: List[tuple], batch_idx: int, group_indices: Dict, cognate_groups: Dict) -> Dict[tuple, bool]:
        """
        Validate a single batch with grouped cognate structure for LLM.
        This will request nested arrays like [[true,false], [true], [false,true,false]]
        """
        try:
            # Build prompt showing grouped structure
            group_prompts = []
            group_counter = 1
            
            # We need to organize this batch by groups for the prompt
            batch_by_groups = {}
            for candidate in cognate_batch:
                search_word = candidate[0]
                learning_lang = candidate[1] 
                native_lang = candidate[3]
                search_key = (search_word, learning_lang, native_lang)
                
                if search_key not in batch_by_groups:
                    batch_by_groups[search_key] = []
                batch_by_groups[search_key].append(candidate)
            
            # Create language names mapping
            lang_names = {
                'eng': 'English', 'spa': 'Spanish', 'fra': 'French', 
                'ita': 'Italian', 'por': 'Portuguese', 'deu': 'German'
            }
            
            for search_key, group_candidates in batch_by_groups.items():
                search_word, learning_lang, native_lang = search_key
                lang1_name = lang_names.get(learning_lang, learning_lang)
                lang2_name = lang_names.get(native_lang, native_lang)
                
                cognate_list = [f'"{candidate[2]}"' for candidate in group_candidates]
                context = group_candidates[0][4]  # Use first sentence as context
                
                group_prompts.append(
                    f'{group_counter}. "{search_word}" ({lang1_name}) → [{", ".join(cognate_list)}] ({lang2_name}) - Context: "{context}"'
                )
                group_counter += 1
            
            user_prompt = '\n'.join(group_prompts)
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": COGNATE_VALIDATION_PROMPT + "\n\nRespond with ONLY a JSON array of arrays. Each sub-array contains boolean values for the cognates of one word, in the same order as presented. Example: [[true, false], [true], [false, true, false]]. Do NOT use markdown formatting or code blocks."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=len(cognate_batch) * 15,
                temperature=0
            )
            
            raw_response = response.choices[0].message.content
            if raw_response:
                raw_response = raw_response.strip()
            else:
                raise ValueError("Empty response from OpenAI")
            
            # Raw response will be processed and logged in summary below
            
            # Extract JSON from markdown code blocks if present
            json_content = raw_response
            if "```json" in raw_response:
                # Extract content between ```json and ```
                start_marker = "```json"
                end_marker = "```"
                start_idx = raw_response.find(start_marker) + len(start_marker)
                end_idx = raw_response.find(end_marker, start_idx)
                if start_idx > len(start_marker) - 1 and end_idx > start_idx:
                    json_content = raw_response[start_idx:end_idx].strip()
                    logger.info("Extracted JSON from markdown: %s", json_content)
            elif "```" in raw_response:
                # Handle plain ``` blocks
                parts = raw_response.split("```")
                if len(parts) >= 2:
                    json_content = parts[1].strip()
                    logger.info("Extracted JSON from code block: %s", json_content)
            
            # Parse the nested JSON response
            try:
                nested_response = json.loads(json_content)
                if not isinstance(nested_response, list):
                    raise ValueError("Response is not a list")
                logger.info("Successfully parsed JSON with %d groups", len(nested_response))
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.warning("Failed to parse nested JSON response for batch %d: %s", batch_idx, parse_error)
                logger.warning("Attempted to parse: %s", json_content)
                # Fallback to accepting all - THIS IS THE BUG!
                logger.error("CRITICAL: Falling back to accepting all pairs as true due to parse failure!")
                return {pair: True for pair in cognate_batch}
            
            # Map nested results back to individual candidates
            results = {}
            group_idx = 0
            
            for search_key, group_candidates in batch_by_groups.items():
                if group_idx < len(nested_response):
                    group_results = nested_response[group_idx]
                    if isinstance(group_results, list):
                        for i, candidate in enumerate(group_candidates):
                            if i < len(group_results):
                                results[candidate] = bool(group_results[i])
                            else:
                                results[candidate] = True  # Default
                    else:
                        # Single boolean instead of array
                        for candidate in group_candidates:
                            results[candidate] = bool(group_results)
                else:
                    # Missing group results
                    for candidate in group_candidates:
                        results[candidate] = True
                
                group_idx += 1
            
            # Handle any remaining candidates not covered
            for candidate in cognate_batch:
                if candidate not in results:
                    results[candidate] = True
            
            # Log consolidated LLM validation results
            logger.info("LLM Validation Results - Batch %d:", batch_idx)
            group_idx = 0
            for search_key, group_candidates in batch_by_groups.items():
                search_word = search_key[0]
                cognate_results = []
                context_phrase = group_candidates[0][4] if group_candidates else ""  # Get context from first candidate
                
                for candidate in group_candidates:
                    cognate_word = candidate[2]  # The cognate word
                    llm_result = results.get(candidate, True)
                    status = "✓" if llm_result else "✗"
                    cognate_results.append(f"'{cognate_word}': {status}")
                
                logger.info("  \"%s\" | '%s' → [%s]", context_phrase, search_word, ", ".join(cognate_results))
                group_idx += 1
            
            return results
            
        except Exception as e:
            logger.error("Grouped batch %d validation failed: %s", batch_idx, str(e))
            return {pair: True for pair in cognate_batch}
    
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
            
            logger.info(f"Loaded {self.cognates_df.shape[0]:,} cognate rows")
            
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
    
    def _calculate_cognate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity score between two words for cognate disambiguation.
        Uses string similarity to prioritize more similar cognates.
        
        Args:
            word1: First word to compare
            word2: Second word to compare
            
        Returns:
            Similarity score between 0 and 1 (1 being identical)
        """
        # Normalize words to lowercase for comparison
        word1 = word1.lower().strip()
        word2 = word2.lower().strip()
        
        # Use SequenceMatcher for similarity scoring
        similarity = SequenceMatcher(None, word1, word2).ratio()
        
        return similarity
    
    def _select_best_cognate_post_llm(self, validated_candidates: List[tuple], search_word: str) -> tuple:
        """
        Select the best cognate from LLM-validated candidates using similarity scoring.
        This method only runs AFTER LLM validation, not before.
        
        Args:
            validated_candidates: List of LLM-validated cognate tuples
            search_word: The original word we're searching cognates for
            
        Returns:
            Best cognate tuple based on similarity
        """
        if len(validated_candidates) == 1:
            return validated_candidates[0]
        
        # Calculate similarities for validated candidates
        similarities = []
        for candidate in validated_candidates:
            cognate_word = candidate[2]  # The cognate word
            similarity = self._calculate_cognate_similarity(search_word, cognate_word)
            similarities.append((candidate, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Log disambiguation decision  
        logger.info("Post-LLM similarity disambiguation for '%s' (%d valid candidates):", search_word, len(validated_candidates))
        for i, (candidate, similarity) in enumerate(similarities[:3]):  # Show top 3
            cognate_word = candidate[2]
            marker = " ← SELECTED" if i == 0 else ""
            logger.info("  %d. '%s' (similarity: %.3f)%s", i + 1, cognate_word, similarity, marker)
        
        return similarities[0][0]  # Return best candidate
    
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
    
    def compute_token_scores(self, token_info: dict, learning_lang: str, native_lang: str, cognate_validation_results: Optional[Dict] = None) -> dict:
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
            
            # Use lemma (base form) for cognate search only for nouns to cover plural, otherwise use the word
            if (stanza_pos == 'NOUN' or stanza_pos == 'ADJ') and token_info.get('lemma'):
                lemma = token_info.get('lemma')
                search_word = lemma.lower() if lemma else word
                logger.info("Using lemma '%s' for noun cognate search (original: '%s')", search_word, word)
            else:
                search_word = word
            
            cognates = self.find_cognates(search_word, learning_lang, native_lang)
            if len(cognates) > 0:
                # Look for pre-computed cognate results first (to avoid re-disambiguation)
                cognate_pair_key = None
                candidate_cognate = None
                is_valid_cognate = True  # Default to true if no validation results
                
                if cognate_validation_results is not None:
                    # Try to find this cognate pair in the validation results
                    # The validation results should contain the already-disambiguated cognate
                    logger.debug("Looking for validated cognates for search_word='%s', learning_lang='%s', native_lang='%s'", 
                               search_word, learning_lang, native_lang)
                    logger.debug("Available validation keys: %s", list(cognate_validation_results.keys())[:5])  # Show first 5
                    
                    for pair_key, validation_result in cognate_validation_results.items():
                        logger.debug("Checking pair_key: %s with validation_result: %s", pair_key, validation_result)
                        if (pair_key[0] == search_word and 
                            pair_key[1].lower() == learning_lang.lower() and 
                            pair_key[3].lower() == native_lang.lower()):
                            if validation_result:  # Only use if actually validated as true
                                candidate_cognate = pair_key[2]  # The pre-selected cognate
                                is_valid_cognate = validation_result
                                cognate_pair_key = pair_key
                                logger.info("FOUND MATCH: search_word='%s' -> candidate_cognate='%s', is_valid=%s", 
                                           search_word, candidate_cognate, is_valid_cognate)
                                break
                            else:
                                logger.info("MATCH but validation_result=False for: %s", pair_key)
                
                # If not found in validation results, no cognates available for this token
                if candidate_cognate is None:
                    # No cognates were found or validated for this search term
                    logger.debug("No validated cognates found for '%s' (%s->%s)", search_word, learning_lang, native_lang)
                    # Continue without cognate boost
                    pass
                else:
                    # Store the cognate found before LLM validation
                    cognate_before_LLM = candidate_cognate
                
                # Validation result was already determined above
                
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
        logger.info("Document sentence-wise tokenization complete - processing %d sentences", len(sentences_data))
        
        # First pass: collect all potential cognate candidates (no disambiguation yet)
        cognate_groups = {}  # Dict[search_word_key, Set[cognate_candidates]]
        cognate_contexts = {}  # Dict[search_word_key, sentence_context]
        cognate_eligible_pos = {'NOUN', 'VERB', 'ADV', 'ADJ'}
        
        for sentence_data in sentences_data:
            sentence_text = sentence_data['text']
            for token_info in sentence_data['tokens']:
                stanza_pos = token_info.get('pos', 'unknown')
                
                if stanza_pos in cognate_eligible_pos and stanza_pos != 'PUNCT':
                    word = token_info['text'].lower()
                    # Use lemma (base form) for cognate search only for nouns
                    if (stanza_pos == 'NOUN' or stanza_pos == 'ADJ') and token_info.get('lemma'):
                        search_word = token_info.get('lemma').lower()
                    else:
                        search_word = word
                    cognates = self.find_cognates(search_word, learning_language, native_language)
                    
                    if len(cognates) > 0:
                        # Create a key for this search word context
                        search_key = (search_word, learning_language, native_language)
                        
                        if search_key not in cognate_groups:
                            cognate_groups[search_key] = set()  # Use set to ensure uniqueness
                            cognate_contexts[search_key] = sentence_text  # Store the actual sentence context
                            
                            # Add ALL unique cognate candidates for this search word with POS filtering
                            for i in range(len(cognates)):
                                cognate_row = cognates.row(i, named=True)
                                
                                # Get concept ID and extract POS from first letter
                                concept_id = cognate_row.get('concept id', '')
                                if concept_id:
                                    concept_pos_letter = concept_id[0].lower()
                                    # Map concept ID first letter to POS categories
                                    concept_pos_mapping = {
                                        'n': 'NOUN',
                                        'v': 'VERB', 
                                        'a': 'ADJ',
                                        'r': 'ADV'
                                    }
                                    concept_pos = concept_pos_mapping.get(concept_pos_letter)
                                    
                                    # Check POS compatibility
                                    if concept_pos and concept_pos != stanza_pos:
                                        logger.info("POS mismatch: Stanza '%s' vs Concept '%s' for '%s' -> '%s', skipping", 
                                                   stanza_pos, concept_pos, search_word, 
                                                   cognate_row['word 1'] if cognate_row['lang 1'].lower() == native_language.lower() else cognate_row['word 2'])
                                        continue  # Skip this candidate due to POS mismatch
                                
                                if cognate_row['lang 1'].lower() == native_language.lower():
                                    candidate_cognate = cognate_row['word 1']
                                else:
                                    candidate_cognate = cognate_row['word 2']
                                
                                # Create tuple for LLM validation (no sentence context for uniqueness)
                                cognate_candidate = (search_word, learning_language, candidate_cognate, native_language)
                                cognate_groups[search_key].add(cognate_candidate)
        
        # Convert sets to lists and add actual sentence context for LLM validation
        final_cognate_groups = {}
        for search_key, unique_candidates in cognate_groups.items():
            final_cognate_groups[search_key] = []
            actual_context = cognate_contexts[search_key]  # Use the actual sentence where this word was found
            for candidate in unique_candidates:
                # Add the actual sentence context from where this word was found
                candidate_with_context = candidate + (actual_context,)
                final_cognate_groups[search_key].append(candidate_with_context)
        
        # Flatten groups for batch validation
        flat_cognate_list = []
        group_indices = {}  # Map to reconstruct groups from flat results
        current_index = 0
        
        for search_key, candidates in final_cognate_groups.items():
            group_indices[search_key] = (current_index, current_index + len(candidates))
            flat_cognate_list.extend(candidates)
            current_index += len(candidates)
        
        # Log summary statistics with POS filtering info
        total_unique_candidates = len(flat_cognate_list)
        unique_search_words = len(final_cognate_groups)
        
        logger.info("Found %d unique search words with %d total unique cognate candidates for LLM validation (after POS filtering)", 
                   unique_search_words, total_unique_candidates)
        
        # Validate all cognates in batch and reconstruct grouped results
        if flat_cognate_list:
            cognate_validation_results = self.validate_cognates_batch(flat_cognate_list, group_indices, final_cognate_groups)
        else:
            cognate_validation_results = {}
        
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