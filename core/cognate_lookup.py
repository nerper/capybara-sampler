"""
CogNet cognate lookup functionality using Polars for efficient data handling.
Downloads and processes the CogNet v2.0 dataset automatically.
"""

import logging
import polars as pl
import csv
import requests
import zipfile
import wordfreq
from pathlib import Path
from typing import Optional, Set
import os
from .tokenizer import ISO_TO_STANZA_MAPPING
from .constants import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Mapping from Stanza Universal POS to WordNet POS
STANZA_TO_WORDNET_POS = {
    'NOUN': 'n',
    'VERB': 'v', 
    'ADJ': 'a',
    'ADV': 'r'
}


class CognateLoader:
    """Handles loading and querying cognate data from CogNet TSV dataset."""
    
    COGNET_URL = "https://github.com/kbatsuren/CogNet/raw/master/CogNet-v2.0.zip"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.tsv_path = self.data_dir / "CogNet-v2.0.tsv"
        self._cognate_data = {}
        self._full_dataset = None
        
    def preload_dataset(self):
        """Preload the full CogNet dataset into memory."""
        logger.info("Preloading CogNet dataset...")
        self._load_full_dataset()
        logger.info("CogNet dataset preloading complete")
    
    def _extract_pos_from_concept_id(self, concept_id: str) -> Optional[str]:
        """
        Extract POS from CogNet concept ID.
        
        Args:
            concept_id: CogNet concept ID (e.g., 'n01321824')
            
        Returns:
            WordNet POS letter ('n', 'v', 'a', 'r') or None if invalid
        """
        if not concept_id or len(concept_id) < 1:
            return None
        
        pos_letter = concept_id[0].lower()
        if pos_letter in ['n', 'v', 'a', 's', 'r']:
            # Convert 's' (satellite adjective) to 'a' (adjective)
            return 'a' if pos_letter == 's' else pos_letter
        
        return None
    
    def _stanza_to_wordnet_pos(self, stanza_pos: str) -> Optional[str]:
        """
        Convert Stanza Universal POS to WordNet POS.
        
        Args:
            stanza_pos: Stanza Universal POS tag (e.g., 'NOUN', 'VERB')
            
        Returns:
            WordNet POS letter ('n', 'v', 'a', 'r') or None if not mappable
        """
        return STANZA_TO_WORDNET_POS.get(stanza_pos.upper())
    
    def _pos_matches(self, stanza_pos: str, concept_id: str) -> bool:
        """
        Check if Stanza POS matches CogNet concept ID POS.
        
        Args:
            stanza_pos: Stanza Universal POS tag
            concept_id: CogNet concept ID
            
        Returns:
            True if POS tags match, False otherwise
        """
        if not stanza_pos or not concept_id:
            return True  # Allow if either is missing
        
        wordnet_pos = self._stanza_to_wordnet_pos(stanza_pos)
        concept_pos = self._extract_pos_from_concept_id(concept_id)
        
        if wordnet_pos is None or concept_pos is None:
            return True  # Allow if conversion fails
        
        return wordnet_pos == concept_pos
        
    def _download_cognet_data(self) -> bool:
        """Download and extract CogNet dataset if not present."""
        if self.tsv_path.exists():
            logger.info("CogNet dataset already exists at %s", self.tsv_path)
            return True
            
        logger.info("CogNet dataset not found, starting download from %s", self.COGNET_URL)
        zip_path = self.data_dir / "CogNet-v2.0.zip"
        
        try:
            # Download the zip file
            logger.info("Downloading CogNet zip file...")
            response = requests.get(self.COGNET_URL, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download complete, extracting TSV file...")
            
            # Extract the TSV file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("CogNet-v2.0.tsv", self.data_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            logger.info("CogNet dataset downloaded and extracted successfully to %s", self.tsv_path)
            return True
            
        except Exception as e:
            logger.error("Error downloading CogNet dataset: %s", str(e))
            return False
    
    def _load_full_dataset(self) -> pl.DataFrame:
        """Load the complete CogNet dataset using chunked processing."""
        if self._full_dataset is not None:
            logger.debug("Using cached CogNet dataset")
            return self._full_dataset
            
        if not self.tsv_path.exists():
            if not self._download_cognet_data():
                # Return empty dataframe if download failed
                logger.error("Failed to download CogNet dataset, returning empty DataFrame")
                return pl.DataFrame({
                    'concept id': pl.Series([], dtype=pl.Utf8),
                    'lang 1': pl.Series([], dtype=pl.Utf8),
                    'word 1': pl.Series([], dtype=pl.Utf8), 
                    'lang 2': pl.Series([], dtype=pl.Utf8),
                    'word 2': pl.Series([], dtype=pl.Utf8),
                    'translit 1': pl.Series([], dtype=pl.Utf8),
                    'translit 2': pl.Series([], dtype=pl.Utf8)
                })
        
        try:
            logger.info("Loading CogNet dataset from %s using polars with language filtering", self.tsv_path)
            
            # Get supported languages dynamically
            target_langs = set(SUPPORTED_LANGUAGES.keys())
            logger.info("Filtering for languages: %s", target_langs)
            
            # Use polars lazy evaluation for efficient filtering
            self._full_dataset = (
                pl.scan_csv(
                    self.tsv_path,
                    separator="\t",
                    has_header=True,
                    encoding="utf8",
                    truncate_ragged_lines=True,
                    quote_char=None,
                )
                .filter(
                    # Filter for supported languages and minimum word length
                    pl.col("lang 1").is_in(list(target_langs)) &
                    pl.col("lang 2").is_in(list(target_langs)) &
                    (pl.col("word 1").str.lengths() >= 3) &
                    (pl.col("word 2").str.lengths() >= 3)
                )
                .collect()
            )
            
            logger.info("Polars filtering complete: loaded %d filtered pairs", len(self._full_dataset))
            
            return self._full_dataset
            
        except Exception as e:
            logger.error("Error loading CogNet dataset: %s", str(e))
            return pl.DataFrame({
                'concept id': pl.Series([], dtype=pl.Utf8),
                'lang 1': pl.Series([], dtype=pl.Utf8),
                'word 1': pl.Series([], dtype=pl.Utf8),
                'lang 2': pl.Series([], dtype=pl.Utf8), 
                'word 2': pl.Series([], dtype=pl.Utf8),
                'translit 1': pl.Series([], dtype=pl.Utf8),
                'translit 2': pl.Series([], dtype=pl.Utf8)
            })
    
    def get_cognate_data(self, learning_lang: str, native_lang: str) -> pl.DataFrame:
        """Get cognate data for a specific language pair."""
        lang_pair = f"{learning_lang}-{native_lang}"
        
        if lang_pair not in self._cognate_data:
            logger.debug("Loading cognate data for language pair: %s", lang_pair)
            full_dataset = self._load_full_dataset()
            
            if full_dataset.is_empty():
                logger.warning("No cognate data available for %s", lang_pair)
                self._cognate_data[lang_pair] = full_dataset
            else:
                # Filter for the specific language pair
                langs = [learning_lang, native_lang]
                filtered_data = full_dataset.filter(
                    pl.col("lang 1").is_in(langs) & pl.col("lang 2").is_in(langs)
                )
                self._cognate_data[lang_pair] = filtered_data
                logger.info("Filtered %d cognate pairs for language pair %s", 
                           len(filtered_data), lang_pair)
                
        return self._cognate_data[lang_pair]
    
    def find_cognates_for_word(self, word: str, learning_lang: str, native_lang: str, pos: Optional[str] = None) -> list:
        """
        Find all cognates for a given word using the preloaded filtered dataset with POS validation.
        
        Args:
            word: Word to find cognates for
            learning_lang: Learning language code (3-letter ISO)
            native_lang: Native language code (3-letter ISO)
            pos: Optional Stanza POS tag for validation (e.g., 'NOUN', 'VERB')
            
        Returns:
            List of cognate words in the other language
        """
        # Use the preloaded and filtered dataset
        dataset = self._load_full_dataset()
        
        if dataset.is_empty():
            logger.debug("No dataset available for cognate search")
            return []
        
        try:
            # Case-insensitive search for both possible orientations using preloaded data
            word_lower = word.lower()
            result = dataset.filter(
                (
                    (pl.col("lang 1") == learning_lang)
                    & (pl.col("word 1").str.to_lowercase() == word_lower)
                    & (pl.col("lang 2") == native_lang)
                )
                |
                (
                    (pl.col("lang 2") == learning_lang)
                    & (pl.col("word 2").str.to_lowercase() == word_lower)
                    & (pl.col("lang 1") == native_lang)
                )
            )
            
            if result.is_empty():
                logger.debug("No cognate matches found for '%s' (%s->%s)", word, learning_lang, native_lang)
                return []
            
            logger.info("Found %d cognate matches for '%s' (%s->%s):", len(result), word, learning_lang, native_lang)
            
            # Log all matching rows with complete data
            for i in range(len(result)):
                row = result.row(i, named=True)
                logger.info("  Row %d: %s", i + 1, dict(row))
            
            # Extract cognate candidates with POS validation
            valid_cognates = []
            
            for i in range(len(result)):
                row = result.row(i, named=True)
                concept_id = row.get("concept id", "")
                
                # Check POS validation if provided
                if pos and not self._pos_matches(pos, concept_id):
                    # Determine the cognate pair for debugging
                    if row["lang 1"] == learning_lang and row["word 1"].lower() == word_lower:
                        cognate_pair = f"{row['word 1']} <-> {row['word 2']}"
                    elif row["lang 2"] == learning_lang and row["word 2"].lower() == word_lower:
                        cognate_pair = f"{row['word 2']} <-> {row['word 1']}"
                    else:
                        cognate_pair = f"{row['word 1']} <-> {row['word 2']}"
                    
                    logger.info("  Skipping cognate pair '%s' due to POS mismatch: Stanza='%s', CogNet='%s'", 
                               cognate_pair, pos, self._extract_pos_from_concept_id(concept_id))
                    continue
                
                # Determine which word is the cognate
                if row["lang 1"] == learning_lang and row["word 1"].lower() == word_lower:
                    candidate_cognate = row["word 2"]
                    cognate_pair = f"{row['word 1']} <-> {row['word 2']}"
                elif row["lang 2"] == learning_lang and row["word 2"].lower() == word_lower:
                    candidate_cognate = row["word 1"]
                    cognate_pair = f"{row['word 2']} <-> {row['word 1']}"
                else:
                    continue
                
                valid_cognates.append(candidate_cognate)
                logger.info("  Accepted cognate pair '%s' (POS matches)", cognate_pair)
            
            if valid_cognates:
                logger.info("After POS filtering: %d valid cognates", len(valid_cognates))
                return [valid_cognates[0]]  # Return first cognate
            else:
                logger.info("No cognates passed POS validation")
                return []
                
        except Exception as e:
            logger.error("Error searching for cognates: %s", str(e))
            return []
    
    def get_available_languages(self) -> Set[str]:
        """Get set of all available language codes in the dataset."""
        full_dataset = self._load_full_dataset()
        
        if full_dataset.is_empty():
            return set()
            
        lang1_codes = set(full_dataset.select("lang 1").to_series().unique().to_list())
        lang2_codes = set(full_dataset.select("lang 2").to_series().unique().to_list())
        
        return lang1_codes.union(lang2_codes)


# Global cognate loader instance
cognate_loader = CognateLoader()