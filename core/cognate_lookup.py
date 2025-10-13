"""
CogNet cognate lookup functionality using Polars for efficient data handling.
Downloads and processes the CogNet v2.0 dataset automatically.
"""

import logging
import polars as pl
import requests
import zipfile
from pathlib import Path
from typing import Optional, Set
import os

logger = logging.getLogger(__name__)


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
        """Load the complete CogNet dataset."""
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
            logger.info("Loading CogNet dataset from %s", self.tsv_path)
            self._full_dataset = pl.read_csv(
                self.tsv_path,
                separator="\t",
                has_header=True,
                ignore_errors=True,
                truncate_ragged_lines=True
            )
            logger.info("Loaded %d cognate pairs from CogNet dataset", len(self._full_dataset))
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
    
    def is_cognate(self, word1: str, word2: str, learning_lang: str, native_lang: str) -> bool:
        """
        Check if two words are cognates based on CogNet data.
        
        Args:
            word1: Word in learning language
            word2: Word in native language  
            learning_lang: Learning language code (3-letter ISO)
            native_lang: Native language code (3-letter ISO)
            
        Returns:
            True if the words are cognates, False otherwise
        """
        cognate_data = self.get_cognate_data(learning_lang, native_lang)
        
        if cognate_data.is_empty():
            return False
        
        # Look for cognate pairs in both directions
        # In CogNet, if words appear together they are cognates (no explicit cognacy field)
        result = cognate_data.filter(
            (
                (pl.col("word 1") == word1) & 
                (pl.col("word 2") == word2) &
                (pl.col("lang 1") == learning_lang) &
                (pl.col("lang 2") == native_lang)
            ) |
            (
                (pl.col("word 1") == word2) &
                (pl.col("word 2") == word1) & 
                (pl.col("lang 1") == native_lang) &
                (pl.col("lang 2") == learning_lang)
            )
        )
        
        if len(result) > 0:
            logger.info("Cognate found for '%s' <-> '%s' (%s-%s):", word1, word2, learning_lang, native_lang)
            # Log all matching rows with complete data
            for i in range(len(result)):
                row = result.row(i, named=True)
                logger.info("  Row %d: %s", i + 1, dict(row))
            return True
        
        return False
    
    def find_cognates_for_word(self, word: str, learning_lang: str, native_lang: str) -> list:
        """
        Find all cognates for a given word in the target language.
        Similar to the example code: if multiple concept IDs exist, pick the first one.
        
        Args:
            word: Word to find cognates for
            learning_lang: Learning language code (3-letter ISO)
            native_lang: Native language code (3-letter ISO)
            
        Returns:
            List of cognate words in the other language
        """
        cognate_data = self.get_cognate_data(learning_lang, native_lang)
        
        if cognate_data.is_empty():
            return []
        
        # Find matches for the word (following your example pattern)
        result = cognate_data.filter(
            (pl.col("word 1") == word) & 
            (pl.col("lang 1") == learning_lang)
        )
        
        if len(result) == 0:
            # Try the reverse direction
            result = cognate_data.filter(
                (pl.col("word 2") == word) & 
                (pl.col("lang 2") == learning_lang)
            )
            
            if len(result) > 0:
                logger.info("Found %d cognate matches for '%s' (%s):", len(result), word, learning_lang)
                # Log all matching rows with complete data
                for i in range(len(result)):
                    row = result.row(i, named=True)
                    logger.info("  Row %d: %s", i + 1, dict(row))
                
                # If multiple concept IDs, take the first one
                first_result = result.head(1)
                cognates = first_result.select("word 1").to_series().to_list()
                if len(result) > 1:
                    logger.info("Multiple concept IDs found, using first one: '%s'", cognates[0] if cognates else 'none')
                return cognates
        else:
            logger.info("Found %d cognate matches for '%s' (%s):", len(result), word, learning_lang)
            # Log all matching rows with complete data
            for i in range(len(result)):
                row = result.row(i, named=True)
                logger.info("  Row %d: %s", i + 1, dict(row))
            
            # If multiple concept IDs, take the first one  
            first_result = result.head(1)
            cognates = first_result.select("word 2").to_series().to_list()
            if len(result) > 1:
                logger.info("Multiple concept IDs found, using first one: '%s'", cognates[0] if cognates else 'none')
            return cognates
        
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