"""
OpenAI-based cognate detection for intelligent language analysis.
"""

import logging
import json
import os
import time
from typing import List, Dict, Optional
import openai
from openai import OpenAI

# Configure detailed logging
logger = logging.getLogger(__name__)

# Track module loading time
MODULE_LOAD_START = time.time()
logger.debug("Loading OpenAI cognate detector module...")

# Time the OpenAI import
openai_import_start = time.time()
logger.debug("Importing OpenAI library...")
logger.debug("OpenAI library imported in %.3f seconds", time.time() - openai_import_start)

# Valid POS tags for cognate detection
VALID_POS_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

COGNATE_DETECTION_PROMPT = """Respond only with a valid JSON object following this schema: {"results":[{"learning_language":"<3-letter ISO code>","native_language":"<3-letter ISO code>","tokens":[{"text":"<token text>","cognate_status":"true_cognate"|"false_cognate"|null,"cognate":"<cognate word or null>"}]}]} Rules: 1) Translate the entire phrase from the learning language into the native language. 2) Compare each token in the original phrase with semantically equivalent words in the translation to detect cognates, ensuring context alignment (e.g., verb vs. adjective). 3) ONLY PROCESS AND RETURN RESULTS for tokens with POS: nouns, verbs, adjectives, and adverbs. Ignore all other tokens (pronouns, determiners, punctuation, etc.). 4) "cognate_status": "true_cognate" → valid POS and verified cognate in meaning and form; "false_cognate" → similar form but different meaning (false friend); null → no clear or verified relationship or no similarity. 5) "cognate": corresponding native word if "true_cognate" or "false_cognate", else null. 6) "learning_language" and "native_language" must match the input; return null if uncertain. 7) Return only JSON, no explanations. 8) Only include tokens with valid POS in the results array - omit all others."""


class OpenAICognateDetector:
    """Handles cognate detection using OpenAI API."""
    
    def __init__(self):
        init_start = time.time()
        logger.debug("Initializing OpenAI cognate detector...")
        
        self.client = None
        self._initialize_client()
        
        init_time = time.time() - init_start
        logger.debug("OpenAI cognate detector initialized in %.3f seconds", init_time)
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from environment."""
        client_init_start = time.time()
        logger.debug("Checking for OpenAI API key in environment...")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables - cognate detection will be disabled")
            return
        
        logger.debug("OpenAI API key found, creating client...")
        
        try:
            client_creation_start = time.time()
            self.client = OpenAI(api_key=api_key)
            client_creation_time = time.time() - client_creation_start
            
            total_init_time = time.time() - client_init_start
            logger.info("✓ OpenAI client initialized successfully in %.3f seconds (%.3f for creation)", 
                       total_init_time, client_creation_time)
            
        except Exception as e:
            init_time = time.time() - client_init_start
            logger.error("✗ Failed to initialize OpenAI client after %.3f seconds: %s", init_time, str(e))
    
    def _filter_tokens_by_pos(self, tokens: List[Dict]) -> List[Dict]:
        """Filter tokens to only include valid POS tags for cognate detection."""
        filtered = []
        for token in tokens:
            pos = token.get('pos')
            if pos in VALID_POS_TAGS:
                filtered.append(token)
                logger.debug("Including token '%s' with POS '%s'", token.get('text'), pos)
            else:
                logger.debug("Skipping token '%s' with POS '%s' (not in valid set)", token.get('text'), pos)
        
        logger.info("Filtered %d tokens to %d with valid POS tags", len(tokens), len(filtered))
        return filtered
    
    def detect_cognates_batch(self, requests: List[Dict]) -> Dict:
        """
        Detect cognates for a batch of phrase requests using OpenAI API.
        
        Args:
            requests: List of {"phrase": str, "learning_language": str, "native_language": str, "tokens": List[Dict]}
        
        Returns:
            Dictionary with cognate detection results
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"results": []}
        
        # Prepare the request payload for OpenAI
        openai_requests = []
        for req in requests:
            # Filter tokens by POS
            filtered_tokens = self._filter_tokens_by_pos(req.get('tokens', []))
            
            if not filtered_tokens:
                logger.info("No valid tokens for cognate detection in phrase: '%s'", req.get('phrase', ''))
                continue
            
            # Only include the filtered tokens (with valid POS) in the request
            openai_requests.append({
                "phrase": req["phrase"],
                "learning_language": req["learning_language"], 
                "native_language": req["native_language"],
                "valid_pos_tokens": [{"text": token["text"], "pos": token["pos"]} for token in filtered_tokens]
            })
        
        if not openai_requests:
            logger.info("No requests with valid tokens for OpenAI API")
            return {"results": []}
        
        # Prepare the input for OpenAI (only send phrases and valid tokens info)
        input_data = {"requests": [
            {
                "phrase": req["phrase"],
                "learning_language": req["learning_language"],
                "native_language": req["native_language"]
            } for req in openai_requests
        ]}
        
        try:
            logger.info("Sending %d requests to OpenAI for cognate detection", len(openai_requests))
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": COGNATE_DETECTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": json.dumps(input_data)
                    }
                ],
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content
            logger.debug("OpenAI response: %s", response_content)
            
            # Check if response content is None
            if response_content is None:
                logger.error("OpenAI returned empty response content")
                return {"results": []}
            
            # Parse the JSON response
            cognate_results = json.loads(response_content)
            logger.info("Successfully received cognate detection results from OpenAI")
            
            return cognate_results
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse OpenAI response as JSON: %s", str(e))
            return {"results": []}
        except Exception as e:
            logger.error("Error calling OpenAI API: %s", str(e))
            return {"results": []}
    
    def find_cognates_for_token(self, token_text: str, token_pos: str, phrase: str, 
                                learning_lang: str, native_lang: str) -> tuple[bool, Optional[str]]:
        """
        Find cognate for a single token by calling OpenAI with the full phrase context.
        
        Args:
            token_text: The token to find cognates for
            token_pos: POS tag of the token
            phrase: Full phrase containing the token
            learning_lang: Learning language code
            native_lang: Native language code
        
        Returns:
            Tuple of (has_cognate: bool, cognate_word: Optional[str])
        """
        # Check if POS is valid
        if token_pos not in VALID_POS_TAGS:
            logger.debug("Token '%s' has invalid POS '%s' for cognate detection", token_text, token_pos)
            return False, None
        
        # Create a single request
        request = {
            "phrase": phrase,
            "learning_language": learning_lang,
            "native_language": native_lang,
            "tokens": [{"text": token_text, "pos": token_pos}]
        }
        
        # Get cognate detection results
        results = self.detect_cognates_batch([request])
        
        # Parse results for this specific token
        if not results.get("results"):
            return False, None
        
        result = results["results"][0]
        tokens = result.get("tokens", [])
        
        # Find our token in the results
        for token_result in tokens:
            if token_result.get("text", "").lower() == token_text.lower():
                cognate_status = token_result.get("cognate_status")
                cognate_word = token_result.get("cognate")
                
                if cognate_status == "true_cognate":
                    logger.info("Found true cognate for '%s': '%s'", token_text, cognate_word)
                    return True, cognate_word
                elif cognate_status == "false_cognate":
                    logger.info("Found false cognate for '%s': '%s'", token_text, cognate_word)
                    return False, cognate_word
                else:
                    logger.debug("No cognate found for '%s'", token_text)
                    return False, None
        
        logger.debug("Token '%s' not found in OpenAI results", token_text)
        return False, None


# Global detector instance  
logger.debug("Creating global OpenAI cognate detector instance...")
detector_creation_start = time.time()
openai_cognate_detector = OpenAICognateDetector()
detector_creation_time = time.time() - detector_creation_start

total_module_time = time.time() - MODULE_LOAD_START
logger.debug("OpenAI cognate detector module loaded completely in %.3f seconds (%.3f for detector creation)", 
            total_module_time, detector_creation_time)