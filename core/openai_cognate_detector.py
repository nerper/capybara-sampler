"""
OpenAI-based cognate detection for intelligent language analysis.
"""

import logging
import json
import os
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# Valid POS tags for cognate detection
VALID_POS_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

COGNATE_DETECTION_PROMPT = """
Respond with compact JSON. No markdown formatting.

For each sentence:
1. Translate from learning_language to native_language
2. Find cognates for nouns, verbs, adjectives, adverbs only
3. COGNATES MUST LOOK SIMILAR (same spelling/sound) - meaning alone is not enough
4. "T" = true cognate (LOOKS similar AND same meaning), "F" = false cognate (LOOKS similar but different meaning), null = no visual similarity

Examples of TRUE cognates: "hospital"↔"hospital", "animal"↔"animal", "chocolate"↔"chocolate"
Examples of FALSE cognates: "embarrassed"↔"embarazada" (look similar, different meanings)
Examples of NULL: "happy"↔"feliz", "went"↔"fui" (different appearance, even if same meaning)

Format: [sentence_index, "translated_text", [[token, cognate_type, cognate_word], ...]]

Example:

INPUT: {"learning_language": "eng", "native_language": "spa", "content": "I am embarrassed. I went to school."}

OUTPUT: [[0, "Estoy avergonzada.", [["I", null, null], ["am", null, null], ["embarrassed", "F", "embarazada"]]], [1, "Fui a la escuela.", [["I", null, null], ["went", null, null], ["to", null, null], ["school", "T", "escuela"]]]]
"""


class OpenAICognateDetector:
    """Handles cognate detection using OpenAI API."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: %s", str(e))
    
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
    
    def detect_cognates_for_sentences(self, sentences_data: List[Dict], learning_lang: str, native_lang: str) -> Dict:
        """
        Detect cognates for multiple sentences using OpenAI API.
        
        Args:
            sentences_data: List of sentence dicts with 'text', 'index', and 'tokens'
            learning_lang: Learning language code
            native_lang: Native language code
        
        Returns:
            Dictionary with cognate detection results organized by sentence
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"results": []}
        
        # Prepare sentences with only valid POS tokens
        processed_sentences = []
        for sent_data in sentences_data:
            filtered_tokens = self._filter_tokens_by_pos(sent_data.get('tokens', []))
            
            if not filtered_tokens:
                logger.info("No valid tokens for cognate detection in sentence %d", sent_data['index'])
                continue
            
            processed_sentences.append({
                "index": sent_data['index'],
                "text": sent_data['text'],
                "learning_language": learning_lang,
                "native_language": native_lang,
                "valid_pos_tokens": [{"text": token["text"], "pos": token["pos"]} for token in filtered_tokens]
            })
        
        if not processed_sentences:
            logger.info("No sentences with valid tokens for OpenAI API")
            return {"results": []}
        
        # Prepare the input for OpenAI
        input_data = {
            "learning_language": learning_lang,
            "native_language": native_lang,
            "sentences": [{
                "index": sent["index"],
                "text": sent["text"]
            } for sent in processed_sentences]
        }
        
        try:
            logger.info("Sending %d sentences to OpenAI for cognate detection", len(processed_sentences))
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                temperature=0
            )
            
            response_content = response.choices[0].message.content
            logger.info("🔍 OpenAI RAW RESPONSE: %s", response_content)
            
            # Check if response content is None
            if response_content is None:
                logger.error("OpenAI returned empty response content")
                return {"results": []}
            
            # Strip markdown code blocks if present
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "").strip()
            
            # Parse the JSON response
            raw_results = json.loads(response_content)
            logger.info("Successfully received cognate detection results from OpenAI")
            
            # Handle both compact array format and object format
            converted_results = {"results": []}
            
            for sentence_data in raw_results:
                # Check if it's object format: {"index": 0, "translated_text": "...", "cognates": [...]}
                if isinstance(sentence_data, dict):
                    index = sentence_data.get("index", 0)
                    translated_text = sentence_data.get("translated_text", "")
                    token_data = sentence_data.get("cognates", [])
                    
                    tokens = []
                    for token_tuple in token_data:
                        if len(token_tuple) >= 3:
                            token_text, cognate_type, cognate_word = token_tuple[0], token_tuple[1], token_tuple[2]
                            
                            # Convert "T"/"F" back to full strings
                            if cognate_type == "T":
                                cognate_status = "true_cognate"
                            elif cognate_type == "F":
                                cognate_status = "false_cognate"
                            else:
                                cognate_status = None
                            
                            tokens.append({
                                "text": token_text,
                                "cognate_status": cognate_status,
                                "cognate": cognate_word
                            })
                    
                    converted_results["results"].append({
                        "index": index,
                        "translated_text": translated_text,
                        "tokens": tokens
                    })
                
                # Handle compact array format: [index, translated_text, [[token, cognate_type, cognate_word], ...]]
                elif isinstance(sentence_data, list) and len(sentence_data) >= 3:
                    index, translated_text, token_data = sentence_data[0], sentence_data[1], sentence_data[2]
                    
                    tokens = []
                    for token_tuple in token_data:
                        if len(token_tuple) >= 3:
                            token_text, cognate_type, cognate_word = token_tuple[0], token_tuple[1], token_tuple[2]
                            
                            # Convert "T"/"F" back to full strings
                            if cognate_type == "T":
                                cognate_status = "true_cognate"
                            elif cognate_type == "F":
                                cognate_status = "false_cognate"
                            else:
                                cognate_status = None
                            
                            tokens.append({
                                "text": token_text,
                                "cognate_status": cognate_status,
                                "cognate": cognate_word
                            })
                    
                    converted_results["results"].append({
                        "index": index,
                        "translated_text": translated_text,
                        "tokens": tokens
                    })
            
            return converted_results
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse OpenAI response as JSON: %s", str(e))
            return {"results": []}
        except Exception as e:
            logger.error("Error calling OpenAI API: %s", str(e))
            return {"results": []}

    def detect_cognates_batch(self, requests: List[Dict]) -> Dict:
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
                temperature=0
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
openai_cognate_detector = OpenAICognateDetector()