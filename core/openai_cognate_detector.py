"""
OpenAI-based cognate detection for intelligent language analysis.
"""

import logging
import json
import os
from typing import List, Dict, Optional
from openai import OpenAI
from .constants import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Valid POS tags for cognate detection
VALID_POS_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

COGNATE_DETECTION_PROMPT = """
Respond with compact JSON. No markdown formatting.

For each sentence:
1. Translate from learning_language to native_language
2. Find cognates for nouns, verbs, adjectives, adverbs only
3. "T" = true cognate, "F" = false cognate, null = no visual similarity

A true cognate is a word in two languages that shares a common origin and has the same or very similar form and meaning.

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
                    token_data = sentence_data.get("tokens")

                    
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

# Argos-first prompt (model should not translate)
ARGOS_FIRST_PROMPT = """
Respond with compact JSON. No markdown formatting.

You are given, for each item, the original text (learning language) and its translation (native language).
Your task: For tokens that are nouns, verbs, adjectives, or adverbs, determine if the original token has a visually similar counterpart in the translation and classify:
- "T" = true cognate: shared origin, similar form and meaning (some example are, but not limited to: animal-animale, problem-problema, città-ciudad, family-famiglia)
- null = no visual similarity

Output per sentence: [sentence_index, [[token, cognate_type, cognate_word], ...]]
- For batch phrase checks, return a list mirroring requests order, each item: [[token, cognate_type, cognate_word], ...]
"""


class OpenAICognateDetectorArgosFirst(OpenAICognateDetector):
    """OpenAI cognate detector that uses Argos Translate first for translations."""

    _ISO3_TO_ARGOS = {
        "eng": "en",
        "spa": "es",
        "ita": "it",
        "fra": "fr",
    }

    def __init__(self):
        super().__init__()
        self._argos_translate = None  # lazy init
        self._argos_install_attempted = set()  # track attempted installs for pairs

    def _ensure_argos(self):
        if self._argos_translate is not None:
            return
        try:
            import argostranslate.package  # type: ignore
            import argostranslate.translate  # type: ignore

            def _translate(text: str, src_iso3: str, tgt_iso3: str):
                src = self._ISO3_TO_ARGOS.get(src_iso3, src_iso3)
                tgt = self._ISO3_TO_ARGOS.get(tgt_iso3, tgt_iso3)
                try:
                    available = argostranslate.translate.get_installed_languages()  # type: ignore
                    installed_codes = [getattr(l, "code", None) for l in available]
                    src_lang = next((l for l in available if getattr(l, "code", None) == src), None)
                    tgt_lang = next((l for l in available if getattr(l, "code", None) == tgt), None)
                    if not src_lang or not tgt_lang:
                        logger.warning(
                            "Argos missing languages for %s->%s. Installed: %s",
                            src,
                            tgt,
                            installed_codes,
                        )
                        return None
                    pair = src_lang.get_translation(tgt_lang)
                    if not pair:
                        logger.warning(
                            "Argos translation pair not installed for %s->%s. Installed: %s",
                            src,
                            tgt,
                            installed_codes,
                        )
                        return None
                    return pair.translate(text)
                except Exception:
                    return None

            self._argos_translate = _translate
            try:
                installed = argostranslate.translate.get_installed_languages()  # type: ignore
                logger.info(
                    "Argos ready. Installed languages: %s",
                    [getattr(l, "code", None) for l in installed],
                )
            except Exception:
                logger.info("Argos translate ready (installed languages unknown)")
        except Exception as e:
            self._argos_translate = lambda text, s, t: None
            logger.info("Argos not available: %s", str(e))

    def _argos_translate_many(self, items, learning_lang: str, native_lang: str):
        self._ensure_argos()
        # Try to ensure the pair is installed before translating
        try:
            self._ensure_argos_pair(learning_lang, native_lang)
        except Exception:
            pass
        out = []
        for it in items:
            text = it["text"]
            trans = None
            try:
                trans = self._argos_translate(text, learning_lang, native_lang) if self._argos_translate else None
            except Exception:
                trans = None
            out.append(trans or "")
        return out

    def _ensure_argos_pair(self, src_iso3: str, tgt_iso3: str) -> bool:
        """Ensure a specific translation pair is installed; auto-install if missing.
        Returns True if pair is available after this call, else False.
        """
        self._ensure_argos()
        try:
            import argostranslate.package  # type: ignore
            import argostranslate.translate  # type: ignore

            src = self._ISO3_TO_ARGOS.get(src_iso3, src_iso3)
            tgt = self._ISO3_TO_ARGOS.get(tgt_iso3, tgt_iso3)

            available = argostranslate.translate.get_installed_languages()  # type: ignore
            src_lang = next((l for l in available if getattr(l, "code", None) == src), None)
            tgt_lang = next((l for l in available if getattr(l, "code", None) == tgt), None)
            if src_lang and tgt_lang and src_lang.get_translation(tgt_lang):
                return True

            pair_key = (src, tgt)
            if pair_key in self._argos_install_attempted:
                logger.warning("Argos pair %s->%s still missing; install previously attempted", src, tgt)
                return False

            logger.info("Attempting Argos auto-install for pair %s->%s", src, tgt)
            self._argos_install_attempted.add(pair_key)

            # Update package index and install matching package
            argostranslate.package.update_package_index()  # type: ignore
            packages = argostranslate.package.get_available_packages()  # type: ignore
            candidates = [p for p in packages if getattr(p, "from_code", None) == src and getattr(p, "to_code", None) == tgt]

            if not candidates:
                logger.warning("No Argos packages available for %s->%s after index update", src, tgt)
                return False

            try:
                candidates[0].install()  # type: ignore
                logger.info("Argos installed package for %s->%s", src, tgt)
            except Exception as e:
                logger.error("Argos install failed for %s->%s: %s", src, tgt, str(e))
                return False

            # Recheck availability
            available = argostranslate.translate.get_installed_languages()  # type: ignore
            src_lang = next((l for l in available if getattr(l, "code", None) == src), None)
            tgt_lang = next((l for l in available if getattr(l, "code", None) == tgt), None)
            if src_lang and tgt_lang and src_lang.get_translation(tgt_lang):
                logger.info("Argos pair %s->%s ready after install", src, tgt)
                return True
            logger.warning("Argos pair %s->%s not ready after install", src, tgt)
            return False
        except Exception as e:
            logger.error("Argos ensure-pair error for %s->%s: %s", src_iso3, tgt_iso3, str(e))
            return False

    def preload_supported_pairs(self):
        """Pre-install Argos translation pairs for supported languages (all ordered pairs)."""
        try:
            langs = list(SUPPORTED_LANGUAGES.keys())
            total = 0
            ready = 0
            for src in langs:
                for tgt in langs:
                    if src == tgt:
                        continue
                    total += 1
                    if self._ensure_argos_pair(src, tgt):
                        ready += 1
            logger.info("Argos preload complete: %d/%d pairs ready", ready, total)
        except Exception as e:
            logger.error("Argos preload error: %s", str(e))

    def detect_cognates_for_sentences(self, sentences_data: List[Dict], learning_lang: str, native_lang: str) -> Dict:
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"results": []}

        # Translate ALL sentences first so translated_text is always populated
        all_sentences = [{"index": s["index"], "text": s["text"]} for s in sentences_data]
        all_translated = self._argos_translate_many(all_sentences, learning_lang, native_lang)
        idx_to_translated_all = {
            item["index"]: all_translated[i] for i, item in enumerate(all_sentences)
        }
        try:
            empty_idx = [it["index"] for i, it in enumerate(all_sentences) if not (all_translated[i] or "").strip()]
            if empty_idx:
                logger.warning(
                    "Argos returned empty translations for %d/%d sentences (pair %s->%s): %s",
                    len(empty_idx),
                    len(all_sentences),
                    learning_lang,
                    native_lang,
                    empty_idx[:10],
                )
        except Exception:
            pass

        # Prepare the subset to send to OpenAI (those with valid POS tokens)
        processed = []
        for sent in sentences_data:
            filtered = self._filter_tokens_by_pos(sent.get("tokens", []))
            if not filtered:
                continue
            processed.append({"index": sent["index"], "text": sent["text"]})

        # If no sentences have valid tokens, still return Argos translations for all
        if not processed:
            return {
                "results": [
                    {
                        "index": item["index"],
                        "translated_text": idx_to_translated_all.get(item["index"], ""),
                        "tokens": [],
                    }
                    for item in all_sentences
                ]
            }

        translated = self._argos_translate_many(processed, learning_lang, native_lang)
        idx_to_translated = {item["index"]: translated[i] for i, item in enumerate(processed)}

        input_data = {
            "learning_language": learning_lang,
            "native_language": native_lang,
            "sentences": [
                {
                    "index": item["index"],
                    "original_text": item["text"],
                    "translated_text": translated[i],
                }
                for i, item in enumerate(processed)
            ],
        }

        try:
            # Log only the pairs (original -> translated), not the system prompt
            try:
                logger.info(
                    "Cognate input pairs (sentences): %s",
                    json.dumps(
                        [
                            {
                                "index": s["index"],
                                "original_text": s["original_text"],
                                "translated_text": s["translated_text"],
                            }
                            for s in input_data.get("sentences", [])
                        ],
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                pass

            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": ARGOS_FIRST_PROMPT},
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content
            if not content:
                return {"results": []}
            # Log raw OpenAI response (full content)
            try:
                logger.info("OpenAI raw response (sentences): %s", content)
            except Exception:
                pass
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            raw = json.loads(content)
            out = {"results": []}
            seen_indexes = set()
            for item in raw:
                # Supported formats:
                # 1) [index, [[token, cognate_type, cognate_word], ...]]
                # 2) {"index": int, "tokens": [[token, cognate_type, cognate_word], ...]}
                if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], list):
                    idx, token_data = item[0], item[1]
                elif isinstance(item, dict):
                    idx = item.get("index", 0)
                    token_data = item.get("tokens") or []
                else:
                    continue

                tokens = []
                for tok in token_data:
                    if len(tok) >= 3:
                        tok_text, ctype, cword = tok[0], tok[1], tok[2]
                        if ctype == "T":
                            status = "true_cognate"
                        elif ctype == "F":
                            status = "false_cognate"
                        else:
                            status = None
                        tokens.append({"text": tok_text, "cognate_status": status, "cognate": cword})

                # Populate translated_text from Argos only (do not rely on model output)
                out["results"].append({
                    "index": idx,
                    "translated_text": idx_to_translated_all.get(idx, idx_to_translated.get(idx, "")),
                    "tokens": tokens,
                })
                seen_indexes.add(idx)

            # Add entries for sentences that were not sent to OpenAI (no valid tokens),
            # but still want translated_text populated via Argos
            for item in all_sentences:
                idx = item["index"]
                if idx not in seen_indexes:
                    out["results"].append(
                        {
                            "index": idx,
                            "translated_text": idx_to_translated_all.get(idx, ""),
                            "tokens": [],
                        }
                    )

            return out
        except Exception as e:
            logger.error("Argos-first detect error: %s", str(e))
            return {"results": []}

    def detect_cognates_batch(self, requests: List[Dict]) -> Dict:
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"results": []}

        filtered_reqs = []
        for req in requests:
            if self._filter_tokens_by_pos(req.get("tokens", [])):
                filtered_reqs.append(req)

        if not filtered_reqs:
            return {"results": []}

        # Translate each phrase independently to respect per-request language pairs
        self._ensure_argos()
        # Try to ensure each pair is installed before translating
        req_payload = []
        for r in filtered_reqs:
            translated_text = None
            try:
                try:
                    self._ensure_argos_pair(r["learning_language"], r["native_language"])
                except Exception:
                    pass
                translated_text = self._argos_translate(r["phrase"], r["learning_language"], r["native_language"]) if self._argos_translate else None
            except Exception:
                translated_text = None
            req_payload.append({
                "original_text": r["phrase"],
                "translated_text": translated_text or "",
                "learning_language": r["learning_language"],
                "native_language": r["native_language"],
            })

        input_data = {"requests": req_payload}

        try:
            # Log only the pairs (original -> translated), not the system prompt
            try:
                logger.debug(
                    "Cognate input pairs (batch): %s",
                    json.dumps(
                        [
                            {
                                "original_text": r.get("original_text", ""),
                                "translated_text": r.get("translated_text", ""),
                            }
                            for r in input_data.get("requests", [])
                        ],
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                pass

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ARGOS_FIRST_PROMPT},
                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content
            if not content:
                return {"results": []}
            # Log raw OpenAI response (full content)
            try:
                logger.info("OpenAI raw response (batch): %s", content)
            except Exception:
                pass
            raw = json.loads(content)
            # Expect a list of per-request token arrays; normalize to include only tokens
            normalized = []
            for item in raw:
                if isinstance(item, list):
                    token_data = item
                elif isinstance(item, dict):
                    token_data = item.get("tokens") or []
                else:
                    token_data = []
                tokens = []
                for tok in token_data:
                    if len(tok) >= 3:
                        tok_text, ctype, cword = tok[0], tok[1], tok[2]
                        if ctype == "T":
                            status = "true_cognate"
                        elif ctype == "F":
                            status = "false_cognate"
                        else:
                            status = None
                        tokens.append({"text": tok_text, "cognate_status": status, "cognate": cword})
                normalized.append({"tokens": tokens})
            return {"results": normalized}
        except Exception as e:
            logger.error("Argos-first batch detect error: %s", str(e))
            return {"results": []}


# Global detector instance (switch to Argos-first)
openai_cognate_detector = OpenAICognateDetectorArgosFirst()
