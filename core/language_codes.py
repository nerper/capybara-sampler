"""
Locale aliases and ISO 639-3 canonical codes for API requests.

Clients may send BCP-47 tags (e.g. en-US) or short forms; scoring uses canonical 3-letter codes.
"""

from __future__ import annotations

from .constants import SUPPORTED_LANGUAGES

# Request string (lowercase, hyphens) -> canonical 3-letter code used everywhere downstream.
# Includes BCP-47 variants from product labels and common ISO 639-1 fallbacks.
ALIASES_TO_CANONICAL: dict[str, str] = {
    # English
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    "en_us": "eng",
    "en_gb": "eng",
    # Spanish
    "es": "spa",
    "es-us": "spa",
    "es-es": "spa",
    "es-419": "spa",
    "es_mx": "spa",
    "es-mx": "spa",
    # French
    "fr": "fra",
    "fr-fr": "fra",
    "fr-ca": "fra",
    # German / Italian
    "de": "deu",
    "it": "ita",
    # Portuguese
    "pt": "por",
    "pt-br": "por",
    "pt_br": "por",
    # Russian
    "ru": "rus",
    # Dutch / Polish (ISO 639-1)
    "nl": "nld",
    "pl": "pol",
    # Japanese / Korean / Chinese (Simplified)
    "ja": "jpn",
    "ko": "kor",
    "zh": "cmn",
    "zh-cn": "cmn",
    "zh-hans": "cmn",
    "zh_sg": "cmn",
    "zh-sg": "cmn",
    "cmn": "cmn",
    "zho": "cmn",
    # Arabic / Hebrew
    "ar": "arb",
    "ar-eg": "arb",
    "ar_eg": "arb",
    "he": "heb",
    "iw": "heb",
}


def normalize_language_request(raw: str) -> str:
    """
    Normalize a client language string to a canonical 3-letter code.

    Unknown strings are lowercased and hyphen-normalized but not otherwise changed,
    so validation against SUPPORTED_LANGUAGES can still reject them.
    """
    s = raw.strip().replace("_", "-")
    key = s.lower()
    if not key:
        return key

    if key in SUPPORTED_LANGUAGES:
        return key
    return ALIASES_TO_CANONICAL.get(key, key)


def alias_map_for_api() -> dict[str, str]:
    """All accepted non-canonical request tokens -> canonical code (for GET /languages)."""
    return dict(ALIASES_TO_CANONICAL)


# Stanza `lang=` argument per canonical ISO 639-3 code (see Stanza multilingual models).
ISO_TO_STANZA_LANG: dict[str, str] = {
    "ita": "it",
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "rus": "ru",
    "jpn": "ja",
    "kor": "ko",
    "cmn": "zh-hans",
    "arb": "ar",
    "heb": "he",
}

# wordfreq `zipf_frequency` language tag (may differ from Stanza, e.g. Chinese uses "zh").
ISO_TO_WORDFREQ_LANG: dict[str, str] = {
    "ita": "it",
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "rus": "ru",
    "jpn": "ja",
    "kor": "ko",
    "cmn": "zh",
    "arb": "ar",
    "heb": "he",
}

# Processors known to load for each Stanza language pack in this project.
DEFAULT_STANZA_PROCESSORS = "tokenize,mwt,pos,lemma,ner"
STANZA_PROCESSORS_BY_STANZA_LANG: dict[str, str] = {
    "ja": "tokenize,pos,lemma,ner",
    "ko": "tokenize,pos,lemma",
    "zh-hans": "tokenize,pos,lemma,ner",
}
