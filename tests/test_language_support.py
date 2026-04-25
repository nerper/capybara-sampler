"""Locale aliases and canonical language registry."""

import pytest

from core.constants import SUPPORTED_LANGUAGES
from core.language_codes import (
    ISO_TO_STANZA_LANG,
    ISO_TO_WORDFREQ_LANG,
    normalize_language_request,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("en-US", "eng"),
        ("en-gb", "eng"),
        ("es-419", "spa"),
        ("es-ES", "spa"),
        ("fr-CA", "fra"),
        ("fr-fr", "fra"),
        ("pt-BR", "por"),
        ("pt_br", "por"),
        ("ja", "jpn"),
        ("zh-Hans", "cmn"),
        ("zh-cn", "cmn"),
        ("ar-EG", "arb"),
        ("he", "heb"),
        ("ru", "rus"),
        ("de", "deu"),
        ("eng", "eng"),
        ("cmn", "cmn"),
    ],
)
def test_normalize_locale_aliases(raw: str, expected: str) -> None:
    assert normalize_language_request(raw) == expected


def test_all_supported_codes_self_normalize() -> None:
    for code in SUPPORTED_LANGUAGES:
        assert normalize_language_request(code) == code


def test_iso_mappings_cover_supported() -> None:
    for code in SUPPORTED_LANGUAGES:
        assert code in ISO_TO_STANZA_LANG
        assert code in ISO_TO_WORDFREQ_LANG


def test_frequency_lookup_no_crash_japanese() -> None:
    """wordfreq may require MeCab for ja; scorer must not raise."""
    from core.score_model import FamiliarityScorer

    scorer = FamiliarityScorer()
    n, z = scorer.get_frequency_score("テスト", "jpn")
    assert 0.0 <= n <= 1.0
    assert isinstance(z, float)
