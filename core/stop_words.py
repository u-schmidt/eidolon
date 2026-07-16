"""Language-specific stop word lists for tag extraction.

Prompts to the LLM are always in English.
Tag extraction supports multiple languages via configurable stop word lists.
"""

from __future__ import annotations

# Default stop words per language.
# Extend or override via configuration.
_STOP_WORDS: dict[str, set[str]] = {
    "de": {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einen", "einer", "eines",
        "und", "oder", "aber", "denn", "doch",
        "mit", "von", "aus", "bei", "nach", "vor",
        "auf", "über", "unter", "neben", "zwischen",
        "ich", "du", "er", "sie", "es", "wir", "ihr",
        "mich", "mir", "dich", "dir", "sich", "uns",
        "nicht", "kein", "keine", "ist", "war", "hat",
        "auch", "nur", "schon", "noch", "mal", "dann",
        "lass", "lässt", "lassen", "macht", "machen",
        "wäre", "hätte", "könnte", "würde",
        "dass", "weil", "wenn", "als", "wie",
        "ja", "nein", "ok", "okay", "jep", "genau",
        "bitte", "danke", "vielen", "gern", "gerne",
        "halt", "eben", "doch", "schon", "mal",
        "einfach", "eigentlich", "irgendwie",
        "also", "trotzdem", "deshalb", "deswegen",
        "übrigens", "übrigens",
    },
    "en": {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "up", "about",
        "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might",
        "i", "you", "he", "she", "it", "we", "they",
        "me", "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those",
        "not", "no", "nor", "so", "very", "just", "then", "now",
        "also", "too", "only", "well", "really", "quite",
        "some", "any", "each", "every", "all", "both", "few",
        "more", "most", "other", "such", "what", "which", "who",
        "when", "where", "why", "how",
        "if", "because", "as", "until", "while", "though",
        "here", "there", "ok", "okay", "yes", "no", "please",
        "thanks", "thank",
    },
}


def get_stop_words(language: str = "en") -> set[str]:
    """Return the stop word set for the given language.

    Falls back to English if the language is not defined.
    """
    return _STOP_WORDS.get(language, _STOP_WORDS["en"])


def register_stop_words(language: str, words: set[str]) -> None:
    """Add or replace stop words for a language at runtime."""
    _STOP_WORDS[language] = words
