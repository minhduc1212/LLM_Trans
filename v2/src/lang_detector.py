"""
lang_detector.py - Detect source language of input text.

Uses Unicode block frequency counting — no external libraries needed.
Works for: EN, ZH (Simplified/Traditional), JA, KO.
Falls back to 'en' for other scripts (Latin-heavy).

Why not use langdetect/langid?
  → Avoid extra deps; Unicode blocks are 100% reliable for CJK detection.
"""

import re
from collections import Counter


# Unicode block ranges
_BLOCK_CJK_UNIFIED      = (0x4E00,  0x9FFF)   # Core CJK — shared by ZH, JA, KO
_BLOCK_CJK_EXT_A        = (0x3400,  0x4DBF)
_BLOCK_CJK_EXT_B        = (0x20000, 0x2A6DF)
_BLOCK_HIRAGANA         = (0x3041,  0x309F)   # JA only
_BLOCK_KATAKANA         = (0x30A0,  0x30FF)   # JA only
_BLOCK_HANGUL_SYLLABLES = (0xAC00,  0xD7AF)   # KO only
_BLOCK_HANGUL_JAMO      = (0x1100,  0x11FF)   # KO only
_BLOCK_LATIN            = (0x0041,  0x007A)   # EN/Latin


def _in_range(cp: int, block: tuple[int, int]) -> bool:
    return block[0] <= cp <= block[1]


def _is_cjk(cp: int) -> bool:
    return (
        _in_range(cp, _BLOCK_CJK_UNIFIED) or
        _in_range(cp, _BLOCK_CJK_EXT_A) or
        _in_range(cp, _BLOCK_CJK_EXT_B)
    )


def detect_source_language(text: str) -> str:
    """
    Returns one of: 'en', 'zh', 'ja', 'ko'

    Detection logic (priority order):
      1. Any Hiragana/Katakana → 'ja'  (unique to Japanese)
      2. Any Hangul → 'ko'             (unique to Korean)
      3. CJK ratio > 15% of chars → 'zh' (Chinese uses CJK without kana/hangul)
      4. Default → 'en'
    """
    if not text:
        return 'en'

    # Sample first 2000 chars for speed
    sample = text[:2000]
    total = len(sample)

    hiragana_count = 0
    katakana_count = 0
    hangul_count = 0
    cjk_count = 0

    for ch in sample:
        cp = ord(ch)
        if _in_range(cp, _BLOCK_HIRAGANA):
            hiragana_count += 1
        elif _in_range(cp, _BLOCK_KATAKANA):
            katakana_count += 1
        elif _in_range(cp, _BLOCK_HANGUL_SYLLABLES) or _in_range(cp, _BLOCK_HANGUL_JAMO):
            hangul_count += 1
        elif _is_cjk(cp):
            cjk_count += 1

    # Japanese: has hiragana or katakana
    if hiragana_count + katakana_count > 5:
        return 'ja'

    # Korean: has hangul
    if hangul_count > 5:
        return 'ko'

    # Chinese: mostly CJK (>15% of all chars)
    if total > 0 and cjk_count / total > 0.15:
        return 'zh'

    return 'en'


def get_source_lang_label(lang_code: str) -> str:
    """Human-readable label for logging."""
    return {
        'en': 'English 🇬🇧',
        'zh': 'Tiếng Trung 🇨🇳',
        'ja': 'Tiếng Nhật 🇯🇵',
        'ko': 'Tiếng Hàn 🇰🇷',
    }.get(lang_code, lang_code)


# Quick test
if __name__ == '__main__':
    tests = [
        ("Hello world, this is English text.", 'en'),
        ("这是一段中文文字，用于测试语言检测功能。", 'zh'),
        ("これは日本語のテキストです。ひらがなとカタカナが含まれています。", 'ja'),
        ("이것은 한국어 텍스트입니다. 언어 감지를 테스트합니다.", 'ko'),
    ]
    for text, expected in tests:
        detected = detect_source_language(text)
        status = '✅' if detected == expected else '❌'
        print(f"{status} Expected={expected}, Got={detected}: {text[:40]}")