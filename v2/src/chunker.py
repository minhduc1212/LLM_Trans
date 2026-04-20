"""
chunker.py - Smart text splitter that preserves paragraph structure.

Key behaviors:
  - Splits at \n\n (double newline) AND at single \n before chapter titles
  - Chapter titles (第X章, Chapter X, 卷X, Arc X, etc.) always start a new paragraph
  - Language-aware chars-per-token so CJK chunks aren't oversized
  - Never splits mid-paragraph (fallback to sentence-level for oversized paragraphs)

Chars-per-token by script (measured against Gemma/LLaMA SentencePiece):
  EN/Latin : ~4.0   ZH : ~1.5   JA : ~2.0   KO : ~2.0
"""

import re
from typing import List


# ── Language-aware chars-per-token ratios ─────────────────────────────────────
_CPT = {'en': 4.0, 'vi': 3.5, 'zh': 1.5, 'ja': 2.0, 'ko': 2.0}
_CPT_DEFAULT = 3.5

# ── Chapter title patterns (triggers paragraph break even on single \n) ───────
_CHAPTER_TITLE_RE = re.compile(
    r'^('
    r'第\s*[\d一二三四五六七八九十百千万零〇]+\s*[章节卷话回篇集幕]'    # ZH: 第2068章 / 第一节 / 第3卷
    r'|[序前后终](?:章|篇|记|言|话)'     # ZH: 序章 / 前言 / 终章
    r'|[上中下](?:卷|篇|册)'            # ZH: 上卷 / 下篇
    r'|Chapter\s+\d+'                   # EN: Chapter 12
    r'|Volume\s+\d+'                    # EN: Volume 3
    r'|Arc\s+\d+'                       # EN: Arc 5
    r'|Prologue|Epilogue|Interlude'     # EN specials
    r'|第\s*\d+\s*話'                   # JA: 第3話
    r'|제\s*\d+\s*장'                   # KO: 제5장
    r')',
    re.IGNORECASE | re.UNICODE,
)


def _is_chapter_title(line: str) -> bool:
    return bool(_CHAPTER_TITLE_RE.match(line.strip()))


# ── Unicode block helpers ──────────────────────────────────────────────────────
def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF

def _is_hiragana_katakana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3041 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF

def _is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF


def detect_script(text: str) -> str:
    sample = text[:500]
    hira_kata = sum(1 for c in sample if _is_hiragana_katakana(c))
    hangul    = sum(1 for c in sample if _is_hangul(c))
    cjk       = sum(1 for c in sample if _is_cjk(c))
    total     = max(len(sample), 1)
    if hira_kata > 3:       return 'ja'
    if hangul    > 3:       return 'ko'
    if cjk / total > 0.12: return 'zh'
    return 'en'


def get_chars_per_token(lang: str) -> float:
    return _CPT.get(lang, _CPT_DEFAULT)


def estimate_tokens(text: str, cpt: float = _CPT_DEFAULT) -> int:
    return max(1, int(len(text) / cpt))


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into logical paragraphs.

    Rules (in order):
      1. \n\n  → always a paragraph boundary
      2. \n    → paragraph boundary ONLY if the next line is a chapter title
      3. Everything else → same paragraph
    """
    # Normalize CRLF
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    lines = text.split('\n')
    paragraphs: List[str] = []
    current_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Empty line = end of current paragraph
        if line.strip() == '':
            if current_lines:
                paragraphs.append('\n'.join(current_lines))
                current_lines = []
            # Skip consecutive empty lines
            while i + 1 < len(lines) and lines[i + 1].strip() == '':
                i += 1
        else:
            # If current line is a chapter title and we already have content → flush first
            if _is_chapter_title(line) and current_lines:
                paragraphs.append('\n'.join(current_lines))
                current_lines = [line]
            # If NEXT line is a chapter title → flush current line as its own paragraph
            elif (current_lines and
                  i + 1 < len(lines) and
                  _is_chapter_title(lines[i + 1])):
                current_lines.append(line)
                paragraphs.append('\n'.join(current_lines))
                current_lines = []
            else:
                current_lines.append(line)
        i += 1

    if current_lines:
        paragraphs.append('\n'.join(current_lines))

    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 2500,
    source_lang: str = 'auto',
) -> List[str]:
    """
    Split text into token-balanced chunks preserving paragraph + chapter boundaries.

    Args:
        text:        Raw input text
        chunk_size:  Target token count per chunk (language-corrected)
        source_lang: 'auto' | 'en' | 'zh' | 'ja' | 'ko'
    """
    if source_lang == 'auto':
        source_lang = detect_script(text)
    cpt = get_chars_per_token(source_lang)

    paragraphs = _split_into_paragraphs(text)

    chunks: List[str] = []
    current_paras: List[str] = []
    current_tokens: int = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para, cpt)

        # Chapter title: always flush previous content first, then start new chunk WITH title
        if _is_chapter_title(para):
            if current_paras:
                chunks.append('\n\n'.join(current_paras))
            current_paras  = [para]
            current_tokens = para_tokens
            continue

        # Oversized single paragraph → sentence-split
        if para_tokens > chunk_size:
            if current_paras:
                chunks.append('\n\n'.join(current_paras))
                current_paras  = []
                current_tokens = 0
            chunks.extend(_split_large_paragraph(para, chunk_size, cpt))
            continue

        # Normal accumulation
        if current_tokens + para_tokens > chunk_size and current_paras:
            chunks.append('\n\n'.join(current_paras))
            current_paras  = [para]
            current_tokens = para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunks.append('\n\n'.join(current_paras))

    return chunks


def _split_large_paragraph(para: str, chunk_size: int, cpt: float) -> List[str]:
    """Fallback: split oversized paragraph at sentence boundaries."""
    endings = re.compile(r'(?<=[.!?。！？…])\s*')
    sentences = [s for s in endings.split(para) if s.strip()]

    sub_chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        st = estimate_tokens(sent, cpt)
        if current_tokens + st > chunk_size and current:
            sub_chunks.append(' '.join(current))
            current        = [sent]
            current_tokens = st
        else:
            current.append(sent)
            current_tokens += st

    if current:
        sub_chunks.append(' '.join(current))

    return sub_chunks


def get_chunk_stats(chunks: List[str], source_lang: str = 'auto') -> dict:
    if not chunks:
        return {'total_chunks': 0, 'min_tokens': 0, 'max_tokens': 0,
                'avg_tokens': 0, 'total_tokens': 0, 'source_lang': source_lang}
    if source_lang == 'auto':
        source_lang = detect_script(chunks[0])
    cpt    = get_chars_per_token(source_lang)
    counts = [estimate_tokens(c, cpt) for c in chunks]
    return {
        'total_chunks':    len(chunks),
        'min_tokens':      min(counts),
        'max_tokens':      max(counts),
        'avg_tokens':      int(sum(counts) / len(counts)),
        'total_tokens':    sum(counts),
        'source_lang':     source_lang,
        'chars_per_token': cpt,
    }