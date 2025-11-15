# rag_engine/sanitizer.py
import re
from typing import Tuple

LATEX_START_TOKENS = [
    r"\\documentclass",
    r"\\begin\{document\}",
    r"%",       # comment often indicates latex content
    r"\\section",
    r"\\title",
    r"\\maketitle",
]

def strip_ollama_metadata(text: str) -> str:
    """
    Remove common trailing metadata (e.g. "thinking=... context=[...]" or "logprobs=...").
    Very defensive: strips a few common patterns found in local LLM client dumps.
    """
    # remove "thinking=... context=[...]" like traces
    text = re.sub(r"'\s*thinking=.*$", "", text, flags=re.S)
    text = re.sub(r"\n?logprobs=.*$", "", text, flags=re.S)
    # remove "context=[...]" huge numeric lists
    text = re.sub(r"context=\[.*\]$", "", text, flags=re.S)
    # remove trailing JSON-like dumps after a closing latex block
    # find last \end{document} and discard anything after that
    m = re.search(r"(\\end\{document\}.*?)($|\n)", text, flags=re.S)
    if m:
        idx = text.rfind(r"\end{document}")
        if idx != -1:
            text = text[: idx + len(r"\end{document}")]
    return text.strip()

def extract_first_latex_block(text: str) -> str:
    """Find earliest occurrence of a latex token and return from there."""
    first_pos = None
    for token in LATEX_START_TOKENS:
        m = re.search(token, text)
        if m:
            pos = m.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    if first_pos is not None:
        return text[first_pos:].strip()
    return text.strip()

def escape_underscores_outside_math(text: str) -> str:
    """
    Escape underscores that are not inside $...$ or \(...\). This is heuristic.
    """
    # split on math regions (naive)
    parts = re.split(r"(\$.*?\$|\\\(.+?\\\))", text, flags=re.S)
    for i, part in enumerate(parts):
        if not (part.startswith("$") or part.startswith("\\(")):
            # escape underscores not already escaped
            part = re.sub(r"(?<!\\)_", r"\\_", part)
            parts[i] = part
    return "".join(parts)

def balance_braces(text: str) -> str:
    # if braces mismatch, try to append closing braces
    open_count = text.count("{")
    close_count = text.count("}")
    if close_count < open_count:
        text += "}" * (open_count - close_count)
    return text

def sanitize(raw: str) -> str:
    """
    Full sanitization pipeline: remove client metadata -> extract LaTeX -> fix common issues.
    """
    t = strip_ollama_metadata(raw)
    t = extract_first_latex_block(t)
    # remove markdown fences
    t = re.sub(r"```+", "", t)
    # remove leading/trailing non-ascii control garbage
    t = t.strip()
    # escape underscores
    t = escape_underscores_outside_math(t)
    # fix unbalanced braces
    t = balance_braces(t)
    return t
