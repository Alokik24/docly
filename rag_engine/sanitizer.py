import re
from typing import Tuple

LATEX_START_TOKENS = [
    r"\\documentclass",
    r"\\begin\{document\}",
    r"%",       
    r"\\section",
    r"\\title",
    r"\\maketitle",
]

def strip_ollama_metadata(text: str) -> str:
    """
    Remove trailing metadata artifacts without affecting LaTeX.
    """
    text = re.sub(r"'\s*thinking=.*$", "", text, flags=re.S)
    text = re.sub(r"\n?logprobs=.*$", "", text, flags=re.S)
    text = re.sub(r"context=\[.*\]$", "", text, flags=re.S)

    # Cut after last \end{document} if model hallucinated metadata after it
    idx = text.rfind("\\end{document}")
    if idx != -1:
        text = text[: idx + len("\\end{document}")]

    return text.strip()


def extract_first_latex_block(text: str) -> str:
    """
    Very mild correction: DO NOT cut aggressively.
    Only trim leading non-LaTeX junk when it clearly precedes content.
    """
    first_pos = None
    for token in LATEX_START_TOKENS:
        m = re.search(token, text)
        if m:
            pos = m.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    if first_pos is not None and first_pos > 40:
        # Only cut if junk >40 chars, avoids slicing valid content accidentally
        return text[first_pos:].strip()
    return text.strip()

def normalize_backslashes(text: str) -> str:
    """
    Convert double-escaped \\ back to single \.
    """
    # Fix specific escaped LaTeX commands
    text = text.replace("\\\\maketitle", "\\maketitle")
    text = text.replace("\\\\begin", "\\begin")
    text = text.replace("\\\\end", "\\end")
    text = text.replace("\\\\newpage", "\\newpage")

    # Generic rule: \\x (letter) → \x
    def repl(m):
        seq = m.group()
        if re.match(r"\\\\[A-Za-z]", seq):
            return "\\" + seq[2:]
        return seq

    return re.sub(r"\\\\.", repl, text)


def normalize_newlines(text: str) -> str:
    """
    Replace literal \n and \n\n with real newlines.
    """
    text = text.replace("\\n\\n", "\n\n")
    text = text.replace("\\n", "\n")
    return text


def strip_forbidden_macros(text: str) -> str:
    """
    Remove macros the model is not allowed to emit in strict/template mode.
    """
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\title\{.*?\}", "", text)
    text = re.sub(r"\\author\{.*?\}", "", text)
    text = re.sub(r"\\date\{.*?\}", "", text)
    return text

def fix_markdown_and_lists(text: str) -> str:
    # Convert markdown bold **text** -> \textbf{text}
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)

    # Convert "- item" to \item
    text = re.sub(r"^\s*-\s+", r"\\item ", text, flags=re.MULTILINE)

    # Wrap itemize if needed
    if "\\item " in text:
        text = "\\begin{itemize}\n" + text + "\n\\end{itemize}"

    return text

def fix_stray_backslashes_before_percent(text: str) -> str:
    return re.sub(r"\\\s*%", "%", text)


def escape_underscores_outside_math(text: str) -> str:
    # unchanged
    parts = re.split(r"(\$.*?\$|\\\(.+?\\\))", text, flags=re.S)
    for i, part in enumerate(parts):
        if not (part.startswith("$") or part.startswith("\\(")):
            part = re.sub(r"(?<!\\)_", r"\\_", part)
            parts[i] = part
    return "".join(parts)


def balance_braces(text: str) -> str:
    # unchanged
    open_count = text.count("{")
    close_count = text.count("}")
    if close_count < open_count:
        text += "}" * (open_count - close_count)
    return text


def sanitize(latex: str) -> str:
    import re

    # --------------------------------------------------
    # 1. Remove model metadata (thinking=..., context=[...])
    # --------------------------------------------------
    latex = re.sub(r"thinking\s*=\s*\S+", "", latex)
    latex = re.sub(r"context\s*=\s*\[.*?\]", "", latex, flags=re.DOTALL)
    latex = re.sub(r"logprobs\s*=\s*\S+", "", latex)
    latex = re.sub(r"model='[^']+'", "", latex)
    latex = re.sub(r"created_at='[^']+'", "", latex)
    latex = re.sub(r"done_reason='[^']+'", "", latex)
    latex = re.sub(r"total_duration=\S+", "", latex)
    latex = re.sub(r"eval_count=\S+", "", latex)
    latex = re.sub(r"eval_duration=\S+", "", latex)
    latex = re.sub(r"done=True", "", latex)
    latex = re.sub(r"done=False", "", latex)
    latex = re.sub(r"load_duration=\S+", "", latex)
    latex = re.sub(r"eval_duration=\S+", "", latex)
    latex = re.sub(r"prompt_", "", latex)
    latex = re.sub(r"response='", "", latex)
    latex = latex.replace("response=", "")

    latex = re.sub(r"\b(load_duration|prompt_|response|done|model_|duration)=\S+", "", latex)

    # remove stray trailing quotes
    latex = latex.strip().strip("'").strip('"')

    # --------------------------------------------------
    # 2. Remove markdown fences
    # --------------------------------------------------
    latex = latex.replace("```", "")

    # --------------------------------------------------
    # 3. Cut off ANYTHING after \end{document}
    # --------------------------------------------------
    idx = latex.rfind("\\end{document}")
    if idx != -1:
        latex = latex[: idx + len("\\end{document}")]

    # --------------------------------------------------
    # 4. Keep only the first valid LaTeX block
    # --------------------------------------------------
    m = re.search(r"(\\documentclass[\s\S]*)", latex)
    if m:
        latex = m.group(1)

    return latex.strip()
