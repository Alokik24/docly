"""
RAGEngine – Phase 2 (patched)
Sanitization + Template Enforcement + DSF + Optional PDF Compilation

Notes on changes:
- Prompt building now supports "template_provided" -> body-only generation.
- Sanitization is always applied (keeps metadata stripping protections).
- Post-generation validation added for strict mode to reject or repair preamble leaks.
- Removed requirement "Start with \documentclass" from strict prompts.
"""

import argparse
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import CONFIG
from .dataset_loader import load_dataset
from .indexer import Indexer
from .retriever import Retriever
from .sanitizer import (
    normalize_backslashes,
    normalize_newlines,
    strip_forbidden_macros,
    fix_markdown_and_lists,
    fix_stray_backslashes_before_percent,
    sanitize
)

from .template_manager import TemplateManager
from .placeholder_filler import fill_placeholders
from .dsf import dsf_to_prompt

# Optional Ollama import
try:
    import ollama
except Exception:
    ollama = None


# ----------------------------------------------------
# Normalize Ollama response safely
# ----------------------------------------------------
def normalize_ollama(resp):
    """Extract text from any Ollama response shape."""
    if isinstance(resp, dict):
        for key in ("response", "text", "output"):
            if key in resp:
                return resp[key]

        # Sometimes nested
        if "choices" in resp and isinstance(resp["choices"], list):
            c = resp["choices"][0]
            for key in ("message", "text", "content"):
                if key in c:
                    return c[key]

        return json.dumps(resp)

    return str(resp)


# ----------------------------------------------------
# Build Phase-2 prompt (supports body-only mode)
# ----------------------------------------------------
def build_prompt(user_request: str, examples: List[dict], template_provided: bool = False) -> str:
    """
    Build a prompt. If template_provided is True, the model is instructed to
    produce ONLY the body (content to go inside \\begin{document} ... \\end{document}).
    """
    parts = [
        "You are a STRICT LaTeX generator.",
        "Output ONLY valid LaTeX source.",
        "NEVER use markdown, NEVER use ``` fences.",
        "Do NOT explain anything, do not add commentary.",
        "",
    ]

    if template_provided:
        parts += [
            "IMPORTANT: A template will wrap your output. YOU MUST ONLY GENERATE",
            "the LaTeX BODY — the content that goes *inside* the document environment.",
            "DO NOT output any of the following anywhere in your response:",
            "- \\documentclass{...}",
            "- \\usepackage{...}",
            "- \\begin{document}",
            "- \\end{document}",
            "- Any preamble-level macros (eg. \\newcommand, \\title, \\author).",
            "Output should start with content elements like \\section{...} or plain paragraphs.",
            "",
        ]
    else:
        # When not using a template, full-document allowed
        parts += [
            "If no template is provided, you may generate a full LaTeX document,",
            "including \\documentclass and preamble as needed.",
            "",
        ]

    parts += [
        "EXAMPLES (for format guidance):",
    ]

    for ex in examples:
        parts += [
            "EXAMPLE_PROMPT:",
            ex.get("user_prompt", ""),
            "EXAMPLE_LATEX:",
            ex.get("latex_output", ""),
            "-" * 30
        ]

    parts += [
        "USER_REQUEST:",
        user_request,
        "Respond ONLY with LaTeX source (respect the body-only rule above if a template is supplied)."
    ]

    return "\n".join(parts)


# ----------------------------------------------------
# Call Ollama
# ----------------------------------------------------
def call_local_ollama(prompt: str, model: str, max_tokens: int = 1500):
    if ollama is None:
        raise RuntimeError("Ollama client missing. Install with: pip install ollama")

    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "num_predict": max_tokens,
            "num_ctx": 1024
        }
    )
    return normalize_ollama(resp)


# ----------------------------------------------------
# Optional: Compile LaTeX → PDF
# ----------------------------------------------------
def try_compile(tex: str, out_pdf_path: Path) -> Optional[str]:
    """Compile LaTeX to PDF using pdflatex if available."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tex_file = td / "doc.tex"
        tex_file.write_text(tex, encoding="utf-8")

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "doc.tex"],
                cwd=td, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            pdf_file = td / "doc.pdf"

            if pdf_file.exists():
                out_pdf_path.write_bytes(pdf_file.read_bytes())
                return None

            return "Compilation finished but PDF missing."

        except FileNotFoundError:
            return "pdflatex not installed."
        except Exception as e:
            return str(e)


# ----------------------------------------------------
# Build FAISS index
# ----------------------------------------------------
def build_index_cmd():
    print("[RAGEngine] Loading dataset:", CONFIG["excel_path"])
    data = load_dataset(CONFIG["excel_path"])
    idx = Indexer(data, CONFIG["sentence_transformer_model"])
    idx.build()
    idx.save(CONFIG["index_path"], CONFIG["meta_path"])
    print("[RAGEngine] Index built successfully.")


# ----------------------------------------------------
# Validation utilities
# ----------------------------------------------------
FORBIDDEN_PREAMBLE_TOKENS = [
    r"\\documentclass",
    r"\\usepackage",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"\\newcommand",
    r"\\title",
    r"\\author",
]


def contains_forbidden_preamble(text: str) -> bool:
    for tok in FORBIDDEN_PREAMBLE_TOKENS:
        if re.search(tok, text):
            return True
    return False


import re  # placed here to keep top of file tidy


# ----------------------------------------------------
# Generate LaTeX
# ----------------------------------------------------
def generate_cmd(
    user_request: Optional[str],
    dsf_path: Optional[str],
    template: str,
    strict: bool,
    compile_pdf: bool,
):
    print("[RAGEngine] Loading index...")

    # Load retriever only if examples are required
    idx = Indexer.load(
        CONFIG["index_path"],
        CONFIG["meta_path"],
        CONFIG["sentence_transformer_model"],
    )
    retriever = Retriever(idx, CONFIG["k"])

    # If DSF JSON is provided, override the user request
    if dsf_path:
        dsf = json.loads(Path(dsf_path).read_text(encoding="utf-8"))
        user_request = dsf_to_prompt(dsf)
        print("[RAGEngine] DSF → Prompt applied.")

    if not user_request:
        raise RuntimeError("No user request provided.")

    # Retrieve examples (can be empty list)
    examples = retriever.retrieve(user_request)
    print(f"[RAGEngine] Retrieved example IDs: {[e['id'] for e in examples]}")

    # Build prompt with template-aware instructions
    template_provided = bool(template)
    prompt = build_prompt(user_request, examples, template_provided=template_provided)

    # Call model
    model = CONFIG["local_llm_model"]
    print(f"[RAGEngine] Calling model: {model}")
    try:
        raw_output = call_local_ollama(prompt, model, max_tokens=2000)
    except Exception as e:
        raise RuntimeError(f"[RAGEngine] Model call failed: {e}")

    # Always sanitize to strip metadata and normalize escapes/newlines
    sanitized = sanitize(raw_output)
    sanitized = normalize_backslashes(sanitized)
    sanitized = normalize_newlines(sanitized)
    sanitized = strip_forbidden_macros(sanitized)
    sanitized = fix_stray_backslashes_before_percent(sanitized)
    sanitized = fix_markdown_and_lists(sanitized)
    sanitized = re.sub(r"^\s*[\}\]]+\s*", "", sanitized)
    # Remove stray environment endings without a matching begin
    sanitized = re.sub(r"\\end\{itemize\}", "", sanitized)
    sanitized = re.sub(r"\\end\{enumerate\}", "", sanitized)
    sanitized = re.sub(r"\\end\{.*?\}", "", sanitized)
    sanitized = re.sub(r"\\begin\{frame\}(\[[^\]]*\])?", "", sanitized)
    sanitized = re.sub(r"\\end\{frame\}", "", sanitized)
    sanitized = sanitized.replace("\\t", "")
    sanitized = sanitized.replace("\t", "")

    # If strict and template_provided: enforce that model DID NOT emit preamble tokens.
    if strict and template_provided:
        if contains_forbidden_preamble(sanitized):
            # Strict mode: prefer to treat this as a failure, but attempt safe body extraction and re-wrap.
            print("[RAGEngine] WARNING: model emitted preamble tokens despite template enforcement in strict mode.")
            print("[RAGEngine] Attempting to extract body-only and re-wrap with the canonical template.")
            # We will still proceed, but this should be treated as a test failure in CI.
            # Let TemplateManager handle extraction (it strips model preamble).
        else:
            print("[RAGEngine] Strict-mode: model output passed preamble token check.")
    # This ensures no random "latex", itemize, frame, beamer garbage appears
    body_start = re.search(r"(\\section|\\subsection|\\paragraph|\\begin\{)", sanitized)
    if body_start:
        sanitized = sanitized[body_start.start():]
    else:
        # As fallback: ensure some minimal output
        sanitized = "\\section{Output}\n" + sanitized
    # Template enforcement (this will strip any model preamble if present)
    tm = TemplateManager()
    final_output = tm.enforce_template(sanitized, template)

    # Post-assembly checks (strict)
    if strict:
        # Ensure only one documentclass (that from template) exists
        docclass_count = len(re.findall(r"\\documentclass", final_output))
        if template_provided:
            # Template should be single source of docclass (count == 1)
            if docclass_count != 1:
                # Fatal under strict mode
                raise RuntimeError(f"[RAGEngine] Strict validation failed: expected exactly 1 \\documentclass from template, found {docclass_count}.")
        else:
            # If no template provided but strict asked, at least ensure 0 or 1 documentclass
            if docclass_count > 1:
                raise RuntimeError(f"[RAGEngine] Strict validation failed: multiple \\documentclass found ({docclass_count}).")

        # Basic check for stray logprobs / token dumps (very defensive)
        # if re.search(r"\b(logprobs|thinking|context)=\b", raw_output):
        #     raise RuntimeError("[RAGEngine] Strict validation failed: model output contains client metadata tokens.")

    # Placeholder filling
    final_output = fill_placeholders(final_output, CONFIG.get("placeholders", {}))

    # Save LaTeX file
    tex_path = Path("last_output.tex")
    tex_path.write_text(final_output, encoding="utf-8")
    print("[RAGEngine] Saved LaTeX to:", tex_path)

    # Optional PDF compile
    if compile_pdf:
        pdf_path = Path("last_output.pdf")
        result = try_compile(final_output, pdf_path)

        if result is None:
            print("[RAGEngine] PDF compiled →", pdf_path)
        else:
            print("[RAGEngine] PDF compile error:")
            print(result)


# ----------------------------------------------------
# CLI Entry point
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="rag_engine")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build_index")
    p_build.set_defaults(func=lambda a: build_index_cmd())

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("user_request", nargs="?", default=None)
    p_gen.add_argument("--dsf", type=str, default=None)
    p_gen.add_argument("--template", type=str, default="article_minimal")
    p_gen.add_argument("--strict", action="store_true")
    p_gen.add_argument("--compile", action="store_true")
    p_gen.set_defaults(func=lambda a: generate_cmd(
        a.user_request, a.dsf, a.template, a.strict, a.compile
    ))

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
