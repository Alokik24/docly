# rag_engine/rag_engine.py

"""
RAGEngine CLI and wrapper for local-only LaTeX generation pipeline.

Usage (from project root):
    python -m rag_engine.rag_engine build_index
    python -m rag_engine.rag_engine generate "Create a LaTeX title page for X"
"""

import argparse
import os
import json
from typing import List

from .config import CONFIG
from .dataset_loader import load_dataset
from .indexer import Indexer
from .retriever import Retriever

# Try importing ollama client; if not available we'll surface an informative error.
try:
    import ollama
except Exception:
    ollama = None


def build_prompt(user_request: str, examples: List[dict]) -> str:
    """
    Build the prompt that instructs the local LLM to output ONLY LaTeX code.
    We include retrieved examples to provide context.
    """
    prompt_parts = [
        "SYSTEM:",
        "You are an unconditional LaTeX generator.",
        "Your output MUST be valid LaTeX code only.",
        "Never output Markdown, never output backticks, never output explanations.",
        "Do NOT talk in natural language.",
        "Start your answer with a LaTeX command like \\documentclass or \\begin.",
        "Do NOT include ``` anywhere.",
        "Do NOT include analysis or commentary.",
        "If the user prompt is unclear, STILL produce a valid minimal LaTeX document.",
        "",
        "If multiple files are required, separate them using comments such as:",
        "% === file: main.tex ===",
        "",
        "EXAMPLES:",
    ]

    for ex in examples:
        prompt_parts.append("\nEXAMPLE_PROMPT:\n")
        prompt_parts.append(ex.get("user_prompt", ""))
        prompt_parts.append("\nEXAMPLE_LATEX:\n")
        prompt_parts.append(ex.get("latex_output", ""))
        prompt_parts.append("\n" + ("-" * 30))

    prompt_parts.append("\nUSER_REQUEST:\n")
    prompt_parts.append(user_request)
    prompt_parts.append("\n\nRespond with LaTeX source only.\n")
    return "\n".join(prompt_parts)


def call_local_ollama(prompt: str, model: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
    if ollama is None:
        raise RuntimeError(
            "Ollama Python client not available. Install with `pip install ollama` and ensure Ollama is running."
        )

    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    )

    # Official structure ALWAYS has resp["response"]
    try:
        return resp["response"]
    except Exception:
        return str(resp)



def save_last_output(text: str, path: str = "last_output.tex"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[RAGEngine] Saved LaTeX output to {path}")


def build_index_cmd():
    print("[RAGEngine] Loading dataset:", CONFIG["excel_path"])
    examples = load_dataset(CONFIG["excel_path"])
    idx = Indexer(examples, CONFIG["sentence_transformer_model"])
    idx.build()
    idx.save(CONFIG["index_path"], CONFIG["meta_path"])
    print("[RAGEngine] Index build complete.")


def generate_cmd(user_request: str):
    print("[RAGEngine] Loading index and metadata...")
    idx = Indexer.load(CONFIG["index_path"], CONFIG["meta_path"], CONFIG["sentence_transformer_model"])
    retriever = Retriever(idx, CONFIG.get("k", 3))

    retrieved = retriever.retrieve(user_request)
    print(f"[RAGEngine] Retrieved {len(retrieved)} examples. IDs: {[r['id'] for r in retrieved]}")

    prompt = build_prompt(user_request, retrieved)

    # call local LLM via Ollama
    model = CONFIG.get("local_llm_model", "mistral")
    print(f"[RAGEngine] Calling local model '{model}' (ollama)...")
    out = call_local_ollama(prompt, model=model)

    # Simple heuristic to keep LaTeX only: look for first latex token
    tokens = ["\\documentclass", "\\begin{document}", "%", "\\section", "\\title", "\\maketitle"]
    first_pos = None
    for t in tokens:
        p = out.find(t)
        if p != -1 and (first_pos is None or p < first_pos):
            first_pos = p
    latex_only = out[first_pos:].strip() if first_pos is not None else out.strip()

    print("\n=== LaTeX Output (preview) ===\n")
    print(latex_only[:4000])
    save_last_output(latex_only)


def main():
    parser = argparse.ArgumentParser(prog="rag_engine")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build_index", help="Build FAISS index from Excel dataset.")
    p_build.set_defaults(func=lambda args: build_index_cmd())

    p_gen = sub.add_parser("generate", help="Generate LaTeX for a user request using the local LLM.")
    p_gen.add_argument("user_request", type=str, help="Freeform user request for LaTeX generation.")
    p_gen.set_defaults(func=lambda args: generate_cmd(args.user_request))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
