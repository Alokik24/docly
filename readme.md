

# RAGEngine

A retrieval-augmented LaTeX generation system for producing structured documents such as assignments, reports, and templates. The system combines dataset-driven retrieval, strict LaTeX generation, sanitization, and template enforcement to produce reliable and compilable LaTeX output using local LLMs.

---

## Overview

RAGEngine is a modular pipeline that retrieves relevant examples from a custom dataset and uses them to condition a local language model. The model generates LaTeX body content, which is then sanitized, validated, and finally wrapped into a predefined LaTeX template. The system supports optional PDF compilation and multiple generation modes through a command-line interface.

The engine uses:

* FAISS index for semantic retrieval
* SentenceTransformer embeddings
* Strict LaTeX generation rules
* A sanitization pipeline to remove unsafe or malformed tokens
* TemplateManager for controlled LaTeX wrapping
* Optional DSF-based structured prompting
* Optional lightweight orchestration layer (LangChain PromptTemplate)

---

## Dataset and Index

The dataset is loaded from an Excel file. Each entry contains:

* id
* user prompt
* keywords
* document type
* document structure
* content elements
* LaTeX output

The system concatenates these fields into a text block suitable for embedding and FAISS indexing. Index construction is handled by `Indexer`, which encodes the dataset using SentenceTransformers and builds a FlatL2 FAISS index.

Command:

<pre class="overflow-visible!" data-start="1767" data-end="1807"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python</span><span> -m rag_engine build_index
</span></span></code></div></div></pre>

---

## Retrieval

`Retriever` performs vector search and applies optional filters such as:

* document type (fuzzy match)
* keyword list (fuzzy match)

Results are ranked using combined similarity and metadata weighting.

---

## Prompt Construction

The engine builds model prompts using retrieved examples. It supports two modes:

1. **Full document generation** (when no template is used)
2. **Body-only generation** (when a LaTeX template is provided)

Body-only mode instructs the model not to output:

* \documentclass
* \usepackage
* \begin{document}
* \end{document}
* title/author/date or newcommand macros

Examples from the dataset are included to establish formatting patterns.

---

## Local Model Integration

RAGEngine interacts with local LLMs through `ollama.generate`.

The `call_local_ollama` wrapper normalizes different possible output structures returned by the Ollama client.

---

## Sanitization

All model outputs pass through a sanitization pipeline that:

* removes metadata artifacts
* normalizes backslashes
* cleans stray newlines
* strips forbidden macros
* fixes malformed list syntax or markdown
* corrects common LaTeX hallucinations such as `oindent`, `extbf`, and `extit`
* removes any remaining unintended environments
* ensures only one preamble exists when strict mode is enabled

The sanitization stage guarantees that final output remains compliant with the expected LaTeX structure.

---

## Template Enforcement

The `TemplateManager` inserts sanitized body content into a selected LaTeX template.

It performs preamble stripping (if the model accidentally generates one) and wraps the content between the template’s `\begin{document}` and `\end{document}`.

Templates are declared in `template_manager.py` and can be extended.

---

## DSF Integration

The engine supports DSF (document specification framework) JSON files.

When provided, the DSF description is converted into a structured prompt using `dsf_to_prompt`.

Example:

<pre class="overflow-visible!" data-start="3826" data-end="3879"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python -m rag_engine generate </span><span>--dsf</span><span> plan</span><span>.json</span><span>
</span></span></code></div></div></pre>

---

## Generation Pipeline

The main LaTeX generation path executed by `generate_cmd` consists of:

1. Load FAISS index and metadata
2. Retrieve relevant examples
3. Build a strict LaTeX prompt
4. Call local LLM
5. Sanitize output
6. Strip or correct preamble if needed
7. Enforce template
8. Validate documentclass rules (strict mode)
9. Fill any placeholders
10. Save output to `last_output.tex`
11. Optionally compile to PDF

Command:

<pre class="overflow-visible!" data-start="4342" data-end="4438"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python -m rag_engine generate </span><span>"create an assignment"</span><span> --</span><span>template</span><span> article_minimal --strict
</span></span></code></div></div></pre>

---

## Orchestrator (Optional Layer)

The `RAGOrchestrator` provides a simplified retrieval → prompt → generation interface using a LangChain PromptTemplate.

It produces body content only and defers template insertion to `TemplateManager`.

Command:

<pre class="overflow-visible!" data-start="4695" data-end="4809"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>python -m rag_engine orchestrate "generate math assignment" </span><span>--doc_type</span><span> assignment </span><span>--keywords</span><span> math,calculus
</span></span></code></div></div></pre>

This mode is lighter than the full engine and is intended for experimentation or cleaner flow control.

---

## File Outputs

RAGEngine writes the final LaTeX source to:

<pre class="overflow-visible!" data-start="4983" data-end="5006"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>last_output.tex
</span></span></code></div></div></pre>

If PDF compilation is enabled, output is written to:

<pre class="overflow-visible!" data-start="5062" data-end="5085"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>last_output.pdf
</span></span></code></div></div></pre>

---

## Requirements

* Python 3.10+
* SentenceTransformers
* FAISS
* Ollama server running a compatible local LLM
* A TeX distribution for optional PDF compilation

---

## Summary

RAGEngine provides a controlled LaTeX generation pipeline that combines retrieval, sanitization, strict prompt design, and template enforcement. It is suitable for generating structured academic content with consistency and flexibility, while maintaining control over the final LaTeX formatting.



python -m rag_engine.rag_engine generate "Create a simple LaTeX document with a title and one paragraph"

python -m rag_engine.rag_engine build_index

python -m rag_engine.test_retrieval

qwen2.5:1.5b-instruct

qwen2.5:3b

python -m rag_engine.rag_engine generate "generate a math assignment latex template" --doc_type assignment --keywords math,proof --template article_minimal --strict
