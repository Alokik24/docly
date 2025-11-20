from langchain_core.prompts import PromptTemplate

from .retriever import Retriever
from .template_manager import TemplateManager
from .sanitizer import sanitize_pipeline   # we'll define below
from .rag_engine import call_local_ollama
from .config import CONFIG


class RAGOrchestrator:
    def __init__(self, indexer):
        self.indexer = indexer
        self.retriever = Retriever(indexer, CONFIG["k"])
        self.tm = TemplateManager()

        self.prompt_template = PromptTemplate.from_template(
            """
You are a STRICT LaTeX generator.
Do NOT use markdown.
ONLY output LaTeX.

# Examples
{example_section}

# User request
{query}

Respond with LaTeX body content only.
"""
        )

    def build_examples(self, examples):
        out = []
        for e in examples:
            out.append(f"PROMPT:\n{e['user_prompt']}\nLATEX:\n{e['latex_output']}\n---")
        return "\n".join(out)

    def generate(self, query, doc_type=None, keywords=None, template="article_minimal"):
        # 1. retrieve
        ex = self.retriever.retrieve(query, doc_type=doc_type, keywords=keywords)

        # 2. format prompt
        example_block = self.build_examples(ex)
        prompt = self.prompt_template.format(example_section=example_block, query=query)

        # 3. run model
        raw = call_local_ollama(prompt, CONFIG["local_llm_model"], max_tokens=1800)

        # 4. sanitize
        clean = sanitize_pipeline(raw)

        # 5. wrap with template
        final = self.tm.enforce_template(clean, template)

        return final
