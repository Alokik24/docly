from langchain_core.prompts import PromptTemplate

from .retriever import Retriever
from .template_manager import TemplateManager
from .sanitizer import sanitize
from .ollama_client import call_local_ollama
from .config import CONFIG


class RAGOrchestrator:
    def __init__(self, indexer):
        self.indexer = indexer
        self.retriever = Retriever(indexer, CONFIG["k"])
        self.tm = TemplateManager()

        # JINJA2 SAFE TEMPLATE (LaTeX braces allowed)
        self.prompt_template = PromptTemplate.from_template(
            """
You are a STRICT LaTeX generator.
Do NOT use markdown or code fences.
ONLY output LaTeX BODY content (inside \\begin{document} ... \\end{document}).

NEVER output:
- \\documentclass
- \\usepackage
- \\begin{document}
- \\end{document}
- title/author/date macros

# Retrieved Examples
{{ example_section }}

# User Request
{{ query }}

Respond with LaTeX BODY ONLY.
""",
            template_format="jinja2"
        )

    def build_examples(self, examples):
        if not examples:
            return "(no examples retrieved)"
        blocks = []
        for e in examples:
            blocks.append(
                f"PROMPT:\n{e.get('user_prompt','')}\n"
                f"LATEX:\n{e.get('latex_output','')}\n"
                "---------------------------"
            )
        return "\n".join(blocks)

    def generate(self, query, doc_type=None, keywords=None, template="article_minimal"):
        examples = self.retriever.retrieve(query, doc_type=doc_type, keywords=keywords)

        prompt = self.prompt_template.format(
            example_section=self.build_examples(examples),
            query=query
        )

        raw = call_local_ollama(prompt, CONFIG["local_llm_model"], max_tokens=1800)
        clean = sanitize(raw)
        final = self.tm.enforce_template(clean, template)

        return final
