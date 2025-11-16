# rag_engine/template_manager.py
from pathlib import Path
from typing import Dict, Optional

DEFAULT_TEMPLATES = {
    "article_minimal": {
        "preamble": r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{lipsum}
% Add your standard macros here
""",
        "postamble": r"""
\end{document}
"""
    },

    "assignment_uff": {
        "preamble": r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\newcommand{\studentname}{\textbf{<STUDENT_NAME>}}
""",
        "postamble": r"\end{document}"
    },
}

class TemplateManager:
    def __init__(self, templates: Optional[Dict] = None):
        self.templates = templates or DEFAULT_TEMPLATES

    def get_preamble(self, name: str = "article_minimal") -> str:
        t = self.templates.get(name)
        if not t:
            raise KeyError(f"Template '{name}' not found")
        return t["preamble"].rstrip()

    def get_postamble(self, name: str = "article_minimal") -> str:
        t = self.templates.get(name)
        if not t:
            raise KeyError(f"Template '{name}' not found")
        return t["postamble"].lstrip()

    def enforce_template(self, latex_body: str, template_name: str = "article_minimal") -> str:
        """
        Correct enforcement:
        - Discard model-generated preamble
        - Keep ONLY body
        - Insert into template exactly once
        """

        pre = self.get_preamble(template_name)
        post = self.get_postamble(template_name)

        # Remove model-generated preambles/preambles
        if "\\begin{document}" in latex_body:
            parts = latex_body.split("\\begin{document}", 1)
            cleaned = parts[1]
        else:
            cleaned = latex_body

        # Remove model-generated end
        cleaned = cleaned.replace("\\end{document}", "")

        cleaned = cleaned.strip()

        # Final assembly
        return f"{pre}\n\\begin{{document}}\n{cleaned}\n{post}"
