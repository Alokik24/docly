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
        return t["preamble"]

    def get_postamble(self, name: str = "article_minimal") -> str:
        t = self.templates.get(name)
        if not t:
            raise KeyError(f"Template '{name}' not found")
        return t["postamble"]

    def enforce_template(self, latex_body: str, template_name: str = "article_minimal") -> str:
        """
        If the model output already contains a \documentclass, keep it but ensure
        preamble packages are included. If not, wrap the body with template preamble/postamble.
        """
        pre = self.get_preamble(template_name)
        post = self.get_postamble(template_name)
        if "\\documentclass" in latex_body:
            # ensure preamble includes required packages - naive way: if pre contains unique lines,
            # add them before first \begin{document}
            if "\\begin{document}" in latex_body:
                parts = latex_body.split("\\begin{document}", 1)
                head = parts[0]
                tail = "\\begin{document}" + parts[1]
                # inject any missing lines from pre that are not already in head
                for line in pre.splitlines():
                    if line.strip() and line.strip() not in head:
                        head = line + "\n" + head
                return head + tail
            else:
                # has documentclass but no begin{document}, just prepend pre and append post
                return pre + "\n" + latex_body + "\n" + post
        else:
            # no documentclass: wrap in template
            return pre + "\n" + latex_body + "\n" + post
