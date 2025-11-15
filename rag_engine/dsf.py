# rag_engine/dsf.py
from typing import Dict, Any, List

def dsf_to_prompt(dsf: Dict[str, Any]) -> str:
    """
    Convert a small JSON-like DSF into a prompt that describes the document structure.
    Example DSF:
    {
      "document_type": "assignment",
      "title": "AD1 - Math",
      "author": "Alice",
      "sections": [
         {"title": "Questão 1", "instructions": "Generate one short problem"},
         {"title": "Questão 2", "instructions": "Two-part problem"}
      ]
    }
    """
    lines: List[str] = []
    lines.append(f"Document type: {dsf.get('document_type','document')}")
    if "title" in dsf:
        lines.append(f"Title: {dsf['title']}")
    if "author" in dsf:
        lines.append(f"Author: {dsf['author']}")
    lines.append("Sections:")
    for i, s in enumerate(dsf.get("sections", []), start=1):
        lines.append(f"  - Section {i}: {s.get('title','')}")
        if s.get("instructions"):
            lines.append(f"    Instructions: {s['instructions']}")
    if dsf.get("notes"):
        lines.append("Notes:")
        lines.append(dsf["notes"])
    lines.append("\nProduce only LaTeX matching this structure.")
    return "\n".join(lines)
