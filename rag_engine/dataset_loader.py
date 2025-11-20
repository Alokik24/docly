# rag_engine/dataset_loader.py

from typing import List, Dict
import pandas as pd


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def load_dataset(excel_path: str) -> List[Dict]:
    """
    Load the dataset from the provided Excel path and return a list of examples.

    Each example is a dict with:
      - id
      - text  (concatenated fields used for embedding)
      - user_prompt
      - latex_output
      - (optional other fields preserved)
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    examples = []

    for _, row in df.iterrows():
        id_ = _safe_str(row.get("id"))
        user_prompt = _safe_str(row.get("user_prompt"))
        keywords = _safe_str(row.get("keywords"))
        doc_type = _safe_str(row.get("doc_type"))
        doc_struct = _safe_str(row.get("document_structure"))
        content_elems = _safe_str(row.get("content_elements"))
        latex_output = _safe_str(row.get("latex_output"))

        text = "\n---\n".join(
            [
                f"DOC_ID: {id_}",
                f"DOC_TYPE: {doc_type}",
                f"PROMPT: {user_prompt}",
                f"KEYWORDS: {keywords}",
                f"STRUCTURE: {doc_struct}",
                f"ELEMENTS: {content_elems}",
            ]
        )

        examples.append(
        {
            "id": id_,
            "text": text,
            "user_prompt": user_prompt,
            "latex_output": latex_output,
            "doc_type": doc_type.lower(),
            "keywords": [kw.strip().lower() for kw in keywords.split(",")] if keywords else [],
            "structure": doc_struct,
            "content_elements": content_elems,
        }
    )


    return examples
