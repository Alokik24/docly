# rag_engine/placeholder_filler.py
def fill_placeholders(tex: str, values: dict) -> str:
    """
    Replace <PLACEHOLDER> patterns with user-provided values.
    Missing values stay as is.
    """
    for key, val in values.items():
        tex = tex.replace(f"<{key}>", val)
    return tex
