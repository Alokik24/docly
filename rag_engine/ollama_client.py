# rag_engine/ollama_client.py

import json

try:
    import ollama
except:
    ollama = None

def normalize_ollama(resp):
    if isinstance(resp, dict):
        for key in ("response", "text", "output"):
            if key in resp:
                return resp[key]
        if "choices" in resp:
            c = resp["choices"][0]
            for key in ("message", "text", "content"):
                if key in c:
                    return c[key]
        return json.dumps(resp)
    return str(resp)

def call_local_ollama(prompt, model, max_tokens=1500):
    if ollama is None:
        raise RuntimeError("Ollama not installed")

    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "num_predict": max_tokens,
            "num_ctx": 1024
        }
    )
    return normalize_ollama(resp)
