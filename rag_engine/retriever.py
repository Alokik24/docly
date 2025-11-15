# rag_engine/retriever.py

from typing import List, Dict
import numpy as np


class Retriever:
    """
    Simple retriever that uses the indexer (with embedder and FAISS index)
    to return top-k examples for a query.
    """

    def __init__(self, indexer, k: int = 3):
        if not getattr(indexer, "_built", False) or indexer.index is None:
            raise RuntimeError("Indexer must be built/loaded before creating Retriever.")
        self.indexer = indexer
        self.embedder = indexer.embedder
        self.index = indexer.index
        self.meta = indexer.examples
        self.k = k

    def retrieve(self, query: str) -> List[Dict]:
        """
        Return up to k nearest examples (as list of dicts).
        """
        q_emb = self.embedder.encode([query])
        q_emb = np.asarray(q_emb, dtype="float32")
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        # ensure k does not exceed number of items in index
        k = min(self.k, len(self.meta))
        D, I = self.index.search(q_emb, k)

        indices = I[0].tolist()
        results = []
        for idx in indices:
            if 0 <= idx < len(self.meta):
                results.append(self.meta[idx])
        return results
