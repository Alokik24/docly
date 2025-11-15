# rag_engine/indexer.py

from typing import List, Dict, Optional
import os
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Indexer:
    """
    Builds, saves, and loads a FAISS index for a list of examples.
    Each example is expected to be a dict containing at least 'text'.
    """

    def __init__(self, examples: List[Dict], model_name: str):
        self.examples = examples
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self._built = False

    def build(self):
        texts = [ex["text"] for ex in self.examples]
        print(f"[Indexer] Encoding {len(texts)} documents with '{self.model_name}'...")
        emb = self.embedder.encode(texts, show_progress_bar=True)
        emb = np.asarray(emb, dtype="float32")

        if emb.ndim != 2:
            raise RuntimeError(f"Embeddings expected to be 2D array, got shape {emb.shape}")

        self.dim = emb.shape[1]
        print(f"[Indexer] Embedding dimension = {self.dim}")

        # create simple flat L2 index (suitable for small-medium datasets)
        index = faiss.IndexFlatL2(self.dim)
        index.add(emb)
        self.index = index
        self._built = True
        print(f"[Indexer] Built FAISS index (n={index.ntotal})")

    def save(self, index_path: str, meta_path: str):
        if self.index is None:
            raise RuntimeError("Index not built; call build() before save().")

        # ensure directory exists
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.examples, f)
        print(f"[Indexer] Saved index -> {index_path}")
        print(f"[Indexer] Saved metadata -> {meta_path}")

    @staticmethod
    def load(index_path: str, meta_path: str, model_name: str) -> "Indexer":
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        inst = Indexer(meta, model_name)
        inst.index = index
        inst.dim = index.d
        # instantiate embedder for queries
        inst.embedder = SentenceTransformer(model_name)
        inst._built = True
        print(f"[Indexer] Loaded index (n={index.ntotal}, dim={inst.dim}), embedder='{model_name}'")
        return inst
