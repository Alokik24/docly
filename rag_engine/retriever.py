# rag_engine/retriever.py

from typing import List, Dict, Optional
import numpy as np
import faiss


class Retriever:
    """
    Retriever with:
    - fuzzy metadata filtering
    - FAISS search on filtered subset
    - weighted scoring combining embedding + metadata
    """

    def __init__(self, indexer, k: int = 3):
        if not getattr(indexer, "_built", False) or indexer.index is None:
            raise RuntimeError("Indexer must be built/loaded before creating Retriever.")

        self.indexer = indexer
        self.embedder = indexer.embedder
        self.index = indexer.index
        self.meta = indexer.examples
        self.dim = indexer.dim
        self.k = k

        self.all_embeddings = self._extract_all_embeddings()

    def _extract_all_embeddings(self):
        total = self.index.ntotal
        emb = np.zeros((total, self.dim), dtype="float32")
        for i in range(total):
            emb[i] = self.index.reconstruct(i)
        return emb

    # ---------------------------------------------------------
    # FUZZY FILTERING + SEARCH
    # ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:

        # Encode query to embedding
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        # ----------------------------------------------
        # 1. FUZZY METADATA FILTER
        # ----------------------------------------------
        filtered_indices = []
        for i, meta in enumerate(self.meta):

            # Fuzzy doc_type matching
            if doc_type:
                dt = meta.get("doc_type", "").lower()
                if doc_type.lower() not in dt:
                    continue

            # Fuzzy keyword matching
            if keywords:
                meta_kw = [kw.lower() for kw in meta.get("keywords", [])]
                if not any(
                    kw.lower() in mk.lower()
                    for kw in keywords
                    for mk in meta_kw
                ):
                    continue

            filtered_indices.append(i)

        # If no metadata matched → fallback to entire dataset
        if not filtered_indices:
            filtered_indices = list(range(len(self.meta)))

        # ----------------------------------------------
        # 2. FAISS SEARCH ON FILTERED SUBSET
        # ----------------------------------------------
        filtered_emb = self.all_embeddings[filtered_indices]

        subset_index = faiss.IndexFlatL2(self.dim)
        subset_index.add(filtered_emb)

        k = min(top_k or self.k, len(filtered_indices))

        D, I = subset_index.search(q_emb, k)

        # ----------------------------------------------
        # 3. Weighted scoring (embedding + metadata)
        # ----------------------------------------------
        combined = []
        for local_idx, dist in zip(I[0], D[0]):
            global_idx = filtered_indices[local_idx]

            # convert distance → similarity
            emb_sim = 1.0 / (1.0 + dist)

            meta_score = 0.0

            # strong boost if doc_type fuzzy matches
            if doc_type:
                dt = self.meta[global_idx].get("doc_type", "").lower()
                if doc_type.lower() in dt:
                    meta_score += 0.5

            # keyword fuzzy overlap scoring
            if keywords:
                mk = [k.lower() for k in self.meta[global_idx].get("keywords", [])]
                overlap = sum(
                    1 for kw in keywords
                    for mkw in mk
                    if kw.lower() in mkw.lower()
                )
                meta_score += 0.1 * overlap

            final_score = 0.8 * emb_sim + 0.2 * meta_score
            combined.append((global_idx, final_score))

        combined.sort(key=lambda x: x[1], reverse=True)

        # ----------------------------------------------
        # 4. Return final results
        # ----------------------------------------------
        return [self.meta[idx] for idx, _ in combined[:k]]
