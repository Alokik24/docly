# test_retrieval.py

import pickle
from .indexer import Indexer
from .retriever import Retriever
from .config import CONFIG


def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)


# -------------------------------------------------------
# Load index + metadata
# -------------------------------------------------------
print_header("1. Loading index + metadata")

idx = Indexer.load(
    CONFIG["index_path"],
    CONFIG["meta_path"],
    CONFIG["sentence_transformer_model"],
)
retr = Retriever(idx)

print("Index loaded.")
print("Total items:", len(idx.examples))


# -------------------------------------------------------
# Test 1: Metadata correctness (doc_type, keywords)
# -------------------------------------------------------
print_header("2. Checking metadata formatting")

meta = idx.examples[0]
print("Sample entry:", meta)

if "doc_type" in meta and "keywords" in meta:
    print("PASS: Metadata fields present.")
else:
    print("FAIL: Missing metadata fields.")

print("doc_type:", meta.get("doc_type"))
print("keywords:", meta.get("keywords"))


# -------------------------------------------------------
# Test 2: Retrieval baseline (no filters)
# -------------------------------------------------------
print_header("3. Testing baseline retrieval")

baseline = retr.retrieve("create an assignment")
print("Retrieved IDs:", [e["id"] for e in baseline])

if len(baseline) > 0:
    print("PASS: Baseline retrieval working.")
else:
    print("FAIL: Baseline retrieval returned empty.")


# -------------------------------------------------------
# Test 3: doc_type filtering (FUZZY)
# -------------------------------------------------------
print_header("4. Testing doc_type filter (fuzzy)")

res_doc = retr.retrieve(
    "generate school assignment",
    doc_type="assignment"
)

print("Retrieved doc_type:", [e["doc_type"] for e in res_doc])

if all("assignment" in e["doc_type"].lower() for e in res_doc):
    print("PASS: doc_type fuzzy filtering works.")
else:
    print("FAIL: doc_type fuzzy filtering incorrect.")


# -------------------------------------------------------
# Test 4: keyword filtering (FUZZY)
# -------------------------------------------------------
print_header("5. Testing keyword filter (fuzzy)")

test_kw = ["math"]

res_kw = retr.retrieve(
    "math homework",
    keywords=test_kw
)

print("Retrieved keywords:", [e["keywords"] for e in res_kw])


def fuzzy_kw_match(query_keywords, meta_keywords):
    return any(
        qk.lower() in mk.lower()
        for qk in query_keywords
        for mk in meta_keywords
    )


if len(res_kw) == 0:
    print("WARNING: No items matched the keyword filter.")
elif all(fuzzy_kw_match(test_kw, e["keywords"]) for e in res_kw):
    print("PASS: Keyword fuzzy filtering works.")
else:
    print("FAIL: Keyword fuzzy filtering incorrect.")


# -------------------------------------------------------
# Test 5: Weighted ranking
# -------------------------------------------------------
print_header("6. Testing weighted ranking (manual check)")

res_rank = retr.retrieve(
    "math assignment",
    doc_type="assignment",
    keywords=["math"]
)

print("Top items (id, keywords):")
for e in res_rank:
    print(e["id"], e["keywords"])

print("Check visually:")
print("- Results with math keywords should appear at top")
print("- All doc_type fields should be 'assignment'")
