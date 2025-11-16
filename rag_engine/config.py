from pathlib import Path

# folder containing rag_engine
PACKAGE_ROOT = Path(__file__).resolve().parent

# project root (one level above rag_engine)
PROJECT_ROOT = PACKAGE_ROOT.parent

CONFIG = {
    "excel_path": str(PROJECT_ROOT / "data" / "docly_dataset2(AutoRecovered).xlsx"),

    "sentence_transformer_model": "all-MiniLM-L6-v2",
    "local_llm_model": "qwen2.5:1.5b-instruct",
    "index_path": str(PACKAGE_ROOT / "index.faiss"),
    "meta_path": str(PACKAGE_ROOT / "meta.pkl"),

    "k": 3,
    "faiss_index_type": "IndexFlatL2",

    # Optional placeholder values
    "placeholders": {
        "STUDENT_NAME": "Alokik Garg",
        "TITLE": "My Document"
    }
}
