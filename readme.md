python -m rag_engine.rag_engine generate "Create a simple LaTeX document with a title and one paragraph"

python -m rag_engine.rag_engine build_index

python -m rag_engine.test_retrieval

qwen2.5:1.5b-instruct

qwen2.5:3b

python -m rag_engine.rag_engine generate "generate a math assignment latex template" --doc_type assignment --keywords math,proof --template article_minimal --strict
