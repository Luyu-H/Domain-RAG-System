# Domain-RAG-System

## Designed Medical RAG System
Run the system using the scripts in /scripts. The corresponding implementation code is in the files under /src.

### 1. Preprocess Data
Process raw data and create chunks:
```bash
python scripts/preprocess_data.py \
    --pubmed_path "data/BioASQ/corpus_subset.json" \
    --openfda_path "data/OpenFDA Drug data/OpenFDA_corpus.json" \
    --kaggle_path "data/kaggle_drug_data/processed/extracted_docs.json" \
    --output_dir data/processed \
    --max_chunk_size 512 \
    --overlap 50
```

**Output:**
- `data/processed/documents.jsonl` - Processed documents
- `data/processed/chunks.jsonl` - Document chunks
- `data/processed/drug_mapping.json` - Drug name mappings
- `data/processed/preprocessing_stats.json` - Statistics

### 2. Build Index
Build hybrid index (vector + BM25) for retrieval:
```bash
python scripts/build_index.py \
    --chunks_path data/processed_for_our_rag/chunks.jsonl \
    --embedding_model pritamdeka/S-PubMedBert-MS-MARCO \
    --output_dir data/indices
```

**Output:**
- `data/indices/bm25_index.pkl` - BM25 index
- `data/indices/index_metadata.json` - Index metadata
- `data/vector_db/` - Qdrant vector database

### 3. Query the Index (Retrieval Only)
Test retrieval without generation:
```bash
python scripts/query.py \
    --query "What are the side effects of aspirin?" \
    --top_k 5 \
    --fusion_method rrf \
    --reranker_kind crossencoder \
    --rerank_top_n 30 \
    --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Options:**
- `--query`: Query text (required)
- `--top_k`: Number of results to return (default: 5)
- `--fusion_method`: 'rrf' or 'weighted' (default: 'rrf')
- `--vector_weight`: Weight for vector search (for weighted fusion)
- `--bm25_weight`: Weight for BM25 search (for weighted fusion)
- `--output`: Save results to JSON file
- `--reranker_kind`: none | simple | crossencoder.
simple: cosine on current embedder (fast, no extra deps).
crossencoder: pairwise Cross-Encoder scoring (best accuracy).
- `--rerank_top_n`: Size of candidate pool to rerank. Typical range: 20–50.
- `--cross_model`: encode model from Hugging Face or local

### 4. Complete RAG Pipeline (Retrieval + Generation)
Run the complete RAG system with answer generation:

**Using Template Generator (No LLM required):**
```bash
python scripts/rag.py \
    --query "What are the side effects of aspirin?" \
    --top_k 5
```

**Using OpenAI (requires API key):**
```bash
python scripts/rag.py \
    --query "What is diabetes?" \
    --use_llm \
    --model_type openai \
    --model_name gpt-3.5-turbo \
    --api_key YOUR_API_KEY \
    --reranker_kind crossencoder \
    --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Using Anthropic Claude:**
```bash
python scripts/rag.py \
    --query "Treatment for hypertension" \
    --use_llm \
    --model_type anthropic \
    --model_name claude-3-sonnet-20240229 \
    --api_key YOUR_API_KEY \
    --reranker_kind crossencoder \
    --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Options:**
- `--query`: Query text (required)
- `--use_llm`: Use LLM for generation (otherwise use template)
- `--model_type`: 'openai', 'anthropic', 'huggingface', 'local', or 'template'
- `--model_name`: Model identifier
- `--api_key`: API key for cloud LLM services
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens in response (default: 500)
- `--top_k`: Number of documents to retrieve (default: 5)
- `--fusion_method`: 'rrf' or 'weighted' (default: 'rrf')
- `--reranker_kind`: none | simple | crossencoder.
- `--rerank_top_n`: Size of candidate pool to rerank. Typical range: 20–50.
- `--cross_model`: encode model from Hugging Face or local
- `--output`: Save results to JSON file
- `--verbose`: Show detailed retrieval results