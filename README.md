# Domain-RAG-System

## Designed Medical RAG System
Run the system using the scripts in /scripts. The corresponding implementation code is in the files under /src.

### 1. Preprocess Data
Process raw data from three different sources and create chunks:
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

### 5. Run Comprehensive Evaluation of Designed RAG System
Evaluate the RAG system on test datasets and compare with baseline results:

```bash
python evaluation/comprehensive_evaluation.py
```

**Output Files:**
- `results/openfda_rag_test_results.json` - RAG system results for OpenFDA
- `results/kaggle_rag_test_results.json` - RAG system results for Kaggle
- `results/bioasq_rag_test_results.json` - RAG system results for BioASQ (if run separately)
- `results/comprehensive_evaluation.json` - Complete evaluation report
- `results/comprehensive_evaluation_report.md` - Markdown summary report

**Run BioASQ Evaluation Separately:**
To run BioASQ evaluation separately (it has 200 queries and may take longer):
```bash
python evaluation/run_bioasq_evaluation.py
```

This will generate `results/bioasq_rag_test_results.json` which will be automatically used by the comprehensive evaluation script.

**Note:** The evaluation script automatically runs RAG system evaluation if result files don't exist. To force re-evaluation, delete the existing result files in the `results/` directory.

### 6. Run Evaluation of Baselines
#### For BioASQ Dataset:
```bash
python baselines/faiss/FAISS_BioASQ.py
```
#### For OpenFDA Dataset:
```bash
python baselines/faiss/openfda_faiss_test.py
```
#### For Kaggle Dataset:
```bash
python baselines/Qdrant/Qdrant_test.py
```