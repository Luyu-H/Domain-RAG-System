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