"""
Data preprocessing pipeline:
    1. Load data from multiple sources
    2. Build drug mapping
    3. Chunk documents
    4. Save processed data
"""

import json
import sys
from pathlib import Path
import argparse

from src.preprocessing import (
    DataLoader, 
    DocumentChunker,
    MedicalTermNormalizer
)


def main(args):
    print("="*60)
    print("Medical RAG Data Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    loader = DataLoader(
        pubmed_path=args.pubmed_path,
        openfda_path=args.openfda_path,
        kaggle_path=args.kaggle_path
    )
    documents = loader.load_all()
    
    # Step 2: Build drug mapping
    print("\n2. Building medical term normalizer...")
    normalizer = MedicalTermNormalizer()
    normalizer.build_drug_mapping_from_data(documents)
    
    # Save drug mapping
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    drug_mapping_path = output_dir / 'drug_mapping.json'
    normalizer.save_drug_mapping(str(drug_mapping_path))
    
    # Step 3: Chunk documents
    print("\n3. Chunking documents...")
    chunker = DocumentChunker(
        max_chunk_size=args.max_chunk_size,
        overlap=args.overlap,
        sentence_split=True
    )
    chunks = chunker.chunk_documents(documents)
    
    # Step 4: Save processed data
    print("\n4. Saving processed data...")
    
    # Save documents
    documents_path = output_dir / 'documents.jsonl'
    with open(documents_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(documents)} documents to {documents_path}")
    
    # Save chunks
    chunks_path = output_dir / 'chunks.jsonl'
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(chunks)} chunks to {chunks_path}")
    
    # Save statistics
    stats = {
        'num_documents': len(documents),
        'num_chunks': len(chunks),
        'num_drug_mappings': len(normalizer.drug_mapping),
        'sources': {},
        'chunk_types': {}
    }
    
    for doc in documents:
        stats['sources'][doc.source] = stats['sources'].get(doc.source, 0) + 1
    
    for chunk in chunks:
        stats['chunk_types'][chunk.chunk_type] = stats['chunk_types'].get(chunk.chunk_type, 0) + 1
    
    stats_path = output_dir / 'preprocessing_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to {stats_path}")
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Drug mappings: {len(normalizer.drug_mapping)}")
    print("\nSource distribution:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")
    print("\nTop 10 chunk types:")
    top_types = sorted(stats['chunk_types'].items(), key=lambda x: x[1], reverse=True)[:10]
    for ctype, count in top_types:
        print(f"  {ctype}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess medical RAG data')
    
    parser.add_argument('--pubmed_path', type=str, 
                       default='data/BioASQ/corpus_subset.json',
                       help='Path to PubMed data')
    parser.add_argument('--openfda_path', type=str,
                       default='data/OpenFDA Drug data/OpenFDA_corpus.json',
                       help='Path to OpenFDA data')
    parser.add_argument('--kaggle_path', type=str,
                       default='data/kaggle_drug_data/processed/extracted_docs.json',
                       help='Path to Kaggle drug data')
    
    parser.add_argument('--output_dir', type=str,
                       default='data/processed_for_our_rag',
                       help='Output directory for processed data')
    
    parser.add_argument('--max_chunk_size', type=int, default=512,
                       help='Maximum chunk size in characters')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Overlap size between chunks')
    
    args = parser.parse_args()
    
    main(args)