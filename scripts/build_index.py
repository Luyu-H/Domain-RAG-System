import json
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from indexing import (
    MedicalEmbedder,
    VectorIndexer,
    BM25Indexer,
    HybridIndexer
)
from preprocessing import MedicalTermNormalizer


def load_chunks(chunks_path: str):
    """Load chunks from JSONL file"""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks


def main(args):
    print("="*60)
    print("Medical RAG Index Building Pipeline")
    print("="*60)
    
    print(f"\n1. Loading chunks from {args.chunks_path}...")
    chunks = load_chunks(args.chunks_path)
    print(f"Loaded {len(chunks)} chunks")
    
    print(f"\n2. Initializing embedding model: {args.embedding_model}...")
    embedder = MedicalEmbedder(
        model_name=args.embedding_model,
        batch_size=args.batch_size
    )
    
    print(f"\n3. Initializing indexers...")
    # Vector
    vector_indexer = VectorIndexer(
        collection_name=args.collection_name,
        embedder=embedder,
        storage_path=args.vector_storage_path,
        distance_metric=args.distance_metric
    )
    # BM25 
    normalizer = MedicalTermNormalizer()
    if Path(args.drug_mapping_path).exists():
        print(f"Loading drug mapping from {args.drug_mapping_path}")
        with open(args.drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    
    bm25_indexer = BM25Indexer(
        k1=args.bm25_k1,
        b=args.bm25_b,
        medical_normalizer=normalizer
    )
    
    # Hybrid indexer
    hybrid_indexer = HybridIndexer(
        vector_indexer=vector_indexer,
        bm25_indexer=bm25_indexer,
        storage_path=args.output_dir
    )
    
    print(f"\n4. Building hybrid index...")
    hybrid_indexer.index_chunks(chunks)
    
    # Save BM25 index
    bm25_path = Path(args.output_dir) / 'bm25_index.pkl'
    hybrid_indexer.save_bm25(str(bm25_path))
    
    # Save index metadata
    metadata = {
        'num_chunks': len(chunks),
        'embedding_model': args.embedding_model,
        'embedding_dim': embedder.embedding_dim,
        'collection_name': args.collection_name,
        'bm25_params': {'k1': args.bm25_k1, 'b': args.bm25_b},
        'distance_metric': args.distance_metric
    }
    
    metadata_path = Path(args.output_dir) / 'index_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved index metadata to {metadata_path}")
    
    print("\n" + "="*60)
    print("Index Building Complete!")
    print("="*60)
    stats = hybrid_indexer.get_stats()
    print("\nIndex Statistics:")
    print(json.dumps(stats, indent=2))
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"- Vector DB: {args.vector_storage_path}")
    print(f"- BM25 index: {bm25_path}")
    print(f"- Metadata: {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build hybrid index for medical RAG')
    
    parser.add_argument('--chunks_path', type=str,
                       default='data/processed_for_our_rag/chunks.jsonl',
                       help='Path to chunks JSONL file')
    parser.add_argument('--drug_mapping_path', type=str,
                       default='data/processed_for_our_rag/drug_mapping.json',
                       help='Path to drug mapping JSON')
    
    parser.add_argument('--embedding_model', type=str,
                       default='pritamdeka/S-PubMedBert-MS-MARCO',
                       choices=[
                           'pritamdeka/S-PubMedBert-MS-MARCO',
                           'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                           'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                           'all-MiniLM-L6-v2'
                       ],
                       help='Embedding model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding generation')
    
    parser.add_argument('--collection_name', type=str,
                       default='medical_rag',
                       help='Qdrant collection name')
    parser.add_argument('--vector_storage_path', type=str,
                       default='data/vector_db',
                       help='Path to store vector database')
    parser.add_argument('--distance_metric', type=str,
                       default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Distance metric for vector search')
    
    parser.add_argument('--bm25_k1', type=float, default=1.5,
                       help='BM25 k1 parameter')
    parser.add_argument('--bm25_b', type=float, default=0.75,
                       help='BM25 b parameter')
    
    parser.add_argument('--output_dir', type=str,
                       default='data/indices',
                       help='Output directory for indices')
    
    args = parser.parse_args()
    
    main(args)