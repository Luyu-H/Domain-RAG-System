"""
Query script for RAG system.
Loads the built index and performs retrieval.
"""

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
from preprocessing import MedicalTermNormalizer, QueryPreprocessor


def load_index_metadata(metadata_path: str):
    """Load index metadata"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_hybrid_indexer(
    indices_dir: str,
    metadata: dict,
    drug_mapping_path: str = None
) -> HybridIndexer:
    """
    Load hybrid indexer from saved indices
    
    Args:
        indices_dir: directory containing indices
        metadata: index metadata
        drug_mapping_path: path to drug mapping file
    """
    indices_path = Path(indices_dir)
    
    print("Loading embedding model...")
    embedder = MedicalEmbedder(
        model_name=metadata['embedding_model'],
        batch_size=32
    )
    
    print("Loading vector indexer...")
    vector_indexer = VectorIndexer(
        collection_name=metadata['collection_name'],
        embedder=embedder,
        storage_path=metadata.get('vector_storage_path', 'data/vector_db'),
        distance_metric=metadata['distance_metric']
    )
    
    print("Loading BM25 indexer...")
    normalizer = MedicalTermNormalizer()
    if drug_mapping_path and Path(drug_mapping_path).exists():
        print(f"Loading drug mapping from {drug_mapping_path}")
        with open(drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    
    bm25_params = metadata.get('bm25_params', {'k1': 1.5, 'b': 0.75})
    bm25_indexer = BM25Indexer(
        k1=bm25_params['k1'],
        b=bm25_params['b'],
        medical_normalizer=normalizer
    )
    
    # Load BM25 index
    bm25_path = indices_path / 'bm25_index.pkl'
    if bm25_path.exists():
        print(f"Loading BM25 index from {bm25_path}")
        bm25_indexer.load(str(bm25_path))
    else:
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
    
    # Create hybrid indexer
    hybrid_indexer = HybridIndexer(
        vector_indexer=vector_indexer,
        bm25_indexer=bm25_indexer,
        storage_path=indices_dir
    )
    
    return hybrid_indexer


def main(args):
    print("="*60)
    print("Medical RAG Query System")
    print("="*60)
    
    # Load metadata
    metadata_path = Path(args.indices_dir) / 'index_metadata.json'
    if not metadata_path.exists():
        print(f"Error: Index metadata not found at {metadata_path}")
        print("Please run build_index.py first to build the index.")
        return
    
    print(f"\n1. Loading index metadata from {metadata_path}...")
    metadata = load_index_metadata(str(metadata_path))
    print(f"   Index contains {metadata['num_chunks']} chunks")
    print(f"   Embedding model: {metadata['embedding_model']}")
    
    # Load indexer
    print(f"\n2. Loading hybrid indexer...")
    hybrid_indexer = load_hybrid_indexer(
        indices_dir=args.indices_dir,
        metadata=metadata,
        drug_mapping_path=args.drug_mapping_path
    )
    
    # Initialize query preprocessor
    print(f"\n3. Initializing query preprocessor...")
    normalizer = MedicalTermNormalizer()
    if args.drug_mapping_path and Path(args.drug_mapping_path).exists():
        with open(args.drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    query_preprocessor = QueryPreprocessor(medical_normalizer=normalizer)
    
    # Process query
    print(f"\n4. Processing query...")
    print(f"   Query: {args.query}")
    
    preprocessed = query_preprocessor.preprocess(args.query)
    print(f"   Normalized: {preprocessed['normalized']}")
    if preprocessed['medical_terms']:
        print(f"   Medical terms: {preprocessed['medical_terms']}")
    
    # Search
    # Use cleaned query if normalized is too different, otherwise use original
    # The normalized query might be over-normalized, so we prefer cleaned or original
    search_query = preprocessed.get('cleaned') or args.query
    if preprocessed.get('normalized') and len(preprocessed['normalized']) < len(search_query) * 2:
        # Only use normalized if it's not excessively long (over-normalization check)
        search_query = preprocessed['normalized']
    
    print(f"\n5. Searching index (top_k={args.top_k}, fusion={args.fusion_method})...")
    results = hybrid_indexer.search(
        query=search_query,
        top_k=args.top_k,
        fusion_method=args.fusion_method,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        filters=args.filters
    )
    
    # Display results
    print(f"\n" + "="*60)
    print(f"Retrieval Results (Top {len(results)})")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.4f}")
        print(f"    Chunk ID: {result['chunk_id']}")
        print(f"    Source: {result['source']} | Type: {result['chunk_type']}")
        print(f"    Text: {result['text'][:200]}...")
        if result.get('metadata'):
            print(f"    Metadata: {result['metadata']}")
    
    # Save results if output path specified
    if args.output:
        output_data = {
            'query': args.query,
            'preprocessed': preprocessed,
            'num_results': len(results),
            'results': results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "="*60)
    print("Query complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the medical RAG index')
    
    parser.add_argument('--query', type=str, required=True,
                       help='Query text')
    parser.add_argument('--indices_dir', type=str,
                       default='data/indices',
                       help='Directory containing indices')
    parser.add_argument('--drug_mapping_path', type=str,
                       default='data/processed_for_our_rag/drug_mapping.json',
                       help='Path to drug mapping JSON')
    
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--fusion_method', type=str,
                       default='rrf',
                       choices=['rrf', 'weighted'],
                       help='Fusion method for hybrid search')
    parser.add_argument('--vector_weight', type=float, default=0.5,
                       help='Weight for vector search (for weighted fusion)')
    parser.add_argument('--bm25_weight', type=float, default=0.5,
                       help='Weight for BM25 search (for weighted fusion)')
    
    parser.add_argument('--filters', type=str, default=None,
                       help='Metadata filters as JSON string, e.g., \'{"source": "pubmed"}\'')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path to save results as JSON')
    
    args = parser.parse_args()
    
    # Parse filters if provided
    if args.filters:
        import json
        args.filters = json.loads(args.filters)
    else:
        args.filters = None
    
    main(args)

