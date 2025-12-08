"""
Complete RAG Pipeline: Retrieval + Generation
This script combines retrieval and generation to provide complete RAG functionality.
"""

import json
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import query functions (duplicated here to avoid import issues)
def load_index_metadata(metadata_path: str):
    """Load index metadata"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_hybrid_indexer(indices_dir: str, metadata: dict, drug_mapping_path: str = None):
    """Load hybrid indexer from saved indices"""
    from indexing import MedicalEmbedder, VectorIndexer, BM25Indexer, HybridIndexer
    from preprocessing import MedicalTermNormalizer
    
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
    
    bm25_path = indices_path / 'bm25_index.pkl'
    if bm25_path.exists():
        print(f"Loading BM25 index from {bm25_path}")
        bm25_indexer.load(str(bm25_path))
    else:
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
    
    hybrid_indexer = HybridIndexer(
        vector_indexer=vector_indexer,
        bm25_indexer=bm25_indexer,
        storage_path=indices_dir
    )
    
    return hybrid_indexer
from preprocessing import MedicalTermNormalizer, QueryPreprocessor
from generation import AnswerGenerator, TemplateGenerator


def main(args):
    print("="*60)
    print("Medical RAG System - Complete Pipeline")
    print("="*60)
    
    # ========== Step 1: Load Index ==========
    metadata_path = Path(args.indices_dir) / 'index_metadata.json'
    if not metadata_path.exists():
        print(f"Error: Index metadata not found at {metadata_path}")
        print("Please run build_index.py first to build the index.")
        return
    
    print(f"\n[1/4] Loading index...")
    metadata = load_index_metadata(str(metadata_path))
    print(f"   Index contains {metadata['num_chunks']} chunks")
    
    hybrid_indexer = load_hybrid_indexer(
        indices_dir=args.indices_dir,
        metadata=metadata,
        drug_mapping_path=args.drug_mapping_path
    )
    
    # ========== Step 2: Preprocess Query ==========
    print(f"\n[2/4] Preprocessing query...")
    print(f"   Query: {args.query}")
    
    normalizer = MedicalTermNormalizer()
    if args.drug_mapping_path and Path(args.drug_mapping_path).exists():
        with open(args.drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    
    query_preprocessor = QueryPreprocessor(medical_normalizer=normalizer)
    preprocessed = query_preprocessor.preprocess(args.query)
    
    print(f"   Normalized: {preprocessed['normalized']}")
    if preprocessed['medical_terms']:
        print(f"   Medical terms: {preprocessed['medical_terms']}")
    
    # ========== Step 3: Retrieve Relevant Documents ==========
    print(f"\n[3/4] Retrieving relevant documents...")
    print(f"   Top-K: {args.top_k}, Fusion: {args.fusion_method}")
    
    # Use cleaned query if normalized is too different, otherwise use original
    # The normalized query might be over-normalized, so we prefer cleaned or original
    search_query = preprocessed.get('cleaned') or args.query
    if preprocessed.get('normalized') and len(preprocessed['normalized']) < len(search_query) * 2:
        # Only use normalized if it's not excessively long (over-normalization check)
        search_query = preprocessed['normalized']
    
    results = hybrid_indexer.search(
        query=search_query,
        top_k=args.top_k,
        fusion_method=args.fusion_method,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        filters=args.filters
    )
    
    print(f"   Retrieved {len(results)} documents")
    
    # Display retrieval results
    if args.verbose:
        print(f"\n   Top retrieval results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   [{i}] {result['source']} | Score: {result['score']:.4f}")
            print(f"       {result['text'][:100]}...")
    
    # ========== Step 4: Generate Answer ==========
    print(f"\n[4/4] Generating answer...")
    
    # Initialize generator
    if args.use_llm and args.model_type != 'template':
        generator = AnswerGenerator(
            model_type=args.model_type,
            model_name=args.model_name,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else:
        print("   Using template generator (no LLM)")
        generator = TemplateGenerator()
    
    # Generate answer
    answer_result = generator.generate(
        query=args.query,
        context=results,
        **args.generator_kwargs
    )
    
    # ========== Display Results ==========
    print("\n" + "="*60)
    print("RAG System Results")
    print("="*60)
    
    print(f"\n📝 Question:")
    print(f"   {args.query}")
    
    print(f"\n💡 Answer:")
    print(f"   {answer_result['answer']}")
    
    print(f"\n📚 Sources ({len(answer_result['sources'])} documents):")
    for i, (source_id, result) in enumerate(zip(answer_result['sources'], results), 1):
        print(f"   [{i}] {result['source']} | {result['chunk_type']} | Score: {result['score']:.4f}")
        if args.verbose:
            print(f"       {result['text'][:150]}...")
    
    if answer_result.get('metadata'):
        print(f"\n🔧 Metadata:")
        for key, value in answer_result['metadata'].items():
            print(f"   {key}: {value}")
    
    # ========== Save Results ==========
    if args.output:
        output_data = {
            'query': args.query,
            'preprocessed': preprocessed,
            'retrieval': {
                'num_results': len(results),
                'results': results
            },
            'generation': answer_result
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {args.output}")
    
    print("\n" + "="*60)
    print("RAG Pipeline Complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete RAG Pipeline: Retrieval + Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query with template generator
  python scripts/rag.py --query "What are the side effects of aspirin?"
  
  # Query with OpenAI (requires API key)
  python scripts/rag.py --query "What is diabetes?" \\
      --use_llm --model_type openai --model_name gpt-3.5-turbo \\
      --api_key YOUR_API_KEY
  
  # Query with custom parameters
  python scripts/rag.py --query "Treatment for hypertension" \\
      --top_k 10 --fusion_method weighted --vector_weight 0.7 \\
      --output results/rag_output.json
        """
    )
    
    # Query arguments
    parser.add_argument('--query', type=str, required=True,
                       help='Query text')
    
    # Index arguments
    parser.add_argument('--indices_dir', type=str,
                       default='data/indices',
                       help='Directory containing indices')
    parser.add_argument('--drug_mapping_path', type=str,
                       default='data/processed_for_our_rag/drug_mapping.json',
                       help='Path to drug mapping JSON')
    
    # Retrieval arguments
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--fusion_method', type=str,
                       default='rrf',
                       choices=['rrf', 'weighted'],
                       help='Fusion method for hybrid search')
    parser.add_argument('--vector_weight', type=float, default=0.5,
                       help='Weight for vector search (for weighted fusion)')
    parser.add_argument('--bm25_weight', type=float, default=0.5,
                       help='Weight for BM25 search (for weighted fusion)')
    parser.add_argument('--filters', type=str, default=None,
                       help='Metadata filters as JSON string')
    
    # Generation arguments
    parser.add_argument('--use_llm', action='store_true',
                       help='Use LLM for generation (otherwise use template)')
    parser.add_argument('--model_type', type=str,
                       default='openai',
                       choices=['openai', 'anthropic', 'huggingface', 'local', 'template'],
                       help='LLM model type')
    parser.add_argument('--model_name', type=str,
                       default='gpt-3.5-turbo',
                       help='Model name/identifier')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for cloud LLM services')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Maximum tokens in response')
    parser.add_argument('--generator_kwargs', type=str, default='{}',
                       help='Additional generator kwargs as JSON string')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output path to save results as JSON')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed retrieval results')
    
    args = parser.parse_args()
    
    # Parse JSON arguments
    if args.filters:
        args.filters = json.loads(args.filters)
    else:
        args.filters = None
    
    try:
        args.generator_kwargs = json.loads(args.generator_kwargs)
    except:
        args.generator_kwargs = {}
    
    main(args)

