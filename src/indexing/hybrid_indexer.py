from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import json

from .vector_indexer import VectorIndexer
from .bm25_indexer import BM25Indexer


class HybridIndexer:
    """
    Hybrid retrieval combining dense retrieval (vector search) and sparse retrieval (BM25).
    Uses Reciprocal Rank Fusion (RRF) for score fusion.
    """
    
    def __init__(self, vector_indexer: VectorIndexer = None, bm25_indexer: BM25Indexer = None,
                 storage_path: str = './data/indices'):
        """
        Args:
            vector_indexer: VectorIndexer instance
            bm25_indexer: BM25Indexer instance
            storage_path: path to store indices
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_indexer = vector_indexer
        self.bm25_indexer = bm25_indexer
    
    def index_chunks(self, chunks: List):
        """
        Index chunks with both vector and BM25.
        
        Args:
            chunks: list of Chunk objects
        """
        print("\n" + "="*60)
        print("Building Hybrid Index")
        print("="*60)
        
        # Index with vector indexer
        if self.vector_indexer:
            print("\n[1/2] Building vector index...")
            self.vector_indexer.create_collection(force_recreate=True)
            self.vector_indexer.index_chunks(chunks)
        else:
            print("\nWarning: No vector indexer provided")
        
        # Index with BM25
        if self.bm25_indexer:
            print("\n[2/2] Building BM25 index...")
            self.bm25_indexer.index_chunks(chunks)
        else:
            print("\nWarning: No BM25 indexer provided")
        
        print("\n" + "="*60)
        print("Hybrid index built successfully!")
        print("="*60)
    
    def search(self, query: str, top_k: int = 10,
               vector_weight: float = 0.5, bm25_weight: float = 0.5,
               filters: Dict = None, fusion_method: str = 'rrf') -> List[Dict]:
        """
        Hybrid search with score fusion.
        
        Args:
            query: query text
            top_k: number of results to return
            vector_weight: weight for vector search scores
            bm25_weight: weight for BM25 scores
            filters: metadata filters
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (weighted sum)
            
        Returns:
            list of fused search results
        """
        vector_results = []
        bm25_results = []
        
        if self.vector_indexer:
            vector_results = self.vector_indexer.search(
                query, 
                top_k=top_k * 2,  # Retrieve more for fusion
                filters=filters
            )
        
        if self.bm25_indexer:
            bm25_results = self.bm25_indexer.search(
                query,
                top_k=top_k * 2,
                filters=filters
            )
        
        # Fuse results
        if fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(
                vector_results, 
                bm25_results,
                top_k=top_k
            )
        else:  # weighted
            fused_results = self._weighted_fusion(
                vector_results,
                bm25_results,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                top_k=top_k
            )
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, vector_results: List[Dict], bm25_results: List[Dict],
                                 top_k: int = 10, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF), score = sum(1 / (k + rank))
        
        Args:
            vector_results: results from vector search
            bm25_results: results from BM25 search
            top_k: number of results to return
            k: constant for RRF (typically 60)
            
        Returns:
            fused and ranked results
        """
        rrf_scores = {}
        
        # Add vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Build result map
        result_map = {}
        for result in vector_results + bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id not in result_map:
                result_map[chunk_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for chunk_id, score in sorted_ids[:top_k]:
            result = result_map[chunk_id].copy()
            result['score'] = float(score)
            result['fusion_method'] = 'rrf'
            fused_results.append(result)
        
        return fused_results
    
    def _weighted_fusion(self, vector_results: List[Dict], bm25_results: List[Dict],
                        vector_weight: float = 0.5, bm25_weight: float = 0.5,
                        top_k: int = 10) -> List[Dict]:
        """
        Weighted score fusion.
        
        Args:
            vector_results: results from vector search
            bm25_results: results from BM25 search
            vector_weight: weight for vector scores
            bm25_weight: weight for BM25 scores
            top_k: number of results to return
            
        Returns:
            fused and ranked results
        """
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return results
            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
            return results
        
        vector_results = normalize_scores(vector_results)
        bm25_results = normalize_scores(bm25_results)
        
        # Combine scores
        combined_scores = {}
        result_map = {}
        
        for result in vector_results:
            chunk_id = result['chunk_id']
            score = result.get('normalized_score', result['score'])
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + vector_weight * score
            result_map[chunk_id] = result
        
        for result in bm25_results:
            chunk_id = result['chunk_id']
            score = result.get('normalized_score', result['score'])
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + bm25_weight * score
            if chunk_id not in result_map:
                result_map[chunk_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for chunk_id, score in sorted_ids[:top_k]:
            result = result_map[chunk_id].copy()
            result['score'] = float(score)
            result['fusion_method'] = 'weighted'
            fused_results.append(result)
        
        return fused_results
    
    def save_bm25(self, filepath: str = None):
        """Save BM25 index"""
        if filepath is None:
            filepath = self.storage_path / 'bm25_index.pkl'
        
        if self.bm25_indexer:
            self.bm25_indexer.save(filepath)
    
    def load_bm25(self, filepath: str = None):
        """Load BM25 index"""
        if filepath is None:
            filepath = self.storage_path / 'bm25_index.pkl'
        
        if self.bm25_indexer:
            self.bm25_indexer.load(filepath)
    
    def get_stats(self) -> Dict:
        """Get statistics for both indices"""
        stats = {}
        
        if self.vector_indexer:
            stats['vector'] = self.vector_indexer.get_stats()
        
        if self.bm25_indexer:
            stats['bm25'] = self.bm25_indexer.get_stats()
        
        return stats


if __name__ == '__main__':
    # Test hybrid indexer
    import sys
    sys.path.append('.')
    
    from src.preprocessing import DocumentChunker, DataLoader
    from src.indexing.embedder import MedicalEmbedder
    
    print("Loading test data...")
    loader = DataLoader(
        pubmed_path='data/BioASQ/corpus_subset.json',
        openfda_path='data/OpenFDA Drug data/OpenFDA_corpus.json',
        kaggle_path='data/kaggle_drug_data/processed/extracted_docs.json'
    )
    documents = loader.load_all()[:20]  # Test with first 20 docs
    
    print("\nChunking...")
    chunker = DocumentChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    print(f"\nCreating hybrid index with {len(chunks)} chunks...")
    
    # Initialize components
    embedder = MedicalEmbedder(model_name='pritamdeka/S-PubMedBert-MS-MARCO')
    vector_indexer = VectorIndexer(
        collection_name='test_hybrid',
        embedder=embedder,
        storage_path='./data/test_hybrid_db'
    )
    bm25_indexer = BM25Indexer(k1=1.5, b=0.75)
    
    # Create hybrid indexer
    hybrid_indexer = HybridIndexer(
        vector_indexer=vector_indexer,
        bm25_indexer=bm25_indexer,
        storage_path='./data/test_hybrid_indices'
    )
    
    # Index chunks
    hybrid_indexer.index_chunks(chunks)
    
    # Test search with different fusion methods
    query = "What are the side effects of aspirin?"
    
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)
    
    # RRF fusion
    print("\n[1] Reciprocal Rank Fusion (RRF):")
    rrf_results = hybrid_indexer.search(query, top_k=5, fusion_method='rrf')
    for i, result in enumerate(rrf_results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']} | Type: {result['chunk_type']}")
        print(f"   Text: {result['text'][:120]}...")
    
    # Weighted fusion
    print("\n[2] Weighted Fusion (0.7 vector + 0.3 BM25):")
    weighted_results = hybrid_indexer.search(
        query, 
        top_k=5, 
        fusion_method='weighted',
        vector_weight=0.7,
        bm25_weight=0.3
    )
    for i, result in enumerate(weighted_results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']} | Type: {result['chunk_type']}")
        print(f"   Text: {result['text'][:120]}...")
    
    # Print stats
    print(f"\n{'='*60}")
    print("Index Statistics:")
    print('='*60)
    stats = hybrid_indexer.get_stats()
    print(json.dumps(stats, indent=2))