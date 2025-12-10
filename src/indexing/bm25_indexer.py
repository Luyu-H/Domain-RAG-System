from typing import List, Dict, Tuple
import pickle
from pathlib import Path
import numpy as np
from collections import defaultdict
import re

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not installed. Install with: pip install rank-bm25")

import sys
sys.path.append('.')
from src.preprocessing import MedicalTermNormalizer


class BM25Indexer:
    """
    BM25 indexer with medical domain optimizations
        - Medical term preservation
        - Custom tokenization
        - Configurable BM25 parameters
        - Persistent storage
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, medical_normalizer: MedicalTermNormalizer = None):
        """
        Args:
            k1: BM25 parameter controlling term frequency saturation (default: 1.5)
            b: BM25 parameter controlling document length normalization (default: 0.75)
            medical_normalizer: MedicalTermNormalizer for term normalization
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")
        
        self.k1 = k1
        self.b = b
        self.normalizer = medical_normalizer or MedicalTermNormalizer()
        
        # Will be set during indexing
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        
        print(f"Initialized BM25Indexer (k1={k1}, b={b})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with medical term preservation
        
        Args:
            text: input text
            
        Returns:
            list of tokens
        """
        text = text.lower()
        
        # Preserve medical terms (don't split them)
        # Replace common medical abbreviations to keep them intact
        preserved_terms = []
        for abbr in self.normalizer.abbreviations.keys():
            if abbr.lower() in text:
                preserved_terms.append(abbr.lower())
        
        # Simple tokenization: split by non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter very short tokens (but keep medical abbreviations)
        filtered_tokens = []
        for token in tokens:
            if len(token) >= 2 or token in preserved_terms:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def index_chunks(self, chunks: List):
        """
        Args:
            chunks: list of Chunk objects or dictionaries
        """
        print(f"\nIndexing {len(chunks)} chunks with BM25...")
        
        self.chunks = chunks

        from tqdm import tqdm
        iterator = tqdm(chunks, desc="Tokenizing")
        
        for chunk in iterator:
            if hasattr(chunk, 'text'):
                text = chunk.text
            else:
                text = chunk['text']
            
            tokens = self.tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        print(f"Successfully indexed {len(chunks)} chunks")
        
        avg_doc_len = np.mean([len(doc) for doc in self.tokenized_corpus])
        print(f"Average document length: {avg_doc_len:.2f} tokens")
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """
        Args:
            query: query text
            top_k: number of results to return
            filters: metadata filters (e.g., {'source': 'pubmed'})
            
        Returns:
            list of search results with BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_chunks() first.")
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # Get more for filtering
        
        # Format results with filtering
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            
            if hasattr(chunk, 'to_dict'):
                chunk_dict = chunk.to_dict()
            else:
                chunk_dict = chunk
            
            if filters:
                skip = False
                for key, value in filters.items():
                    if chunk_dict.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            result = {
                'chunk_id': chunk_dict.get('chunk_id', f'chunk_{idx}'),
                'doc_id': chunk_dict.get('doc_id', ''),
                'source': chunk_dict.get('source', ''),
                'chunk_type': chunk_dict.get('chunk_type', ''),
                'text': chunk_dict.get('text', ''),
                'metadata': chunk_dict.get('metadata', {}),
                'score': float(scores[idx])
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, filepath: str):
        """Save BM25 index to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BM25 index to {filepath}")
    
    def load(self, filepath: str):
        """Load BM25 index from disk"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        
        print(f"Loaded BM25 index from {filepath}")
        print(f"Index contains {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.bm25 is None:
            return {'error': 'Index not built'}
        
        return {
            'num_documents': len(self.chunks),
            'avg_doc_length': float(np.mean([len(doc) for doc in self.tokenized_corpus])),
            'vocab_size': len(set(token for doc in self.tokenized_corpus for token in doc)),
            'k1': self.k1,
            'b': self.b
        }


if __name__ == '__main__':
    # Test BM25 indexer
    from src.preprocessing import DocumentChunker, DataLoader
    
    print("Loading test data...")
    loader = DataLoader(
        pubmed_path='data/BioASQ/corpus_subset.json',
        openfda_path='data/OpenFDA Drug data/OpenFDA_corpus.json',
        kaggle_path='data/kaggle_drug_data/processed/extracted_docs.json'
    )
    documents = loader.load_all()[:10]  # Test with first 10 docs
    
    print("\nChunking...")
    chunker = DocumentChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    print(f"\nCreating BM25 index with {len(chunks)} chunks...")
    indexer = BM25Indexer(k1=1.5, b=0.75)
    indexer.index_chunks(chunks)
    
    print("\nTesting search...")
    query = "aspirin side effects"
    results = indexer.search(query, top_k=5)
    
    print(f"\nTop 5 BM25 results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Type: {result['chunk_type']}")
        print(f"   Text: {result['text'][:150]}...")
    
    stats = indexer.get_stats()
    print(f"\nIndex stats: {stats}")
    
    # Test save/load
    print("\nTesting save/load...")
    indexer.save('data/test_bm25_index.pkl')
    
    new_indexer = BM25Indexer()
    new_indexer.load('data/test_bm25_index.pkl')
    print("Load successful!")