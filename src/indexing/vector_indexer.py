from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Vector indexing will not work.")

from .embedder import MedicalEmbedder


class VectorIndexer:
    """
    Vector indexing using Qdrant vector database
        - Medical domain embeddings
        - Metadata filtering
        - Batch indexing
        - Persistent storage
    """
    
    def __init__(self,
                 collection_name: str = 'medical_docs',
                 embedder: MedicalEmbedder = None,
                 storage_path: str = './data/vector_db',
                 distance_metric: str = 'cosine',
                 use_in_memory: bool = False):
        """
        Args:
            collection_name: name of the Qdrant collection
            embedder: MedicalEmbedder instance
            storage_path: path to store vector database
            distance_metric: 'cosine', 'euclidean', or 'dot'
            use_in_memory: if True, use in-memory storage (for testing)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.embedder = embedder or MedicalEmbedder()
        self.use_in_memory = use_in_memory
        
        if not use_in_memory:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = None
        
        distance_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
            'dot': Distance.DOT
        }
        self.distance_metric = distance_map.get(distance_metric.lower(), Distance.COSINE)
        
        if use_in_memory:
            self.client = QdrantClient(":memory:")
            print("Initialized in-memory Qdrant client")
        else:
            self.client = QdrantClient(path=str(self.storage_path))
            print(f"Initialized persistent Qdrant client at: {self.storage_path}")
        
        print(f"Collection: {collection_name}")
        print(f"Embedding model: {self.embedder.model_name}")
    
    def create_collection(self, force_recreate: bool = True):
        """
        Args:
            force_recreate: if True, delete existing collection and create new
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and force_recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.embedding_dim,
                        distance=self.distance_metric
                    )
                )
                print(f"Collection created with dimension: {self.embedder.embedding_dim}")
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def index_chunks(self, chunks: List, batch_size: int = 100):
        """
        Index chunks with embeddings
        
        Args:
            chunks: list of Chunk objects or dictionaries
            batch_size: batch size for indexing
        """
        print(f"\nIndexing {len(chunks)} chunks...")
        
        texts = []
        for chunk in chunks:
            if hasattr(chunk, 'text'):
                texts.append(chunk.text)
            else:
                texts.append(chunk['text'])
        
        print("Generating embeddings...")
        embeddings = self.embedder.encode(texts)
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert chunk to dictionary if it's an object
            if hasattr(chunk, 'to_dict'):
                chunk_dict = chunk.to_dict()
            else:
                chunk_dict = chunk
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'chunk_id': chunk_dict.get('chunk_id', f'chunk_{idx}'),
                    'doc_id': chunk_dict.get('doc_id', ''),
                    'source': chunk_dict.get('source', ''),
                    'chunk_type': chunk_dict.get('chunk_type', ''),
                    'text': chunk_dict.get('text', ''),
                    'metadata': chunk_dict.get('metadata', {})
                }
            )
            points.append(point)
        
        print(f"Uploading to Qdrant (batch_size={batch_size})...")
        try:
            for i in tqdm(range(0, len(points), batch_size), disable=not True):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"Successfully indexed {len(chunks)} chunks")
            
            # Print collection info
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Collection points count: {collection_info.points_count}")
        except Exception as e:
            print(f"Error during indexing: {e}")
            raise
    
    def search(self,
               query: str,
               top_k: int = 10,
               filters: Dict = None,
               score_threshold: float = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: query text
            top_k: number of results to return
            filters: metadata filters (e.g., {'source': 'pubmed'})
            score_threshold: minimum similarity score
            
        Returns:
            list of search results with scores
        """
        query_embedding = self.embedder.encode(query)
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        search_params = {}
        if score_threshold is not None:
            search_params['score_threshold'] = score_threshold
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            **search_params
        )
        # Handle QueryResponse
        if hasattr(search_result, 'points'):
            hits = search_result.points
        else:
            hits = search_result
        
        # Format results
        results = []
        for hit in hits:
            result = {
                'chunk_id': hit.payload['chunk_id'],
                'doc_id': hit.payload['doc_id'],
                'source': hit.payload['source'],
                'chunk_type': hit.payload['chunk_type'],
                'text': hit.payload['text'],
                'metadata': hit.payload['metadata'],
                'score': hit.score
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'embedding_dim': self.embedder.embedding_dim,
                'model_name': self.embedder.model_name,
                'storage_path': str(self.storage_path) if self.storage_path else 'in-memory'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    
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
    
    print(f"\nCreating vector index with {len(chunks)} chunks...")
    
    # Use in-memory for testing to avoid PyTorch issues
    indexer = VectorIndexer(
        collection_name='test_medical',
        storage_path='./data/test_vector_db',
        use_in_memory=False  # Set to True for faster testing
    )
    
    indexer.create_collection(force_recreate=True)
    indexer.index_chunks(chunks)
    
    print("\nTesting search...")
    query = "What are the side effects of aspirin?"
    
    try:
        results = indexer.search(query, top_k=5)
        
        print(f"\nTop 5 results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   Type: {result['chunk_type']}")
            print(f"   Text: {result['text'][:150]}...")
        
        stats = indexer.get_stats()
        print(f"\nIndex stats: {stats}")
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()