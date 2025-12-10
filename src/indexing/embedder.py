from typing import List, Union, Dict
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


class MedicalEmbedder:
    """
    Generate embeddings using medical domain-specific models.
    
    Candidate models:
        - 'pubmedbert_marco': 'pritamdeka/S-PubMedBert-MS-MARCO', Best for retrieval,
        - 'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        - 'biobert': 'dmis-lab/biobert-v1.1',
        - 'sapbert': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext', Good for entity matching,
        - 'bioclinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        - 'baseline': 'all-MiniLM-L6-v2', Fast baseline
    """
    
    def __init__(self, model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO',
                 batch_size: int = 32, max_length: int = 512):
        """
        Args:
            model_name: HuggingFace model name or local path
            batch_size: batch size for encoding
            max_length: maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_length
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to default model: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: single text or list of texts
            normalize: normalize embeddings to unit length
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode with batching
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def encode_chunks(self, chunks: List[Dict], text_key: str = 'text') -> np.ndarray:
        """
        Encode chunks (from DocumentChunker).
        
        Args:
            chunks: list of chunk dictionaries
            text_key: key for text field in chunk dict
            
        Returns:
            numpy array of embeddings
        """
        texts = [chunk[text_key] if isinstance(chunk, dict) else chunk.text 
                for chunk in chunks]
        return self.encode(texts)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'device': self.device
        }


if __name__ == '__main__':
    # Test embedder
    embedder = MedicalEmbedder(model_name='pritamdeka/S-PubMedBert-MS-MARCO')
    
    test_texts = [
        "Aspirin is used to treat pain and inflammation",
        "Common side effects include nausea and headache",
        "Contraindicated in patients with bleeding disorders"
    ]
    
    print("\nTesting embedder...")
    embeddings = embedder.encode(test_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings)
    print(f"\nSimilarity matrix:\n{sim}")