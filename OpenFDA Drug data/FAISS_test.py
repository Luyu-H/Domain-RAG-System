#!/usr/bin/env python3
"""
Simplified vector storage testing script
Mainly tests FAISS, other systems require additional setup
"""

import json
import numpy as np
import time
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_faiss_retrieval():
    """Test FAISS vector retrieval"""
    logger.info("Starting FAISS vector retrieval test...")
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load data
        with open('extracted_drug_documents.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        with open('test_queries.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            queries = test_data['queries']
            ground_truth = test_data['ground_truth']
        
        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
        
        # Load embedding model
        logger.info("Loading Sentence Transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate document embeddings
        logger.info("Generating document embeddings...")
        texts = [doc['text'] for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use inner product similarity
        
        # Normalize embedding vectors
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created, containing {index.ntotal} vectors")
        
        # Test queries
        results = {}
        
        for query in queries:
            logger.info(f"Test queries: {query['query']}")
            start_time = time.time()
            
            # Generate query vector
            query_embedding = model.encode([query['query']])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = index.search(query_embedding.astype('float32'), 10)
            
            # Get results
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(documents):
                    retrieved_docs.append({
                        'doc_id': documents[idx]['id'],
                        'score': float(score),
                        'brand_name': documents[idx]['brand_name'],
                        'generic_name': documents[idx]['generic_name'],
                        'text_preview': documents[idx]['text'][:200] + "..."
                    })
            
            query_time = time.time() - start_time
            
            # Calculate evaluation metrics
            ground_truth_ids = set(ground_truth[query['id']])
            retrieved_ids = [doc['doc_id'] for doc in retrieved_docs]
            intersection = set(retrieved_ids) & ground_truth_ids
            
            precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[query['id']] = {
                'query': query['query'],
                'description': query['description'],
                'difficulty': query['difficulty'],
                'query_time': query_time,
                'retrieved_docs': retrieved_docs,
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'relevant_retrieved': len(intersection),
                    'total_retrieved': len(retrieved_ids),
                    'total_relevant': len(ground_truth_ids)
                }
            }
            
            logger.info(f"Query completed - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
        
        return results
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please run: pip install faiss-cpu sentence-transformers")
        return None
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None

def print_results_summary(results: Dict[str, Any]):
    """Print results summary"""
    if not results:
        print("No test results")
        return
    
    print("\n" + "="*80)
    print("FAISS Vector Retrieval Test Results")
    print("="*80)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_time = 0
    
    for query_id, result in results.items():
        metrics = result['metrics']
        print(f"\nQuery {query_id}: {result['description']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Query: {result['query']}")
        print(f"Performance metrics:")
        print(f"  - Precision: {metrics['precision']:.3f}")
        print(f"  - Recall: {metrics['recall']:.3f}")
        print(f"  - F1 Score: {metrics['f1_score']:.3f}")
        print(f"  - Query time: {result['query_time']:.3f}seconds")
        print(f"  - Relevant documents retrieved: {metrics['relevant_retrieved']}/{metrics['total_relevant']}")
        
        print(f"\nRetrieval results (top 5):")
        for i, doc in enumerate(result['retrieved_docs'][:5], 1):
            print(f"  {i}. {doc['brand_name']} ({doc['generic_name']}) - Score: {doc['score']:.3f}")
            print(f"     Preview: {doc['text_preview']}")
        
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1_score']
        total_time += result['query_time']
    
    # Calculate average metrics
    num_queries = len(results)
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = total_f1 / num_queries
    avg_time = total_time / num_queries
    
    print(f"\n" + "="*80)
    print("Overall Performance Summary")
    print("="*80)
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    print(f"Average Query time: {avg_time:.3f} seconds")
    print(f"Total Queries: {num_queries}")

def save_detailed_results(results: Dict[str, Any], output_file: str):
    """Save detailed results"""
    if not results:
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed results saved to: {output_file}")

def main():
    """Main function"""
    print("Starting FAISS vector retrieval test...")
    
    # Run tests
    results = test_faiss_retrieval()
    
    if results:
        # Print results
        print_results_summary(results)
        
        # Save results
        save_detailed_results(results, 'faiss_test_results.json')
        
        print(f"\nTest completed! Results saved to faiss_test_results.json")
    else:
        print("Test failed, please check error messages")

if __name__ == "__main__":
    main()
