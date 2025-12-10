#!/usr/bin/env python3
"""
OpenFDA FAISS Vector Retrieval Test Script
Using OpenFDA corpus and test queries for FAISS retrieval testing
"""

import json
import numpy as np
import time
from typing import List, Dict, Any
import logging
import os
import sys

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_openfda_data():
    """Load OpenFDA corpus data"""
    corpus_path = "/Users/doupeihao/Desktop/CSE291a/Domain-RAG-System/data/OpenFDA Drug data/OpenFDA_corpus.json"
    queries_path = "/Users/doupeihao/Desktop/CSE291a/Domain-RAG-System/data/OpenFDA Drug data/openfda_test_queries.json"
    
    logger.info("Loading OpenFDA corpus data...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    logger.info("Loading OpenFDA test queries...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    logger.info(f"Corpus contains {len(corpus_data)} documents")
    logger.info(f"Test queries contain {len(queries_data['questions'])} queries")
    
    return corpus_data, queries_data

def create_document_index(corpus_data):
    """Create document index for corpus"""
    doc_index = {}
    for i, doc in enumerate(corpus_data):
        doc_index[doc['id']] = i
    return doc_index

def test_faiss_retrieval():
    """Test FAISS vector retrieval"""
    logger.info("Starting OpenFDA FAISS vector retrieval test...")
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load data
        corpus_data, queries_data = load_openfda_data()
        doc_index = create_document_index(corpus_data)
        
        # Load embedding model
        logger.info("Loading Sentence Transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate document embeddings
        logger.info("Generating document embeddings...")
        texts = [doc['text'] for doc in corpus_data]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use inner product similarity
        
        # Normalize embedding vectors
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        # Test queries
        results = {}
        questions = queries_data['questions']
        
        for question in questions:
            query_id = question['id']
            query_text = question['body']
            ground_truth_docs = set(question['documents'])
            ideal_answer = question['ideal_answer']
            
            logger.info(f"Processing query {query_id}: {query_text}")
            start_time = time.time()
            
            # Generate query vector
            query_embedding = model.encode([query_text])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = index.search(query_embedding.astype('float32'), 10)
            
            # Get retrieval results
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(corpus_data):
                    doc = corpus_data[idx]
                    retrieved_docs.append({
                        'doc_id': doc['id'],
                        'score': float(score),
                        'brand_name': doc['brand_name'],
                        'generic_name': doc['generic_name'],
                        'text_preview': doc['text'][:300] + "...",
                        'metadata': doc['metadata']
                    })
            
            query_time = time.time() - start_time
            
            # Calculate evaluation metrics
            retrieved_ids = [doc['doc_id'] for doc in retrieved_docs]
            intersection = set(retrieved_ids) & ground_truth_docs
            
            precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(intersection) / len(ground_truth_docs) if ground_truth_docs else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[query_id] = {
                'query': query_text,
                'ideal_answer': ideal_answer,
                'ground_truth_docs': list(ground_truth_docs),
                'query_time': query_time,
                'retrieved_docs': retrieved_docs,
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'relevant_retrieved': len(intersection),
                    'total_retrieved': len(retrieved_ids),
                    'total_relevant': len(ground_truth_docs)
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
        import traceback
        traceback.print_exc()
        return None

def print_results_summary(results: Dict[str, Any]):
    """Print results summary"""
    if not results:
        print("No test results")
        return
    
    print("\n" + "="*80)
    print("OpenFDA FAISS Vector Retrieval Test Results")
    print("="*80)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_time = 0
    
    for query_id, result in results.items():
        metrics = result['metrics']
        print(f"\nQuery {query_id}: {result['query']}")
        print(f"Ideal Answer: {result['ideal_answer'][0]}")
        print(f"Performance Metrics:")
        print(f"  - Precision: {metrics['precision']:.3f}")
        print(f"  - Recall: {metrics['recall']:.3f}")
        print(f"  - F1 Score: {metrics['f1_score']:.3f}")
        print(f"  - Query Time: {result['query_time']:.3f} seconds")
        print(f"  - Relevant Documents Retrieved: {metrics['relevant_retrieved']}/{metrics['total_relevant']}")
        
        print(f"\nRetrieval Results (Top 5):")
        for i, doc in enumerate(result['retrieved_docs'][:5], 1):
            print(f"  {i}. {doc['brand_name']} ({doc['generic_name']}) - Score: {doc['score']:.3f}")
            print(f"     Preview: {doc['text_preview']}")
        
        # Check if retrieved documents are in ground truth
        print(f"\nGround Truth Documents: {result['ground_truth_docs']}")
        retrieved_ids = [doc['doc_id'] for doc in result['retrieved_docs']]
        print(f"Retrieved Document IDs: {retrieved_ids}")
        
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
    print(f"Average Query Time: {avg_time:.3f} seconds")
    print(f"Total Queries: {num_queries}")

def save_detailed_results(results: Dict[str, Any], output_file: str):
    """Save detailed results"""
    if not results:
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed results saved to: {output_file}")

def manual_quality_assessment(results: Dict[str, Any]):
    """Manual retrieval quality assessment"""
    print("\n" + "="*80)
    print("Manual Retrieval Quality Assessment")
    print("="*80)
    
    for query_id, result in results.items():
        print(f"\nQuery {query_id}: {result['query']}")
        print(f"Ideal Answer: {result['ideal_answer'][0]}")
        
        print(f"\nRetrieval Results Quality Analysis:")
        retrieved_docs = result['retrieved_docs']
        
        for i, doc in enumerate(retrieved_docs[:5], 1):
            is_relevant = doc['doc_id'] in result['ground_truth_docs']
            relevance_marker = "✓ Relevant" if is_relevant else "✗ Not Relevant"
            
            print(f"  {i}. {relevance_marker} {doc['brand_name']} - Score: {doc['score']:.3f}")
            print(f"     Document ID: {doc['doc_id']}")
            print(f"     Preview: {doc['text_preview'][:200]}...")
            
            # Simple content relevance check
            query_words = set(result['query'].lower().split())
            doc_words = set(doc['text_preview'].lower().split())
            word_overlap = len(query_words.intersection(doc_words))
            print(f"     Keyword Overlap: {word_overlap} words")
        
        print(f"\nGround Truth Documents Retrieved:")
        for gt_doc in result['ground_truth_docs']:
            found = any(doc['doc_id'] == gt_doc for doc in retrieved_docs)
            print(f"  {gt_doc}: {'✓ Found' if found else '✗ Not Found'}")

def main():
    """Main function"""
    print("Starting OpenFDA FAISS vector retrieval test...")
    
    # Run tests
    results = test_faiss_retrieval()
    
    if results:
        # Print results
        print_results_summary(results)
        
        # Manual quality assessment
        manual_quality_assessment(results)
        
        # Save results
        output_file = "/Users/doupeihao/Desktop/CSE291a/Domain-RAG-System/results/openfda_faiss_test_results.json"
        save_detailed_results(results, output_file)
        
        print(f"\nTest completed! Results saved to {output_file}")
    else:
        print("Test failed, please check error messages")

if __name__ == "__main__":
    main()
