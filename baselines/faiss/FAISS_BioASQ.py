#!/usr/bin/env python3
"""
BioASQ FAISS Vector Retrieval Evaluation Script
Comprehensive evaluation with Precision, Recall, F1, Hit@K, and MRR metrics
"""

import json
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_pubmed_id(url: str) -> str:
    """Extract PubMed ID from URL"""
    # URL format: http://www.ncbi.nlm.nih.gov/pubmed/30242830
    return url.strip().split('/')[-1]

def load_bioasq_data(corpus_path: str, queries_path: str):
    """Load BioASQ corpus and query data"""
    logger.info("Loading BioASQ corpus data...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    logger.info("Loading BioASQ test queries...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    # Process corpus - combine title and abstract as text
    for doc in corpus_data:
        doc['text'] = f"{doc['title']} {doc['abstract']}"
        doc['pubmed_id'] = extract_pubmed_id(doc['link'])
    
    # Process queries - extract PubMed IDs from document URLs
    for question in queries_data['questions']:
        question['ground_truth_ids'] = [extract_pubmed_id(url) for url in question['documents']]
    
    logger.info(f"Corpus contains {len(corpus_data)} documents")
    logger.info(f"Test queries contain {len(queries_data['questions'])} questions")
    
    return corpus_data, queries_data

def create_pubmed_index(corpus_data: List[Dict]) -> Dict[str, int]:
    """Create PubMed ID to corpus index mapping"""
    pubmed_to_idx = {}
    for i, doc in enumerate(corpus_data):
        pubmed_to_idx[doc['pubmed_id']] = i
    return pubmed_to_idx

def calculate_metrics(retrieved_ids: List[str], ground_truth_ids: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """Calculate comprehensive retrieval metrics"""
    metrics = {}
    
    # Convert to sets for intersection calculations
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids)
    
    # Calculate intersection
    intersection = retrieved_set & ground_truth_set
    
    # Precision, Recall, F1
    metrics['precision'] = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
    metrics['recall'] = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    # Hit@K - whether any relevant document appears in top-K results
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        metrics[f'hit@{k}'] = 1.0 if (top_k & ground_truth_set) else 0.0
    
    # MRR (Mean Reciprocal Rank)
    # Find the rank of the first relevant document
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in ground_truth_set:
            reciprocal_rank = 1.0 / rank
            break
    metrics['mrr'] = reciprocal_rank
    
    # Additional statistics
    metrics['relevant_retrieved'] = len(intersection)
    metrics['total_retrieved'] = len(retrieved_ids)
    metrics['total_relevant'] = len(ground_truth_set)
    
    return metrics

def test_faiss_retrieval(corpus_path: str, queries_path: str, top_k: int = 10):
    """Test FAISS vector retrieval on BioASQ dataset"""
    logger.info("Starting BioASQ FAISS vector retrieval evaluation...")
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load data
        corpus_data, queries_data = load_bioasq_data(corpus_path, queries_path)
        pubmed_to_idx = create_pubmed_index(corpus_data)
        
        # Load embedding model
        logger.info("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate document embeddings
        logger.info("Generating document embeddings...")
        texts = [doc['text'] for doc in corpus_data]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {index.ntotal} vectors (dimension: {dimension})")
        
        # Test queries
        results = {}
        questions = queries_data['questions']
        
        logger.info(f"Processing {len(questions)} queries...")
        for idx, question in enumerate(questions, 1):
            query_id = question['id']
            query_text = question['body']
            query_type = question.get('type', 'unknown')
            ground_truth_ids = question['ground_truth_ids']
            
            if idx % 10 == 0:
                logger.info(f"Processing query {idx}/{len(questions)}")
            
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = model.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            # Get retrieval results
            retrieved_docs = []
            retrieved_ids = []
            for score, idx_val in zip(scores[0], indices[0]):
                if idx_val < len(corpus_data):
                    doc = corpus_data[idx_val]
                    retrieved_docs.append({
                        'pubmed_id': doc['pubmed_id'],
                        'score': float(score),
                        'title': doc['title'],
                        'link': doc['link']
                    })
                    retrieved_ids.append(doc['pubmed_id'])
            
            query_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_metrics(retrieved_ids, ground_truth_ids, k_values=[1, 3, 5, 10])
            
            results[query_id] = {
                'query': query_text,
                'type': query_type,
                'ideal_answer': question.get('ideal_answer', []),
                'ground_truth_ids': ground_truth_ids,
                'retrieved_ids': retrieved_ids,
                'retrieved_docs': retrieved_docs,
                'query_time': query_time,
                'metrics': metrics
            }
        
        logger.info("Retrieval evaluation completed!")
        return results
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please run: pip install faiss-cpu sentence-transformers")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def aggregate_metrics_by_type(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics by question type"""
    type_metrics = defaultdict(lambda: defaultdict(list))
    
    for query_id, result in results.items():
        q_type = result['type']
        metrics = result['metrics']
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                type_metrics[q_type][metric_name].append(metric_value)
    
    # Calculate averages
    aggregated = {}
    for q_type, metrics_dict in type_metrics.items():
        aggregated[q_type] = {}
        aggregated[q_type]['count'] = len(metrics_dict['precision'])
        
        for metric_name, values in metrics_dict.items():
            aggregated[q_type][metric_name] = np.mean(values)
    
    return aggregated

def print_detailed_results(results: Dict[str, Any]):
    """Print detailed results for each query"""
    print("\n" + "="*100)
    print("DETAILED QUERY RESULTS")
    print("="*100)
    
    for query_id, result in results.items():
        print(f"\n{'─'*100}")
        print(f"Query ID: {query_id}")
        print(f"Question Type: {result['type']}")
        print(f"Question: {result['query']}")
        print(f"\nIdeal Answer: {result['ideal_answer'][0] if result['ideal_answer'] else 'N/A'}")
        
        metrics = result['metrics']
        print(f"\nMetrics:")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall: {metrics['recall']:.4f}")
        print(f"  • F1 Score: {metrics['f1_score']:.4f}")
        print(f"  • MRR: {metrics['mrr']:.4f}")
        print(f"  • Hit@1: {metrics['hit@1']:.4f}")
        print(f"  • Hit@3: {metrics['hit@3']:.4f}")
        print(f"  • Hit@5: {metrics['hit@5']:.4f}")
        print(f"  • Hit@10: {metrics['hit@10']:.4f}")
        print(f"  • Query Time: {result['query_time']:.4f}s")
        print(f"  • Relevant Retrieved: {metrics['relevant_retrieved']}/{metrics['total_relevant']}")
        
        print(f"\nTop-5 Retrieved Documents:")
        for i, doc in enumerate(result['retrieved_docs'][:5], 1):
            is_relevant = doc['pubmed_id'] in result['ground_truth_ids']
            marker = "✓" if is_relevant else "✗"
            print(f"  {i}. [{marker}] Score: {doc['score']:.4f} | PMID: {doc['pubmed_id']}")
            print(f"     Title: {doc['title'][:80]}...")
        
        print(f"\nGround Truth Documents: {len(result['ground_truth_ids'])}")
        for gt_id in result['ground_truth_ids']:
            found = gt_id in result['retrieved_ids']
            if found:
                rank = result['retrieved_ids'].index(gt_id) + 1
                print(f"  • {gt_id}: Found at rank {rank}")
            else:
                print(f"  • {gt_id}: Not retrieved")

def print_summary_by_type(type_metrics: Dict[str, Dict[str, float]]):
    """Print summary metrics aggregated by question type"""
    print("\n" + "="*100)
    print("METRICS BY QUESTION TYPE")
    print("="*100)
    
    # Sort by question type
    sorted_types = sorted(type_metrics.items())
    
    for q_type, metrics in sorted_types:
        print(f"\n{'─'*100}")
        print(f"Question Type: {q_type.upper()}")
        print(f"   Number of Questions: {int(metrics['count'])}")
        print(f"\n   Performance Metrics:")
        print(f"   ├─ Precision:     {metrics['precision']:.4f}")
        print(f"   ├─ Recall:        {metrics['recall']:.4f}")
        print(f"   ├─ F1 Score:      {metrics['f1_score']:.4f}")
        print(f"   ├─ MRR:           {metrics['mrr']:.4f}")
        print(f"   ├─ Hit@1:         {metrics['hit@1']:.4f}")
        print(f"   ├─ Hit@3:         {metrics['hit@3']:.4f}")
        print(f"   ├─ Hit@5:         {metrics['hit@5']:.4f}")
        print(f"   └─ Hit@10:        {metrics['hit@10']:.4f}")

def print_overall_summary(results: Dict[str, Any]):
    """Print overall summary statistics"""
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    
    total_queries = len(results)
    all_metrics = defaultdict(list)
    
    for result in results.values():
        for metric_name, metric_value in result['metrics'].items():
            if isinstance(metric_value, (int, float)):
                all_metrics[metric_name].append(metric_value)
    
    print(f"\nOverall Performance (across {total_queries} queries):\n")
    print(f"   {'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"   {'-'*68}")
    
    metrics_order = ['precision', 'recall', 'f1_score', 'mrr', 'hit@1', 'hit@3', 'hit@5', 'hit@10']
    
    for metric_name in metrics_order:
        if metric_name in all_metrics:
            values = all_metrics[metric_name]
            print(f"   {metric_name:<20} {np.mean(values):<12.4f} {np.std(values):<12.4f} "
                  f"{np.min(values):<12.4f} {np.max(values):<12.4f}")
    
    # Query time statistics
    query_times = [r['query_time'] for r in results.values()]
    print(f"\nQuery Time Statistics:")
    print(f"   • Average: {np.mean(query_times):.4f}s")
    print(f"   • Total: {np.sum(query_times):.2f}s")
    print(f"   • Min: {np.min(query_times):.4f}s")
    print(f"   • Max: {np.max(query_times):.4f}s")

def save_results(results: Dict[str, Any], type_metrics: Dict[str, Dict], output_dir: str):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, 'bioasq_faiss_test_results.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to: {detailed_path}")
    
    # Save aggregated metrics by type
    aggregated_path = os.path.join(output_dir, 'bioasq_faiss_metrics_by_type.json')
    with open(aggregated_path, 'w', encoding='utf-8') as f:
        json.dump(type_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics by type saved to: {aggregated_path}")
    
    # Save summary CSV
    csv_path = os.path.join(output_dir, 'bioasq_faiss_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("question_type,count,precision,recall,f1_score,mrr,hit@1,hit@3,hit@5,hit@10\n")
        # Data
        for q_type, metrics in sorted(type_metrics.items()):
            f.write(f"{q_type},{int(metrics['count'])},{metrics['precision']:.4f},"
                   f"{metrics['recall']:.4f},{metrics['f1_score']:.4f},{metrics['mrr']:.4f},"
                   f"{metrics['hit@1']:.4f},{metrics['hit@3']:.4f},"
                   f"{metrics['hit@5']:.4f},{metrics['hit@10']:.4f}\n")
    logger.info(f"Summary CSV saved to: {csv_path}")

def main():
    """Main function"""
    # Configuration
    CORPUS_PATH = "./data/BioASQ/corpus_subset.json"
    QUERIES_PATH = "./data/BioASQ/bioasq_subset.json"
    OUTPUT_DIR = "./results"
    TOP_K = 10
    
    print("="*100)
    print("BioASQ FAISS Vector Retrieval Evaluation")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  • Corpus: {CORPUS_PATH}")
    print(f"  • Queries: {QUERIES_PATH}")
    print(f"  • Output: {OUTPUT_DIR}")
    print(f"  • Top-K: {TOP_K}")
    print(f"  • Embedding Model: all-MiniLM-L6-v2")
    print(f"  • Index Type: FAISS IndexFlatIP")
    
    # Run evaluation
    results = test_faiss_retrieval(CORPUS_PATH, QUERIES_PATH, top_k=TOP_K)
    
    if results:
        # Aggregate metrics by question type
        type_metrics = aggregate_metrics_by_type(results)
        
        # Print all results
        print_overall_summary(results)
        print_summary_by_type(type_metrics)
        print_detailed_results(results)
        
        # Save results
        save_results(results, type_metrics, OUTPUT_DIR)
        
        print(f"\n{'='*100}")
        print("Evaluation completed successfully!")
        print(f"{'='*100}\n")
    else:
        print("\nEvaluation failed. Please check error messages above.\n")

if __name__ == "__main__":
    main()