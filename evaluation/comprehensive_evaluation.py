#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script - Evaluate the entire RAG system
Evaluates three datasets: OpenFDA, Kaggle Drug Data, BioASQ
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import statistics

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

def load_json(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_retrieval_metrics(ground_truth: List[str], retrieved: List[str]) -> Dict[str, float]:
    """Calculate retrieval metrics"""
    gt_set = set(ground_truth)
    k = len(retrieved)
    
    if k == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hit@k": 0.0,
            "mrr": 0.0
        }
    
    # True positives
    tp = sum(1 for r in retrieved if r in gt_set)
    
    # Precision
    precision = tp / k if k > 0 else 0.0
    
    # Recall
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Hit@K
    hit_at_k = 1.0 if tp > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in gt_set:
            mrr = 1.0 / i
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit@k": hit_at_k,
        "mrr": mrr
    }

def calculate_baseline_metrics(baseline_data: Dict, dataset_type: str) -> Dict[str, float]:
    """Calculate overall metrics from baseline results"""
    all_metrics = []
    
    if dataset_type == "openfda":
        # OpenFDA format: {query_id: {metrics: {...}, retrieved_docs: [...], ground_truth_docs: [...]}}
        for qid, result in baseline_data.items():
            if isinstance(result, dict):
                # Always calculate metrics from retrieved_docs and ground_truth_docs to ensure completeness
                gt_docs = result.get("ground_truth_docs", [])
                retrieved_docs = [doc.get("doc_id", "") for doc in result.get("retrieved_docs", [])]
                if gt_docs and retrieved_docs:
                    metrics = calculate_retrieval_metrics(gt_docs, retrieved_docs)
                    # Add query time if available
                    if "query_time" in result:
                        metrics["query_time"] = result["query_time"]
                    all_metrics.append(metrics)
                elif "metrics" in result:
                    # Fallback to metrics if docs not available
                    metrics = result["metrics"].copy()
                    # Normalize field names (f1_score -> f1)
                    if "f1_score" in metrics and "f1" not in metrics:
                        metrics["f1"] = metrics.pop("f1_score")
                    # Ensure hit@k and mrr are present, calculate if missing
                    if "hit@k" not in metrics or "mrr" not in metrics:
                        gt_docs = result.get("ground_truth_docs", [])
                        retrieved_docs = [doc.get("doc_id", "") for doc in result.get("retrieved_docs", [])]
                        if gt_docs and retrieved_docs:
                            full_metrics = calculate_retrieval_metrics(gt_docs, retrieved_docs)
                            metrics["hit@k"] = full_metrics.get("hit@k", 0)
                            metrics["mrr"] = full_metrics.get("mrr", 0)
                    all_metrics.append(metrics)
    
    elif dataset_type == "kaggle":
        # Kaggle format: {per_query: [{metrics: {...}, ...}], overall: {...}}
        if "overall" in baseline_data:
            return baseline_data["overall"]
        elif "per_query" in baseline_data:
            for query_result in baseline_data["per_query"]:
                if "metrics" in query_result:
                    all_metrics.append(query_result["metrics"])
    
    if all_metrics:
        query_times = [m.get("query_time", m.get("query_time_sec", 0)) for m in all_metrics if "query_time" in m or "query_time_sec" in m]
        avg_query_time = statistics.mean(query_times) if query_times else 0.0
        
        return {
            "precision": statistics.mean([m.get("precision", 0) for m in all_metrics]),
            "recall": statistics.mean([m.get("recall", 0) for m in all_metrics]),
            "f1": statistics.mean([m.get("f1", 0) for m in all_metrics]),
            "hit@k": statistics.mean([m.get("hit@k", 0) for m in all_metrics]),
            "mrr": statistics.mean([m.get("mrr", 0) for m in all_metrics]),
            "avg_query_time": avg_query_time,
            "total_queries": len(all_metrics)
        }
    
    return None

def compare_with_baseline(current: Dict, baseline: Dict, dataset_name: str) -> Dict:
    """Compare current results with baseline results"""
    comparison = {
        "dataset": dataset_name,
        "baseline": {},
        "current": {},
        "improvement": {},
        "relative_improvement": {}
    }
    
    metrics_to_compare = ["precision", "recall", "f1", "hit@k", "mrr"]
    
    for metric in metrics_to_compare:
        baseline_val = baseline.get(metric, 0)
        current_val = current.get(metric, 0)
        improvement = current_val - baseline_val
        relative_improvement = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        
        comparison["baseline"][metric] = baseline_val
        comparison["current"][metric] = current_val
        comparison["improvement"][metric] = improvement
        comparison["relative_improvement"][metric] = relative_improvement
    
    # Query time comparison
    baseline_time = baseline.get("avg_query_time", 0)
    current_time = current.get("avg_query_time", 0)
    if baseline_time > 0 and current_time > 0:
        time_change = current_time - baseline_time
        time_change_pct = (time_change / baseline_time * 100)
        comparison["baseline"]["avg_query_time"] = baseline_time
        comparison["current"]["avg_query_time"] = current_time
        comparison["improvement"]["avg_query_time"] = time_change
        comparison["relative_improvement"]["avg_query_time"] = time_change_pct
    
    return comparison

def load_index_metadata(metadata_path: str):
    """Load index metadata from file."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_hybrid_indexer_local(
    indices_dir: str,
    metadata: dict,
    drug_mapping_path: Optional[str] = None
):
    """Load hybrid indexer - local copy of function from scripts/query.py"""
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / 'src'))
        
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
            with open(drug_mapping_path, 'r', encoding='utf-8') as f:
                normalizer.drug_mapping = json.load(f)
        
        bm25_params = metadata.get('bm25_params', {'k1': 1.5, 'b': 0.75})
        bm25_indexer = BM25Indexer(
            k1=bm25_params.get('k1', 1.5),
            b=bm25_params.get('b', 0.75),
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
    except Exception as e:
        print(f"Error loading hybrid indexer: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_rag_evaluation_openfda(
    indices_dir: str = "data/indices",
    drug_mapping_path: str = "data/processed_for_our_rag/drug_mapping.json",
    top_k: int = 10,
    fusion_method: str = "rrf",
    reranker_kind: str = "simple",
    rerank_top_n: int = 50,
    output_path: str = "results/openfda_rag_test_results.json"
) -> Dict[str, Any]:
    """Run RAG system evaluation on OpenFDA dataset"""
    print("Running RAG system evaluation for OpenFDA...")
    
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / 'src'))
        
        from preprocessing import MedicalTermNormalizer, QueryPreprocessor
        from reranker import build_reranker
    except Exception as e:
        print(f"Error importing RAG system modules: {e}")
        print("Please ensure the RAG system is properly set up.")
        import traceback
        traceback.print_exc()
        return None
    
    # Load queries
    queries_path = "data/OpenFDA Drug data/openfda_test_queries.json"
    queries = load_json(queries_path)
    
    # Load index
    metadata_path = Path(indices_dir) / 'index_metadata.json'
    if not metadata_path.exists():
        print(f"Error: Index metadata not found at {metadata_path}")
        print("Please run build_index.py first to build the index.")
        return None
    
    print(f"Loading index from {indices_dir}...")
    metadata = load_index_metadata(str(metadata_path))
    hybrid_indexer = load_hybrid_indexer_local(
        indices_dir=indices_dir,
        metadata=metadata,
        drug_mapping_path=drug_mapping_path if Path(drug_mapping_path).exists() else None
    )
    
    # Initialize query preprocessor
    normalizer = MedicalTermNormalizer()
    if Path(drug_mapping_path).exists():
        with open(drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    query_preprocessor = QueryPreprocessor(medical_normalizer=normalizer)
    
    # Build reranker if needed
    reranker = None
    if reranker_kind != 'none':
        reranker = build_reranker(
            kind=reranker_kind,
            top_n=rerank_top_n,
            cross_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            embedder=hybrid_indexer.vector_indexer.embedder
        )
    
    # Evaluate each query
    results = {}
    questions = queries.get("questions", [])
    
    for q in questions:
        qid = q["id"]
        query_text = q.get("body", "")
        gt_docs = q.get("documents", [])
        
        print(f"Processing query {qid}: {query_text[:60]}...")
        
        # Preprocess query
        preprocessed = query_preprocessor.preprocess(query_text)
        search_query = preprocessed.get('cleaned') or query_text
        if preprocessed.get('normalized') and len(preprocessed['normalized']) < len(search_query) * 2:
            search_query = preprocessed['normalized']
        
        # Retrieve
        start_time = time.time()
        fetch_k = max(top_k, rerank_top_n if reranker_kind != 'none' else top_k)
        retrieved_results = hybrid_indexer.search(
            query=search_query,
            top_k=fetch_k,
            fusion_method=fusion_method,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        # Rerank if needed
        if reranker and retrieved_results:
            retrieved_results = reranker.rerank(query_text, retrieved_results, top_k=top_k)
        else:
            retrieved_results = retrieved_results[:top_k]
        
        query_time = time.time() - start_time
        
        # Extract document IDs
        # For OpenFDA, use doc_id field (OpenFDA document IDs)
        retrieved_docs = []
        for r in retrieved_results:
            # Try doc_id first (for OpenFDA), then chunk_id (for Kaggle), then source as fallback
            doc_id = r.get('doc_id', '') or r.get('chunk_id', '') or r.get('source', '')
            # Remove source prefix if present (e.g., "openfda_" or "kaggle_")
            if doc_id and '_' in doc_id:
                # Check if it's a prefixed ID (e.g., "openfda_xxx" or "kaggle_xxx")
                parts = doc_id.split('_', 1)
                if len(parts) == 2 and parts[0] in ['openfda', 'kaggle', 'pubmed']:
                    doc_id = parts[1]  # Remove prefix
            if doc_id:
                retrieved_docs.append({
                    'doc_id': doc_id,
                    'score': r.get('score', 0.0),
                    'text_preview': r.get('text', '')[:300] + "...",
                    'metadata': r.get('metadata', {})
                })
        
        retrieved_ids = [doc['doc_id'] for doc in retrieved_docs]
        
        # Calculate metrics
        metrics = calculate_retrieval_metrics(gt_docs, retrieved_ids)
        
        results[qid] = {
            'query': query_text,
            'ideal_answer': q.get('ideal_answer', []),
            'ground_truth_docs': gt_docs,
            'query_time': query_time,
            'retrieved_docs': retrieved_docs,
            'metrics': {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'hit@k': metrics['hit@k'],
                'mrr': metrics['mrr'],
                'relevant_retrieved': len(set(gt_docs) & set(retrieved_ids)),
                'total_retrieved': len(retrieved_ids),
                'total_relevant': len(gt_docs)
            }
        }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"RAG evaluation results saved to {output_path}")
    return results

def run_rag_evaluation_kaggle(
    indices_dir: str = "data/indices",
    drug_mapping_path: str = "data/processed_for_our_rag/drug_mapping.json",
    top_k: int = 5,
    fusion_method: str = "rrf",
    reranker_kind: str = "simple",
    rerank_top_n: int = 50,
    output_path: str = "results/kaggle_rag_test_results.json"
) -> Dict[str, Any]:
    """Run RAG system evaluation on Kaggle dataset"""
    print("Running RAG system evaluation for Kaggle...")
    
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / 'src'))
        
        from preprocessing import MedicalTermNormalizer, QueryPreprocessor
        from reranker import build_reranker
    except Exception as e:
        print(f"Error importing RAG system modules: {e}")
        print("Please ensure the RAG system is properly set up.")
        import traceback
        traceback.print_exc()
        return None
    
    # Load queries
    queries_path = "data/kaggle_drug_data/processed/test_queries_formatted.json"
    queries = load_json(queries_path)
    
    # Load index
    metadata_path = Path(indices_dir) / 'index_metadata.json'
    if not metadata_path.exists():
        print(f"Error: Index metadata not found at {metadata_path}")
        print("Please run build_index.py first to build the index.")
        return None
    
    print(f"Loading index from {indices_dir}...")
    metadata = load_index_metadata(str(metadata_path))
    hybrid_indexer = load_hybrid_indexer_local(
        indices_dir=indices_dir,
        metadata=metadata,
        drug_mapping_path=drug_mapping_path if Path(drug_mapping_path).exists() else None
    )
    
    # Initialize query preprocessor
    normalizer = MedicalTermNormalizer()
    if Path(drug_mapping_path).exists():
        with open(drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    query_preprocessor = QueryPreprocessor(medical_normalizer=normalizer)
    
    # Build reranker if needed
    reranker = None
    if reranker_kind != 'none':
        reranker = build_reranker(
            kind=reranker_kind,
            top_n=rerank_top_n,
            cross_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            embedder=hybrid_indexer.vector_indexer.embedder
        )
    
    # Evaluate each query
    per_query_results = []
    questions = queries.get("questions", [])
    
    for q in questions:
        qid = q["id"]
        query_text = q.get("body", "")
        gt_docs = q.get("documents", [])
        qtype = q.get("type", "unknown")
        
        print(f"Processing query {qid} ({qtype}): {query_text[:60]}...")
        
        # Preprocess query
        preprocessed = query_preprocessor.preprocess(query_text)
        search_query = preprocessed.get('cleaned') or query_text
        if preprocessed.get('normalized') and len(preprocessed['normalized']) < len(search_query) * 2:
            search_query = preprocessed['normalized']
        
        # Retrieve
        start_time = time.time()
        fetch_k = max(top_k, rerank_top_n if reranker_kind != 'none' else top_k)
        retrieved_results = hybrid_indexer.search(
            query=search_query,
            top_k=fetch_k,
            fusion_method=fusion_method,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        # Rerank if needed
        if reranker and retrieved_results:
            retrieved_results = reranker.rerank(query_text, retrieved_results, top_k=top_k)
        else:
            retrieved_results = retrieved_results[:top_k]
        
        query_time = time.time() - start_time
        
        # Extract chunk IDs
        # For Kaggle, use chunk_id field
        retrieved_ids = []
        topk = []
        for rank, r in enumerate(retrieved_results, start=1):
            # Try chunk_id first (for Kaggle), then doc_id, then source as fallback
            chunk_id = r.get('chunk_id', '') or r.get('doc_id', '') or r.get('source', '')
            if chunk_id:
                retrieved_ids.append(chunk_id)
                topk.append({
                    'rank': rank,
                    'score': float(r.get('score', 0.0)),
                    'chunk_id': chunk_id,
                    'preview': r.get('text', '')[:240],
                    'metadata': r.get('metadata', {})
                })
        
        # Calculate metrics
        metrics = calculate_retrieval_metrics(gt_docs, retrieved_ids)
        
        per_query_results.append({
            'id': qid,
            'type': qtype,
            'query': query_text,
            'k': top_k,
            'query_time_sec': query_time,
            'ground_truth': gt_docs,
            'retrieved_ids': retrieved_ids,
            'metrics': {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'hit@k': metrics['hit@k'],
                'mrr': metrics['mrr']
            },
            'topk': topk,
            'ideal_answer': q.get('ideal_answer', '')
        })
    
    # Calculate overall metrics
    if per_query_results:
        overall = {
            'precision': statistics.mean([r['metrics']['precision'] for r in per_query_results]),
            'recall': statistics.mean([r['metrics']['recall'] for r in per_query_results]),
            'f1': statistics.mean([r['metrics']['f1'] for r in per_query_results]),
            'hit@k': statistics.mean([r['metrics']['hit@k'] for r in per_query_results]),
            'mrr': statistics.mean([r['metrics']['mrr'] for r in per_query_results])
        }
    else:
        overall = {}
    
    results = {
        'model': 'RAG System (Hybrid + Reranker)',
        'k': top_k,
        'per_query': per_query_results,
        'overall': overall
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"RAG evaluation results saved to {output_path}")
    return results

def run_rag_evaluation_bioasq(
    indices_dir: str = "data/indices",
    drug_mapping_path: str = "data/processed_for_our_rag/drug_mapping.json",
    top_k: int = 10,
    fusion_method: str = "rrf",
    reranker_kind: str = "simple",
    rerank_top_n: int = 50,
    output_path: str = "results/bioasq_rag_test_results.json"
) -> Dict[str, Any]:
    """Run RAG system evaluation on BioASQ dataset"""
    print("Running RAG system evaluation for BioASQ...")
    
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / 'src'))
        
        from preprocessing import MedicalTermNormalizer, QueryPreprocessor
        from reranker import build_reranker
    except Exception as e:
        print(f"Error importing RAG system modules: {e}")
        print("Please ensure the RAG system is properly set up.")
        import traceback
        traceback.print_exc()
        return None
    
    # Load queries
    queries_path = "data/BioASQ/bioasq_subset.json"
    queries = load_json(queries_path)
    
    # Load index
    metadata_path = Path(indices_dir) / 'index_metadata.json'
    if not metadata_path.exists():
        print(f"Error: Index metadata not found at {metadata_path}")
        print("Please run build_index.py first to build the index.")
        return None
    
    print(f"Loading index from {indices_dir}...")
    metadata = load_index_metadata(str(metadata_path))
    hybrid_indexer = load_hybrid_indexer_local(
        indices_dir=indices_dir,
        metadata=metadata,
        drug_mapping_path=drug_mapping_path if Path(drug_mapping_path).exists() else None
    )
    
    # Initialize query preprocessor
    normalizer = MedicalTermNormalizer()
    if Path(drug_mapping_path).exists():
        with open(drug_mapping_path, 'r') as f:
            normalizer.drug_mapping = json.load(f)
    query_preprocessor = QueryPreprocessor(medical_normalizer=normalizer)
    
    # Build reranker if needed
    reranker = None
    if reranker_kind != 'none':
        reranker = build_reranker(
            kind=reranker_kind,
            top_n=rerank_top_n,
            cross_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            embedder=hybrid_indexer.vector_indexer.embedder
        )
    
    # Extract PubMed ID from URL
    def extract_pubmed_id(url: str) -> str:
        if isinstance(url, str) and "pubmed" in url:
            return url.split("/")[-1]
        return url
    
    # Evaluate each query
    per_query_results = []
    questions = queries.get("questions", [])
    
    for q in questions:
        qid = q.get("id", "")
        query_text = q.get("body", "")
        # Ground truth documents - extract PubMed IDs from URLs
        gt_docs = [extract_pubmed_id(url) for url in q.get("documents", [])]
        qtype = q.get("type", "unknown")
        
        print(f"Processing query {qid} ({qtype}): {query_text[:60]}...")
        
        # Preprocess query
        preprocessed = query_preprocessor.preprocess(query_text)
        search_query = preprocessed.get('cleaned') or query_text
        if preprocessed.get('normalized') and len(preprocessed['normalized']) < len(search_query) * 2:
            search_query = preprocessed['normalized']
        
        # Retrieve
        start_time = time.time()
        fetch_k = max(top_k, rerank_top_n if reranker_kind != 'none' else top_k)
        retrieved_results = hybrid_indexer.search(
            query=search_query,
            top_k=fetch_k,
            fusion_method=fusion_method,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        # Rerank if needed
        if reranker and retrieved_results:
            retrieved_results = reranker.rerank(query_text, retrieved_results, top_k=top_k)
        else:
            retrieved_results = retrieved_results[:top_k]
        
        query_time = time.time() - start_time
        
        # Extract document IDs
        # For BioASQ, we need to extract PubMed IDs from doc_id
        # The doc_id in retrieval results should be the PubMed ID (numeric string)
        retrieved_ids = []
        topk = []
        for rank, r in enumerate(retrieved_results, start=1):
            # Get doc_id (should be PubMed ID for BioASQ documents)
            doc_id = r.get('doc_id', '')
            
            # If doc_id is empty, try chunk_id or source
            if not doc_id:
                doc_id = r.get('chunk_id', '') or r.get('source', '')
            
            # Remove source prefix if present (e.g., "pubmed_30242830" -> "30242830")
            if doc_id and '_' in doc_id:
                parts = doc_id.split('_', 1)
                if len(parts) == 2 and parts[0] in ['openfda', 'kaggle', 'pubmed']:
                    doc_id = parts[1]  # Remove prefix
            
            # For BioASQ, doc_id should be a PubMed ID (numeric string)
            # If it's not numeric, try to extract from metadata
            if doc_id:
                if doc_id.isdigit():
                    # It's a PubMed ID
                    retrieved_ids.append(doc_id)
                else:
                    # Try to extract PubMed ID from metadata
                    metadata = r.get('metadata', {})
                    # Check metadata for doc_id or pubmed_id
                    meta_doc_id = metadata.get('doc_id', '') or metadata.get('pubmed_id', '')
                    if meta_doc_id and str(meta_doc_id).isdigit():
                        retrieved_ids.append(str(meta_doc_id))
                    elif doc_id:
                        # Use as is (might be a chunk ID, but we'll try to match)
                        retrieved_ids.append(doc_id)
                
                topk.append({
                    'rank': rank,
                    'score': float(r.get('score', 0.0)),
                    'doc_id': doc_id,
                    'preview': r.get('text', '')[:240],
                    'metadata': r.get('metadata', {})
                })
        
        # Calculate metrics
        metrics = calculate_retrieval_metrics(gt_docs, retrieved_ids)
        
        per_query_results.append({
            'id': qid,
            'type': qtype,
            'query': query_text,
            'k': top_k,
            'query_time_sec': query_time,
            'ground_truth': gt_docs,
            'retrieved_ids': retrieved_ids,
            'metrics': {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'hit@k': metrics['hit@k'],
                'mrr': metrics['mrr']
            },
            'topk': topk,
            'ideal_answer': q.get('ideal_answer', [])
        })
    
    # Calculate overall metrics
    if per_query_results:
        overall = {
            'precision': statistics.mean([r['metrics']['precision'] for r in per_query_results]),
            'recall': statistics.mean([r['metrics']['recall'] for r in per_query_results]),
            'f1': statistics.mean([r['metrics']['f1'] for r in per_query_results]),
            'hit@k': statistics.mean([r['metrics']['hit@k'] for r in per_query_results]),
            'mrr': statistics.mean([r['metrics']['mrr'] for r in per_query_results])
        }
    else:
        overall = {}
    
    results = {
        'model': 'RAG System (Hybrid + Reranker)',
        'k': top_k,
        'per_query': per_query_results,
        'overall': overall
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"RAG evaluation results saved to {output_path}")
    return results

def evaluate_openfda(run_rag: bool = True):
    """Evaluate OpenFDA dataset using RAG system results"""
    print("=" * 80)
    print("Evaluating OpenFDA Drug Data")
    print("=" * 80)
    
    queries_path = "data/OpenFDA Drug data/openfda_test_queries.json"
    
    # Try to load RAG system results first
    rag_results_path = "results/openfda_rag_test_results.json"
    results_path = rag_results_path
    
    if run_rag and not Path(rag_results_path).exists():
        print("RAG system results not found. Running RAG system evaluation...")
        rag_results = run_rag_evaluation_openfda()
        if rag_results is None:
            print("Warning: Failed to run RAG evaluation. Using baseline results if available.")
            results_path = "results/openfda_faiss_test_results.json"
        else:
            results_path = rag_results_path
    elif not Path(rag_results_path).exists():
        print("RAG system results not found. Using baseline results.")
        results_path = "results/openfda_faiss_test_results.json"
    else:
        print(f"Loading RAG system results from {rag_results_path}")
    
    if not Path(results_path).exists():
        print(f"Error: Results file not found at {results_path}")
        return {"dataset": "OpenFDA Drug Data", "overall": {}, "per_query": []}
    
    queries = load_json(queries_path)
    results = load_json(results_path)
    
    all_metrics = []
    per_query_metrics = []
    
    for q in queries["questions"]:
        qid = q["id"]
        gt_docs = q.get("documents", [])
        
        if qid not in results:
            continue
        
        result = results[qid]
        retrieved_docs = [doc["doc_id"] for doc in result.get("retrieved_docs", [])]
        
        metrics = calculate_retrieval_metrics(gt_docs, retrieved_docs)
        metrics["query_id"] = qid
        metrics["query"] = q.get("body", "")
        metrics["query_time"] = result.get("query_time", 0.0)
        metrics["total_relevant"] = len(gt_docs)
        metrics["total_retrieved"] = len(retrieved_docs)
        metrics["relevant_retrieved"] = len(set(gt_docs) & set(retrieved_docs))
        
        all_metrics.append(metrics)
        per_query_metrics.append({
            "id": qid,
            "query": q.get("body", ""),
            "type": q.get("type", "unknown"),
            **metrics
        })
    
    # Calculate overall metrics
    if all_metrics:
        overall = {
            "precision": statistics.mean([m["precision"] for m in all_metrics]),
            "recall": statistics.mean([m["recall"] for m in all_metrics]),
            "f1": statistics.mean([m["f1"] for m in all_metrics]),
            "hit@k": statistics.mean([m["hit@k"] for m in all_metrics]),
            "mrr": statistics.mean([m["mrr"] for m in all_metrics]),
            "avg_query_time": statistics.mean([m["query_time"] for m in all_metrics]),
            "total_queries": len(all_metrics)
        }
    else:
        overall = {}
    
    return {
        "dataset": "OpenFDA Drug Data",
        "overall": overall,
        "per_query": per_query_metrics
    }

def evaluate_kaggle(run_rag: bool = True):
    """Evaluate Kaggle Drug Data dataset using RAG system results"""
    print("=" * 80)
    print("Evaluating Kaggle Drug Data")
    print("=" * 80)
    
    queries_path = "data/kaggle_drug_data/processed/test_queries_formatted.json"
    
    # Try to load RAG system results first
    rag_results_path = "results/kaggle_rag_test_results.json"
    results_path = rag_results_path
    
    if run_rag and not Path(rag_results_path).exists():
        print("RAG system results not found. Running RAG system evaluation...")
        rag_results = run_rag_evaluation_kaggle()
        if rag_results is None:
            print("Warning: Failed to run RAG evaluation. Using baseline results if available.")
            results_path = "data/kaggle_drug_data/processed/qdrant_test_results.json"
        else:
            results_path = rag_results_path
    elif not Path(rag_results_path).exists():
        print("RAG system results not found. Using baseline results.")
        results_path = "data/kaggle_drug_data/processed/qdrant_test_results.json"
    else:
        print(f"Loading RAG system results from {rag_results_path}")
    
    if not Path(results_path).exists():
        print(f"Error: Results file not found at {results_path}")
        return {"dataset": "Kaggle Drug Data", "overall": {}, "per_type": {}, "per_query": []}
    
    queries = load_json(queries_path)
    results = load_json(results_path)
    
    # Extract per_query data from results
    per_query_results = results.get("per_query", [])
    
    # Create mapping from query ID to results
    result_map = {r["id"]: r for r in per_query_results}
    
    all_metrics = []
    per_query_metrics = []
    type_metrics = defaultdict(list)
    
    for q in queries["questions"]:
        qid = q["id"]
        gt_docs = q.get("documents", [])
        qtype = q.get("type", "unknown")
        
        if qid not in result_map:
            continue
        
        result = result_map[qid]
        retrieved_docs = result.get("retrieved_ids", [])
        
        # Use existing metrics from results, or calculate if not available
        if "metrics" in result:
            metrics = result["metrics"].copy()
        else:
            metrics = calculate_retrieval_metrics(gt_docs, retrieved_docs)
        
        metrics["query_id"] = qid
        metrics["query"] = q.get("body", "")
        metrics["query_time"] = result.get("query_time_sec", 0.0)
        metrics["total_relevant"] = len(gt_docs)
        metrics["total_retrieved"] = len(retrieved_docs)
        metrics["relevant_retrieved"] = len(set(gt_docs) & set(retrieved_docs))
        
        all_metrics.append(metrics)
        type_metrics[qtype].append(metrics)
        
        per_query_metrics.append({
            "id": qid,
            "query": q.get("body", ""),
            "type": qtype,
            **metrics
        })
    
    # Calculate overall metrics
    if all_metrics:
        overall = {
            "precision": statistics.mean([m["precision"] for m in all_metrics]),
            "recall": statistics.mean([m["recall"] for m in all_metrics]),
            "f1": statistics.mean([m["f1"] for m in all_metrics]),
            "hit@k": statistics.mean([m["hit@k"] for m in all_metrics]),
            "mrr": statistics.mean([m["mrr"] for m in all_metrics]),
            "avg_query_time": statistics.mean([m["query_time"] for m in all_metrics]),
            "total_queries": len(all_metrics)
        }
    else:
        overall = results.get("overall", {})
    
    # Calculate metrics by type
    type_overall = {}
    for qtype, metrics_list in type_metrics.items():
        if metrics_list:
            type_overall[qtype] = {
                "precision": statistics.mean([m["precision"] for m in metrics_list]),
                "recall": statistics.mean([m["recall"] for m in metrics_list]),
                "f1": statistics.mean([m["f1"] for m in metrics_list]),
                "hit@k": statistics.mean([m["hit@k"] for m in metrics_list]),
                "mrr": statistics.mean([m["mrr"] for m in metrics_list]),
                "count": len(metrics_list)
            }
    
    return {
        "dataset": "Kaggle Drug Data",
        "overall": overall,
        "per_type": type_overall,
        "per_query": per_query_metrics
    }

def evaluate_bioasq(run_rag: bool = True):
    """Evaluate BioASQ dataset using RAG system results"""
    print("=" * 80)
    print("Evaluating BioASQ Dataset")
    print("=" * 80)
    
    corpus_path = "data/BioASQ/corpus_subset.json"
    queries_path = "data/BioASQ/bioasq_subset.json"
    
    # Try to load RAG system results first
    rag_results_path = "results/bioasq_rag_test_results.json"
    results_path = rag_results_path
    
    if run_rag and not Path(rag_results_path).exists():
        print("RAG system results not found. Running RAG system evaluation...")
        rag_results = run_rag_evaluation_bioasq()
        if rag_results is None:
            print("Warning: Failed to run RAG evaluation. Checking for other result files...")
            # Try to find other retrieval results file
            possible_result_paths = [
                "results/bioasq_test_results.json",
                "results/bioasq_retrieval_results.json",
                "data/BioASQ/bioasq_test_results.json"
            ]
            results_path = None
            for path in possible_result_paths:
                if Path(path).exists():
                    results_path = path
                    break
        else:
            results_path = rag_results_path
    elif not Path(rag_results_path).exists():
        print("RAG system results not found. Checking for other result files...")
        # Try to find other retrieval results file
        possible_result_paths = [
            "results/bioasq_test_results.json",
            "results/bioasq_retrieval_results.json",
            "data/BioASQ/bioasq_test_results.json"
        ]
        results_path = None
        for path in possible_result_paths:
            if Path(path).exists():
                results_path = path
                break
    else:
        print(f"Loading RAG system results from {rag_results_path}")
        results_path = rag_results_path
    
    corpus = load_json(corpus_path)
    queries_data = load_json(queries_path)
    
    # Create mapping from document ID to index
    doc_id_map = {}
    for idx, doc in enumerate(corpus):
        doc_id = doc.get("id", "")
        if doc_id:
            doc_id_map[doc_id] = idx
    
    # Extract PubMed ID (from URL)
    def extract_pubmed_id(url: str) -> str:
        if isinstance(url, str) and "pubmed" in url:
            return url.split("/")[-1]
        return url
    
    questions = queries_data.get("questions", [])
    
    print(f"Loaded {len(questions)} questions")
    print(f"Corpus contains {len(corpus)} documents")
    print(f"Document ID mapping contains {len(doc_id_map)} entries")
    
    # Statistics
    type_counts = defaultdict(int)
    for q in questions:
        qtype = q.get("type", "unknown")
        type_counts[qtype] += 1
    
    print(f"\nQuestion type distribution:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")
    
    # If retrieval results exist, perform full evaluation
    if results_path:
        print(f"\nFound retrieval results at: {results_path}")
        results = load_json(results_path)
        
        # Handle different result formats
        if isinstance(results, dict):
            # Check if it's per_query format (like Kaggle)
            if "per_query" in results:
                per_query_results = results.get("per_query", [])
                result_map = {r.get("id", ""): r for r in per_query_results}
            # Check if it's query_id key format (like OpenFDA)
            else:
                result_map = results
        else:
            result_map = {}
        
        all_metrics = []
        per_query_metrics = []
        type_metrics = defaultdict(list)
        
        for q in questions:
            qid = q.get("id", "")
            # Ground truth documents - extract PubMed IDs from URLs
            gt_docs = [extract_pubmed_id(url) for url in q.get("documents", [])]
            qtype = q.get("type", "unknown")
            
            if qid not in result_map:
                continue
            
            result = result_map[qid]
            
            # Handle different result formats
            if "retrieved_ids" in result:
                retrieved_docs = result.get("retrieved_ids", [])
            elif "retrieved_docs" in result:
                retrieved_docs = [doc.get("doc_id", "") for doc in result.get("retrieved_docs", [])]
            elif "retrieved" in result:
                retrieved_docs = [h.get("chunk_id", "") for h in result.get("retrieved", [])]
            else:
                continue
            
            # Calculate metrics
            if "metrics" in result:
                metrics = result["metrics"].copy()
            else:
                metrics = calculate_retrieval_metrics(gt_docs, retrieved_docs)
            
            metrics["query_id"] = qid
            metrics["query"] = q.get("body", "")
            metrics["query_time"] = result.get("query_time", result.get("query_time_sec", 0.0))
            metrics["total_relevant"] = len(gt_docs)
            metrics["total_retrieved"] = len(retrieved_docs)
            metrics["relevant_retrieved"] = len(set(gt_docs) & set(retrieved_docs))
            metrics["has_ideal_answer"] = bool(q.get("ideal_answer"))
            metrics["ideal_answer_count"] = len(q.get("ideal_answer", [])) if isinstance(q.get("ideal_answer"), list) else (1 if q.get("ideal_answer") else 0)
            
            all_metrics.append(metrics)
            type_metrics[qtype].append(metrics)
            
            per_query_metrics.append({
                "id": qid,
                "query": q.get("body", ""),
                "type": qtype,
                **metrics
            })
        
        # Calculate overall metrics
        if all_metrics:
            overall = {
                "precision": statistics.mean([m["precision"] for m in all_metrics]),
                "recall": statistics.mean([m["recall"] for m in all_metrics]),
                "f1": statistics.mean([m["f1"] for m in all_metrics]),
                "hit@k": statistics.mean([m["hit@k"] for m in all_metrics]),
                "mrr": statistics.mean([m["mrr"] for m in all_metrics]),
                "avg_query_time": statistics.mean([m["query_time"] for m in all_metrics]),
                "total_queries": len(all_metrics)
            }
        else:
            overall = results.get("overall", {})
        
        # Calculate metrics by type
        type_overall = {}
        for qtype, metrics_list in type_metrics.items():
            if metrics_list:
                type_overall[qtype] = {
                    "precision": statistics.mean([m["precision"] for m in metrics_list]),
                    "recall": statistics.mean([m["recall"] for m in metrics_list]),
                    "f1": statistics.mean([m["f1"] for m in metrics_list]),
                    "hit@k": statistics.mean([m["hit@k"] for m in metrics_list]),
                    "mrr": statistics.mean([m["mrr"] for m in metrics_list]),
                    "count": len(metrics_list)
                }
        
        return {
            "dataset": "BioASQ",
            "overall": overall,
            "per_type": type_overall,
            "per_query": per_query_metrics,
            "statistics": {
                "total_questions": len(questions),
                "total_documents": len(corpus),
                "question_types": dict(type_counts),
                "avg_documents_per_question": statistics.mean([
                    len(q.get("documents", [])) for q in questions
                ]) if questions else 0.0,
                "evaluated_queries": len(all_metrics)
            }
        }
    else:
        # No retrieval results found, return statistics only
        print("\nNo retrieval results found. Only providing dataset statistics.")
        print("To perform full evaluation, run retrieval on BioASQ dataset first.")
        return {
            "dataset": "BioASQ",
            "statistics": {
                "total_questions": len(questions),
                "total_documents": len(corpus),
                "question_types": dict(type_counts),
                "avg_documents_per_question": statistics.mean([
                    len(q.get("documents", [])) for q in questions
                ]) if questions else 0.0,
                "questions_with_ideal_answer": sum(1 for q in questions if q.get("ideal_answer")),
                "questions_with_snippets": sum(1 for q in questions if q.get("snippets"))
            },
            "note": "BioASQ dataset retrieval results not found. Please run retrieval evaluation first to get full metrics."
        }

def generate_report():
    """Generate comprehensive evaluation report"""
    print("\n" + "=" * 80)
    print("RAG System Comprehensive Evaluation Report Generation")
    print("=" * 80 + "\n")
    
    # Evaluate each dataset
    openfda_results = evaluate_openfda()
    kaggle_results = evaluate_kaggle()
    bioasq_results = evaluate_bioasq()
    
    # Generate report
    report = {
        "evaluation_summary": {
            "date": str(Path(__file__).stat().st_mtime),
            "datasets": ["OpenFDA Drug Data", "Kaggle Drug Data", "BioASQ"]
        },
        "openfda": openfda_results,
        "kaggle": kaggle_results,
        "bioasq": bioasq_results,
        "comparison": {
            "openfda_vs_kaggle": {
                "note": "Comparison between two datasets with baseline results"
            }
        }
    }
    
    # Comparative analysis
    if openfda_results.get("overall") and kaggle_results.get("overall"):
        ofda_overall = openfda_results["overall"]
        kgl_overall = kaggle_results["overall"]
        
        report["comparison"]["openfda_vs_kaggle"] = {
            "precision": {
                "openfda": ofda_overall.get("precision", 0),
                "kaggle": kgl_overall.get("precision", 0),
                "difference": kgl_overall.get("precision", 0) - ofda_overall.get("precision", 0)
            },
            "recall": {
                "openfda": ofda_overall.get("recall", 0),
                "kaggle": kgl_overall.get("recall", 0),
                "difference": kgl_overall.get("recall", 0) - ofda_overall.get("recall", 0)
            },
            "f1": {
                "openfda": ofda_overall.get("f1", 0),
                "kaggle": kgl_overall.get("f1", 0),
                "difference": kgl_overall.get("f1", 0) - ofda_overall.get("f1", 0)
            },
            "hit@k": {
                "openfda": ofda_overall.get("hit@k", 0),
                "kaggle": kgl_overall.get("hit@k", 0),
                "difference": kgl_overall.get("hit@k", 0) - ofda_overall.get("hit@k", 0)
            },
            "mrr": {
                "openfda": ofda_overall.get("mrr", 0),
                "kaggle": kgl_overall.get("mrr", 0),
                "difference": kgl_overall.get("mrr", 0) - ofda_overall.get("mrr", 0)
            }
        }
    
    # Baseline comparison
    report["baseline_comparison"] = {}
    
    # OpenFDA baseline comparison
    baseline_openfda_path = "results/openfda_faiss_test_results.json"
    if Path(baseline_openfda_path).exists() and openfda_results.get("overall"):
        baseline_openfda = load_json(baseline_openfda_path)
        baseline_openfda_metrics = calculate_baseline_metrics(baseline_openfda, "openfda")
        if baseline_openfda_metrics:
            report["baseline_comparison"]["openfda"] = compare_with_baseline(
                openfda_results["overall"], 
                baseline_openfda_metrics,
                "OpenFDA"
            )
    
    # Kaggle baseline comparison
    baseline_kaggle_path = "data/kaggle_drug_data/processed/qdrant_test_results.json"
    if Path(baseline_kaggle_path).exists() and kaggle_results.get("overall"):
        baseline_kaggle = load_json(baseline_kaggle_path)
        baseline_kaggle_metrics = calculate_baseline_metrics(baseline_kaggle, "kaggle")
        if baseline_kaggle_metrics:
            report["baseline_comparison"]["kaggle"] = compare_with_baseline(
                kaggle_results["overall"],
                baseline_kaggle_metrics,
                "Kaggle"
            )
    
    # Save JSON report
    output_path = Path("results/comprehensive_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Evaluation report saved to: {output_path}")
    
    # Generate Markdown report
    generate_markdown_report(report)
    
    return report

def generate_markdown_report(report: Dict):
    """Generate evaluation report in Markdown format"""
    md_content = []
    
    md_content.append("# RAG System Comprehensive Evaluation Report\n")
    md_content.append("## Overview\n")
    md_content.append("This report provides a comprehensive evaluation of the RAG system's performance on three medical domain datasets.\n")
    
    # OpenFDA results
    md_content.append("\n## 1. OpenFDA Drug Data Evaluation Results\n")
    ofda = report["openfda"]
    if ofda.get("overall"):
        overall = ofda["overall"]
        md_content.append("### Overall Metrics\n")
        md_content.append("| Metric | Value |\n")
        md_content.append("|--------|-------|\n")
        md_content.append(f"| Precision | {overall.get('precision', 0):.4f} |\n")
        md_content.append(f"| Recall | {overall.get('recall', 0):.4f} |\n")
        md_content.append(f"| F1 Score | {overall.get('f1', 0):.4f} |\n")
        md_content.append(f"| Hit@K | {overall.get('hit@k', 0):.4f} |\n")
        md_content.append(f"| MRR | {overall.get('mrr', 0):.4f} |\n")
        md_content.append(f"| Average Query Time (seconds) | {overall.get('avg_query_time', 0):.4f} |\n")
        md_content.append(f"| Total Queries | {overall.get('total_queries', 0)} |\n")
    
    # Detailed results for each query
    if ofda.get("per_query"):
        md_content.append("\n### Detailed Results per Query\n")
        md_content.append("| Query ID | Query | Precision | Recall | F1 | Hit@K | MRR |\n")
        md_content.append("|----------|-------|-----------|--------|----|----|----|\n")
        for q in ofda["per_query"][:10]:  # Show only first 10
            md_content.append(
                f"| {q.get('id', '')} | {q.get('query', '')[:50]}... | "
                f"{q.get('precision', 0):.4f} | {q.get('recall', 0):.4f} | "
                f"{q.get('f1', 0):.4f} | {q.get('hit@k', 0):.4f} | {q.get('mrr', 0):.4f} |\n"
            )
        if len(ofda["per_query"]) > 10:
            md_content.append(f"\n*Note: Only showing first 10 queries out of {len(ofda['per_query'])} total queries*\n")
    
    # Kaggle results
    md_content.append("\n## 2. Kaggle Drug Data Evaluation Results\n")
    kgl = report["kaggle"]
    if kgl.get("overall"):
        overall = kgl["overall"]
        md_content.append("### Overall Metrics\n")
        md_content.append("| Metric | Value |\n")
        md_content.append("|--------|-------|\n")
        md_content.append(f"| Precision | {overall.get('precision', 0):.4f} |\n")
        md_content.append(f"| Recall | {overall.get('recall', 0):.4f} |\n")
        md_content.append(f"| F1 Score | {overall.get('f1', 0):.4f} |\n")
        md_content.append(f"| Hit@K | {overall.get('hit@k', 0):.4f} |\n")
        md_content.append(f"| MRR | {overall.get('mrr', 0):.4f} |\n")
        md_content.append(f"| Average Query Time (seconds) | {overall.get('avg_query_time', 0):.4f} |\n")
        md_content.append(f"| Total Queries | {overall.get('total_queries', 0)} |\n")
    
    # Results by type
    if kgl.get("per_type"):
        md_content.append("\n### Metrics by Question Type\n")
        md_content.append("| Type | Precision | Recall | F1 | Hit@K | MRR | Count |\n")
        md_content.append("|------|-----------|--------|----|----|----|----|\n")
        for qtype, metrics in kgl["per_type"].items():
            md_content.append(
                f"| {qtype} | {metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} | "
                f"{metrics.get('hit@k', 0):.4f} | {metrics.get('mrr', 0):.4f} | "
                f"{metrics.get('count', 0)} |\n"
            )
    
    # BioASQ results
    md_content.append("\n## 3. BioASQ Dataset Evaluation Results\n")
    bioasq = report["bioasq"]
    
    # If evaluation results exist, show them
    if bioasq.get("overall"):
        overall = bioasq["overall"]
        md_content.append("### Overall Metrics\n")
        md_content.append("| Metric | Value |\n")
        md_content.append("|--------|-------|\n")
        md_content.append(f"| Precision | {overall.get('precision', 0):.4f} |\n")
        md_content.append(f"| Recall | {overall.get('recall', 0):.4f} |\n")
        md_content.append(f"| F1 Score | {overall.get('f1', 0):.4f} |\n")
        md_content.append(f"| Hit@K | {overall.get('hit@k', 0):.4f} |\n")
        md_content.append(f"| MRR | {overall.get('mrr', 0):.4f} |\n")
        md_content.append(f"| Average Query Time (seconds) | {overall.get('avg_query_time', 0):.4f} |\n")
        md_content.append(f"| Total Queries Evaluated | {overall.get('total_queries', 0)} |\n")
        
        # Results by type
        if bioasq.get("per_type"):
            md_content.append("\n### Metrics by Question Type\n")
            md_content.append("| Type | Precision | Recall | F1 | Hit@K | MRR | Count |\n")
            md_content.append("|------|-----------|--------|----|----|----|----|\n")
            for qtype, metrics in bioasq["per_type"].items():
                md_content.append(
                    f"| {qtype} | {metrics.get('precision', 0):.4f} | "
                    f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} | "
                    f"{metrics.get('hit@k', 0):.4f} | {metrics.get('mrr', 0):.4f} | "
                    f"{metrics.get('count', 0)} |\n"
                )
    
    # Dataset statistics
    if bioasq.get("statistics"):
        stats = bioasq["statistics"]
        md_content.append("\n### Dataset Statistics\n")
        md_content.append("| Metric | Value |\n")
        md_content.append("|--------|-------|\n")
        md_content.append(f"| Total Questions | {stats.get('total_questions', 0)} |\n")
        md_content.append(f"| Total Documents | {stats.get('total_documents', 0)} |\n")
        md_content.append(f"| Average Documents per Question | {stats.get('avg_documents_per_question', 0):.2f} |\n")
        if "evaluated_queries" in stats:
            md_content.append(f"| Evaluated Queries | {stats.get('evaluated_queries', 0)} |\n")
        if "questions_with_ideal_answer" in stats:
            md_content.append(f"| Questions with Ideal Answer | {stats.get('questions_with_ideal_answer', 0)} |\n")
        if "questions_with_snippets" in stats:
            md_content.append(f"| Questions with Snippets | {stats.get('questions_with_snippets', 0)} |\n")
        
        if stats.get("question_types"):
            md_content.append("\n### Question Type Distribution\n")
            md_content.append("| Type | Count |\n")
            md_content.append("|------|-------|\n")
            for qtype, count in stats["question_types"].items():
                md_content.append(f"| {qtype} | {count} |\n")
    
    # Note if no evaluation results
    if bioasq.get("note"):
        md_content.append(f"\n**Note:** {bioasq.get('note', '')}\n")
    
    # Comparative analysis
    if report.get("comparison") and report["comparison"].get("openfda_vs_kaggle"):
        md_content.append("\n## 4. Dataset Comparison Analysis\n")
        comp = report["comparison"]["openfda_vs_kaggle"]
        
        if "precision" in comp:
            md_content.append("### OpenFDA vs Kaggle Metrics Comparison\n")
            md_content.append("| Metric | OpenFDA | Kaggle | Difference |\n")
            md_content.append("|--------|---------|--------|------------|\n")
            for metric in ["precision", "recall", "f1", "hit@k", "mrr"]:
                if metric in comp:
                    m = comp[metric]
                    diff = m.get("difference", 0)
                    diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
                    md_content.append(
                        f"| {metric.capitalize()} | {m.get('openfda', 0):.4f} | "
                        f"{m.get('kaggle', 0):.4f} | {diff_str} |\n"
                    )
    
    # Baseline comparison
    baseline_comp = report.get("baseline_comparison", {})
    has_openfda = baseline_comp.get("openfda") is not None
    has_kaggle = baseline_comp.get("kaggle") is not None
    
    if baseline_comp and (has_openfda or has_kaggle):
        md_content.append("\n## 5. Baseline Comparison\n")
        md_content.append("This section compares the current evaluation results with baseline results.\n")
        md_content.append("\n**Note**: Currently, the evaluation results shown are from baseline systems. ")
        md_content.append("When RAG system results are available, they will be compared against these baseline metrics.\n")
        
        # OpenFDA baseline comparison
        if "openfda" in baseline_comp:
            comp = baseline_comp["openfda"]
            md_content.append("\n### OpenFDA: Current vs Baseline\n")
            md_content.append("| Metric | Baseline | Current | Improvement | Relative Improvement |\n")
            md_content.append("|--------|----------|---------|-------------|---------------------|\n")
            for metric in ["precision", "recall", "f1", "hit@k", "mrr"]:
                baseline_val = comp["baseline"].get(metric, 0)
                current_val = comp["current"].get(metric, 0)
                improvement = comp["improvement"].get(metric, 0)
                rel_improvement = comp["relative_improvement"].get(metric, 0)
                improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.0000"
                rel_improvement_str = f"{rel_improvement:+.2f}%" if rel_improvement != 0 else "0.00%"
                md_content.append(
                    f"| {metric.capitalize()} | {baseline_val:.4f} | {current_val:.4f} | "
                    f"{improvement_str} | {rel_improvement_str} |\n"
                )
            if "avg_query_time" in comp["baseline"]:
                baseline_time = comp["baseline"]["avg_query_time"]
                current_time = comp["current"]["avg_query_time"]
                time_improvement = comp["improvement"]["avg_query_time"]
                time_rel_improvement = comp["relative_improvement"]["avg_query_time"]
                time_str = f"{time_improvement:+.4f}" if time_improvement != 0 else "0.0000"
                time_rel_str = f"{time_rel_improvement:+.2f}%" if time_rel_improvement != 0 else "0.00%"
                md_content.append(
                    f"| Avg Query Time (s) | {baseline_time:.4f} | {current_time:.4f} | "
                    f"{time_str} | {time_rel_str} |\n"
                )
        
        # Kaggle baseline comparison
        if "kaggle" in baseline_comp:
            comp = baseline_comp["kaggle"]
            md_content.append("\n### Kaggle: Current vs Baseline\n")
            md_content.append("| Metric | Baseline | Current | Improvement | Relative Improvement |\n")
            md_content.append("|--------|----------|---------|-------------|---------------------|\n")
            for metric in ["precision", "recall", "f1", "hit@k", "mrr"]:
                baseline_val = comp["baseline"].get(metric, 0)
                current_val = comp["current"].get(metric, 0)
                improvement = comp["improvement"].get(metric, 0)
                rel_improvement = comp["relative_improvement"].get(metric, 0)
                improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.0000"
                rel_improvement_str = f"{rel_improvement:+.2f}%" if rel_improvement != 0 else "0.00%"
                md_content.append(
                    f"| {metric.capitalize()} | {baseline_val:.4f} | {current_val:.4f} | "
                    f"{improvement_str} | {rel_improvement_str} |\n"
                )
            if "avg_query_time" in comp["baseline"]:
                baseline_time = comp["baseline"]["avg_query_time"]
                current_time = comp["current"]["avg_query_time"]
                time_improvement = comp["improvement"]["avg_query_time"]
                time_rel_improvement = comp["relative_improvement"]["avg_query_time"]
                time_str = f"{time_improvement:+.4f}" if time_improvement != 0 else "0.0000"
                time_rel_str = f"{time_rel_improvement:+.2f}%" if time_rel_improvement != 0 else "0.00%"
                md_content.append(
                    f"| Avg Query Time (s) | {baseline_time:.4f} | {current_time:.4f} | "
                    f"{time_str} | {time_rel_str} |\n"
                )
        
        # Summary of baseline comparison
        md_content.append("\n### Baseline Comparison Summary\n")
        if "openfda" in baseline_comp and "kaggle" in baseline_comp:
            ofda_comp = baseline_comp["openfda"]
            kgl_comp = baseline_comp["kaggle"]
            
            # Calculate overall improvements
            ofda_f1_improvement = ofda_comp["improvement"].get("f1", 0)
            kgl_f1_improvement = kgl_comp["improvement"].get("f1", 0)
            
            md_content.append(f"- **OpenFDA F1 Improvement**: {ofda_f1_improvement:+.4f} ")
            if ofda_f1_improvement > 0:
                md_content.append("(improved)\n")
            elif ofda_f1_improvement < 0:
                md_content.append("(degraded)\n")
            else:
                md_content.append("(no change)\n")
            
            md_content.append(f"- **Kaggle F1 Improvement**: {kgl_f1_improvement:+.4f} ")
            if kgl_f1_improvement > 0:
                md_content.append("(improved)\n")
            elif kgl_f1_improvement < 0:
                md_content.append("(degraded)\n")
            else:
                md_content.append("(no change)\n")
            
            md_content.append("\n**Note**: The baseline results shown above represent the performance of the baseline retrieval system. ")
            md_content.append("When RAG system results are available, they will be compared against these baseline metrics to show improvements.\n")
    
    # Detailed analysis
    md_content.append("\n## 6. Detailed Analysis\n")
    
    # OpenFDA analysis
    if ofda.get("per_query"):
        md_content.append("\n### OpenFDA Dataset Analysis\n")
        zero_f1_count = sum(1 for q in ofda["per_query"] if q.get("f1", 0) == 0)
        md_content.append(f"- Total queries: {len(ofda['per_query'])}\n")
        md_content.append(f"- Completely failed queries (F1=0): {zero_f1_count} ({zero_f1_count/len(ofda['per_query'])*100:.1f}%)\n")
        md_content.append(f"- Successfully retrieved relevant documents: {len(ofda['per_query']) - zero_f1_count} ({(len(ofda['per_query']) - zero_f1_count)/len(ofda['per_query'])*100:.1f}%)\n")
        
        # Find best and worst performing queries
        sorted_queries = sorted(ofda["per_query"], key=lambda x: x.get("f1", 0), reverse=True)
        if sorted_queries:
            best = sorted_queries[0]
            md_content.append(f"\n**Best performing query:**\n")
            md_content.append(f"- ID: {best.get('id', '')}\n")
            md_content.append(f"- Query: {best.get('query', '')}\n")
            md_content.append(f"- F1: {best.get('f1', 0):.4f}, Precision: {best.get('precision', 0):.4f}, Recall: {best.get('recall', 0):.4f}\n")
    
    # Kaggle analysis
    if kgl.get("per_type"):
        md_content.append("\n### Kaggle Dataset Analysis by Type\n")
        md_content.append("Different question types show significant performance variations:\n")
        for qtype, metrics in sorted(kgl["per_type"].items(), key=lambda x: x[1].get("f1", 0), reverse=True):
            f1 = metrics.get("f1", 0)
            md_content.append(f"- **{qtype}**: F1={f1:.4f}, Precision={metrics.get('precision', 0):.4f}, Recall={metrics.get('recall', 0):.4f} ({metrics.get('count', 0)} queries)\n")
    
    # BioASQ analysis
    if bioasq.get("per_query"):
        md_content.append("\n### BioASQ Dataset Analysis\n")
        zero_f1_count = sum(1 for q in bioasq["per_query"] if q.get("f1", 0) == 0)
        md_content.append(f"- Total queries evaluated: {len(bioasq['per_query'])}\n")
        md_content.append(f"- Completely failed queries (F1=0): {zero_f1_count} ({zero_f1_count/len(bioasq['per_query'])*100:.1f}%)\n")
        md_content.append(f"- Successfully retrieved relevant documents: {len(bioasq['per_query']) - zero_f1_count} ({(len(bioasq['per_query']) - zero_f1_count)/len(bioasq['per_query'])*100:.1f}%)\n")
        
        # Find best performing query
        sorted_queries = sorted(bioasq["per_query"], key=lambda x: x.get("f1", 0), reverse=True)
        if sorted_queries:
            best = sorted_queries[0]
            md_content.append(f"\n**Best performing query:**\n")
            md_content.append(f"- ID: {best.get('id', '')}\n")
            md_content.append(f"- Query: {best.get('query', '')[:100]}...\n")
            md_content.append(f"- F1: {best.get('f1', 0):.4f}, Precision: {best.get('precision', 0):.4f}, Recall: {best.get('recall', 0):.4f}\n")
    
    if bioasq.get("per_type"):
        md_content.append("\n### BioASQ Dataset Analysis by Type\n")
        md_content.append("Performance across different question types:\n")
        for qtype, metrics in sorted(bioasq["per_type"].items(), key=lambda x: x[1].get("f1", 0), reverse=True):
            f1 = metrics.get("f1", 0)
            md_content.append(f"- **{qtype}**: F1={f1:.4f}, Precision={metrics.get('precision', 0):.4f}, Recall={metrics.get('recall', 0):.4f} ({metrics.get('count', 0)} queries)\n")
    
    # Conclusions and recommendations
    md_content.append("\n## 7. Conclusions and Recommendations\n")
    md_content.append("\n### Key Findings\n")
    
    if ofda.get("overall") and kgl.get("overall"):
        ofda_f1 = ofda["overall"].get("f1", 0)
        kgl_f1 = kgl["overall"].get("f1", 0)
        
        md_content.append(f"1. **Overall Performance Comparison**:\n")
        md_content.append(f"   - Kaggle dataset: F1={kgl_f1:.4f}, Precision={kgl['overall'].get('precision', 0):.4f}, Recall={kgl['overall'].get('recall', 0):.4f}\n")
        md_content.append(f"   - OpenFDA dataset: F1={ofda_f1:.4f}, Precision={ofda['overall'].get('precision', 0):.4f}, Recall={ofda['overall'].get('recall', 0):.4f}\n")
        
        if kgl_f1 > ofda_f1:
            md_content.append(f"   - **Conclusion**: Kaggle dataset performs better than OpenFDA dataset, with F1 score {(kgl_f1-ofda_f1)*100:.2f} percentage points higher\n")
        else:
            md_content.append(f"   - **Conclusion**: OpenFDA dataset performs better than Kaggle dataset, with F1 score {(ofda_f1-kgl_f1)*100:.2f} percentage points higher\n")
        
        md_content.append(f"\n2. **Retrieval Efficiency**:\n")
        md_content.append(f"   - Kaggle average query time: {kgl['overall'].get('avg_query_time', 0):.4f} seconds\n")
        md_content.append(f"   - OpenFDA average query time: {ofda['overall'].get('avg_query_time', 0):.4f} seconds\n")
        
        md_content.append(f"\n3. **Recall Analysis**:\n")
        ofda_recall = ofda["overall"].get("recall", 0)
        kgl_recall = kgl["overall"].get("recall", 0)
        md_content.append(f"   - Kaggle Recall: {kgl_recall:.4f} ({kgl_recall*100:.1f}%)\n")
        md_content.append(f"   - OpenFDA Recall: {ofda_recall:.4f} ({ofda_recall*100:.1f}%)\n")
        md_content.append(f"   - Both datasets show relatively low recall rates, indicating that the baseline retrieval method failed to effectively retrieve all relevant documents\n")
    
    # Include BioASQ in comparison if results are available
    if bioasq.get("overall"):
        bioasq_f1 = bioasq["overall"].get("f1", 0)
        bioasq_precision = bioasq["overall"].get("precision", 0)
        bioasq_recall = bioasq["overall"].get("recall", 0)
        
        md_content.append(f"\n4. **BioASQ Dataset Performance**:\n")
        md_content.append(f"   - BioASQ dataset: F1={bioasq_f1:.4f}, Precision={bioasq_precision:.4f}, Recall={bioasq_recall:.4f}\n")
        md_content.append(f"   - Average query time: {bioasq['overall'].get('avg_query_time', 0):.4f} seconds\n")
        md_content.append(f"   - Total queries evaluated: {bioasq['overall'].get('total_queries', 0)}\n")
        
        # Compare with other datasets if available
        if ofda.get("overall") and kgl.get("overall"):
            md_content.append(f"\n5. **Three-Dataset Comparison**:\n")
            datasets = [
                ("Kaggle", kgl["overall"].get("f1", 0)),
                ("OpenFDA", ofda["overall"].get("f1", 0)),
                ("BioASQ", bioasq_f1)
            ]
            datasets_sorted = sorted(datasets, key=lambda x: x[1], reverse=True)
            md_content.append(f"   - Best performing: {datasets_sorted[0][0]} (F1={datasets_sorted[0][1]:.4f})\n")
            md_content.append(f"   - Second: {datasets_sorted[1][0]} (F1={datasets_sorted[1][1]:.4f})\n")
            md_content.append(f"   - Third: {datasets_sorted[2][0]} (F1={datasets_sorted[2][1]:.4f})\n")
    
    md_content.append("\n### Improvement Recommendations\n")
    md_content.append("Based on the evaluation results, the following improvement directions are recommended:\n\n")
    md_content.append("1. **Retrieval Strategy Optimization**:\n")
    md_content.append("   - Implement hybrid retrieval combining vector retrieval and BM25 sparse retrieval\n")
    md_content.append("   - Use domain-specific embedding models (e.g., S-PubMedBERT) instead of general models\n")
    md_content.append("   - Consider query expansion and synonym matching\n\n")
    
    md_content.append("2. **Reranking**:\n")
    md_content.append("   - Add cross-encoder reranking models to improve Top-K result precision\n")
    md_content.append("   - Use learning-to-rank methods\n\n")
    
    md_content.append("3. **Query Preprocessing**:\n")
    md_content.append("   - Implement medical term standardization and normalization\n")
    md_content.append("   - Handle drug name variants and abbreviations\n")
    md_content.append("   - Query intent recognition and classification\n\n")
    
    md_content.append("4. **Data Quality**:\n")
    md_content.append("   - Improve document chunking strategies to ensure important information is not split\n")
    md_content.append("   - Add metadata filtering (e.g., document type, source)\n")
    md_content.append("   - Consider document quality scoring\n\n")
    
    md_content.append("5. **Evaluation Metrics Extension**:\n")
    md_content.append("   - Add NDCG (Normalized Discounted Cumulative Gain) metric\n")
    md_content.append("   - Evaluate answer generation quality (if generation step is included)\n")
    md_content.append("   - Analyze performance differences across different query types\n")
    
    # Save Markdown report
    md_path = Path("results/comprehensive_evaluation_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    print(f"✓ Markdown report saved to: {md_path}")

if __name__ == "__main__":
    report = generate_report()
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
