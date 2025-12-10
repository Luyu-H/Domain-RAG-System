#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RAG system evaluation on BioASQ dataset
This script evaluates the RAG system on BioASQ test queries and saves results.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_evaluation import run_rag_evaluation_bioasq

if __name__ == "__main__":
    print("=" * 80)
    print("BioASQ RAG System Evaluation")
    print("=" * 80)
    print()
    
    # Run evaluation
    results = run_rag_evaluation_bioasq(
        indices_dir="data/indices",
        drug_mapping_path="data/processed_for_our_rag/drug_mapping.json",
        top_k=10,
        fusion_method="rrf",
        reranker_kind="simple",
        rerank_top_n=50,
        output_path="results/bioasq_rag_test_results.json"
    )
    
    if results:
        print("\n" + "=" * 80)
        print("Evaluation completed successfully!")
        print("=" * 80)
        print(f"\nResults saved to: results/bioasq_rag_test_results.json")
        print(f"\nOverall Metrics:")
        overall = results.get("overall", {})
        if overall:
            print(f"  Precision: {overall.get('precision', 0):.4f}")
            print(f"  Recall: {overall.get('recall', 0):.4f}")
            print(f"  F1 Score: {overall.get('f1', 0):.4f}")
            print(f"  Hit@K: {overall.get('hit@k', 0):.4f}")
            print(f"  MRR: {overall.get('mrr', 0):.4f}")
        print(f"\nTotal queries evaluated: {len(results.get('per_query', []))}")
    else:
        print("\n" + "=" * 80)
        print("Evaluation failed. Please check error messages above.")
        print("=" * 80)
        sys.exit(1)

