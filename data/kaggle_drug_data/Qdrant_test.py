#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time, logging
from pathlib import Path
from typing import Any, Dict, List

BASE = Path("./processed")
CORPUS_PATH    = BASE / "extracted_corpus.json"
QUESTIONS_PATH = BASE / "test_queries_formatted.json"
OUT_JSON       = BASE / "qdrant_test_results.json"
COLLECTION     = "kaggle_drugs_corpus_qdrant"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("QDRANT_SBERT_TEST")

def eval_metrics(gt_ids: List[str], hit_ids: List[str]) -> Dict[str, float]:
    gt = list(dict.fromkeys(gt_ids)); gt_set = set(gt)
    k = max(1, len(hit_ids))
    tp = sum(1 for h in hit_ids if h in gt_set)
    prec = tp / k
    rec  = tp / max(1, len(gt))
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    rr = 0.0
    for i, h in enumerate(hit_ids, start=1):
        if h in gt_set: rr = 1.0/i; break
    hitk = 1.0 if tp > 0 else 0.0
    return {"precision":prec,"recall":rec,"f1":f1,"hit@k":hitk,"mrr":rr}

def qdrant_retrieval_test(k: int = 5, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    qjson  = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = qjson.get("questions", [])
    log.info(f"Loaded corpus={len(corpus)} chunks, questions={len(questions)}")

    # Payloads
    texts = [c.get("text","") for c in corpus]
    payloads = []
    for c in corpus:
        md = c.get("metadata", {}) or {}
        p = {"chunk_id": c.get("chunk_id"), "preview": (c.get("text","").replace("\n"," ")[:240] if c.get("text") else "")}
        for key in ("drug_name","medical_condition","chunk_kind","rating","no_of_reviews","rx_otc","pregnancy_category"):
            if key in md: p[key] = md[key]
        payloads.append(p)

    # Embeddings (SBERT)
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    if X.size == 0: raise RuntimeError("No vectors produced for corpus")
    dim = X.shape[1]
    log.info(f"Embedded corpus | dim={dim} | model={model_name}")

    # Qdrant in-memory
    client = QdrantClient(location=":memory:")
    if COLLECTION in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE))

    # Upsert
    ids = list(range(len(texts)))
    B = 512
    for i in range(0, len(ids), B):
        sl = slice(i, min(i+B, len(ids)))
        client.upsert(COLLECTION, points=rest.Batch(ids=ids[sl],
                                                    vectors=X[sl].tolist(),
                                                    payloads=payloads[sl]))
    # Queries
    qtexts = [q.get("body","") for q in questions]
    QX = model.encode(qtexts, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    # Search & eval
    results = {"model": model_name, "k": k, "per_query": []}
    for qi, q in enumerate(questions):
        start = time.time()
        hits = client.search(COLLECTION, query_vector=QX[qi].tolist(), limit=k, with_payload=True)
        elapsed = round(time.time() - start, 4)

        retrieved, topk = [], []
        for rank, h in enumerate(hits, start=1):
            cid = h.payload.get("chunk_id"); retrieved.append(cid)
            topk.append({"rank":rank, "score":float(h.score), "chunk_id":cid,
                         "preview":h.payload.get("preview",""),
                         "metadata":{k:v for k,v in (h.payload or {}).items() if k not in ("preview","chunk_id")}})
        m = eval_metrics(q.get("documents", []), retrieved)
        results["per_query"].append({
            "id": q.get("id", str(qi+1)), "type": q.get("type",""),
            "query": q.get("body",""), "k": k, "query_time_sec": elapsed,
            "ground_truth": q.get("documents", []), "retrieved_ids": retrieved,
            "metrics": m, "topk": topk, "ideal_answer": q.get("ideal_answer","")
        })
        log.info(f"Q{q.get('id', qi+1)} | P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                 f"Hit@k={m['hit@k']:.3f} MRR={m['mrr']:.3f} | {elapsed}s")

    # Aggregate
    agg = {"precision":0.0,"recall":0.0,"f1":0.0,"hit@k":0.0,"mrr":0.0}
    n = max(1, len(results["per_query"]))
    for r in results["per_query"]:
        for k_ in agg: agg[k_] += r["metrics"][k_]
    for k_ in agg: agg[k_] = round(agg[k_] / n, 4)
    results["overall"] = agg

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== QDRANT (SBERT) Retrieval Test ===")
    print("Model:", model_name, "| k=", k)
    print("Overall:", results["overall"])
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    qdrant_retrieval_test(k=args.k, model_name=args.model)
