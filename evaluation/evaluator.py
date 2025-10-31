#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation only (no retrieval).
Reads:
  - questions_formatted.json  (schema like dataformatting.json; may include: exact_answer, ideal_answer)
  - test_results.json      (output from retrieval)

Computes:
  (A) Retrieval metrics: Precision@k, Recall@k, F1, Hit@k, MRR, Support-Coverage
  (B) Answer-aware metrics per type:
      - yesno: exact match on {"yes","no"}, token-F1 vs snippets
      - factoid: exact match (case/space normalized) + token-F1
      - list: set precision/recall/F1 (items detected in snippets)
      - summary: ROUGE-L-like (LCS) + token-F1 vs ideal_answer
Saves: eval_faiss.json
"""

import json, math, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# edit path to test
BASE = Path("./data/") 
QUESTIONS_PATH  = BASE / "questions_formatted.json"
RETRIEVED_PATH  = BASE / "test_results.json"
OUT_EVAL        = BASE / "eval_retrieval.json"

# ---------------- text utils ----------------
_WS = re.compile(r"\s+")
def norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())

def tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", norm_text(s)) if t]

def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred); g = tokenize(gold)
    if not p or not g: return 0.0
    # bag overlap
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    prec = overlap / len(p)
    rec  = overlap / len(g)
    return (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

def lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l(pred: str, gold: str) -> float:
    p = tokenize(pred); g = tokenize(gold)
    if not p or not g: return 0.0
    L = lcs(p, g)
    prec = L / len(p); rec = L / len(g)
    return (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

# ---------------- retrieval metrics ----------------
def metrics_retrieval(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    gt = list(dict.fromkeys(gt_ids))
    gt_set = set(gt)
    k = max(1, len(retrieved_ids))
    tp = sum(1 for r in retrieved_ids if r in gt_set)
    prec = tp / k
    rec  = tp / max(1, len(gt))
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    rr   = 0.0
    for i, r in enumerate(retrieved_ids, start=1):
        if r in gt_set: rr = 1.0 / i; break
    hitk = 1.0 if tp > 0 else 0.0
    return {"precision":prec, "recall":rec, "f1":f1, "hit@k":hitk, "mrr":rr}

def support_coverage(snippets: List[Dict[str, Any]], retrieved_ids: List[str]) -> float:
    """Fraction of gold snippets whose 'document' is present in retrieved ids."""
    gold_docs = [s.get("document") for s in (snippets or []) if s.get("document")]
    if not gold_docs: return 0.0
    rset = set(retrieved_ids)
    covered = sum(1 for d in gold_docs if d in rset)
    return covered / len(gold_docs)

# ---------------- answer-aware evaluation by type ----------------
def eval_yesno(retrieved_text: str, exact: Any, ideal: Any) -> Dict[str, float]:
    out = {}
    if isinstance(exact, str):
        gold = norm_text(exact)
        # detect yes/no in retrieved text (naive)
        pred_yes = " yes " in (" " + norm_text(retrieved_text) + " ")
        pred_no  = " no "  in (" " + norm_text(retrieved_text) + " ")
        pred = "yes" if pred_yes and not pred_no else ("no" if pred_no and not pred_yes else "")
        out["em"] = 1.0 if pred and pred == gold else 0.0
        out["token_f1_exact"] = token_f1(pred, gold) if pred else 0.0
    if isinstance(ideal, str):
        out["token_f1_ideal"] = token_f1(retrieved_text, ideal)
        out["rougeL_ideal"]   = rouge_l(retrieved_text, ideal)
    return out

def eval_factoid(retrieved_text: str, exact: Any, ideal: Any) -> Dict[str, float]:
    out = {}
    if isinstance(exact, str):
        out["em"] = 1.0 if norm_text(exact) in norm_text(retrieved_text) else 0.0
        out["token_f1_exact"] = token_f1(retrieved_text, exact)
    if isinstance(ideal, str):
        out["token_f1_ideal"] = token_f1(retrieved_text, ideal)
        out["rougeL_ideal"]   = rouge_l(retrieved_text, ideal)
    return out

def eval_list(retrieved_text: str, exact: Any, ideal: Any) -> Dict[str, float]:
    """exact may be list of strings; score by set P/R/F1 if items appear in retrieved text."""
    out = {}
    if isinstance(exact, list) and exact:
        gold = [norm_text(x) for x in exact if isinstance(x, str)]
        found = set()
        textn = norm_text(retrieved_text)
        for g in gold:
            if g and g in textn:
                found.add(g)
        tp = len(found)
        prec = tp / max(1, len(set(found)))  # here predicted list == found
        rec  = tp / max(1, len(set(gold)))
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        out.update({"list_precision":prec, "list_recall":rec, "list_f1":f1})
    if isinstance(ideal, str):
        out["token_f1_ideal"] = token_f1(retrieved_text, ideal)
        out["rougeL_ideal"]   = rouge_l(retrieved_text, ideal)
    return out

def eval_summary(retrieved_text: str, exact: Any, ideal: Any) -> Dict[str, float]:
    out = {}
    if isinstance(ideal, str):
        out["token_f1_ideal"] = token_f1(retrieved_text, ideal)
        out["rougeL_ideal"]   = rouge_l(retrieved_text, ideal)
    return out

TYPE_EVAL = {
    "yesno": eval_yesno,
    "factoid": eval_factoid,
    "list": eval_list,
    "summary": eval_summary,
}

# ---------------- main evaluation ----------------
def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")).get("questions", [])
    retrieved = json.loads(RETRIEVED_PATH.read_text(encoding="utf-8"))

    # map query id -> retrieved ids & join a simple "retrieved_text" from those ids
    per_query = {q["id"]: q for q in retrieved.get("per_query", [])}
    # optional: load corpus to build a richer retrieved_text (skipped here for speed)
    # if you want, concatenate top-5 previews stored in retrieval output itself:
    # (we only stored ids+scores, so retrieved_text is left empty unless you add previews)

    out = {"per_query": [], "overall": {}}
    agg_r = {"precision":0,"recall":0,"f1":0,"hit@k":0,"mrr":0}
    agg_a = {}  # will average over present keys

    for q in questions:
        qid = q.get("id")
        qtype = q.get("type","").lower()
        gt_docs = q.get("documents", []) or []
        exact = q.get("exact_answer", None)      # OPTIONAL field
        ideal = q.get("ideal_answer", None)      # OPTIONAL/Recommended

        r = per_query.get(qid, {})
        hits = r.get("retrieved", [])
        retrieved_ids = [h["chunk_id"] for h in hits]

        # retrieval metrics
        rm = metrics_retrieval(gt_docs, retrieved_ids)
        cov = support_coverage(q.get("snippets", []), retrieved_ids)

        # build a naive retrieved_text by concatenating the ids themselves (or integrate corpus/preview for better text metrics)
        # For stronger answer-aware metrics, add previews in retrieval_faiss.py and concat them here.
        retrieved_text = " ".join(retrieved_ids)

        # answer-aware by type (graceful if type is unknown)
        ae = {}
        fn = TYPE_EVAL.get(qtype)
        if fn:
            ae = fn(retrieved_text, exact, ideal)

        out["per_query"].append({
            "id": qid,
            "type": qtype,
            "retrieval": rm,
            "support_coverage": cov,
            "answer_eval": ae
        })

        for k in agg_r: agg_r[k] += rm[k]
        for k,v in ae.items():
            agg_a[k] = agg_a.get(k, 0.0) + v

    n = max(1, len(out["per_query"]))
    out["overall"]["retrieval"] = {k: round(v/n, 4) for k,v in agg_r.items()}
    if agg_a:
        out["overall"]["answer_eval"] = {k: round(v/n, 4) for k,v in agg_a.items()}

    OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_EVAL.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] evaluation saved -> {OUT_EVAL}")

if __name__ == "__main__":
    main()
