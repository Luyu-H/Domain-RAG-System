#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build one query per request type (fixed order) and compute Top-5 ground-truth chunks.
Input  : JSON array corpus  -> ./data/kaggle_drug_data/processed/extracted_corpus.json
Output : JSON with queries, ground_truth_top5, answers_top5, summaries
         -> ./data/kaggle_drug_data/processed/test_queries_ifcase_top5.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter


class GeneralQueryTop5Builder:
    def __init__(self, corpus_path: str, out_path: str):
        self.corpus_path = Path(corpus_path)
        self.out_path = Path(out_path)
        self.corpus: List[Dict[str, Any]] = []
        self.picks: Dict[str, List[str]] = {}
        self.results: Dict[str, Any] = {
            "queries": [],
            "ground_truth_top5": {},
            "answers_top5": {},
            "summaries": {}
        }

    # ---------------- I/O ----------------
    @staticmethod
    def _load_array(path: Path) -> List[Dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _brief(text: str, n: int = 240) -> str:
        s = (text or "").replace("\n", " ").strip()
        return s if len(s) <= n else (s[:n] + "…")

    @staticmethod
    def _simple_rank(items: List[Tuple[float, Dict]]) -> List[Dict]:
        items.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in items if isinstance(s, (int, float)) and s > 0]

    # ---------------- Discover frequent entities ----------------
    def _discover_entities(self) -> Dict[str, List[str]]:
        cond_counter = Counter()
        drug_counter = Counter()
        class_counter = Counter()

        for c in self.corpus:
            m = c.get("metadata", {})
            drug = (m.get("drug_name") or "").strip().lower()
            cond = (m.get("medical_condition") or "").strip().lower()
            if drug: drug_counter[drug] += 1
            if cond: cond_counter[cond] += 1

            # parse classes from facts text "Classes: x, y"
            if m.get("chunk_kind") == "facts":
                t = c.get("text", "")
                if "Classes:" in t:
                    try:
                        part = t.split("Classes:", 1)[1].split("\n", 1)[0]
                        for item in [x.strip().lower() for x in part.split(",") if x.strip()]:
                            class_counter[item] += 1
                    except Exception:
                        pass

        def topk(cnt: Counter, k: int) -> List[str]:
            return [x for x, _ in cnt.most_common(k) if x]

        return {
            "top_conditions": topk(cond_counter, 10),
            "top_drugs": topk(drug_counter, 20),
            "top_classes": topk(class_counter, 10),
        }

    # ---------------- Build fixed-order queries ----------------
    def _build_queries(self) -> List[Dict[str, Any]]:
        drugA = "doxycycline" if "doxycycline" in self.picks["top_drugs"] else (self.picks["top_drugs"][0] if self.picks["top_drugs"] else "")
        drugB = "spironolactone" if "spironolactone" in self.picks["top_drugs"] else drugA
        condA = "acne" if "acne" in self.picks["top_conditions"] else (self.picks["top_conditions"][0] if self.picks["top_conditions"] else "")
        clazz = self.picks["top_classes"][0] if self.picks["top_classes"] else ""

        return [
            {"id":"query_1","type":"drug_side_effects","query":f"What are the serious and common side effects of {drugA}?","params":{"drug_name":drugA},"expected_fields":["side_effects_serious","side_effects_common"]},
            {"id":"query_2","type":"condition_best_rated","query":f"Among drugs used for {condA.capitalize()}, which are best rated by patients?","params":{"condition":condA},"expected_fields":["facts"]},
            {"id":"query_3","type":"drug_facts","query":f"Is {drugB} Rx or OTC, what is its pregnancy category, and what are its brand names?","params":{"drug_name":drugB},"expected_fields":["facts"]},
            {"id":"query_4","type":"drug_alternatives","query":f"What are alternative drugs related to {drugA} (with links)?","params":{"drug_name":drugA},"expected_fields":["related"]},
            {"id":"query_5","type":"pregnancy_safe_options","query":f"For {condA.capitalize()}, which options look pregnancy-safe?","params":{"condition":condA},"expected_fields":["facts"]},
            {"id":"query_6","type":"otc_options","query":f"Which OTC options exist for {condA.capitalize()}?","params":{"condition":condA},"expected_fields":["facts"]},
            {"id":"query_7","type":"alcohol_caution","query":f"Does {drugA} have alcohol cautions?","params":{"drug_name":drugA},"expected_fields":["facts"]},
            {"id":"query_8","type":"class_based_options","query":f"List drugs that belong to the class: {clazz}","params":{"drug_class":clazz},"expected_fields":["facts"]},
        ]

    # ---------------- Rank by request type (if-elif router) ----------------
    def _rank_top5_for_query(self, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        qtype = q["type"]
        params = q.get("params", {})
        items: List[Tuple[float, Dict[str, Any]]] = []

        if qtype == "drug_side_effects":
            drug = params["drug_name"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("drug_name","").lower()==drug and m.get("chunk_kind") in ("side_effects_serious","side_effects_common"):
                    base = 5 if m.get("chunk_kind")=="side_effects_serious" else 3
                    t = (c.get("text") or "").lower()
                    if any(k in t for k in ("nausea","rash","headache","diarrhea")):
                        base += 1
                    items.append((base, c))

        elif qtype == "condition_best_rated":
            cond = params["condition"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("medical_condition","").lower()==cond and m.get("chunk_kind")=="facts":
                    r = m.get("rating") or 0.0
                    n = m.get("no_of_reviews") or 0.0
                    score = r*100.0 + n
                    items.append((score, c))

        elif qtype == "drug_facts":
            drug = params["drug_name"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("drug_name","").lower()==drug and m.get("chunk_kind")=="facts":
                    s = 0
                    if (m.get("rx_otc") or "").lower() in ("rx","otc"): s += 2
                    if m.get("pregnancy_category") in ("generally_safe","caution","avoid"): s += 2
                    if "Brands:" in (c.get("text") or ""): s += 2
                    items.append((s, c))

        elif qtype == "drug_alternatives":
            drug = params["drug_name"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("drug_name","").lower()==drug and m.get("chunk_kind")=="related":
                    t = c.get("text","")
                    score = 2*t.count("http") + t.count("\n- ")
                    items.append((score, c))

        elif qtype == "pregnancy_safe_options":
            cond = params["condition"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("medical_condition","").lower()==cond and m.get("chunk_kind")=="facts":
                    pc = m.get("pregnancy_category")
                    score = 10 if pc=="generally_safe" else (6 if pc=="caution" else 0)
                    items.append((score, c))

        elif qtype == "otc_options":
            cond = params["condition"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("medical_condition","").lower()==cond and m.get("chunk_kind")=="facts":
                    score = 8 if (m.get("rx_otc","").lower()=="otc") else 0
                    items.append((score, c))

        elif qtype == "alcohol_caution":
            drug = params["drug_name"].lower()
            for c in self.corpus:
                m = c.get("metadata", {})
                if m.get("drug_name","").lower()==drug and m.get("chunk_kind")=="facts":
                    s = 0
                    if m.get("alcohol"): s += 3
                    if "alcohol" in (c.get("text","").lower()): s += 2
                    items.append((s, c))

        elif qtype == "class_based_options":
            clazz = (params.get("drug_class") or "").lower()
            if clazz:
                for c in self.corpus:
                    m = c.get("metadata", {})
                    if m.get("chunk_kind")=="facts" and clazz in (c.get("text","").lower()):
                        s = 0
                        if "Brands:" in (c.get("text") or ""): s += 2
                        if m.get("rating") is not None: s += 1
                        items.append((s, c))

        ranked = self._simple_rank(items)[:5]
        return ranked

    # ---------------- Orchestration ----------------
    def generate(self) -> Dict[str, Any]:
        # load + discover + build queries
        self.corpus = self._load_array(self.corpus_path)
        self.picks = self._discover_entities()
        queries = self._build_queries()

        # compute top-5 for each query in given order
        for q in queries:
            ranked = self._rank_top5_for_query(q)

            self.results["queries"].append({
                "id": q["id"],
                "type": q["type"],
                "query": q["query"],
                "params": q.get("params", {}),
                "expected_fields": q.get("expected_fields", [])
            })

            self.results["ground_truth_top5"][q["id"]] = [c["chunk_id"] for c in ranked]
            self.results["answers_top5"][q["id"]] = [{
                "chunk_id": c["chunk_id"],
                "preview": self._brief(c.get("text","")),
                "metadata": c.get("metadata", {})
            } for c in ranked]
            self.results["summaries"][q["id"]] = " ".join(("• " + self._brief(c.get("text",""), 180)) for c in ranked)

        # save
        self._save_json(self.out_path, self.results)
        return self.results


# ---------------- Minimal main() ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./processed/extracted_corpus.json")
    ap.add_argument("--out", default="./processed/test_queries_types.json")
    args = ap.parse_args()

    builder = GeneralQueryTop5Builder(corpus_path=args.corpus, out_path=args.out)
    results = builder.generate()

    print(f"[OK] wrote {args.out}")
    for q in results["queries"]:
        print(f" - {q['id']} | {q['type']:<22} | GT={len(results['ground_truth_top5'][q['id']])} | {q['query']}")


if __name__ == "__main__":
    main()
