#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drugs.com dataset processor for RAG.

- Reads `drugs_side_effects_drugs_com.csv`
- Normalizes into doc-level JSON and chunk-level JSON/TSV
- Prints analysis similar to your openFDA extractor main()

Stdlib only.
"""

import csv
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Optional


# ============ Utilities ============

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()

def _safe_float(x: Optional[str]):
    try:
        return float(x) if x not in (None, "") else None
    except Exception:
        return None

def split_list(s: str, sep=",") -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(sep) if x.strip()]

def parse_related(s: str) -> List[Dict[str, str]]:
    """
    Parse: "name: https://link | name2: https://link2"
    """
    out = []
    if not s:
        return out
    for part in s.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, url = part.split(":", 1)
            out.append({"name": name.strip(), "url": url.strip()})
        else:
            out.append({"name": part, "url": ""})
    return out

def normalize_pregnancy(cat: str) -> str:
    c = (cat or "").strip().upper()
    if c in {"X", "D"}:
        return "avoid"
    if c in {"C"}:
        return "caution"
    if c in {"A", "B"}:
        return "generally_safe"
    return "unknown"

def extract_side_effect_lists(text: str) -> Dict[str, List[str]]:
    """
    Heuristic to split SERIOUS vs COMMON side effects from long consumer text.
    Tuned for drugs.com style paragraphs.
    """
    if not text:
        return {"serious": [], "common": []}
    t = " ".join(text.split())

    # Common block (e.g., "Common side effects ... include:")
    common = []
    mc = re.search(r"(Common\s+side\s+effects.*?include:)(.*)", t, flags=re.I)
    if mc:
        common_text = mc.group(2)
        common = re.split(r"[;•·]|\s*,\s*|\.\s+", common_text)
        common = [re.sub(r"^[\-\u2022]\s*", "", s).strip(" .;") for s in common if s.strip()]

    # Serious block (e.g., "may cause serious side effects. ... Common ...")
    serious = []
    ms = re.search(r"may\s+cause\s+serious\s+side\s+effects\.(.*?)(Common|$)", t, flags=re.I)
    if ms:
        serious_text = ms.group(1)
        serious = re.split(r"[;•·]|\s*,\s*|\.\s+", serious_text)
        serious = [re.sub(r"^[\-\u2022]\s*", "", s).strip(" .;") for s in serious if s.strip()]

    # Fallback serious
    if not serious:
        alt = re.search(r"Call your doctor at once if you have:(.*?)(Common|$)", t, flags=re.I)
        if alt:
            st = alt.group(1)
            serious = [s.strip(" .;") for s in re.split(r"[;•·]|\s*,\s*|\.\s+", st) if s.strip()]

    # dedupe & truncate a bit
    def uniq(xs):
        seen = set()
        out = []
        for x in xs:
            k = x.lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x)
        return out[:40]

    return {"serious": uniq(serious), "common": uniq(common)}

def summarize(text: str, max_chars: int = 900) -> str:
    if not text:
        return ""
    s = " ".join(text.split())
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_period = cut.rfind(".")
    return cut[:last_period + 1] if last_period >= 200 else (cut + "…")


# ============ Processor Class ============

class DrugsComDataProcessor:
    """
    A class that:
      - loads the drugs.com CSV
      - constructs doc-level records
      - splits into retrieval chunks with metadata
      - saves to json/TSV
      - exposes stats for analysis
    """

    EXPECTED_COLS = [
        "drug_name","medical_condition","side_effects","generic_name","drug_classes",
        "brand_names","activity","rx_otc","pregnancy_category","csa","alcohol",
        "related_drugs","medical_condition_description","rating","no_of_reviews",
        "drug_link","medical_condition_url"
    ]

    def __init__(self, input_csv: str):
        self.input_csv = Path(input_csv)
        self.docs: List[Dict] = []
        self.chunks: List[Dict] = []

    # ---------- Load & Process ----------

    def _iter_rows(self) -> Iterable[Dict]:
        with self.input_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, quotechar='"')
            # warn if schema differs
            missing = [c for c in self.EXPECTED_COLS if c not in reader.fieldnames]
            if missing:
                print(f"[WARN] Missing columns in {self.input_csv.name}: {missing}")
            for row in reader:
                yield row

    def _make_doc(self, row: Dict) -> Dict:
        drug = (row.get("drug_name") or "").strip()
        cond = (row.get("medical_condition") or "").strip()
        doc_id = md5(f"{drug}|{cond}")

        side_groups = extract_side_effect_lists(row.get("side_effects") or "")

        return {
            "doc_id": doc_id,
            "drug_name": drug,
            "generic_name": (row.get("generic_name") or "").strip(),
            "medical_condition": cond,
            "drug_classes": split_list(row.get("drug_classes","")),
            "brand_names": split_list(row.get("brand_names","")),
            "rx_otc": (row.get("rx_otc") or "").strip(),  # "Rx" / "OTC"
            "pregnancy_category": normalize_pregnancy(row.get("pregnancy_category","")),
            "pregnancy_category_raw": (row.get("pregnancy_category") or "").strip(),
            "csa": (row.get("csa") or "").strip(),
            "alcohol": (row.get("alcohol") or "").strip(),
            "related_drugs": parse_related(row.get("related_drugs","")),
            "activity": (row.get("activity") or "").strip(),
            "rating": _safe_float(row.get("rating")),
            "no_of_reviews": _safe_float(row.get("no_of_reviews")),
            "links": {
                "drug": (row.get("drug_link") or "").strip(),
                "condition": (row.get("medical_condition_url") or "").strip()
            },
            "condition_summary": summarize(row.get("medical_condition_description") or "", 900),
            "side_effects_structured": side_groups,
            "side_effects_raw": (row.get("side_effects") or "").strip()
        }

    def _mk_chunk(self, doc: Dict, kind: str, text: str) -> Dict:
        base_meta = {
            "doc_id": doc["doc_id"],
            "drug_name": doc["drug_name"],
            "generic_name": doc["generic_name"],
            "medical_condition": doc["medical_condition"],
            "rx_otc": doc["rx_otc"],
            "pregnancy_category": doc["pregnancy_category"],
            "rating": doc["rating"],
            "no_of_reviews": doc["no_of_reviews"],
            "drug_link": doc["links"]["drug"],
            "chunk_kind": kind
        }
        chunk_id = md5(f"{doc['doc_id']}|{kind}|{len(text)}")
        return {"chunk_id": chunk_id, "text": text, "metadata": base_meta}

    def _make_chunks(self, doc: Dict) -> List[Dict]:
        chunks = []

        # 1) Facts
        facts = []
        if doc["drug_classes"]:
            facts.append("Classes: " + ", ".join(doc["drug_classes"]))
        if doc["brand_names"]:
            facts.append("Brands: " + ", ".join(doc["brand_names"]))
        facts.append(f"Rx/OTC: {doc['rx_otc'] or 'Unknown'}; Pregnancy: {doc['pregnancy_category']} ({doc['pregnancy_category_raw'] or 'n/a'})")
        if doc["rating"] is not None:
            facts.append(f"Rating: {doc['rating']} ({int(doc['no_of_reviews'] or 0)} reviews)")
        fact_text = f"{doc['drug_name']} — {doc['medical_condition']}\n" + "\n".join(facts)
        chunks.append(self._mk_chunk(doc, "facts", fact_text))

        # 2) Serious SE
        serious = doc["side_effects_structured"]["serious"]
        if serious:
            text = f"Serious side effects of {doc['drug_name']}:\n- " + "\n- ".join(serious)
            chunks.append(self._mk_chunk(doc, "side_effects_serious", text))

        # 3) Common SE
        common = doc["side_effects_structured"]["common"]
        if common:
            text = f"Common side effects of {doc['drug_name']}:\n- " + "\n- ".join(common)
            chunks.append(self._mk_chunk(doc, "side_effects_common", text))

        # 4) Condition overview
        if doc["condition_summary"]:
            text = f"{doc['medical_condition']} — overview:\n{doc['condition_summary']}"
            chunks.append(self._mk_chunk(doc, "condition_overview", text))

        # 5) Related
        rel = doc["related_drugs"]
        if rel:
            lines = [f"- {r['name']} ({r['url']})" if r['url'] else f"- {r['name']}" for r in rel]
            text = f"Related drugs to {doc['drug_name']} for {doc['medical_condition']}:\n" + "\n".join(lines)
            chunks.append(self._mk_chunk(doc, "related", text))

        return chunks

    # ---------- Public API ----------

    def extract(self, limit: Optional[int] = None) -> None:
        """
        Build self.docs and self.chunks from the CSV.
        """
        self.docs.clear()
        self.chunks.clear()

        for i, row in enumerate(self._iter_rows()):
            if limit is not None and i >= limit:
                break
            doc = self._make_doc(row)
            self.docs.append(doc)
            self.chunks.extend(self._make_chunks(doc))

    def save(self, out_docs: str, out_corpus_json: str, out_corpus_tsv: str) -> None:
        """
        Persist docs and chunks.
        """
        p_docs = Path(out_docs)
        p_cj = Path(out_corpus_json)
        p_ct = Path(out_corpus_tsv)

        p_docs.parent.mkdir(parents=True, exist_ok=True)
        p_cj.parent.mkdir(parents=True, exist_ok=True)
        p_ct.parent.mkdir(parents=True, exist_ok=True)

        # Write as JSON
        with p_docs.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

        with p_cj.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        with p_ct.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                clean_text = c["text"].replace("\n", " ").strip()
                f.write(f"{c['chunk_id']}\t{clean_text}\n")

    # ---------- Analyses ----------

    def stats(self) -> Dict[str, float]:
        """
        Compute analysis metrics similar to your openFDA example.
        Returns a dict for easier testing/printing.
        """
        n_docs = len(self.docs)
        n_chunks = len(self.chunks)

        # average document "length": use len of combined basic facts + raw SE
        lengths = []
        for d in self.docs:
            L = len((d.get("side_effects_raw") or "")) + len(" ".join(d.get("drug_classes", []))) + len(" ".join(d.get("brand_names", []))) + len(d.get("condition_summary", "") or "")
            lengths.append(L)
        avg_len = (sum(lengths) / n_docs) if n_docs else 0.0

        # coverage stats
        has_serious = sum(1 for d in self.docs if d["side_effects_structured"]["serious"])
        has_common  = sum(1 for d in self.docs if d["side_effects_structured"]["common"])
        has_classes = sum(1 for d in self.docs if d["drug_classes"])
        has_brands  = sum(1 for d in self.docs if d["brand_names"])
        has_rating  = sum(1 for d in self.docs if d["rating"] is not None)

        rx_count    = sum(1 for d in self.docs if (d["rx_otc"] or "").lower() == "rx")
        otc_count   = sum(1 for d in self.docs if (d["rx_otc"] or "").lower() == "otc")

        preg_buckets = {
            "generally_safe": 0,
            "caution": 0,
            "avoid": 0,
            "unknown": 0
        }
        for d in self.docs:
            preg_buckets[d["pregnancy_category"]] = preg_buckets.get(d["pregnancy_category"], 0) + 1

        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "avg_length_chars": avg_len,
            "has_serious_count": has_serious,
            "has_common_count": has_common,
            "has_classes_count": has_classes,
            "has_brands_count": has_brands,
            "has_rating_count": has_rating,
            "rx_count": rx_count,
            "otc_count": otc_count,
            **{f"preg_{k}": v for k, v in preg_buckets.items()}
        }


# ============ Main (like your example) ============

def main():
    """Main function"""
    input_file = "./drugs_side_effects_drugs_com.csv"
    out_dir = Path("./processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_docs = out_dir / "extracted_docs.json"
    out_corpus_json = out_dir / "extracted_corpus.json"
    out_corpus_tsv = out_dir / "extracted_corpus.tsv"

    # Extract data
    extractor = DrugsComDataProcessor(input_file)
    extractor.extract(limit=None)  # or set e.g., limit=1000 for a fast run
    extractor.save(str(out_docs), str(out_corpus_json), str(out_corpus_tsv))

    # Display statistics
    stats = extractor.stats()
    n = max(1, stats["documents"])

    print("\nData extraction completed:")
    print(f"- Total documents: {stats['documents']}")
    print(f"- Total chunks: {stats['chunks']}")
    print(f"- Average document length: {stats['avg_length_chars']:.0f} characters")

    # Feature coverage (similar style)
    print("- Document feature statistics:")
    def pct(x): return 100.0 * x / n
    print(f"  Has SERIOUS side effects: {stats['has_serious_count']} ({pct(stats['has_serious_count']):.1f}%)")
    print(f"  Has COMMON side effects:  {stats['has_common_count']} ({pct(stats['has_common_count']):.1f}%)")
    print(f"  Has Drug classes:         {stats['has_classes_count']} ({pct(stats['has_classes_count']):.1f}%)")
    print(f"  Has Brand names:          {stats['has_brands_count']} ({pct(stats['has_brands_count']):.1f}%)")
    print(f"  Has Ratings:              {stats['has_rating_count']} ({pct(stats['has_rating_count']):.1f}%)")
    print(f"  Rx-only:                  {stats['rx_count']} ({pct(stats['rx_count']):.1f}%)")
    print(f"  OTC:                      {stats['otc_count']} ({pct(stats['otc_count']):.1f}%)")

    # Pregnancy buckets
    print("- Pregnancy category buckets (normalized):")
    print(f"  Generally safe (A/B):     {stats['preg_generally_safe']}")
    print(f"  Caution (C):              {stats['preg_caution']}")
    print(f"  Avoid (D/X):              {stats['preg_avoid']}")
    print(f"  Unknown:                  {stats['preg_unknown']}")

    # Output paths
    print("\nOutputs:")
    print(f"  {out_docs}")
    print(f"  {out_corpus_json}")
    print(f"  {out_corpus_tsv}\n")


if __name__ == "__main__":
    main()
