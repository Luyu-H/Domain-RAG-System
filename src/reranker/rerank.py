# src/reranker.py
import math
from typing import List, Dict, Optional, Iterable, Tuple, Literal

# Optional cross-encoder (pip install sentence-transformers)
try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except Exception:
    _CE_AVAILABLE = False

# Reuse your embedder
try:
    from src.indexing.embedder import MedicalEmbedder
except Exception:
    from indexing.embedder import MedicalEmbedder  # fallback


class BaseReranker:
    """Minimal rerank interface."""
    def __init__(self, top_n: int = 50, name: str = "base"):
        self.top_n = top_n
        self.name = name

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        """Return scores for (query, text) pairs."""
        raise NotImplementedError

    def rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank results[:top_n] and return top_k."""
        if not results:
            return results
        pool = results[: min(self.top_n, len(results))]
        pairs = [(query, c.get("text", "") or "") for c in pool]
        scores = self.score_pairs(pairs)
        rescored = []
        for c, s in zip(pool, scores):
            r = c.copy()
            r["rerank_score"] = float(s)
            r["fusion_method"] = (r.get("fusion_method") or "pre") + "+rerank"
            rescored.append(r)
        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_k]


class SimpleSimilarityReranker(BaseReranker):
    """Cosine rerank using your MedicalEmbedder."""
    def __init__(self, embedder: Optional[MedicalEmbedder] = None, top_n: int = 50):
        super().__init__(top_n=top_n, name="simple-sim")
        self.embedder = embedder or MedicalEmbedder()
        self.dim = self.embedder.embedding_dim

    @staticmethod
    def _cosine(a, b) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        return 0.0 if (da == 0 or db == 0) else (num / (da * db))

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        q = pairs[0][0]
        q_vec = self.embedder.encode(q)[0]
        texts = [p for _, p in pairs]
        d_vecs = self.embedder.encode(texts)
        return [self._cosine(q_vec, dv) for dv in d_vecs]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder rerank (requires sentence-transformers)."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 50):
        if not _CE_AVAILABLE:
            raise ImportError("sentence_transformers is not installed.")
        super().__init__(top_n=top_n, name="cross-encoder")
        self.model = CrossEncoder(model_name)

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


def build_reranker(
    kind: Literal["none", "simple", "crossencoder"] = "simple",
    top_n: int = 50,
    cross_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    embedder: Optional[MedicalEmbedder] = None,
) -> Optional[BaseReranker]:
    """Factory for rerankers."""
    if kind == "none":
        return None
    if kind == "crossencoder":
        try:
            return CrossEncoderReranker(model_name=cross_model, top_n=top_n)
        except Exception:
            return SimpleSimilarityReranker(embedder=embedder, top_n=top_n)
    return SimpleSimilarityReranker(embedder=embedder, top_n=top_n)


# ------------------------------------------------------------
if __name__ == "__main__":
    import sys, json, argparse
    from pathlib import Path

    # Make project src importable
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    # Lazy imports to avoid overhead when used as a library
    try:
        from src.preprocessing import DocumentChunker, DataLoader
        from src.indexing.embedder import MedicalEmbedder
        from src.indexing.vector_indexer import VectorIndexer
        from src.indexing.bm25_indexer import BM25Indexer
        from src.indexing.hybrid_indexer import HybridIndexer
    except Exception:
        # Fallback if project exposes modules directly
        from preprocessing import DocumentChunker, DataLoader  # type: ignore
        from indexing.embedder import MedicalEmbedder           # type: ignore
        from indexing.vector_indexer import VectorIndexer       # type: ignore
        from indexing.bm25_indexer import BM25Indexer           # type: ignore
        from indexing.hybrid_indexer import HybridIndexer       # type: ignore

    def _print_results(title: str, results, k: int = 5):
        print(f"\n{title}")
        for i, r in enumerate(results[:k], 1):
            s = r.get("score", 0.0)
            rs = r.get("rerank_score")
            txt = (r.get("text", "") or "").replace("\n", " ")
            if rs is None:
                print(f"{i}. score={s:.4f} | {txt[:160]}{'...' if len(txt)>160 else ''}")
            else:
                print(f"{i}. rerank={rs:.4f} | fused={s:.4f} | {txt[:160]}{'...' if len(txt)>160 else ''}")

    ap = argparse.ArgumentParser(description="Reranker quick test on top of hybrid retrieval")
    ap.add_argument("--pubmed", type=str, default="data/BioASQ/corpus_subset.json")
    ap.add_argument("--openfda", type=str, default="data/OpenFDA Drug data/OpenFDA_corpus.json")
    ap.add_argument("--kaggle", type=str, default="data/kaggle_drug_data/processed/extracted_docs.json")
    ap.add_argument("--sample_docs", type=int, default=20)
    ap.add_argument("--collection", type=str, default="test_rerank_hybrid")
    ap.add_argument("--vec_db", type=str, default="data/test_rerank_hybrid_db")
    ap.add_argument("--idx_dir", type=str, default="data/test_rerank_hybrid_indices")
    ap.add_argument("--model", type=str, default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--query", type=str, default="What are the side effects of aspirin?")
    ap.add_argument("--fusion", type=str, choices=["rrf", "weighted"], default="rrf")
    ap.add_argument("--vec_w", type=float, default=0.7)
    ap.add_argument("--bm25_w", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--reranker_kind", type=str, choices=["none", "simple", "crossencoder"], default="simple")
    ap.add_argument("--rerank_top_n", type=int, default=50)
    ap.add_argument("--cross_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = ap.parse_args()

    # Load a small mixed dataset
    loader = DataLoader(pubmed_path=args.pubmed, openfda_path=args.openfda, kaggle_path=args.kaggle)
    documents = loader.load_all()[:args.sample_docs]

    # Chunk documents
    chunker = DocumentChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk_documents(documents)

    # Build hybrid index (vector + BM25)
    embedder = MedicalEmbedder(model_name=args.model)
    vector_indexer = VectorIndexer(
        collection_name=args.collection,
        embedder=embedder,
        storage_path=args.vec_db
    )
    bm25_indexer = BM25Indexer(k1=1.5, b=0.75)
    hybrid_indexer = HybridIndexer(
        vector_indexer=vector_indexer,
        bm25_indexer=bm25_indexer,
        storage_path=args.idx_dir
    )
    hybrid_indexer.index_chunks(chunks)

    # Baseline retrieve
    base = hybrid_indexer.search(
        args.query,
        top_k=args.top_k,
        fusion_method=args.fusion,
        vector_weight=args.vec_w,
        bm25_weight=args.bm25_w
    )
    _print_results(f"Top-{args.top_k} (fusion={args.fusion})", base, k=args.top_k)

    # Optional rerank
    if args.reranker_kind != "none" and base:
        rr = build_reranker(
            kind=args.reranker_kind,
            top_n=args.rerank_top_n,
            cross_model=args.cross_model,
            embedder=embedder
        )
        if rr:
            reranked = rr.rerank(args.query, base, top_k=args.top_k)
            _print_results(
                f"Top-{args.top_k} after rerank (kind={args.reranker_kind}, pool={args.rerank_top_n})",
                reranked, k=args.top_k
            )
