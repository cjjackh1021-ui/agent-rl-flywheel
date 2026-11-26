"""Hybrid Okapi BM25 + dense vector + Reciprocal Rank Fusion (RRF) retriever."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

_EN_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.-]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class DocumentChunk:
    """Single document unit stored in the hybrid knowledge base."""

    doc_id: str
    group_id: str
    title: str
    text: str


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text into lowercase words plus CJK unigrams/bigrams."""
    tokens: list[str] = []
    for match in _EN_TOKEN_RE.findall(text):
        tokens.append(match.lower())
    for run in _CJK_RUN_RE.findall(text):
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


def default_hashed_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Deterministic zero-dependency L2-normalized character n-gram hashing encoder."""
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        norm_text = re.sub(r"\s+", " ", text.lower()).strip()
        grams: list[str] = []
        grams.extend(tokenize_mixed(norm_text))
        for n in (2, 3):
            for i in range(max(0, len(norm_text) - n + 1)):
                grams.append(norm_text[i : i + n])
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            mat[row_idx, bucket] += sign
        norm = float(np.linalg.norm(mat[row_idx]))
        if norm > 0.0:
            mat[row_idx] /= norm
    return mat


class HybridRetriever:
    """BM25 + dense cosine similarity fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self._embed_fn = embed_fn or default_hashed_embed

        self._doc_tokens: list[list[str]] = [
            tokenize_mixed(f"{c.title} {c.text}") for c in self.chunks
        ]
        self._doc_tf: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / max(1, len(self._doc_lens))

        df: Counter[str] = Counter()
        for tf in self._doc_tf:
            for term in tf:
                df[term] += 1
        n_docs = len(self.chunks)
        self._idf: dict[str, float] = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

        if self.chunks:
            corpus_texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            emb = np.asarray(self._embed_fn(corpus_texts), dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms > 0.0, norms, 1.0)
            self._doc_emb = emb / norms
        else:
            self._doc_emb = np.zeros((0, 256), dtype=np.float32)

    def _bm25_scores(self, query: str) -> list[float]:
        q_tokens = tokenize_mixed(query)
        scores = [0.0] * len(self.chunks)
        if not q_tokens or not self.chunks:
            return scores
        for i, tf in enumerate(self._doc_tf):
            dl = self._doc_lens[i]
            s = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self._avgdl)))
                s += idf * (freq * (self.k1 + 1.0)) / max(1e-6, denom)
            scores[i] = s
        return scores

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        """Retrieve top-k chunks ranked by BM25 + dense vector RRF fusion."""
        if not self.chunks or not query.strip():
            return []
        bm25 = self._bm25_scores(query)
        bm25_order = sorted(range(len(self.chunks)), key=lambda i: (-bm25[i], i))

        q_emb = np.asarray(self._embed_fn([query]), dtype=np.float32)[0]
        q_norm = float(np.linalg.norm(q_emb))
        if q_norm > 0.0:
            q_emb = q_emb / q_norm
        dense = (self._doc_emb @ q_emb).tolist()
        dense_order = sorted(range(len(self.chunks)), key=lambda i: (-dense[i], i))

        rrf_scores = [0.0] * len(self.chunks)
        for rank, doc_idx in enumerate(bm25_order, start=1):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank)
        for rank, doc_idx in enumerate(dense_order, start=1):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank)

        fused_order = sorted(range(len(self.chunks)), key=lambda i: (-rrf_scores[i], i))
        results: list[dict[str, object]] = []
        for idx in fused_order[: max(1, top_k)]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "snippet": chunk.text,
                    "score": round(float(rrf_scores[idx]), 6),
                }
            )
        return results
