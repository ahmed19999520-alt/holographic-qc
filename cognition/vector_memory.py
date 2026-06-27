from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class MemoryConfig:
    embedding_dim: int = 256
    episodic_capacity: int = 10000
    semantic_capacity: int = 5000
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    consolidation_interval: int = 100
    forgetting_rate: float = 0.001


@dataclass
class MemoryTrace:
    embedding: np.ndarray
    content: Any
    timestamp: float
    access_count: int = 0
    importance: float = 1.0
    episode_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def strength(self, current_time: float, decay_rate: float = 0.001) -> float:
        age = current_time - self.timestamp
        return self.importance * np.exp(-decay_rate * age) * np.log1p(self.access_count)


class EpisodicMemory:
    def __init__(self, config: MemoryConfig):
        self.cfg = config
        self._traces: List[MemoryTrace] = []
        self._embeddings: Optional[np.ndarray] = None
        self._episode_counter: int = 0
        self._consolidation_counter: int = 0

    def encode(self, content: Any, embedding: np.ndarray, importance: float = 1.0) -> str:
        episode_id = f"ep_{self._episode_counter:06d}"
        self._episode_counter += 1
        trace = MemoryTrace(
            embedding=embedding.copy() / (np.linalg.norm(embedding) + 1e-10),
            content=content,
            timestamp=time.perf_counter(),
            importance=importance,
            episode_id=episode_id,
        )
        self._traces.append(trace)
        if len(self._traces) > self.cfg.episodic_capacity:
            self._forget_weakest()
        self._embeddings = None
        self._consolidation_counter += 1
        if self._consolidation_counter % self.cfg.consolidation_interval == 0:
            self._consolidate()
        return episode_id

    def _forget_weakest(self):
        now = time.perf_counter()
        strengths = [t.strength(now, self.cfg.forgetting_rate) for t in self._traces]
        weakest_idx = int(np.argmin(strengths))
        self._traces.pop(weakest_idx)

    def _consolidate(self):
        now = time.perf_counter()
        threshold = 0.01
        self._traces = [
            t for t in self._traces
            if t.strength(now, self.cfg.forgetting_rate) > threshold
        ]
        self._embeddings = None

    def _rebuild_index(self):
        if not self._traces:
            self._embeddings = np.zeros((0, self.cfg.embedding_dim))
            return
        embs = np.stack([t.embedding for t in self._traces])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = embs / norms

    def retrieve(
        self, query: np.ndarray, k: int = None, threshold: float = None
    ) -> List[Tuple[MemoryTrace, float]]:
        if not self._traces:
            return []
        if self._embeddings is None or len(self._embeddings) != len(self._traces):
            self._rebuild_index()
        k = k or self.cfg.retrieval_k
        threshold = threshold or self.cfg.similarity_threshold
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        similarities = self._embeddings @ q_norm
        now = time.perf_counter()
        strengths = np.array([t.strength(now, self.cfg.forgetting_rate) for t in self._traces])
        combined_scores = similarities * (0.7 + 0.3 * np.tanh(strengths))
        top_indices = np.argsort(combined_scores)[::-1][:k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                self._traces[idx].access_count += 1
                results.append((self._traces[idx], sim))
        return results

    def temporal_sequence(self, episode_ids: List[str]) -> List[MemoryTrace]:
        id_set = set(episode_ids)
        matches = [t for t in self._traces if t.episode_id in id_set]
        return sorted(matches, key=lambda t: t.timestamp)

    def statistics(self) -> dict:
        now = time.perf_counter()
        if not self._traces:
            return {"n_traces": 0, "mean_strength": 0.0, "capacity_used": 0.0}
        strengths = [t.strength(now, self.cfg.forgetting_rate) for t in self._traces]
        return {
            "n_traces": len(self._traces),
            "mean_strength": float(np.mean(strengths)),
            "max_strength": float(np.max(strengths)),
            "capacity_used": len(self._traces) / self.cfg.episodic_capacity,
            "mean_access_count": float(np.mean([t.access_count for t in self._traces])),
        }


class SemanticMemory:
    def __init__(self, config: MemoryConfig):
        self.cfg = config
        self._concepts: Dict[str, MemoryTrace] = {}
        self._cluster_centroids: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None

    def store_concept(
        self, concept_name: str, embedding: np.ndarray,
        properties: Dict = None, importance: float = 1.0
    ):
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        if concept_name in self._concepts:
            existing = self._concepts[concept_name]
            alpha = 0.3
            merged_emb = (1 - alpha) * existing.embedding + alpha * norm_emb
            merged_emb /= (np.linalg.norm(merged_emb) + 1e-10)
            existing.embedding = merged_emb
            existing.access_count += 1
            if properties:
                existing.metadata.update(properties)
        else:
            trace = MemoryTrace(
                embedding=norm_emb,
                content=concept_name,
                timestamp=time.perf_counter(),
                importance=importance,
                metadata=properties or {},
            )
            self._concepts[concept_name] = trace
        self._cluster_centroids = None

    def retrieve_concept(self, query: np.ndarray, k: int = None) -> List[Tuple[str, float]]:
        if not self._concepts:
            return []
        k = k or self.cfg.retrieval_k
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        results = []
        for name, trace in self._concepts.items():
            sim = float(np.dot(trace.embedding, q_norm))
            if sim >= self.cfg.similarity_threshold:
                trace.access_count += 1
                results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def analogy(
        self, A: str, B: str, C: str, k: int = 3
    ) -> List[Tuple[str, float]]:
        if not all(x in self._concepts for x in [A, B, C]):
            return []
        emb_A = self._concepts[A].embedding
        emb_B = self._concepts[B].embedding
        emb_C = self._concepts[C].embedding
        query = emb_B - emb_A + emb_C
        exclude = {A, B, C}
        results = []
        for name, trace in self._concepts.items():
            if name in exclude:
                continue
            sim = float(np.dot(trace.embedding, query / (np.linalg.norm(query) + 1e-10)))
            results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def cluster(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        if len(self._concepts) < n_clusters:
            return {0: list(self._concepts.keys())}
        from sklearn.cluster import KMeans
        names = list(self._concepts.keys())
        embs = np.stack([self._concepts[n].embedding for n in names])
        n_clusters = min(n_clusters, len(names))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(embs)
        self._cluster_centroids = km.cluster_centers_
        self._cluster_labels = labels
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for name, label in zip(names, labels):
            clusters[label].append(name)
        return clusters

    def statistics(self) -> dict:
        return {
            "n_concepts": len(self._concepts),
            "capacity_used": len(self._concepts) / self.cfg.semantic_capacity,
            "mean_access_count": float(np.mean([t.access_count for t in self._concepts.values()])) if self._concepts else 0.0,
        }


class VectorMemory:
    def __init__(self, config: MemoryConfig):
        self.cfg = config
        self.episodic = EpisodicMemory(config)
        self.semantic = SemanticMemory(config)
        self._working_memory: deque = deque(maxlen=7)
        self._n_total_stores: int = 0
        self._n_total_retrievals: int = 0

    def store(
        self, content: Any, embedding: np.ndarray,
        memory_type: str = "episodic",
        concept_name: Optional[str] = None,
        importance: float = 1.0,
    ) -> Optional[str]:
        self._n_total_stores += 1
        self._working_memory.append((embedding.copy(), content))
        if memory_type == "episodic":
            return self.episodic.encode(content, embedding, importance)
        elif memory_type == "semantic" and concept_name:
            self.semantic.store_concept(
                concept_name, embedding,
                properties={"content": str(content)[:200]},
                importance=importance,
            )
            return concept_name
        return None

    def retrieve(
        self, query: np.ndarray, memory_type: str = "episodic", k: int = 5
    ) -> List[Tuple[Any, float]]:
        self._n_total_retrievals += 1
        if memory_type == "episodic":
            results = self.episodic.retrieve(query, k=k)
            return [(r.content, sim) for r, sim in results]
        elif memory_type == "semantic":
            return self.semantic.retrieve_concept(query, k=k)
        return []

    def working_memory_contents(self) -> List[Any]:
        return [item[1] for item in self._working_memory]

    def consolidate_working_to_long_term(self):
        for emb, content in self._working_memory:
            self.episodic.encode(content, emb, importance=0.5)
        self._working_memory.clear()

    def statistics(self) -> dict:
        return {
            "episodic": self.episodic.statistics(),
            "semantic": self.semantic.statistics(),
            "working_memory_size": len(self._working_memory),
            "total_stores": self._n_total_stores,
            "total_retrievals": self._n_total_retrievals,
        }