"""
BM25 稀疏关键词索引 — 与 FAISS 稠密向量索引互补

特点：
- 基于 jieba 中文分词构建倒排索引
- BM25 评分：有利于精确匹配专有名词、型号、SKU
- 独立于 FAISS 运行，查询融合在外部处理
- 支持 pickle 序列化持久化

Usage:
    si = SparseIndex()
    si.index_chunk(chunk)
    results = si.search("X1充电器 GN65W", top_k=10)
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import jieba
except ImportError:
    jieba = None


# ─── 停用词（最小集） ───

STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "而且", "或", "或者",
    "的", "地", "得", "之", "与", "及", "以及", "并", "而", "以",
    "可以", "能", "能够", "会", "应该", "需要", "必须",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "this", "that", "these", "those", "it", "its",
}


class SparseIndex:
    """
    BM25 倒排索引。

    不参与查询融合逻辑，只负责：
    1. 建倒排索引（index_chunk）
    2. BM25 检索（search）
    3. 序列化/反序列化
    """

    def __init__(self):
        # 倒排表: term → {chunk_id: term_frequency_in_chunk}
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        # 文档频率: term → 包含该 term 的 chunk 数
        self._doc_freq: dict[str, int] = defaultdict(int)
        # chunk 长度: chunk_id → 该 chunk 的 term 总数
        self._doc_lengths: dict[str, int] = {}
        # 原始文本存储（用于检索结果回显）
        self._chunks: dict[str, str] = {}
        # 总 chunk 数
        self._doc_count: int = 0
        # 平均 chunk 长度
        self._avg_doc_len: float = 0.0

    # ─── token 化 ───

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文分词 + 去停用词 + 保留长度 ≥ 2 的词"""
        if jieba is not None:
            words = list(jieba.cut(text))
            return [
                w.lower().strip()
                for w in words
                if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS
            ]

        # ── 无 jieba 回退：字符 n-gram ──
        import re
        tokens: list[str] = []

        # 1. 提取英文+数字单词（保留原样）
        alpha_words = re.findall(r"[a-zA-Z0-9]+", text)
        tokens.extend(w.lower() for w in alpha_words if len(w) >= 2)

        # 2. 对中文字符做 2-gram + 3-gram 滑动窗口
        chinese_chars = re.findall(r"[一-鿿]", text)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])          # 2-gram
        for i in range(len(chinese_chars) - 2):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1] + chinese_chars[i + 2])  # 3-gram

        # 3. 去停用词 + 去重
        return [
            t for t in tokens
            if t not in STOP_WORDS and len(t) >= 2
        ]

    # ─── 索引构建 ───

    def index_chunk(self, chunk: Any):
        """
        将 Chunk 加入倒排索引。

        Args:
            chunk: Chunk 实例（需要有 chunk_id, content 属性）
        """
        chunk_id = chunk.chunk_id
        content = chunk.content

        self._chunks[chunk_id] = content
        self._doc_count += 1

        terms = self._tokenize(content)
        term_counts: dict[str, int] = defaultdict(int)
        for t in terms:
            term_counts[t] += 1

        self._doc_lengths[chunk_id] = sum(term_counts.values())

        for term, tf in term_counts.items():
            self._inverted[term][chunk_id] = tf
            self._doc_freq[term] += 1

        # 更新平均长度
        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def index_chunks(self, chunks: list[Any]):
        """批量索引 Chunk 列表"""
        for chunk in chunks:
            self.index_chunk(chunk)

    def remove_chunk(self, chunk_id: str):
        """从索引中移除一个 chunk"""
        if chunk_id not in self._chunks:
            return

        content = self._chunks.pop(chunk_id, "")
        terms = self._tokenize(content)
        term_counts: dict[str, int] = defaultdict(int)
        for t in terms:
            term_counts[t] += 1

        for term in term_counts:
            if term in self._inverted:
                self._inverted[term].pop(chunk_id, None)
                if not self._inverted[term]:
                    del self._inverted[term]
                self._doc_freq[term] = max(0, self._doc_freq.get(term, 1) - 1)

        self._doc_lengths.pop(chunk_id, None)
        self._doc_count = max(0, self._doc_count - 1)

        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    # ─── 检索 ───

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        BM25 检索。

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            [(chunk_id, bm25_score), ...]  按分数降序排列
        """
        if not query or not self._chunks:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self._inverted:
                continue

            # IDF
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))

            for chunk_id, tf in self._inverted[term].items():
                doc_len = self._doc_lengths.get(chunk_id, 0)
                # BM25 TF component
                k1, b = 1.5, 0.75
                tf_component = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1))
                )
                scores[chunk_id] += idf * tf_component

        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_with_content(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """检索并返回带内容的结果"""
        ranked = self.search(query, top_k)
        return [
            {
                "chunk_id": chunk_id,
                "score": score,
                "content": self._chunks.get(chunk_id, "")[:500],
            }
            for chunk_id, score in ranked
        ]

    # ─── 持久化 ───

    def save(self, path: str):
        """pickle 序列化保存"""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "inverted": dict(self._inverted),
            "doc_freq": dict(self._doc_freq),
            "doc_lengths": dict(self._doc_lengths),
            "chunks": dict(self._chunks),
            "doc_count": self._doc_count,
            "avg_doc_len": self._avg_doc_len,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> SparseIndex:
        """从 pickle 文件加载"""
        filepath = Path(path)
        if not filepath.exists():
            return cls()

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        si = cls()
        si._inverted = defaultdict(dict, data.get("inverted", {}))
        si._doc_freq = defaultdict(int, data.get("doc_freq", {}))
        si._doc_lengths = data.get("doc_lengths", {})
        si._chunks = data.get("chunks", {})
        si._doc_count = data.get("doc_count", 0)
        si._avg_doc_len = data.get("avg_doc_len", 0.0)

        return si

    # ─── 统计 ───

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_docs": self._doc_count,
            "total_terms": len(self._inverted),
            "avg_doc_length": round(self._avg_doc_len, 2),
            "top_terms": sorted(
                self._doc_freq.items(), key=lambda x: x[1], reverse=True,
            )[:20],
        }
