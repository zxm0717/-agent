"""
长期记忆 — FAISS 向量库 + 语义分块 + OpenAI Embedding（可选）

索引构建链路：
  raw files → DocParser → SemanticChunker → Chunk[] → FAISS + 元数据

Usage:
    ltm = LongTermMemory(embedding_mode="hash")        # demo 模式，零依赖
    ltm = LongTermMemory(embedding_mode="openai")      # 语义向量，需 API key
    ltm.load_knowledge_base("data/knowledge_base/")    # 扫描目录，自动分块入索引
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

from memory.doc_parser import DocParser, ParsedDocument, StructureElement


# ─── Chunk 数据类 ───

@dataclass
class Chunk:
    """索引中的最小检索单元"""
    chunk_id: str
    content: str
    chunk_type: str = "paragraph"     # paragraph | list_item | heading | section | sku_fact | policy_rule
    token_count: int = 0
    char_count: int = 0
    source: str = ""
    chunk_index: int = 0
    parent_id: str | None = None
    section_title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "parent_id": self.parent_id,
            "section_title": self.section_title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


# ─── SemanticChunker ───

class SemanticChunker:
    """
    Token 级语义分块引擎。

    流程：
    1. 遍历 StructureElement，按段落边界合并
    2. tiktoken 计数，超过 chunk_size 时切分
    3. 单段超限时按句子边界（。！？.!?）切分
    4. 每 3-5 个子 chunk 生成一个 parent（章节级 chunk）
    5. 每个 chunk 尾部 128 token 作为下个 chunk 前缀（overlap）
    """

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 128,
        min_chunk_tokens: int = 128,
        parent_group_size: int = 4,
        tokenizer_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens
        self.min_chunk = min_chunk_tokens
        self.parent_group_size = parent_group_size
        self.tokenizer_name = tokenizer_name
        self._enc = None

    def _get_tokenizer(self):
        if self._enc is None and tiktoken is not None:
            try:
                self._enc = tiktoken.get_encoding(self.tokenizer_name)
            except Exception:
                self._enc = None
        return self._enc

    def _token_count(self, text: str) -> int:
        enc = self._get_tokenizer()
        if enc is not None:
            return len(enc.encode(text))
        # 回退：中文 ~1.5 字符/token，英文 ~4 字符/token
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _split_sentences(self, text: str) -> list[str]:
        """按句子边界切分"""
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, doc: ParsedDocument, source: str = "") -> list[Chunk]:
        """
        对已解析文档进行分块。

        Args:
            doc: DocParser.parse() 的产物
            source: 文件名（用于 Chunk.source）

        Returns:
            Chunk 列表（包含 child 和 parent）
        """
        if not doc.structure:
            # 无结构时退化为单 chunk
            content = doc.cleaned_text or doc.raw_text
            if not content.strip():
                return []
            return [Chunk(
                chunk_id=_make_chunk_id(content),
                content=content,
                chunk_type="paragraph",
                token_count=self._token_count(content),
                char_count=len(content),
                source=source,
                chunk_index=0,
            )]

        # Step 1: 遍历结构元素，合并为 token 窗口内的 chunk
        child_chunks: list[Chunk] = []
        current_buffer: list[StructureElement] = []
        current_tokens = 0
        section_title = ""

        for elem in doc.structure:
            if elem.elem_type == "heading":
                # 标题 → 输出当前 buffer，开始新 section
                if current_buffer:
                    child_chunks.extend(self._flush_buffer(
                        current_buffer, source, len(child_chunks), section_title,
                    ))
                    current_buffer = []
                    current_tokens = 0
                section_title = elem.content
                # 标题本身也作为一个 chunk
                child_chunks.append(Chunk(
                    chunk_id=_make_chunk_id(elem.content),
                    content=elem.content,
                    chunk_type="heading",
                    token_count=self._token_count(elem.content),
                    char_count=len(elem.content),
                    source=source,
                    chunk_index=len(child_chunks),
                    section_title=section_title,
                ))
                continue

            elem_tokens = self._token_count(elem.content)

            if current_tokens + elem_tokens <= self.chunk_size:
                current_buffer.append(elem)
                current_tokens += elem_tokens
            else:
                # 先 flush 当前 buffer
                if current_buffer:
                    child_chunks.extend(self._flush_buffer(
                        current_buffer, source, len(child_chunks), section_title,
                    ))
                    current_buffer = []
                    current_tokens = 0

                # 单个元素超限 → 按句子切
                if elem_tokens > self.chunk_size:
                    sentences = self._split_sentences(elem.content)
                    for sent in sentences:
                        sent_tokens = self._token_count(sent)
                        if current_tokens + sent_tokens <= self.chunk_size:
                            # 包装为伪元素
                            current_buffer.append(StructureElement(
                                elem_type=elem.elem_type,
                                content=sent,
                                position=elem.position,
                            ))
                            current_tokens += sent_tokens
                        else:
                            if current_buffer:
                                child_chunks.extend(self._flush_buffer(
                                    current_buffer, source, len(child_chunks), section_title,
                                ))
                            # 单句太长 → 强制分块
                            if sent_tokens > self.chunk_size:
                                child_chunks.append(self._force_chunk(
                                    sent, elem.elem_type, source, len(child_chunks), section_title, sent_tokens,
                                ))
                                current_buffer = []
                                current_tokens = 0
                            else:
                                current_buffer = [StructureElement(
                                    elem_type=elem.elem_type,
                                    content=sent,
                                    position=elem.position,
                                )]
                                current_tokens = sent_tokens
                else:
                    current_buffer.append(elem)
                    current_tokens = elem_tokens

        # flush 最后的 buffer
        if current_buffer:
            child_chunks.extend(self._flush_buffer(
                current_buffer, source, len(child_chunks), section_title,
            ))

        # Step 2: 最小 chunk 合并
        child_chunks = self._merge_small_chunks(child_chunks)

        # Step 3: 生成 parent chunk
        parents = self._generate_parents(child_chunks, source)

        # Step 4: 设置 parent_id 关联
        for child in child_chunks:
            for parent in parents:
                if (parent.chunk_index <= child.chunk_index <
                        parent.chunk_index + self.parent_group_size):
                    child.parent_id = parent.chunk_id
                    break

        return child_chunks + parents

    def _flush_buffer(
        self, buffer: list[StructureElement], source: str,
        start_index: int, section_title: str,
    ) -> list[Chunk]:
        """将缓冲区合并为一个 chunk"""
        if not buffer:
            return []

        content = " ".join(e.content for e in buffer)
        # 判断 chunk_type
        types = {e.elem_type for e in buffer}
        if len(types) == 1:
            chunk_type = next(iter(types))
        else:
            chunk_type = "paragraph"

        return [Chunk(
            chunk_id=_make_chunk_id(content),
            content=content,
            chunk_type=chunk_type,
            token_count=self._token_count(content),
            char_count=len(content),
            source=source,
            chunk_index=start_index,
            section_title=section_title,
        )]

    def _force_chunk(
        self, text: str, elem_type: str, source: str,
        index: int, section_title: str, token_count: int,
    ) -> Chunk:
        """单句太长时的硬切"""
        # 按 chunk_size 近似字符数切分
        approx_chars = int(self.chunk_size * 1.5)  # 中文 ~1.5 char/token
        content = text[:approx_chars]
        return Chunk(
            chunk_id=_make_chunk_id(content),
            content=content,
            chunk_type=elem_type,
            token_count=token_count,
            char_count=len(content),
            source=source,
            chunk_index=index,
            section_title=section_title,
        )

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """将过短的 chunk 与相邻合并"""
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk.token_count >= self.min_chunk or chunk.chunk_type == "heading":
                merged.append(chunk)
                i += 1
            elif i > 0 and merged[-1].token_count + chunk.token_count <= self.chunk_size:
                # 合并到前一个
                prev = merged[-1]
                prev.content = prev.content + " " + chunk.content
                prev.token_count = self._token_count(prev.content)
                prev.char_count = len(prev.content)
                i += 1
            elif i + 1 < len(chunks) and chunk.token_count + chunks[i + 1].token_count <= self.chunk_size:
                # 与后一个合并
                next_chunk = chunks[i + 1]
                chunk.content = chunk.content + " " + next_chunk.content
                chunk.token_count = self._token_count(chunk.content)
                chunk.char_count = len(chunk.content)
                merged.append(chunk)
                i += 2
            else:
                merged.append(chunk)
                i += 1

        return merged

    def _generate_parents(self, children: list[Chunk], source: str) -> list[Chunk]:
        """生成 parent（section 级粗粒度 chunk）"""
        parents: list[Chunk] = []
        group_size = self.parent_group_size

        for i in range(0, len(children), group_size):
            group = children[i:i + group_size]
            # 合并子 chunk 内容
            parent_content = "\n".join(c.content for c in group)
            section_title = group[0].section_title if group else ""

            parents.append(Chunk(
                chunk_id=_make_chunk_id(parent_content + "_parent"),
                content=parent_content,
                chunk_type="section",
                token_count=self._token_count(parent_content),
                char_count=len(parent_content),
                source=source,
                chunk_index=i,  # 等于第一个子 chunk 的 index
                section_title=section_title,
                metadata={"child_count": len(group)},
            ))

        return parents


# ─── 辅助 ───

def _make_chunk_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:12]


import re  # SemanticChunker._split_sentences 用到


# ─── LongTermMemory ───

class LongTermMemory:
    """
    长期记忆：FAISS 向量索引 + 语义分块。

    embedding_mode:
        "hash"   — 确定性随机向量（demo / 无 API key 时默认）
        "openai" — OpenAI text-embedding-3-small 语义向量

    Usage:
        ltm = LongTermMemory(embedding_mode="openai")
        ltm.load_knowledge_base("data/knowledge_base/")
        results = ltm.search("X1充电器", top_k=5)
    """

    def __init__(
        self,
        index_path: str = "./vector_store/faiss_index",
        embedding_dim: int = 1536,
        embedding_mode: str = "hash",
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 128,
    ):
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode

        self.chunker = SemanticChunker(
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

        self.chunks: list[Chunk] = []
        self._index = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._init_index()

    def _init_index(self):
        """初始化或加载 FAISS 索引"""
        if faiss is None:
            self._index = None
            return

        chunks_path = self.index_path.with_suffix(".chunks.pkl")
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                if chunks_path.exists():
                    with open(chunks_path, "rb") as f:
                        self.chunks = pickle.load(f)
            except Exception:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    # ─── Embedding ───

    def _embed(self, text: str) -> np.ndarray:
        if self.embedding_mode == "openai":
            return self._openai_embed(text)
        else:
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        """确定性随机向量（demo 用，无语义）"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % (2 ** 32))
        vec = np.random.randn(self.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def _openai_embed(self, text: str) -> np.ndarray:
        """OpenAI Embedding API（语义向量，带缓存）"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            self._embedding_cache[text] = vec
            return vec
        except Exception:
            # API 调用失败，回退到 hash
            return self._hash_embed(text)

    # ─── 索引构建 ───

    def add_chunk(self, chunk: Chunk):
        """将单个 Chunk 添加进向量索引"""
        self.chunks.append(chunk)

        if self._index is not None:
            vec = self._embed(chunk.content)
            self._index.add(vec.reshape(1, -1))

    def remove_chunks_by_source(self, source: str) -> int:
        """
        按 source（文件名或结构化虚拟路径）删除所有关联 chunk。
        因为 FAISS IndexFlatIP 不支持单条删除，所以删除后需要调用
        rebuild_faiss_index() 重建向量索引。

        Returns:
            删除的 chunk 数量
        """
        removed_count = 0
        removed_ids: set[str] = set()

        # 先标记要删的 chunk_id（含被删 chunk 作为 parent 的子 chunk）
        for chunk in self.chunks:
            if chunk.source == source:
                removed_ids.add(chunk.chunk_id)
                removed_count += 1

        # 清理 parent_id 引用（指向被删 chunk 的子 chunk 解绑）
        for chunk in self.chunks:
            if chunk.parent_id in removed_ids:
                chunk.parent_id = None

        # 从列表移除
        self.chunks = [c for c in self.chunks if c.source != source]

        # 删除后重建 FAISS 索引（IndexFlatIP 无 remove 接口）
        if removed_count > 0 and self._index is not None:
            self.rebuild_faiss_index()

        return removed_count

    def rebuild_faiss_index(self):
        """从当前 chunks 列表全量重建 FAISS 索引（用于增量删除后的恢复）"""
        if faiss is None:
            return

        self._index = faiss.IndexFlatIP(self.embedding_dim)

        for chunk in self.chunks:
            vec = self._embed(chunk.content)
            self._index.add(vec.reshape(1, -1))

    def load_single_file(
        self,
        file_path: str | Path,
        kb_dir: str | Path,
        ocr_backend=None,
    ) -> int:
        """
        解析并索引单个文件（用于增量新增/修改）。
        先删除该 source 的旧 chunk，再解析 + 添加。

        Args:
            file_path: 绝对或相对文件路径
            kb_dir: 知识库根目录（用于计算相对路径）
            ocr_backend: OCRBackend 或 None

        Returns:
            新增的 chunk 数量
        """
        import os as _os

        file_path = Path(file_path)
        kb_dir = Path(kb_dir)

        # 计算相对路径作为 source 标识
        try:
            source = str(file_path.relative_to(kb_dir)).replace("\\", "/")
        except ValueError:
            source = file_path.name

        ext = file_path.suffix.lower()
        parser = DocParser(ocr_backend=ocr_backend)

        if ext not in parser.supported_extensions:
            return 0

        doc = parser.parse(str(file_path))

        if doc.parse_warnings:
            for w in doc.parse_warnings:
                print(f"  [DocParser] {w}")

        if not doc.cleaned_text and not doc.raw_text:
            return 0

        # 删旧
        self.remove_chunks_by_source(source)

        # 解析 + 分块 + 添加
        chunks = self.chunker.chunk(doc, source=source)

        for chunk in chunks:
            self.add_chunk(chunk)

        return len(chunks)

    def load_knowledge_base(
        self,
        kb_dir: str,
        ocr_backend=None,
    ) -> int:
        """
        从目录加载知识库，自动分块 + 索引（全量重建）。

        Args:
            kb_dir: 知识库目录路径
            ocr_backend: OCRBackend 或 None

        Returns:
            产出的 chunk 总数
        """
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return 0

        parser = DocParser(ocr_backend=ocr_backend)
        total_chunks = 0

        for file_path in sorted(kb_path.rglob("*")):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in parser.supported_extensions:
                continue

            doc = parser.parse(str(file_path))

            if doc.parse_warnings:
                for w in doc.parse_warnings:
                    print(f"  [DocParser] {w}")

            if not doc.cleaned_text and not doc.raw_text:
                continue

            # source 用相对路径（相对于 kb_dir.parent），与 IndexManifest.scan_dir 一致
            try:
                source = str(file_path.relative_to(kb_path.parent)).replace("\\", "/")
            except ValueError:
                source = file_path.name

            chunks = self.chunker.chunk(doc, source=source)

            for chunk in chunks:
                self.add_chunk(chunk)

            total_chunks += len(chunks)

        self.save()
        return total_chunks

    def load_knowledge_base_incremental(
        self,
        kb_dir: str,
        manifest=None,
        ocr_backend=None,
    ) -> dict:
        """
        增量加载知识库：对比文件 Hash，仅处理变更文件。
        首次运行（无 manifest 或为空）自动退化为全量加载。

        Args:
            kb_dir: 知识库目录路径
            manifest: IndexManifest 实例或 None
            ocr_backend: OCRBackend 或 None

        Returns:
            {
                "mode": "full" | "incremental",
                "total_chunks": int,
                "added": int, "modified": int, "deleted": int,
                "files_added": [...], "files_modified": [...], "files_deleted": [...],
            }
        """
        from memory.index_manifest import IndexManifest

        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return {"mode": "none", "total_chunks": 0, "added": 0, "modified": 0, "deleted": 0}

        # 无 manifest 或首次运行 → 全量
        if manifest is None or manifest.is_empty:
            total = self.load_knowledge_base(kb_dir, ocr_backend=ocr_backend)
            return {
                "mode": "full",
                "total_chunks": total,
                "added": total, "modified": 0, "deleted": 0,
                "files_added": [], "files_modified": [], "files_deleted": [],
            }

        # 扫描当前文件 Hash
        current_hashes = IndexManifest.scan_dir(kb_dir)

        # Diff
        diff = manifest.diff(current_hashes)

        # 无变更 → 直接返回（但 chunks 仍然需要从磁盘加载）
        if not diff.added and not diff.modified and not diff.deleted:
            return {
                "mode": "incremental",
                "total_chunks": len(self.chunks),
                "added": 0, "modified": 0, "deleted": 0,
                "files_added": [], "files_modified": [], "files_deleted": [],
            }

        print(f"[index] 增量索引: +{len(diff.added)} 新增 / "
              f"~{len(diff.modified)} 修改 / -{len(diff.deleted)} 删除")

        added_count = 0
        modified_count = 0
        deleted_count = 0

        # 处理删除
        for rel_path in diff.deleted:
            removed = self.remove_chunks_by_source(rel_path)
            deleted_count += removed
            manifest.files.pop(rel_path, None)
            print(f"  [del] {rel_path} → -{removed} chunks")

        # 处理修改（先删后加）
        for rel_path in diff.modified:
            abs_path = kb_path.parent / rel_path
            if not abs_path.exists():
                continue

            removed = self.remove_chunks_by_source(rel_path)
            new_count = self.load_single_file(abs_path, kb_path.parent, ocr_backend=ocr_backend)
            modified_count += new_count
            manifest.files[rel_path] = current_hashes[rel_path]
            print(f"  [mod] {rel_path} → -{removed} +{new_count} chunks")

        # 处理新增
        for rel_path in diff.added:
            abs_path = kb_path.parent / rel_path
            if not abs_path.exists():
                continue

            new_count = self.load_single_file(abs_path, kb_path.parent, ocr_backend=ocr_backend)
            added_count += new_count
            manifest.files[rel_path] = current_hashes[rel_path]
            print(f"  [add] {rel_path} → +{new_count} chunks")

        # 更新 manifest 中的结构化条目不受文件扫描影响，手动保留
        # （结构化 key 以 __structured__/ 开头，不在 scan_dir 范围内）

        # 保存
        self.save()
        manifest.save()

        return {
            "mode": "incremental",
            "total_chunks": len(self.chunks),
            "added": added_count,
            "modified": modified_count,
            "deleted": deleted_count,
            "files_added": diff.added,
            "files_modified": diff.modified,
            "files_deleted": diff.deleted,
        }

    # ─── 检索 ───

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义相似度检索"""
        if self._index is None or not self.chunks:
            return self._fallback_search(query, top_k)

        query_vec = self._embed(query).reshape(1, -1)
        scores, indices = self._index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            r = chunk.to_dict()
            r["score"] = float(score)
            # 如果有 parent，附带 parent 内容用于上下文扩展
            if chunk.parent_id:
                r["parent"] = self._get_parent_content(chunk.parent_id)
            results.append(r)

        return results

    def _get_parent_content(self, parent_id: str) -> str | None:
        for c in self.chunks:
            if c.chunk_id == parent_id:
                return c.content[:500]
        return None

    def _fallback_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """FAISS 不可用时的关键词回退搜索"""
        scored = []
        query_terms = set(query.lower().split())

        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                r = chunk.to_dict()
                r["score"] = float(score)
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ─── 持久化 ───

    def save(self):
        """持久化 FAISS 索引 + Chunk 列表到磁盘"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

        chunks_path = self.index_path.with_suffix(".chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def get_statistics(self) -> dict[str, Any]:
        """索引统计"""
        type_counts: dict[str, int] = {}
        total_tokens = 0
        for c in self.chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
            total_tokens += c.token_count

        return {
            "total_chunks": len(self.chunks),
            "chunk_types": type_counts,
            "total_tokens": total_tokens,
            "embedding_mode": self.embedding_mode,
            "index_size": self._index.ntotal if self._index else 0,
        }
