"""
索引清单 — 增量索引的核心：追踪每个文件的 Hash，检测 新增/修改/删除。

Workflow:
  1. startup → scan_dir() 计算当前所有文件 MD5
  2. load_manifest() 读取上次持久化的清单
  3. diff() 产出 added / modified / deleted 三组文件
  4. 仅对变更文件重建索引
  5. save_manifest() 更新清单到磁盘

注意：结构化编码的 chunk（SKU/价格/政策）用虚拟路径标识，
      其在 manifest 中的 key 形如 "__structured__/sku_x1_phone"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IndexManifest:
    """
    文件 → Hash 映射，JSON 持久化。

    结构示例:
    {
      "files": {
        "data/knowledge_base/products/smartphone_x1.txt": "a1b2c3...",
        "__structured__/sku_x1_standard": "e4f5g6...",
      },
      "meta": {
        "last_indexed_at": "2026-06-19T10:00:00",
        "total_chunks": 25,
        "embedding_mode": "hash"
      }
    }
    """

    persist_path: Path
    files: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ─── Hash 计算 ───

    @staticmethod
    def file_hash(filepath: str | Path) -> str:
        """计算单个文件的 MD5 (8KB 分块，适合大文件)"""
        filepath = Path(filepath)
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def content_hash(content: str) -> str:
        """计算字符串内容的 MD5 — 用于结构化数据 chunk"""
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def scan_dir(kb_dir: str | Path) -> dict[str, str]:
        """
        扫描目录，返回 {相对路径: MD5} 映射。
        只扫描支持的文件类型。
        """
        kb_dir = Path(kb_dir)
        if not kb_dir.exists():
            return {}

        supported = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        hashes: dict[str, str] = {}

        for filepath in sorted(kb_dir.rglob("*")):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in supported:
                continue

            rel_path = str(filepath.relative_to(kb_dir.parent)).replace("\\", "/")
            try:
                hashes[rel_path] = IndexManifest.file_hash(filepath)
            except Exception:
                continue

        return hashes

    # ─── Diff ───

    @dataclass
    class Diff:
        """增量变更集"""
        added: list[str] = field(default_factory=list)      # 新文件路径
        modified: list[str] = field(default_factory=list)    # 修改的文件路径
        deleted: list[str] = field(default_factory=list)     # 删除的文件路径（含结构化条目）

    def diff(self, current_hashes: dict[str, str]) -> Diff:
        """
        对比当前文件 Hash 与已持久化的清单，产出变更集。
        """
        result = self.Diff()

        old_set = set(self.files.keys())
        new_set = set(current_hashes.keys())

        # 删除：旧有、新无
        result.deleted = sorted(old_set - new_set)

        for path in sorted(new_set):
            new_hash = current_hashes[path]
            old_hash = self.files.get(path)
            if old_hash is None:
                result.added.append(path)
            elif new_hash != old_hash:
                result.modified.append(path)

        return result

    @property
    def is_empty(self) -> bool:
        """从未索引过（无持久化文件 或 文件为空）"""
        return len(self.files) == 0

    # ─── 结构化数据支持 ───

    def register_structured(self, key: str, content: str):
        """
        注册结构化编码产物的虚拟路径。
        key: "__structured__/sku_x1_standard"
        """
        self.files[key] = self.content_hash(content)

    # ─── 持久化 ───

    def save(self):
        """写入 JSON 到磁盘"""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "files": self.files,
            "meta": self.meta,
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> IndexManifest:
        """从 JSON 加载清单，不存在则返回空"""
        path = Path(path)
        manifest = cls(persist_path=path)

        if not path.exists():
            return manifest

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest.files = data.get("files", {})
            manifest.meta = data.get("meta", {})
        except (json.JSONDecodeError, KeyError):
            pass

        return manifest

    def reset(self):
        """清空清单（用于强制全量重建）"""
        self.files.clear()
        self.meta.clear()
