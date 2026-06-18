"""
文档解析器 — 多格式输入 → 统一文本 + 结构识别

支持的格式：
- .txt / .md ：直接读取，检测标题/段落/列表结构
- .pdf ：PyPDF2 提取文本，扫描页通过 OCR 后端处理
- .png/.jpg/.jpeg/.bmp ：通过 OCR 后端提取文字

噪音清洗：Unicode 规范化、多余空白压缩、控制字符移除。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StructureElement:
    """文档结构元素"""
    elem_type: str          # "heading" | "paragraph" | "list_item"
    content: str
    level: int = 0          # 标题层级 1-3，非标题为 0
    position: int = 0       # 在文档内的序号


@dataclass
class ParsedDocument:
    """解析后的文档"""
    raw_text: str
    cleaned_text: str
    structure: list[StructureElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)


class DocParser:
    """
    多格式文档解析器。

    按文件扩展名路由到对应的解析方法，统一输出 ParsedDocument。
    OCR 后端可选：不配则图片文件在扫描时被跳过。
    """

    # 无需 OCR 即可处理的格式
    TEXT_EXTENSIONS = {".txt", ".md"}
    # 需要 OCR 的格式
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    # PDF（混合：文本页 + 可能的扫描页）
    PDF_EXTENSIONS = {".pdf"}

    HEADING_PATTERNS = [
        # Markdown 风格
        (re.compile(r"^#{1,6}\s+(.+)"), lambda m: (len(m.group(0).split()[0]), m.group(1))),
        # 中文编号: 一、二、三、...
        (re.compile(r"^[一二三四五六七八九十]+[、．.]\s*(.*)"), lambda m: (2, m.group(0))),
        # 括号编号: （一）（二）
        (re.compile(r"^（[一二三四五六七八九十]+）\s*(.*)"), lambda m: (2, m.group(0))),
        # 数字编号+顿号: 1、2.、3)
        (re.compile(r"^(\d+)[、．.)]\s*(.*)"), lambda m: (2, m.group(0))),
    ]

    def __init__(self, ocr_backend=None):
        """
        Args:
            ocr_backend: OCRBackend 实例或 None。
                         None 时图片文件在 parse() 时返回空文档 + warning。
        """
        self.ocr = ocr_backend

    @property
    def supported_extensions(self) -> set[str]:
        """根据当前配置返回支持的文件扩展名"""
        exts = set(self.TEXT_EXTENSIONS) | set(self.PDF_EXTENSIONS)
        if self.ocr is not None:
            exts |= set(self.IMAGE_EXTENSIONS)
        return exts

    # ─── 主入口 ───

    def parse(self, filepath: str) -> ParsedDocument:
        """
        解析单个文件，按扩展名路由。

        Returns:
            ParsedDocument。不支持的格式返回带有 parse_warnings 的空文档。
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        if not path.exists():
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata={"source": path.name, "error": "file_not_found"},
                parse_warnings=[f"文件不存在: {filepath}"],
            )

        file_metadata = {
            "source": path.name,
            "filepath": str(path),
            "extension": ext,
            "file_size": path.stat().st_size,
        }

        if ext in self.TEXT_EXTENSIONS:
            return self._parse_text(path, file_metadata)

        elif ext in self.PDF_EXTENSIONS:
            return self._parse_pdf(path, file_metadata)

        elif ext in self.IMAGE_EXTENSIONS:
            if self.ocr is None:
                return ParsedDocument(
                    raw_text="", cleaned_text="",
                    metadata=file_metadata,
                    parse_warnings=[f"跳过图片文件 (未配置 OCR 后端): {path.name}"],
                )
            return self._parse_image(path, file_metadata)

        else:
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata=file_metadata,
                parse_warnings=[f"不支持的文件格式: {ext}"],
            )

    # ─── 文本文件解析 ───

    def _parse_text(self, path: Path, metadata: dict) -> ParsedDocument:
        """解析 txt/md 文件"""
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = path.read_text(encoding="gbk")
            except Exception:
                return ParsedDocument(
                    raw_text="", cleaned_text="",
                    metadata=metadata,
                    parse_warnings=[f"编码读取失败: {path.name}"],
                )

        cleaned = self._clean(raw_text)
        structure = self._detect_structure(cleaned)

        return ParsedDocument(
            raw_text=raw_text,
            cleaned_text=cleaned,
            structure=structure,
            metadata=metadata,
        )

    # ─── PDF 解析 ───

    def _parse_pdf(self, path: Path, metadata: dict) -> ParsedDocument:
        """解析 PDF 文件：优先用 PyPDF2 提取文本，空页尝试 OCR"""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata=metadata,
                parse_warnings=["PyPDF2 未安装，无法解析 PDF"],
            )

        try:
            reader = PdfReader(str(path))
            pages_text: list[str] = []
            ocr_pages = 0

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if not page_text.strip():
                    # 空页可能是扫描件，尝试 OCR
                    if self.ocr is not None:
                        # 尝试将页面渲染为图片后 OCR
                        # PyPDF2 本身不能渲染，这里只做标记
                        # 生产环境可用 pdf2image 将 PDF 页转为图片再 OCR
                        metadata.setdefault("ocr_needed_pages", []).append(i + 1)
                        ocr_pages += 1
                        continue
                    else:
                        metadata.setdefault("skipped_empty_pages", []).append(i + 1)
                        continue

                pages_text.append(page_text)

            raw_text = "\n\n".join(pages_text)
            warnings = []
            if ocr_pages > 0:
                warnings.append(f"{path.name}: {ocr_pages} 页需要 OCR，当前未处理")
            if metadata.get("skipped_empty_pages"):
                warnings.append(
                    f"{path.name}: {len(metadata['skipped_empty_pages'])} 空白页跳过"
                )

            cleaned = self._clean(raw_text) if raw_text.strip() else ""

            return ParsedDocument(
                raw_text=raw_text,
                cleaned_text=cleaned,
                structure=self._detect_structure(cleaned) if cleaned else [],
                metadata=metadata,
                parse_warnings=warnings,
            )

        except Exception as e:
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata=metadata,
                parse_warnings=[f"PDF 解析失败: {str(e)}"],
            )

    # ─── 图片解析 ───

    def _parse_image(self, path: Path, metadata: dict) -> ParsedDocument:
        """通过 OCR 解析图片文件"""
        if self.ocr is None:
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata=metadata,
                parse_warnings=[f"跳过图片文件 (未配置 OCR): {path.name}"],
            )

        try:
            extracted = self.ocr.process_image(str(path))
        except Exception as e:
            return ParsedDocument(
                raw_text="", cleaned_text="",
                metadata=metadata,
                parse_warnings=[f"OCR 失败: {str(e)}"],
            )

        if not extracted.strip():
            metadata["ocr_empty"] = True

        cleaned = self._clean(extracted)
        structure = self._detect_structure(cleaned) if cleaned else []

        return ParsedDocument(
            raw_text=extracted,
            cleaned_text=cleaned,
            structure=structure,
            metadata={**metadata, "ocr_backend": self.ocr.name},
        )

    # ─── 噪音清洗 ───

    @staticmethod
    def _clean(text: str) -> str:
        """文本噪音清洗"""
        if not text:
            return ""

        # Unicode 规范化
        text = unicodedata.normalize("NFKC", text)

        # 移除控制字符（保留换行和制表）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 压缩多余空行（3+ → 2）
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 行首行尾空白
        text = "\n".join(line.strip() for line in text.splitlines())

        # 多余空白字符
        text = re.sub(r" {2,}", " ", text)

        # 常见水印/页眉模式
        text = re.sub(r"^\d{1,4}\s*/\s*\d{1,4}$", "", text, flags=re.MULTILINE)

        return text.strip()

    # ─── 结构检测 ───

    @staticmethod
    def _detect_structure(text: str) -> list[StructureElement]:
        """
        检测文档结构：标题 / 段落 / 列表项。

        逐行扫描，标记每行的元素类型和层级。
        连续的非特殊行合并为段落。
        """
        if not text.strip():
            return []

        elements: list[StructureElement] = []
        lines = text.splitlines()
        para_buffer: list[str] = []
        position = 0

        def flush_para():
            nonlocal position
            if para_buffer:
                content = " ".join(para_buffer).strip()
                if content:
                    elements.append(StructureElement(
                        elem_type="paragraph",
                        content=content,
                        position=position,
                    ))
                    position += 1
                para_buffer.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_para()
                continue

            # 检查是否为标题
            heading = DocParser._match_heading(stripped)
            if heading:
                flush_para()
                elements.append(StructureElement(
                    elem_type="heading",
                    content=heading[1],
                    level=heading[0],
                    position=position,
                ))
                position += 1
                continue

            # 检查是否为列表项
            if re.match(r"^[-•·\*]\s+", stripped):
                flush_para()
                elements.append(StructureElement(
                    elem_type="list_item",
                    content=re.sub(r"^[-•·\*]\s+", "", stripped),
                    position=position,
                ))
                position += 1
                continue

            # 普通行 → 段落缓冲区
            para_buffer.append(stripped)

        flush_para()
        return elements

    @staticmethod
    def _match_heading(line: str) -> tuple[int, str] | None:
        """匹配标题，返回 (层级, 标题文本) 或 None"""
        for pattern, extractor in DocParser.HEADING_PATTERNS:
            m = pattern.match(line)
            if m:
                level, content = extractor(m)
                level = min(level, 3)  # 最多 3 级标题
                return (level, content.strip())
        return None
