"""
OCR 后端 — 可插拔图片文字提取

双后端设计：
- OpenAIVisionBackend: 复用 GPT-4o Vision，零新依赖，适合低频索引
- PaddleOCRBackend: 离线 OCR，中文优化，适合大批量索引

Usage:
    # 默认（不配 OCR）：图片文件直接跳过
    parser = DocParser()

    # OpenAI Vision 后端
    from langchain_openai import ChatOpenAI
    ocr = OpenAIVisionBackend(ChatOpenAI(model="gpt-4o"))
    parser = DocParser(ocr_backend=ocr)

    # PaddleOCR 后端
    ocr = PaddleOCRBackend()
    parser = DocParser(ocr_backend=ocr)
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class OCRBackend(ABC):
    """OCR 后端抽象接口"""

    @abstractmethod
    def process_image(self, image_path: str) -> str:
        """对单张图片返回提取的文本"""
        ...

    @abstractmethod
    def process_image_bytes(self, image_data: bytes, mime_type: str = "image/png") -> str:
        """对图片二进制数据返回提取的文本（用于 PDF 嵌入图）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称"""
        ...


class OpenAIVisionBackend(OCRBackend):
    """
    在线 OCR，复用 GPT-4o Vision。
    零新依赖，已有 API key 即可用。
    """

    def __init__(self, llm: Any = None):
        self.llm = llm
        self._lazy_llm = None

    @property
    def name(self) -> str:
        return "openai-vision"

    def _get_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        if self._lazy_llm is None:
            try:
                from langchain_openai import ChatOpenAI
                import os
                self._lazy_llm = ChatOpenAI(
                    model=os.getenv("VISION_MODEL_NAME", "gpt-4o"),
                    temperature=0,
                )
            except Exception:
                return None
        return self._lazy_llm

    def process_image(self, image_path: str) -> str:
        """读取图片 → base64 → GPT-4o Vision → 文本"""
        llm = self._get_llm()
        if llm is None:
            return ""

        path = Path(image_path)
        image_data = path.read_bytes()

        # 检测 mime 类型
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")

        return self.process_image_bytes(image_data, mime_type)

    async def _call_vision(self, base64_data: str, mime_type: str) -> str:
        """异步调用 GPT-4o Vision"""
        llm = self._get_llm()
        if llm is None:
            return ""

        from langchain_core.messages import HumanMessage

        image_url = f"data:{mime_type};base64,{base64_data}"

        messages = [HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "请提取这张图片中的所有文字，保持原有段落结构。"
                    "不要添加额外解释，只输出提取的文字内容。"
                    "如果图片中没有文字，返回空字符串。"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            },
        ])]

        response = await llm.ainvoke(messages)
        return response.content.strip()

    def process_image_bytes(self, image_data: bytes, mime_type: str = "image/png") -> str:
        """同步包装，内部转 base64"""
        import asyncio

        encoded = base64.b64encode(image_data).decode("utf-8")

        try:
            loop = asyncio.get_running_loop()
            # 已有运行中的事件循环，用 run_in_executor
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._call_vision(encoded, mime_type), loop,
            )
            return future.result(timeout=30)
        except RuntimeError:
            # 无事件循环，创建一个
            return asyncio.run(self._call_vision(encoded, mime_type))


class PaddleOCRBackend(OCRBackend):
    """
    离线 OCR，基于百度 PaddleOCR。
    安装: pip install paddleocr
    首次运行自动下载模型到 ~/.paddleocr/。
    """

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._ocr = None

    @property
    def name(self) -> str:
        return f"paddleocr-{self.lang}"

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)
            except ImportError:
                raise ImportError(
                    "PaddleOCR 未安装。运行: pip install paddleocr\n"
                    "或使用 OpenAIVisionBackend 作为 OCR 后端。"
                )
        return self._ocr

    def process_image(self, image_path: str) -> str:
        ocr = self._get_ocr()
        result = ocr.ocr(image_path)

        if not result or not result[0]:
            return ""

        # 按行合并，保留段落间距
        lines: list[str] = []
        last_y_center = -1

        for line_info in result[0]:
            if line_info is None:
                continue

            # line_info: [bbox, (text, confidence)]
            _, text_info = line_info
            text = text_info[0].strip()

            if not text:
                continue

            # 用 y 坐标判断是否为同一段落
            y_center = (line_info[0][0][1] + line_info[0][3][1]) / 2
            if lines and abs(y_center - last_y_center) > 30:
                lines.append("")  # 空行分隔段落
            lines.append(text)
            last_y_center = y_center

        return "\n".join(lines)

    def process_image_bytes(self, image_data: bytes, mime_type: str = "image/png") -> str:
        """保存临时文件后调用 OCR"""
        import tempfile

        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/bmp": ".bmp",
        }
        suffix = ext_map.get(mime_type, ".png")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_data)
            tmp_path = f.name

        try:
            return self.process_image(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
