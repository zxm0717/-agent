"""
视觉理解 Agent — 多模态图片/文档分析

支持：
- 商品图片识别（包装、型号、外观）
- 截图分析（错误页面、订单截图、聊天记录）
- 票据识别（发票、收据、保修卡）
- 文档图片 OCR（拍照的说明书、合同页）

依赖 GPT-4o 或同等多模态 LLM。
"""

from __future__ import annotations

import base64
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


# ─── Prompt 模板 ───

VISION_SYSTEM_PROMPT = """你是一个电商智能客服的视觉分析专家。你的任务是分析用户上传的图片，
提取与客服场景相关的关键信息。

根据图片类型提供不同分析：

**商品图片：**
- 识别商品名称、品牌、型号
- 识别外观特征（颜色、尺寸、包装状态）
- 如果有破损/异常，详细描述

**截图：**
- 提取屏幕上的文字（错误信息、订单号、价格等）
- 判断是哪个环节出现问题（支付、物流、账号）
- 提取可操作的信息（订单号、快递单号、客服编号）

**票据/凭证：**
- 提取发票号、日期、金额
- 提取商家名称、税号
- 提取商品清单

**通用规则：**
- 不要编造图片中没有的信息
- 对于模糊不清的文字标注"[模糊]"
- 返回 JSON 格式确保可被下游 Agent 解析
"""

SPECIFIC_PROMPTS = {
    "screenshot": (
        "这是一张用户上传的截图。请分析截图内容："
        "1. 这是什么应用/页面的截图？"
        "2. 截图中显示了什么错误或问题？"
        "3. 提取截图中的订单号、金额、日期等关键数据。"
        "4. 用户可能遇到了什么问题？"
    ),
    "invoice": (
        "这是一张发票/收据图片。请提取以下信息："
        "1. 发票号码和开票日期"
        "2. 销售方名称"
        "3. 商品/服务明细和金额"
        "4. 合计金额（含税/不含税）"
        "5. 是否有异常（涂改、模糊、信息不全）"
    ),
    "product": (
        "这是一张商品图片。请分析："
        "1. 这是什么商品？品牌和型号是什么？"
        "2. 外观状态如何？（全新/有破损/有划痕）"
        "3. 包装是否完好？"
        "4. 是否有序列号、防伪码等可识别信息？"
    ),
    "delivery": (
        "这是一张物流/快递相关图片。请分析："
        "1. 快递单号和物流公司"
        "2. 运单状态（已签收/运输中/异常）"
        "3. 寄件/收件信息（注意保护隐私，只提取城市）"
        "4. 是否有破损或其他异常"
    ),
}


class VisionAnalysisResult:
    """视觉分析结果"""

    def __init__(self, raw_response: str, parsed: dict[str, Any] | None = None):
        self.raw = raw_response
        self.parsed = parsed or {}

    @property
    def description(self) -> str:
        return self.parsed.get("description", self.raw[:500])

    @property
    def extracted_text(self) -> str:
        return self.parsed.get("extracted_text", "")

    @property
    def entities(self) -> list[dict[str, str]]:
        return self.parsed.get("entities", [])

    @property
    def suggested_actions(self) -> list[str]:
        return self.parsed.get("suggested_actions", [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "extracted_text": self.extracted_text,
            "entities": self.entities,
            "suggested_actions": self.suggested_actions,
        }


class VisionAgent:
    """
    视觉理解 Agent。

    处理图片输入，调用多模态 LLM 分析图片内容，
    提取结构化信息供下游 Agent 使用。

    Usage:
        agent = VisionAgent(llm)
        result = await agent.analyze_image(base64_data, "screenshot")
        state = await agent.process(state)
    """

    # 支持的图片格式
    SUPPORTED_MIME_TYPES = {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/gif": "gif",
        "image/webp": "webp",
    }

    # 最大图片大小（base64 解码后）: 10MB
    MAX_IMAGE_BYTES = 10 * 1024 * 1024

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @staticmethod
    def validate_image_base64(image_data: str) -> dict[str, Any]:
        """
        校验 base64 图片的格式和大小。
        返回 {"valid": bool, "error": str | None, "mime_type": str | None, "size_bytes": int}
        """
        # 处理可能的 data URL 前缀
        clean_data = image_data
        detected_mime = "image/png"

        if image_data.startswith("data:"):
            # data:image/png;base64,xxxx
            header, clean_data = image_data.split(",", 1)
            parts = header.split(";")
            if parts[0].startswith("data:"):
                detected_mime = parts[0][5:]

        try:
            decoded = base64.b64decode(clean_data)
        except Exception:
            return {"valid": False, "error": "Base64 解码失败，请确认图片数据正确", "mime_type": None, "size_bytes": 0}

        if len(decoded) > VisionAgent.MAX_IMAGE_BYTES:
            return {
                "valid": False,
                "error": f"图片过大 ({len(decoded) / 1024 / 1024:.1f}MB)，上限 {VisionAgent.MAX_IMAGE_BYTES / 1024 / 1024:.0f}MB",
                "mime_type": detected_mime,
                "size_bytes": len(decoded),
            }

        # 简单的 magic bytes 校验
        magic = decoded[:4]
        if magic[:3] == b"\xff\xd8\xff":
            detected_mime = "image/jpeg"
        elif magic[:4] == b"\x89PNG":
            detected_mime = "image/png"
        elif magic[:4] == b"GIF8":
            detected_mime = "image/gif"

        return {
            "valid": True,
            "error": None,
            "mime_type": detected_mime,
            "size_bytes": len(decoded),
        }

    @trace_agent_call("vision_analyze")
    async def analyze_image(
        self,
        image_base64: str,
        analysis_type: str = "general",
        context: str = "",
    ) -> VisionAnalysisResult:
        """
        分析单张图片。

        Args:
            image_base64: base64 编码的图片（支持 data URL 前缀）
            analysis_type: screenshot | invoice | product | delivery | general
            context: 用户附加的文字说明

        Returns:
            VisionAnalysisResult 包含描述、提取文本、实体、建议操作
        """
        # 处理 data URL 前缀
        clean_data = image_base64
        mime_type = "image/png"
        if image_base64.startswith("data:"):
            header, clean_data = image_base64.split(",", 1)
            parts = header.split(";")
            if parts[0].startswith("data:"):
                mime_type = parts[0][5:]

        # 构建图片 URL
        image_url = f"data:{mime_type};base64,{clean_data}"

        # 选择专用 prompt
        specific_prompt = SPECIFIC_PROMPTS.get(analysis_type, "")

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{VISION_SYSTEM_PROMPT}\n\n"
                    f"{specific_prompt}\n\n"
                    f"用户附加说明: {context}" if context else f"{VISION_SYSTEM_PROMPT}\n\n{specific_prompt}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "auto"},
            },
        ]

        messages = [HumanMessage(content=user_content)]
        response = await self.llm.ainvoke(messages)

        # 尝试解析 JSON 输出
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content)
        except (json.JSONDecodeError, Exception):
            parsed = {"description": response.content, "extracted_text": response.content}

        return VisionAnalysisResult(raw_response=response.content, parsed=parsed)

    @trace_agent_call("vision_process_batch")
    async def process_batch(
        self,
        images: list[str],
        analysis_type: str = "general",
        context: str = "",
    ) -> list[VisionAnalysisResult]:
        """批量分析多张图片"""
        results = []
        for img in images:
            result = await self.analyze_image(img, analysis_type, context)
            results.append(result)
        return results

    @trace_agent_call("vision_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        作为 LangGraph 节点处理状态。
        从 state.images 中读取图片列表，逐张分析。
        """
        images = state.get("images", [])
        messages = state.get("messages", [])

        user_text = ""
        if messages:
            user_text = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        if not images:
            return {
                **state,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "vision": "未提供图片数据。",
                },
            }

        # 判断分析类型
        text_lower = user_text.lower()
        if any(kw in text_lower for kw in ["发票", "invoice", "收据", "receipt"]):
            analysis_type = "invoice"
        elif any(kw in text_lower for kw in ["快递", "物流", "delivery", "包裹"]):
            analysis_type = "delivery"
        elif any(kw in text_lower for kw in ["商品", "产品", "product", "实物"]):
            analysis_type = "product"
        elif any(kw in text_lower for kw in ["截图", "screenshot", "错误", "error"]):
            analysis_type = "screenshot"
        else:
            analysis_type = "general"

        # 校验并分析
        all_results: list[dict[str, Any]] = []
        for img in images:
            validation = self.validate_image_base64(img)
            if not validation["valid"]:
                all_results.append({
                    "error": validation["error"],
                    "description": f"图片校验失败: {validation['error']}",
                })
                continue

            result = await self.analyze_image(img, analysis_type, user_text)
            all_results.append(result.to_dict())

        # 汇总所有图片分析结果
        summary_parts = []
        for i, r in enumerate(all_results, 1):
            if r.get("error"):
                summary_parts.append(f"图片{i}: 处理失败 - {r['error']}")
            else:
                parts = [f"图片{i}分析结果:"]
                if r.get("description"):
                    parts.append(f"  描述: {r['description']}")
                if r.get("extracted_text"):
                    parts.append(f"  提取文字: {r['extracted_text']}")
                if r.get("entities"):
                    parts.append(f"  识别实体: {json.dumps(r['entities'], ensure_ascii=False)}")
                if r.get("suggested_actions"):
                    parts.append(f"  建议操作: {', '.join(r['suggested_actions'])}")
                summary_parts.append("\n".join(parts))

        vision_summary = "\n\n".join(summary_parts) if summary_parts else "图片分析完成，未提取到有效信息。"

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "vision": vision_summary,
            },
            "vision_results": {
                "analysis_type": analysis_type,
                "image_count": len(images),
                "results": all_results,
            },
        }
