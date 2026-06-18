"""
API 数据模型 — Pydantic 请求/响应模型

支持：
- ChatRequest: 原有的纯文本聊天请求
- ChatMultiModalRequest: 多模态聊天请求（支持图片和附件）
- 知识图谱相关请求/响应模型
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ─── 附件模型 ───


class Attachment(BaseModel):
    """多模态附件（PDF、文档等）"""
    data: str = Field(..., description="Base64 编码的文件内容")
    mime_type: str = Field(..., description="MIME 类型，如 image/png, application/pdf")
    filename: str | None = Field(None, description="原始文件名")

    @field_validator("mime_type")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        allowed = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp",
            "application/pdf",
        }
        if v not in allowed:
            raise ValueError(f"不支持的 MIME 类型: {v}。支持: {', '.join(sorted(allowed))}")
        return v


# ─── 聊天请求 ───


class ChatRequest(BaseModel):
    """原有的纯文本聊天请求（向后兼容）"""
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatMultiModalRequest(BaseModel):
    """多模态聊天请求"""
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None
    images: list[str] | None = Field(None, description="Base64 编码的图片列表（支持 data URL 前缀）")
    attachments: list[Attachment] | None = Field(None, description="附件列表（PDF等）")


# ─── 聊天响应 ───


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


# ─── 知识图谱模型 ───


class EntityQuery(BaseModel):
    """图谱实体查询"""
    entity_name: str = Field(..., description="实体名称（模糊匹配）")
    entity_type: str | None = Field(None, description="实体类型过滤")


class GraphBuildRequest(BaseModel):
    """图谱构建请求"""
    documents: list[dict[str, str]] = Field(..., description="文档列表 [{\"content\": \"...\", \"source\": \"...\"}]")


class GraphStatsResponse(BaseModel):
    """图谱统计响应"""
    node_count: int
    edge_count: int
    entity_types: dict[str, int]
    relation_types: dict[str, int]
    is_dag: bool
