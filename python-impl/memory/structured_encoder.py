"""
结构化数据编码器 — 将结构化数据转为可检索的文本 Chunk

支持的编码类型：
- encode_sku: 商品 SKU 参数 → "星耀X1手机，256GB版，曜石黑，4299元"
- encode_compatibility: 配件兼容列表 → "X1兼容：65W充电器(129元)、30W充电板(199元)..."
- encode_policy_rule: 政策规则 → "退换货政策：7天内无理由退换... 例外：已激活软件..."
- encode_price_tier: 价格梯度 → "X1三个版本：256GB/4299元、512GB/5299元、1TB/6299元"

产物是标准 Chunk 实例（chunk_type="sku_fact" / "policy_rule" 等），
与文本分块的产物走同一套索引管线。
"""

from __future__ import annotations

from typing import Any

from memory.long_term import Chunk


class StructuredEncoder:
    """
    结构化数据 → descriptive text chunk。

    调用方产出的 Chunk 可以直接喂给 LongTermMemory.add_chunk()
    和 SparseIndex.index_chunk()。
    """

    # ─── SKU 编码 ───

    @staticmethod
    def encode_sku(
        product_name: str,
        fields: dict[str, Any],
        variant_label: str = "",
    ) -> Chunk:
        """
        将单个 SKU 编码为事实型 chunk。

        Example:
            >>> enc.encode_sku("星耀X1", {"storage":"256GB","price":4299}, "标准版")
            Chunk(content="星耀X1手机，标准版：256GB存储，售价4299元。", ...)
        """
        parts = [f"{product_name}手机"]
        if variant_label:
            parts.append(f"{variant_label}：")

        field_texts = []
        for key, value in fields.items():
            if key == "price" or key == "售价":
                field_texts.append(f"售价{value}元")
            elif key == "storage" or key == "存储":
                field_texts.append(f"{value}存储")
            elif key == "ram" or key == "内存":
                field_texts.append(f"{value}运行内存")
            elif key == "color" or key == "颜色":
                field_texts.append(f"{value}色")
            elif key == "battery" or key == "电池":
                field_texts.append(f"{value}电池")
            elif key == "screen" or key == "屏幕":
                field_texts.append(f"{value}屏幕")
            else:
                field_texts.append(f"{key}: {value}")

        content = "，".join(parts) + "，".join(field_texts) + "。"

        return Chunk(
            chunk_id=_make_id(content),
            content=content,
            chunk_type="sku_fact",
            token_count=0,
            char_count=len(content),
            source="structured_data",
            metadata={
                "doc_type": "product_fact",
                "structured_fields": fields,
                "variant_label": variant_label,
            },
        )

    @staticmethod
    def encode_price_tier(
        product_name: str,
        tiers: list[dict[str, Any]],
    ) -> Chunk:
        """
        将多版本价格信息编码为对比型 chunk。

        Example:
            >>> enc.encode_price_tier("星耀X1", [
            ...     {"label":"256GB","price":4299},
            ...     {"label":"512GB","price":5299},
            ...     {"label":"1TB","price":6299},
            ... ])
            Chunk(content="星耀X1共有三个版本：256GB版4299元、512GB版5299元、1TB版6299元。", ...)
        """
        tier_texts = []
        entity_ids = []
        for t in tiers:
            label = t.get("label", "")
            price = t.get("price", "")
            tier_texts.append(f"{label}版{price}元")
            if t.get("entity_id"):
                entity_ids.append(t["entity_id"])

        content = f"{product_name}共有{len(tiers)}个版本：{'、'.join(tier_texts)}。"

        return Chunk(
            chunk_id=_make_id(content),
            content=content,
            chunk_type="sku_fact",
            token_count=0,
            char_count=len(content),
            source="structured_data",
            metadata={
                "doc_type": "price_tier",
                "entity_ids": entity_ids,
                "structured_fields": {"tiers": tiers},
            },
        )

    # ─── 兼容关系编码 ───

    @staticmethod
    def encode_compatibility(
        product_name: str,
        accessories: list[dict[str, Any]],
    ) -> Chunk:
        """
        将配件兼容列表编码为关系型 chunk。

        Example:
            >>> enc.encode_compatibility("星耀X1", [
            ...     {"name":"65W氮化镓充电器","model":"GN65W","price":129},
            ...     {"name":"30W无线充电板","model":"WC30","price":199},
            ... ])
            Chunk(content="星耀X1兼容以下配件：65W氮化镓充电器(GN65W，129元)、...", ...)
        """
        acc_texts = []
        entity_ids = [a.get("entity_id") for a in accessories if a.get("entity_id")]

        for acc in accessories:
            name = acc.get("name", "")
            model = acc.get("model", "")
            price = acc.get("price", "")

            parts = [name]
            if model:
                parts.append(f"({model}")
                if price:
                    parts.append(f"，{price}元")
                parts.append(")")
            elif price:
                parts.append(f"({price}元)")

            acc_texts.append("".join(parts))

        content = f"{product_name}兼容以下配件：{'、'.join(acc_texts)}。"

        return Chunk(
            chunk_id=_make_id(content),
            content=content,
            chunk_type="sku_fact",
            token_count=0,
            char_count=len(content),
            source="structured_data",
            metadata={
                "doc_type": "compatibility",
                "entity_ids": entity_ids,
                "relation": "compatible_with",
                "structured_fields": {"accessories": accessories},
            },
        )

    # ─── 政策规则编码 ───

    @staticmethod
    def encode_policy_rule(
        policy_name: str,
        conditions: dict[str, str],
        exceptions: list[str] | None = None,
        extra_notes: str = "",
    ) -> Chunk:
        """
        将政策规则表编码为条件型 chunk。

        Example:
            >>> enc.encode_policy_rule("退换货", {
            ...     "时间限制": "签收后7天内可无理由退换",
            ...     "条件": "商品完好、配件齐全、包装完整",
            ... }, exceptions=["已激活的软件产品", "已拆封的个人护理产品"])
        """
        lines = [f"{policy_name}政策："]
        for key, value in conditions.items():
            lines.append(f"{key}：{value}；")

        if exceptions:
            lines.append(f"例外情况：{'、'.join(exceptions)}。")

        if extra_notes:
            lines.append(extra_notes)

        content = " ".join(lines)

        return Chunk(
            chunk_id=_make_id(content),
            content=content,
            chunk_type="policy_rule",
            token_count=0,
            char_count=len(content),
            source="structured_data",
            metadata={
                "doc_type": "policy_rule",
                "structured_fields": {
                    "conditions": conditions,
                    "exceptions": exceptions or [],
                },
            },
        )

    @staticmethod
    def encode_process(
        process_name: str,
        steps: list[str],
        prerequisites: list[str] | None = None,
        outcome: str = "",
    ) -> Chunk:
        """
        将业务流程编码为步骤型 chunk。

        Example:
            >>> enc.encode_process("下单", [
            ...     "选择商品", "加入购物车", "确认订单", "填写地址",
            ...     "选择支付方式", "提交订单", "支付", "等待发货"
            ... ], prerequisites=["已注册账号", "已绑定手机号"])
        """
        steps_text = " → ".join(steps)
        lines = [f"{process_name}流程：{steps_text}。"]

        if prerequisites:
            lines.append(f"前置条件：{'、'.join(prerequisites)}。")

        if outcome:
            lines.append(f"结果：{outcome}。")

        content = " ".join(lines)

        return Chunk(
            chunk_id=_make_id(content),
            content=content,
            chunk_type="policy_rule",
            token_count=0,
            char_count=len(content),
            source="structured_data",
            metadata={
                "doc_type": "process",
                "structured_fields": {
                    "steps": steps,
                    "prerequisites": prerequisites or [],
                    "outcome": outcome,
                },
            },
        )


def _make_id(content: str) -> str:
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()[:12]
