"""
知识图谱存储 — 基于 NetworkX 的嵌入式图存储

用于电商场景的结构化知识管理：
- 实体节点：商品、品类、品牌、规格、政策、流程等
- 关系边：属于、兼容、替代、依赖、适用等
- 支持图遍历查询（1-hop / 2-hop 邻居），多跳推理
- JSON 持久化，可选 Neo4j 升级

与 LongTermMemory 互补：向量库负责语义匹配，图谱负责结构化关系推理。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import networkx as nx


# ─── LLM 实体提取 Prompt ───

EXTRACTION_SYSTEM_PROMPT = """你是一个电商知识图谱构建专家。请从给定的电商文档中提取实体和关系。

实体类型：Product(商品)、Category(品类)、Brand(品牌)、Spec(规格参数)、Policy(政策规则)、
         Process(业务流程)、Service(售后服务)、PriceRange(价格区间)

关系类型：belongs_to(属于品类)、has_brand(品牌是)、has_spec(规格为)、
         compatible_with(兼容搭配)、alternative_to(替代品)、supersedes(升级取代)、
         requires(必要条件)、applies_to(适用范围)、depends_on(前置依赖)、
         exclusions(不适用场景)

请以 JSON 返回：
{
  "entities": [
    {"id": "唯一ID(英文下划线)", "type": "Product|Category|...", "name": "显示名称",
     "properties": {"key": "value"}}
  ],
  "relationships": [
    {"source": "entity_id_a", "target": "entity_id_b", "type": "关系类型",
     "properties": {"note": "补充说明"}}
  ]
}
"""


class KnowledgeGraph:
    """
    嵌入式知识图谱，基于 NetworkX 有向图。

    特点：
    - 纯 Python 实现，零外部服务依赖
    - 内存存储 + JSON 文件持久化
    - 支持 1-hop/2-hop 邻居遍历
    - LLM 辅助实体关系提取（从文档自动建图）
    - 预留 Neo4j 适配器接口

    Usage:
        kg = KnowledgeGraph()
        kg.add_entity("phone_x1", "Product", {"name": "X1手机", "brand": "X"})
        kg.add_relationship("phone_x1", "electronics", "belongs_to")
        neighbors = kg.get_neighbors("phone_x1", hops=2)
    """

    def __init__(
        self,
        persist_path: str | None = "./graph_store/graph.json",
        llm: Any = None,
    ):
        self._graph = nx.DiGraph()
        self.persist_path = Path(persist_path) if persist_path else None
        self.llm = llm

        if self.persist_path and self.persist_path.exists():
            self._load()

    # ─── 实体操作 ───

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str = "",
        properties: dict[str, Any] | None = None,
        source_doc: str = "",
    ) -> str:
        """添加一个实体节点，已存在则更新属性"""
        props = dict(properties or {})
        props["name"] = name or entity_id
        props["type"] = entity_type
        if source_doc:
            props["source_doc"] = source_doc

        if self._graph.has_node(entity_id):
            self._graph.nodes[entity_id].update(props)
        else:
            self._graph.add_node(entity_id, **props)

        return entity_id

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """获取实体信息"""
        if not self._graph.has_node(entity_id):
            return None
        return {"id": entity_id, **self._graph.nodes[entity_id]}

    def search_entities(
        self, query: str, entity_type: str | None = None
    ) -> list[dict[str, Any]]:
        """按名称/类型搜索实体，支持模糊匹配"""
        results = []
        query_lower = query.lower()

        for node_id, attrs in self._graph.nodes(data=True):
            if entity_type and attrs.get("type") != entity_type:
                continue

            name = attrs.get("name", node_id).lower()
            node_type = attrs.get("type", "").lower()

            if query_lower in name or query_lower in node_type or query_lower in node_id.lower():
                results.append({"id": node_id, **attrs})

        # 按名称匹配度排序（完全匹配 > 前缀匹配 > 包含匹配）
        def score(r):
            n = r.get("name", "").lower()
            if n == query_lower:
                return 3
            if n.startswith(query_lower):
                return 2
            return 1

        results.sort(key=score, reverse=True)
        return results

    def entity_exists(self, entity_id: str) -> bool:
        return self._graph.has_node(entity_id)

    # ─── 关系操作 ───

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """添加一条关系边，自动确保两端实体存在"""
        if not self._graph.has_node(source_id):
            self.add_entity(source_id, "Unknown", name=source_id)
        if not self._graph.has_node(target_id):
            self.add_entity(target_id, "Unknown", name=target_id)

        self._graph.add_edge(
            source_id, target_id,
            type=relation_type,
            **(properties or {}),
        )

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[dict]:
        """获取实体关联的所有关系"""
        relationships = []

        if direction in ("outgoing", "both"):
            for _, target, attrs in self._graph.out_edges(entity_id, data=True):
                relationships.append({
                    "source": entity_id,
                    "target": target,
                    "type": attrs.get("type", "unknown"),
                    "direction": "outgoing",
                    "properties": {k: v for k, v in attrs.items() if k != "type"},
                })

        if direction in ("incoming", "both"):
            for source, _, attrs in self._graph.in_edges(entity_id, data=True):
                relationships.append({
                    "source": source,
                    "target": entity_id,
                    "type": attrs.get("type", "unknown"),
                    "direction": "incoming",
                    "properties": {k: v for k, v in attrs.items() if k != "type"},
                })

        return relationships

    # ─── 图遍历 ───

    def get_neighbors(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        hops: int = 1,
    ) -> list[dict[str, Any]]:
        """
        获取指定实体的邻居。
        hops=1: 直接邻居
        hops=2: 二跳邻居（邻居的邻居）
        """
        if not self._graph.has_node(entity_id):
            return []

        # BFS 遍历
        visited: set[str] = {entity_id}
        frontier = {entity_id}
        result_nodes: set[str] = set()

        for hop in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in visited:
                        # 过滤关系类型
                        if relation_types:
                            edge_data = self._graph.get_edge_data(node, neighbor)
                            edge_type = edge_data.get("type", "")
                            if edge_type not in relation_types:
                                continue
                        result_nodes.add(neighbor)
                        next_frontier.add(neighbor)
                        visited.add(neighbor)

                # 也检查入边方向
                for predecessor in self._graph.predecessors(node):
                    if predecessor not in visited:
                        if relation_types:
                            edge_data = self._graph.get_edge_data(predecessor, node)
                            edge_type = edge_data.get("type", "")
                            if edge_type not in relation_types:
                                continue
                        result_nodes.add(predecessor)
                        next_frontier.add(predecessor)
                        visited.add(predecessor)

            frontier = next_frontier

        return [{"id": nid, **self._graph.nodes[nid]} for nid in result_nodes]

    def get_subgraph_context(
        self, entity_ids: list[str], max_hops: int = 2, max_entities: int = 30
    ) -> str:
        """
        将多个实体周围的子图序列化为 LLM 可读的文本上下文。
        用于 GraphRAG Agent 的上下文注入。
        """
        all_nodes: set[str] = set()
        all_edges: list[tuple[str, str, str]] = []

        for eid in entity_ids:
            if not self._graph.has_node(eid):
                continue
            all_nodes.add(eid)

            # BFS 收集子图
            visited: set[str] = {eid}
            frontier = {eid}

            for _ in range(max_hops):
                next_frontier: set[str] = set()
                for node in frontier:
                    for neighbor in self._graph.neighbors(node):
                        all_edges.append((node, neighbor, self._graph[node][neighbor].get("type", "")))
                        if neighbor not in visited and len(all_nodes) < max_entities:
                            all_nodes.add(neighbor)
                            next_frontier.add(neighbor)
                            visited.add(neighbor)
                    for predecessor in self._graph.predecessors(node):
                        all_edges.append((predecessor, node, self._graph[predecessor][node].get("type", "")))
                        if predecessor not in visited and len(all_nodes) < max_entities:
                            all_nodes.add(predecessor)
                            next_frontier.add(predecessor)
                            visited.add(predecessor)
                frontier = next_frontier
                if len(all_nodes) >= max_entities:
                    break

        # 构建文本表示
        lines = ["## 知识图谱上下文\n"]

        lines.append("### 实体节点:")
        for nid in sorted(all_nodes):
            node = self._graph.nodes[nid]
            lines.append(
                f"  [{nid}] ({node.get('type', 'Unknown')}) {node.get('name', nid)}"
            )

        lines.append("\n### 关系边:")
        edges_dedup = list(set(all_edges))
        for src, tgt, rtype in sorted(edges_dedup, key=lambda e: e[2]):
            lines.append(f"  {src} --[{rtype}]--> {tgt}")

        return "\n".join(lines)

    def find_paths(
        self, source_id: str, target_id: str, max_length: int = 4
    ) -> list[list[str]]:
        """查找两实体间的所有路径，用于多跳推理"""
        try:
            paths = list(nx.all_simple_paths(
                self._graph, source_id, target_id, cutoff=max_length,
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    # ─── 图谱构建 ───

    async def build_from_documents(
        self, documents: list[dict[str, str]]
    ) -> int:
        """
        使用 LLM 从文档批量提取实体和关系，自动构建图谱。
        documents: [{"content": "...", "source": "..."}, ...]
        返回提取到的实体总数。
        """
        if self.llm is None:
            return 0

        total_entities = 0
        seen_entities: dict[str, str] = {}  # name→id 去重映射

        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")

            if not content.strip():
                continue

            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                response = await self.llm.ainvoke([
                    SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(content=f"文档来源: {source}\n\n文档内容:\n{content}"),
                ])

                extracted = json.loads(response.content)
            except (json.JSONDecodeError, Exception):
                continue

            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            # 添加实体（名称去重）
            name_to_id: dict[str, str] = {}
            for ent in entities:
                ent_name = ent.get("name", "").strip()
                ent_type = ent.get("type", "Unknown")
                ent_id = ent.get("id", "").strip()

                if not ent_id:
                    continue

                # 去重：同名同类型合并
                dedup_key = f"{ent_name}|{ent_type}".lower()
                if dedup_key in seen_entities:
                    name_to_id[ent_name] = seen_entities[dedup_key]
                    continue

                seen_entities[dedup_key] = ent_id
                name_to_id[ent_name] = ent_id
                self.add_entity(
                    entity_id=ent_id,
                    entity_type=ent_type,
                    name=ent_name,
                    properties=ent.get("properties", {}),
                    source_doc=source,
                )
                total_entities += 1

            # 添加关系
            for rel in relationships:
                src_name = rel.get("source", "")
                tgt_name = rel.get("target", "")

                # 解析：关系中的 source/target 可能是实体名而非 ID
                src_id = name_to_id.get(src_name, src_name)
                tgt_id = name_to_id.get(tgt_name, tgt_name)

                self.add_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel.get("type", "related_to"),
                    properties=rel.get("properties", {}),
                )

        self.save()
        return total_entities

    def build_from_structured(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> int:
        """
        从结构化数据直接构建图谱（无需 LLM）。
        entities: [{"id": "...", "type": "...", "name": "...", "properties": {...}}, ...]
        relationships: [{"source": "...", "target": "...", "type": "...", "properties": {...}}, ...]
        """
        for ent in entities:
            self.add_entity(
                entity_id=ent["id"],
                entity_type=ent.get("type", "Unknown"),
                name=ent.get("name", ent["id"]),
                properties=ent.get("properties", {}),
            )

        for rel in relationships:
            self.add_relationship(
                source_id=rel["source"],
                target_id=rel["target"],
                relation_type=rel.get("type", "related_to"),
                properties=rel.get("properties", {}),
            )

        self.save()
        return len(entities)

    # ─── 持久化 ───

    def save(self) -> None:
        """将图谱保存为 JSON 文件"""
        if self.persist_path is None:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [
                {"id": nid, **attrs}
                for nid, attrs in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": src, "target": tgt, **attrs}
                for src, tgt, attrs in self._graph.edges(data=True)
            ],
        }

        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """从 JSON 文件加载图谱"""
        if self.persist_path is None or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node in data.get("nodes", []):
                nid = node.pop("id")
                self._graph.add_node(nid, **node)

            for edge in data.get("edges", []):
                src = edge.pop("source")
                tgt = edge.pop("target")
                self._graph.add_edge(src, tgt, **edge)
        except (json.JSONDecodeError, KeyError):
            pass

    # ─── 统计与诊断 ───

    def get_statistics(self) -> dict[str, Any]:
        """获取图谱统计信息"""
        entity_types: dict[str, int] = {}
        for _, attrs in self._graph.nodes(data=True):
            etype = attrs.get("type", "Unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        relation_types: dict[str, int] = {}
        for _, _, attrs in self._graph.edges(data=True):
            rtype = attrs.get("type", "unknown")
            relation_types[rtype] = relation_types.get(rtype, 0) + 1

        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
        }

    def clear(self) -> None:
        """清空图谱"""
        self._graph.clear()
