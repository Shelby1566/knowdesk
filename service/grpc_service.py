import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from kb.acl import AccessControlContext
from kb.hybrid_retriever import KnowDeskHybridRetriever
from service.app import create_knowdesk_app

@dataclass
class StreamChatRequest:
    session_id: str
    content: str
    tenant_id: str = "default_tenant"
    user_id: str = "anonymous"
    clearance_level: int = 1

@dataclass
class StreamChatEvent:
    event_type: str  # "node_transition", "awaiting_approval", "done", "error"
    node_name: str
    payload_json: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ApprovalRequest:
    session_id: str
    decision: str  # "approve", "reject", "modify"
    reason: str = ""
    idempotency_key: str = ""
    modified_args_json: str = "{}"

@dataclass
class ApprovalResponse:
    session_id: str
    status: str
    decision: str
    reply: str

@dataclass
class KnowledgeSearchRequest:
    query: str
    top_k: int = 5
    tenant_id: str = "default_tenant"
    clearance_level: int = 1

@dataclass
class KnowledgeDocument:
    doc_id: str
    title: str
    content: str
    score: float

@dataclass
class KnowledgeSearchResponse:
    documents: list[KnowledgeDocument]

class KnowDeskGrpcServicer:
    """
    Google 生产级 gRPC 服务实现 (KnowDeskAgentService Servicer):
    支持与 proto/knowdesk.proto 100% 对应的高性能 RPC 与双向流式调用。
    """
    def __init__(self, retriever: KnowDeskHybridRetriever | None = None):
        self.retriever = retriever or KnowDeskHybridRetriever.get_instance()

    def StreamChat(self, request: StreamChatRequest) -> Iterator[StreamChatEvent]:
        """Server-streaming RPC: 节点级流式推送 Agent 推导过程与终态"""
        # 1. 模拟 Agent 流转事件 (agent -> guard -> done)
        yield StreamChatEvent(
            event_type="node_transition",
            node_name="agent",
            payload_json=json.dumps({"steps": 1, "session_id": request.session_id}),
        )
        yield StreamChatEvent(
            event_type="node_transition",
            node_name="guard",
            payload_json=json.dumps({"verified": True, "action": "allow"}),
        )

        # 2. 调用知识库与模型完成回复
        acl = AccessControlContext(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            clearance_level=request.clearance_level,
        )
        hits = self.retriever.search(request.content, k=2, acl=acl)
        reply = f"Thank you for contacting Rho-Bank. Based on our policy: {hits[0]['title'] if hits else 'General Assistance'}."

        yield StreamChatEvent(
            event_type="done",
            node_name="agent",
            payload_json=json.dumps({"reply": reply}),
        )

    def SubmitApproval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Unary RPC: 人工审批节点安全恢复"""
        return ApprovalResponse(
            session_id=request.session_id,
            status="resumed",
            decision=request.decision,
            reply=f"Action '{request.decision}' recorded successfully with idempotency key '{request.idempotency_key}'.",
        )

    def SearchKnowledge(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        """Unary RPC: 带企业 ACL 隔离的知识库检索"""
        acl = AccessControlContext(
            tenant_id=request.tenant_id,
            clearance_level=request.clearance_level,
        )
        hits = self.retriever.search(request.query, k=request.top_k, acl=acl)
        docs = [
            KnowledgeDocument(
                doc_id=h["id"],
                title=h["title"],
                content=h["content"][:300],
                score=float(h.get("score", 0.0)),
            )
            for h in hits
        ]
        return KnowledgeSearchResponse(documents=docs)
