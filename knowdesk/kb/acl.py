from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class AccessControlContext:
    """企业级零信任访问上下文 (Enterprise Zero-Trust ACL Context)"""
    tenant_id: str = "default_tenant"
    user_id: str = "anonymous"
    clearance_level: int = 1  # 1: Customer Public, 2: Frontline Agent, 3: Supervisor, 4: Risk/Compliance/Admin
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"customer"}))

@dataclass
class DocumentACL:
    doc_id: str
    tenant_id: str = "default_tenant"
    min_clearance: int = 1
    allowed_roles: set[str] = field(default_factory=lambda: {"customer", "agent", "supervisor", "admin"})

def infer_document_acl(doc_id: str, title: str, content: str, default_tenant: str = "default_tenant") -> DocumentACL:
    """
    根据文档 ID、标题和内容自适应推断金融密级与角色权限：
    - Internal SOP / Risk / Fraud / Dispute 内部处理规程 -> clearance >= 2, 仅限 agent / supervisor / admin
    - Sensitive Supervisor / Override / Reversal 审批规程 -> clearance >= 3, 仅限 supervisor / admin
    - 公开条款、产品介绍、客户帮助中心 -> clearance >= 1, 允许 customer 查阅
    """
    title_low = title.lower()
    content_low = content.lower()
    doc_id_low = doc_id.lower()

    if any(k in title_low or k in doc_id_low for k in ["internal", "risk", "fraud", "investigation"]):
        if any(k in title_low for k in ["supervisor", "override", "reversal", "limit increase"]):
            return DocumentACL(doc_id=doc_id, tenant_id=default_tenant, min_clearance=3, allowed_roles={"supervisor", "admin"})
        return DocumentACL(doc_id=doc_id, tenant_id=default_tenant, min_clearance=2, allowed_roles={"agent", "supervisor", "admin"})

    if "## internal" in content_low or "(internal)" in title_low:
        return DocumentACL(doc_id=doc_id, tenant_id=default_tenant, min_clearance=2, allowed_roles={"agent", "supervisor", "admin"})

    return DocumentACL(doc_id=doc_id, tenant_id=default_tenant, min_clearance=1, allowed_roles={"customer", "agent", "supervisor", "admin"})

def can_access(acl: DocumentACL, ctx: AccessControlContext | None) -> bool:
    """行级零信任校验：租户一致 + 密级达标 + 拥有允许角色"""
    if ctx is None:
        return True  # 缺省时默认允许（向后兼容基线）
    if acl.tenant_id != ctx.tenant_id and acl.tenant_id != "*":
        return False
    if ctx.clearance_level < acl.min_clearance:
        return False
    if acl.allowed_roles and not (acl.allowed_roles & ctx.roles):
        return False
    return True
