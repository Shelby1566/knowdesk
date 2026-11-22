import json
from pathlib import Path

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from knowdesk.kb.acl import AccessControlContext, DocumentACL
from knowdesk.kb.chunk import Chunk
from knowdesk.kb.hybrid_retriever import KnowDeskHybridRetriever
from knowdesk.obs.otel_setup import get_memory_exporter, get_tracer
from knowdesk.service.app import build_live_runtime, create_knowdesk_app, request_acl
from tests.test_service_api import _pending_idempotency_key, _scripted_llm


def _tools(counter: dict):
    def freeze(args):
        counter["writes"] += 1
        return {"frozen": args.get("card_id")}

    return {
        "get_user_details": lambda args: {"user_id": "u_01", "verified": True},
        "freeze_card": freeze,
    }


def _app(tmp_path: Path, counter: dict | None = None):
    counter = counter if counter is not None else {"writes": 0}
    application = create_knowdesk_app(
        db_path=tmp_path / "api.sqlite",
        llm_fn=_scripted_llm(),
        tool_funcs=_tools(counter),
    )
    return application, counter


def _open_approval(client: TestClient):
    created = client.post("/sessions", headers={"X-User-Id": "owner_a", "X-Tenant-Id": "tenant_a"})
    assert created.status_code == 200
    assert created.json()["verified"] is False
    session_id = created.json()["session_id"]
    streamed = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "Please freeze my card"},
        headers={"X-User-Id": "owner_a", "X-Tenant-Id": "tenant_a"},
    )
    assert streamed.status_code == 200
    idem_key = _pending_idempotency_key(streamed.text)
    return session_id, idem_key


def test_otel_exporter_redacts_nested_card_and_ssn():
    exporter: InMemorySpanExporter = get_memory_exporter()
    exporter.clear()
    payload = {
        "profile": {
            "card": "4111111111111111",
            "ssn": "123-45-6789",
            "notes": ["reach me at 313-555-0246", {"mail": "fati@protonmail.com"}],
        }
    }
    tracer = get_tracer("knowdesk.test")
    with tracer.start_as_current_span("pii_nested") as span:
        span.set_attribute("payload", json.dumps(payload))
        span.set_attribute("tags", ["ssn 123-45-6789"])
    finished = [item for item in exporter.get_finished_spans() if item.name == "pii_nested"]
    assert len(finished) == 1
    exported = dict(finished[0].attributes)
    nested = json.loads(exported["payload"])
    assert nested["profile"]["card"] == "****-****-****-1111"
    assert nested["profile"]["ssn"] == "***-**-6789"
    assert nested["profile"]["notes"][0] == "reach me at 313-***-0246"
    assert nested["profile"]["notes"][1]["mail"] == "f***@protonmail.com"
    rendered = json.dumps(exported)
    assert "4111111111111111" not in rendered
    assert "123-45-6789" not in rendered
    assert exported["tags"] == ("ssn ***-**-6789",) or exported["tags"] == ["ssn ***-**-6789"]


def test_http_auth_approval_and_output_guard(tmp_path: Path, monkeypatch):
    counter = {"writes": 0}
    application, counter = _app(tmp_path, counter)
    client = TestClient(application)
    session_id, idem_key = _open_approval(client)
    owner = {"X-User-Id": "owner_a", "X-Tenant-Id": "tenant_a"}

    denied = client.post(
        f"/sessions/{session_id}/approvals",
        json={"decision": "approve", "idempotency_key": idem_key},
        headers=owner,
    )
    assert denied.status_code == 403
    assert counter["writes"] == 0

    conflict = client.post(
        f"/sessions/{session_id}/approvals",
        json={"decision": "approve", "idempotency_key": "not-the-pending-key"},
        headers={**owner, "X-Role": "supervisor"},
    )
    assert conflict.status_code == 409
    assert counter["writes"] == 0

    cross_user = client.get(f"/sessions/{session_id}/trace", headers={"X-User-Id": "owner_b", "X-Tenant-Id": "tenant_a"})
    assert cross_user.status_code == 403
    cross_tenant = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "hello"},
        headers={"X-User-Id": "owner_a", "X-Tenant-Id": "tenant_b"},
    )
    assert cross_tenant.status_code == 403

    monkeypatch.setenv("KNOWDESK_API_KEY", "secret-key")
    missing_key = client.post("/sessions", headers=owner)
    assert missing_key.status_code == 401
    monkeypatch.delenv("KNOWDESK_API_KEY", raising=False)

    guarded = create_knowdesk_app(
        db_path=tmp_path / "guard.sqlite",
        llm_fn=lambda messages: {"content": "The card on file is 4532-0151-1283-5320.", "tool_calls": []},
        tool_funcs={},
    )
    guard_client = TestClient(guarded)
    guard_session = guard_client.post("/sessions").json()["session_id"]
    done = guard_client.post(f"/sessions/{guard_session}/messages", json={"content": "What is my card?"})
    assert "4532-0151-1283-5320" not in done.text
    assert "output policy" in done.text


def test_background_job_transitions_without_fake_score(tmp_path: Path):
    application, _counter = _app(tmp_path)
    client = TestClient(application)
    created = client.post("/jobs", json={"job_type": "regression_eval", "params": {}})
    assert created.status_code == 200
    assert created.json()["status"] == "pending"
    fetched = client.get(f"/jobs/{created.json()['job_id']}")
    body = fetched.json()
    assert body["status"] == "completed"
    assert body["history"] == ["pending", "running", "completed"]
    assert body["result"]["dev"] > 0
    assert body["result"]["test"] > 0
    assert "pass_1" not in body["result"]

    failed = client.post("/jobs", json={"job_type": "not_a_real_job"})
    failed_body = client.get(f"/jobs/{failed.json()['job_id']}").json()
    assert failed_body["status"] == "failed"
    assert "pending" in failed_body["history"] and "running" in failed_body["history"]


def test_cross_tenant_acl_filters_at_search_entry(monkeypatch, tmp_path: Path):
    retriever = KnowDeskHybridRetriever.__new__(KnowDeskHybridRetriever)
    retriever.chunk_map = {
        "c1": Chunk("c1", "doc_a", "Fees", "fee", "fee", "card"),
        "c2": Chunk("c2", "doc_b", "Fees", "fee", "fee", "card"),
    }
    retriever.doc_acl = {
        "doc_a": DocumentACL("doc_a", tenant_id="tenant_a", min_clearance=1, allowed_roles={"customer"}),
        "doc_b": DocumentACL("doc_b", tenant_id="tenant_b", min_clearance=1, allowed_roles={"customer"}),
    }

    class _BM25:
        def search(self, query, k=50):
            return [("c1", 5.0), ("c2", 4.0)]

    class _Store:
        def search(self, vector, k=50):
            return [("c1", 0.9), ("c2", 0.8)]

    class _Dense:
        store = _Store()

        def encode_query(self, query):
            return [1.0]

    retriever.bm25 = _BM25()
    retriever.dense = _Dense()
    context = AccessControlContext(tenant_id="tenant_a", user_id="user_a", roles=frozenset({"customer"}))
    hits = retriever.search_chunks("annual fee", k=5, acl=context)
    assert [chunk.doc_id for chunk, _score in hits] == ["doc_a"]

    seen = {}

    class _FakeRetriever:
        def search_sections(self, query, k=10, fusion_mode="adaptive_rrf", acl=None):
            seen["acl"] = acl
            seen["query"] = query
            return []

    monkeypatch.setattr(KnowDeskHybridRetriever, "get_instance", classmethod(lambda cls: _FakeRetriever()))
    runtime = build_live_runtime(tmp_path / "live.sqlite")
    token = request_acl.set(context)
    try:
        runtime["tool_funcs"].get("KB_search")({"query": "annual fee", "top_k": 3})
    finally:
        request_acl.reset(token)
    assert seen["acl"].tenant_id == "tenant_a"
    assert seen["query"] == "annual fee"


def test_sources_do_not_keep_fake_metrics_or_direct_mcp_call():
    app_source = Path("knowdesk/service/app.py").read_text(encoding="utf-8")
    otel_source = Path("knowdesk/obs/otel_setup.py").read_text(encoding="utf-8")
    client_source = Path("knowdesk/mcp/client.py").read_text(encoding="utf-8")
    server_source = Path("knowdesk/mcp/server.py").read_text(encoding="utf-8")
    page = Path("knowdesk/service/static/index.html").read_text(encoding="utf-8")
    assert "0.40" not in app_source
    assert "input_tokens=250" not in app_source
    assert "output_tokens=45" not in app_source
    assert "cached_tokens=120" not in app_source
    assert "from opentelemetry.sdk.trace import" in otel_source
    assert "InMemorySpanExporter" in otel_source
    assert "OTLPSpanExporter" in otel_source
    assert "PIIRedactingSpanProcessor" in otel_source
    assert "self.mcp_server.call_tool" not in client_source
    assert 'mode="legacy"' in client_source
    assert "_MockBankingEnv" not in server_source
    assert "resp.body.getReader" in page
    assert "TextDecoder" in page
    assert "Date.now()" not in page
    assert "await resp.text()" not in page
    assert "decision === 'modify'" in page


def test_banking_env_load_failure_is_explicit(monkeypatch):
    import knowdesk.mcp.server as server

    def explode():
        raise OSError("knowledge base unavailable")

    monkeypatch.setattr(server, "_HAS_TAU2", True)
    monkeypatch.setattr(server, "_load_tau_environment", explode)
    try:
        server._get_banking_env()
        raise AssertionError("missing environment must not fall back to a mock")
    except RuntimeError as exc:
        assert "banking_knowledge" in str(exc)
