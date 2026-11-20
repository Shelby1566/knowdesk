"""One real HTTP turn against create_knowdesk_app(). No fake LLM and no fake tools.

Loads DASHSCOPE_API_KEY from the tau2-bench env file. Prints token counts only.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / "tau2-bench" / ".env", override=False)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from service.app import prefer_direct_dashscope

prefer_direct_dashscope()

if not os.getenv("DASHSCOPE_API_KEY"):
    raise SystemExit("DASHSCOPE_API_KEY is missing")

import httpx
import uvicorn

from kb.hybrid_retriever import KnowDeskHybridRetriever
from service.app import create_knowdesk_app

TICKET = (
    "客户问钻石精英卡的年费在什么条件下可以减免。"
    "请调用 KB_search，query 使用 annual fee waiver，再根据检索结果回答。"
    "不要办理冻结、转账或其他写操作。"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    searches: list[str] = []
    original = KnowDeskHybridRetriever.search_sections

    def wrapped(self, query, k=10, fusion_mode=None, acl=None, **kwargs):
        searches.append(str(query))
        return original(self, query, k=k, fusion_mode=fusion_mode, acl=acl, **kwargs)

    KnowDeskHybridRetriever.search_sections = wrapped  # type: ignore[method-assign]
    db_path = Path("/tmp/knowdesk_smoke_live.sqlite")
    if db_path.exists():
        db_path.unlink()
    app = create_knowdesk_app(db_path=db_path)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1.0)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        raise SystemExit("server did not accept HTTP")

    headers = {"X-User-Id": "smoke", "X-Tenant-Id": "smoke"}
    with httpx.Client(timeout=180.0) as client:
        created = client.post(f"http://127.0.0.1:{port}/sessions", headers=headers)
        created.raise_for_status()
        session_id = created.json()["session_id"]
        sent = client.post(
            f"http://127.0.0.1:{port}/sessions/{session_id}/messages",
            headers=headers,
            json={"content": TICKET},
        )
        sent.raise_for_status()
        body = sent.text
        trace = client.get(f"http://127.0.0.1:{port}/sessions/{session_id}/trace", headers=headers)
        trace.raise_for_status()
    server.should_exit = True
    thread.join(timeout=5)

    if "400" in body or '"event": "error"' in body or "event: error" in body:
        raise SystemExit(f"live turn failed:\n{body[:2000]}")
    summary = trace.json().get("summary") or {}
    tokens = int(summary.get("total_input_tokens") or 0)
    if tokens <= 0:
        raise SystemExit(f"trace has no input tokens: {json.dumps(summary, ensure_ascii=False)}")
    if not searches:
        raise SystemExit("KB_search did not run")
    print(
        json.dumps(
            {
                "session_id": session_id,
                "kb_search_calls": len(searches),
                "total_input_tokens": tokens,
                "total_output_tokens": summary.get("total_output_tokens"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
