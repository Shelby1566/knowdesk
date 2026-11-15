from typing import Any

def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) // 4 + 15 for m in messages)

def compress_context_watermark(
    messages: list[dict[str, Any]],
    high_watermark_tokens: int = 12000,
    keep_recent_messages: int = 6,
) -> tuple[list[dict[str, Any]], bool]:
    """
    M13 水位线上下文压缩：
    1. 未达高水位线时绝不修改历史（100% 保护 M18 前缀缓存）；
    2. 触发水位线时，将早期检索返回的大段文档正文替换为保留 [doc_id + title] 的轻量占位符；
    3. 严禁删除任何 assistant tool_calls 或孤立 tool 消息（确保 tool_call_id 100% 成对合法）。
    """
    if estimate_tokens(messages) < high_watermark_tokens or len(messages) <= keep_recent_messages + 1:
        return list(messages), False

    cutoff = len(messages) - keep_recent_messages
    compressed: list[dict[str, Any]] = []
    did_compress = False

    for idx, msg in enumerate(messages):
        if idx < cutoff and msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            if len(content) > 400:
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                header_preview = " | ".join(lines[:3])[:220]
                placeholder = f"[COMPRESSED_KB_RESULT: retained_headers='{header_preview}', original_chars={len(content)}]"
                new_msg = dict(msg)
                new_msg["content"] = placeholder
                compressed.append(new_msg)
                did_compress = True
                continue
        compressed.append(dict(msg))

    return compressed, did_compress
