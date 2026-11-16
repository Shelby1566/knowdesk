import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# 严格只读白名单：写操作/转账操作绝不允许投机性执行
SAFE_SPECULATIVE_TOOLS = {
    "get_card_info",
    "get_account_balance",
    "get_recent_transactions",
    "search_policy_docs",
    "get_exchange_rates",
}

# 模式识别正则 (<1ms 匹配)
_RE_CARD_LAST4 = re.compile(r"(?:card|ending in|末四位|尾号|卡号)[^\d]{0,10}(\d{4})\b", re.IGNORECASE)
_RE_ACCOUNT_ID = re.compile(r"\b(acc_[a-zA-Z0-9_-]+|user_[a-zA-Z0-9_-]+)\b", re.IGNORECASE)
_RE_POLICY_INTENT = re.compile(
    r"(annual\s*fee|fee\s*waiver|年费|减免|chargeback|dispute|wire\s*limit|转账限额|apr|interest\s*rate)",
    re.IGNORECASE,
)
_RE_TRANSACTION_INTENT = re.compile(
    r"(recent\s*transactions?|transactions?|charges?|statements?|账单|消费记录|交易记录|明细)",
    re.IGNORECASE,
)

@dataclass
class PrefetchItem:
    key: str
    tool_name: str
    args: dict[str, Any]
    data: Any
    latency_ms: float
    created_at: float
    ttl_seconds: float = 60.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

class SpeculativePrefetchEngine:
    """
    投机性并行预执行与预取引擎 (Speculative Tool Execution & Prefetch Engine)
    
    原理：
    用户提交消息到 LLM 首 Token 生成 (TTFT) 通常存在 800ms - 2000ms 纯计算等待空窗。
    本引擎在收到用户提问的瞬间，通过轻量级正则与意图解析器 (<2ms) 预判可能涉及的只读数据，
    与 LLM 首 Token 推理在后台并行异步执行预取 (Prefetch)。
    当主 Agent 稍后分发只读工具时，直接从内存 Cache 命中，实现 0ms 延迟响应，显著提升端到端交互吞吐。
    
    安全红线：
    - 严格遵循只读白名单 (SAFE_SPECULATIVE_TOOLS)，绝不对转账、销户、改密等写操作进行投机性预执行。
    - 设定内存 TTL (默认 60s) 与 LRU 上限，防止未采纳预取产生内存泄漏。
    """

    def __init__(
        self,
        retriever_fn: Callable[[str], Any] | None = None,
        mock_banking_fetcher: Callable[[str, dict[str, Any]], Any] | None = None,
        max_cache_items: int = 200,
        default_ttl_seconds: float = 60.0,
    ) -> None:
        self.retriever_fn = retriever_fn
        self.mock_banking_fetcher = mock_banking_fetcher or self._default_mock_fetcher
        self.max_cache_items = max_cache_items
        self.default_ttl_seconds = default_ttl_seconds

        # session_id -> { cache_key: PrefetchItem }
        self._cache: dict[str, dict[str, PrefetchItem]] = {}
        self._stats = {
            "total_prefetches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "saved_latency_ms": 0.0,
        }

    @staticmethod
    def _default_mock_fetcher(tool_name: str, args: dict[str, Any]) -> Any:
        """默认轻量级 Mock 预取数据生成器"""
        if tool_name == "get_card_info":
            last4 = args.get("last4", "0000")
            return {
                "card_id": f"card_mock_{last4}",
                "last4": last4,
                "product": "diamond_elite" if last4 == "4589" else "sapphire_reserve",
                "status": "ACTIVE",
                "annual_spend": 34500.0 if last4 == "4589" else 12000.0,
                "late_payment_count_12m": 0 if last4 == "4589" else 1,
            }
        if tool_name == "get_recent_transactions":
            return [
                {"id": "tx_001", "amount": 128.50, "merchant": "Whole Foods Market", "status": "SETTLED"},
                {"id": "tx_002", "amount": 45.00, "merchant": "Shell Gas", "status": "SETTLED"},
            ]
        return {"status": "ok", "mock_prefetched": True, "tool": tool_name, "args": args}

    def extract_speculative_intents(self, user_text: str) -> list[dict[str, Any]]:
        """超快速启发式意图与实体抽取器 (<2ms)"""
        intents: list[dict[str, Any]] = []

        # 1. 抽取卡号末四位
        m_card = _RE_CARD_LAST4.search(user_text)
        if m_card:
            last4 = m_card.group(1)
            intents.append({
                "tool_name": "get_card_info",
                "args": {"last4": last4},
                "cache_key": f"get_card_info:{last4}",
            })

        # 2. 抽取交易明细查询意图
        if _RE_TRANSACTION_INTENT.search(user_text):
            intents.append({
                "tool_name": "get_recent_transactions",
                "args": {"limit": 5},
                "cache_key": "get_recent_transactions:limit=5",
            })

        # 3. 抽取政策规则查询意图
        m_policy = _RE_POLICY_INTENT.search(user_text)
        if m_policy:
            topic = m_policy.group(1).lower().strip()
            intents.append({
                "tool_name": "search_policy_docs",
                "args": {"query": topic},
                "cache_key": f"search_policy_docs:{topic}",
            })

        return intents

    async def prefetch_speculative(self, session_id: str, user_text: str) -> list[str]:
        """
        在后台异步触发投机性预取
        返回本次触发预取的 cache_keys 列表
        """
        candidate_intents = self.extract_speculative_intents(user_text)
        if not candidate_intents:
            return []

        if session_id not in self._cache:
            self._cache[session_id] = {}

        tasks = []
        keys = []
        for intent in candidate_intents:
            tool_name = intent["tool_name"]
            # 严格安全白名单校验
            if tool_name not in SAFE_SPECULATIVE_TOOLS:
                continue

            cache_key = intent["cache_key"]
            args = intent["args"]
            keys.append(cache_key)
            tasks.append(self._execute_single_prefetch(session_id, tool_name, args, cache_key))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return keys

    async def _execute_single_prefetch(
        self, session_id: str, tool_name: str, args: dict[str, Any], cache_key: str
    ) -> None:
        t0 = time.perf_counter()
        data: Any = None
        try:
            if tool_name == "search_policy_docs" and self.retriever_fn:
                # 预取知识库
                if asyncio.iscoroutinefunction(self.retriever_fn):
                    data = await self.retriever_fn(args.get("query", ""))
                else:
                    data = self.retriever_fn(args.get("query", ""))
            else:
                # 预取核心银行只读数据
                if asyncio.iscoroutinefunction(self.mock_banking_fetcher):
                    data = await self.mock_banking_fetcher(tool_name, args)
                else:
                    data = self.mock_banking_fetcher(tool_name, args)
        except Exception as e:
            data = {"error": f"speculative_prefetch_failed: {e}"}

        latency_ms = (time.perf_counter() - t0) * 1000.0

        item = PrefetchItem(
            key=cache_key,
            tool_name=tool_name,
            args=args,
            data=data,
            latency_ms=latency_ms,
            created_at=time.time(),
            ttl_seconds=self.default_ttl_seconds,
        )

        sess_cache = self._cache.setdefault(session_id, {})
        sess_cache[cache_key] = item
        self._stats["total_prefetches"] += 1

        # 维持 LRU / 容量上限
        if len(sess_cache) > self.max_cache_items:
            oldest_key = min(sess_cache.keys(), key=lambda k: sess_cache[k].created_at)
            sess_cache.pop(oldest_key, None)

    def claim(self, session_id: str, tool_name: str, args: dict[str, Any]) -> tuple[bool, Any, float]:
        """
        工具真正执行时尝试认领预取缓存：
        - 返回 (is_hit, cached_data, latency_saved_ms)
        - 如果命中且未过期：返回预取数据，记录节约时长
        - 如果未命中或已过期：返回 (False, None, 0.0)
        """
        sess_cache = self._cache.get(session_id)
        if not sess_cache:
            self._stats["cache_misses"] += 1
            return False, None, 0.0

        # 生成匹配 key
        matched_item: PrefetchItem | None = None
        matched_key: str | None = None

        for k, item in sess_cache.items():
            if item.tool_name == tool_name:
                # 检查主要入参一致性 (如 last4 或 query)
                if all(item.args.get(ak) == args.get(ak) for ak in item.args if ak in args):
                    if not item.is_expired:
                        matched_item = item
                        matched_key = k
                        break

        if matched_item and matched_key:
            data = matched_item.data
            saved_ms = matched_item.latency_ms
            # 认领后消费删除，避免重复投机数据滞留
            sess_cache.pop(matched_key, None)
            self._stats["cache_hits"] += 1
            self._stats["saved_latency_ms"] += saved_ms
            return True, data, saved_ms

        self._stats["cache_misses"] += 1
        return False, None, 0.0

    def stats(self) -> dict[str, Any]:
        """导出投机预取实时监控指标"""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = (self._stats["cache_hits"] / total) if total > 0 else 0.0
        return {
            "total_prefetches": self._stats["total_prefetches"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "hit_rate": round(hit_rate, 4),
            "saved_latency_ms": round(self._stats["saved_latency_ms"], 2),
        }
