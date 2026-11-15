import hashlib
import json
import random
import time
from typing import Any, Callable
from tools.schemas import StructuredToolError

class RetryableToolException(Exception):
    def __init__(self, message: str, error_type: str = "timeout"):
        super().__init__(message)
        self.error_type = error_type

class InvalidArgumentToolException(Exception):
    pass

def compute_idempotency_key(session_id: str, tool_name: str, args: dict[str, Any]) -> str:
    canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{session_id}||{tool_name}||{canonical_args}".encode("utf-8")).hexdigest()

class ReliableToolExecutor:
    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng_fn: Callable[[float, float], float] = random.uniform,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep_fn = sleep_fn
        self.rng_fn = rng_fn
        self.idempotency_store: dict[str, str] = {}
        self.last_call_tracker: dict[str, tuple[str, int]] = {}
        self.duplicate_intercept_count: int = 0

    def execute(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        func: Callable[[dict[str, Any]], Any],
        is_write: bool = False,
        allow_idempotent_replay: bool = True,
    ) -> str:
        idem_key = compute_idempotency_key(session_id, tool_name, args)
        if is_write and allow_idempotent_replay and idem_key in self.idempotency_store:
            return self.idempotency_store[idem_key]

        prev_key, count = self.last_call_tracker.get(session_id, ("", 0))
        if idem_key == prev_key and not is_write:
            count += 1
            self.last_call_tracker[session_id] = (idem_key, count)
            if count >= 2:
                self.duplicate_intercept_count += 1
                return StructuredToolError(
                    error_type="duplicate_call",
                    message=f"Tool '{tool_name}' called repeatedly with identical arguments.",
                    retryable=False,
                    hint="Use previous result or change parameters.",
                ).to_tool_message()
        else:
            self.last_call_tracker[session_id] = (idem_key, 1)

        attempt = 0
        while True:
            try:
                raw_out = func(args)
                out_str = raw_out if isinstance(raw_out, str) else json.dumps(raw_out, ensure_ascii=False)
                if is_write:
                    self.idempotency_store[idem_key] = out_str
                return out_str
            except InvalidArgumentToolException as e:
                return StructuredToolError(
                    error_type="invalid_arguments", message=str(e), retryable=False
                ).to_tool_message()
            except RetryableToolException as e:
                if attempt >= self.max_retries:
                    return StructuredToolError(
                        error_type="timeout" if e.error_type == "timeout" else "upstream_error",
                        message=f"Failed after {attempt + 1} attempts: {e}",
                        retryable=True,
                    ).to_tool_message()
                cap = min(self.max_delay, self.base_delay * (2 ** attempt))
                self.sleep_fn(self.rng_fn(0.0, cap))
                attempt += 1
