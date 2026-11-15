from typing import Literal, TypeVar
from pydantic import BaseModel, ValidationError

ErrorType = Literal[
    "invalid_arguments", "unauthorized", "not_found", "timeout",
    "upstream_error", "guard_blocked", "user_rejected", "duplicate_call",
]

class StructuredToolError(BaseModel):
    error_type: ErrorType
    message: str
    retryable: bool = False
    hint: str | None = None
    rule_id: str | None = None

    def to_tool_message(self) -> str:
        return self.model_dump_json()

class SufficiencyOutput(BaseModel):
    sufficient: bool
    missing_aspect: str = ""
    rewritten_query: str = ""

class FaithfulnessVerdict(BaseModel):
    verdict: Literal["faithful", "unfaithful", "cannot_tell"]
    evidence: str = ""
    reason: str = ""

T = TypeVar("T", bound=BaseModel)

def validate_structured_output(
    raw_text: str, model_cls: type[T], fallback: T, failure_counter: list[int] | None = None
) -> tuple[T, bool]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        return model_cls.model_validate_json(cleaned), True
    except (ValidationError, ValueError):
        if failure_counter is not None:
            failure_counter.append(1)
        return fallback, False
