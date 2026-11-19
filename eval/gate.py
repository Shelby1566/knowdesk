"""冻结的 dev / test 切分。调参只能用 dev，test 只用于上报。"""

import json
from pathlib import Path

SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"


class FrozenSplitError(RuntimeError):
    """调参运行碰到了冻结的 test 任务，或切分文件本身不封闭。"""


def load_splits(path: Path = SPLITS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_frozen_integrity(splits: dict) -> None:
    dev = [str(item) for item in splits["dev_ids"]]
    test = [str(item) for item in splits["test_ids"]]
    regression = [str(item) for item in splits["regression_ids"]]
    if len(dev) != len(set(dev)) or len(test) != len(set(test)):
        raise FrozenSplitError("split ids contain duplicates")
    overlap = sorted(set(dev) & set(test))
    if overlap:
        raise FrozenSplitError(f"dev and test overlap: {overlap[:5]}")
    if not set(regression) <= set(dev):
        raise FrozenSplitError("regression ids must be a subset of dev")
    if set(regression) & set(test):
        raise FrozenSplitError("regression ids leak into test")
    if len(dev) + len(test) != int(splits["total_tasks"]):
        raise FrozenSplitError("dev plus test does not equal total_tasks")


def tunable_ids(splits: dict, which: str = "regression10") -> list[str]:
    """调参和开发回归用的任务。test 不在这里。"""
    assert_frozen_integrity(splits)
    if which == "dev":
        return list(splits["dev_ids"])
    if which == "regression":
        return list(splits["regression_ids"])
    if which == "regression10":
        return list(splits["regression_ids"][:10])
    if which == "test":
        raise FrozenSplitError("test split is held out; do not tune on it")
    raise FrozenSplitError(f"unknown tunable split: {which}")


def report_ids(splits: dict, which: str) -> list[str]:
    """上报用。test 可以读，但不能再拿去改提示词或检索参数。"""
    assert_frozen_integrity(splits)
    if which == "test":
        return list(splits["test_ids"])
    return tunable_ids(splits, which)


def require_tunable(task_ids: list[str], splits: dict) -> None:
    """开发脚本在开跑前调用。任务号一旦落在 test 里就停止。"""
    assert_frozen_integrity(splits)
    held_out = set(splits["test_ids"])
    leaked = [task_id for task_id in task_ids if task_id in held_out]
    if leaked:
        raise FrozenSplitError(f"frozen test split leaked into a tuning run: {leaked[:8]}")


def main() -> None:
    splits = load_splits()
    assert_frozen_integrity(splits)
    print(
        json.dumps(
            {
                "gate": "ok",
                "dev": len(splits["dev_ids"]),
                "test": len(splits["test_ids"]),
                "regression": len(splits["regression_ids"]),
                "tune_on": "dev or regression only",
                "report_test_with": "eval.gate.report_ids(splits, 'test')",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
