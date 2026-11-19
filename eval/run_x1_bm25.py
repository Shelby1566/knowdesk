"""Retired local-BM25 script.

The previous version scored KnowDesk's own BM25Index and labeled the first row
as the τ baseline. That file is not a τ BM25Retriever measurement.
"""


def main() -> None:
    raise RuntimeError(
        "eval/run_x1_bm25.py is retired because it did not call τ BM25Retriever. "
        "Run eval/run_x1_full.py."
    )


if __name__ == "__main__":
    main()
