import pytest

from agent.guard import SecurityState
from agent.workflow import project_messages_for_model, stage_directive
from eval.gate import FrozenSplitError, load_splits, report_ids, require_tunable, tunable_ids
from kb.chunk import Chunk
from kb.hybrid_retriever import KnowDeskHybridRetriever
from kb.loader import Document
from kb.sections import format_matched_section


def test_matched_section_is_not_the_full_document():
    full = "FULL DOCUMENT SECRET " * 30
    chunk = Chunk(
        chunk_id="card_1#2",
        doc_id="card_1",
        breadcrumb="Diamond Elite > Fees",
        body="The annual fee is $495.",
        text="Diamond Elite > Fees\nThe annual fee is $495.",
        category="card",
    )
    rendered = format_matched_section(chunk)
    assert "The annual fee is $495." in rendered
    assert "card_1#2" in rendered
    assert full not in rendered

    retriever = KnowDeskHybridRetriever.__new__(KnowDeskHybridRetriever)
    retriever.doc_map = {"card_1": Document("card_1", "Diamond Elite", full, "card")}
    retriever.doc_acl = {}
    retriever.rank_documents = lambda *args, **kwargs: [("card_1", 0.91, chunk)]
    hits = retriever.search_sections("annual fee", k=3)
    assert hits[0]["id"] == "card_1"
    assert hits[0]["chunk_id"] == "card_1#2"
    assert full not in hits[0]["content"]
    assert "The annual fee is $495." in hits[0]["content"]


def test_sop_stage_advances_and_does_not_mutate_history():
    tools = {"KB_search", "unlock_discoverable_agent_tool", "call_discoverable_agent_tool"}
    sec = SecurityState()
    history = [{"role": "user", "content": "Please waive my annual fee."}]
    assert "stage: retrieve" in stage_directive(history, sec, available_tools=tools)

    searched = [{"role": "assistant", "content": "", "tool_calls": [{"name": "KB_search", "arguments": {"query": "fee"}}]}]
    assert "stage: verify" in stage_directive(searched, sec, available_tools=tools)
    sec.verified = True
    assert "stage: unlock" in stage_directive(searched, sec, available_tools=tools)
    sec.unlocked_tools.add("waive_annual_fee")
    assert "stage: fill" in stage_directive(searched, sec, available_tools=tools)
    called = searched + [{"role": "assistant", "tool_calls": [{"name": "call_discoverable_agent_tool"}]}]
    assert "stage: confirm" in stage_directive(called, sec, available_tools=tools)

    original = [{"role": "system", "content": "base policy"}, {"role": "user", "content": "hi"}]
    projected = project_messages_for_model(original, "STAGE CARD")
    assert original[0]["content"] == "base policy"
    assert projected[0]["content"].startswith("base policy")
    assert "STAGE CARD" in projected[0]["content"]
    assert projected[-1] == {"role": "user", "content": "hi"}
    assert stage_directive(history, sec, available_tools={"freeze_card"}) == ""


def test_frozen_split_blocks_test_leak():
    splits = load_splits()
    dev = tunable_ids(splits, "dev")
    held_out = report_ids(splits, "test")
    assert len(dev) == 30
    assert len(held_out) == 67
    assert not set(dev) & set(held_out)
    require_tunable(tunable_ids(splits, "regression10"), splits)
    with pytest.raises(FrozenSplitError):
        tunable_ids(splits, "test")
    with pytest.raises(FrozenSplitError):
        require_tunable([held_out[0]], splits)


def test_section_retrieval_replaces_document_body():
    pytest.importorskip("tau2")
    from adapters.tau2_adapter import install_section_retrieval, restore_builtin_retrieval
    from tau2.knowledge.pipeline import RetrievalPipeline
    from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever

    class FakeRetriever:
        def search_sections(self, query, k=10, **kwargs):
            return [{"id": "doc_a", "content": "SECTION ONLY", "score": 0.4}]

    class RetrieverSelf:
        query_key = "query"
        top_k = 5

    state = {"doc_content_map": {"doc_a": "FULL DOCUMENT"}}

    class Pipe:
        pass

    pipe = Pipe()
    pipe.state = state
    install_section_retrieval(FakeRetriever())
    try:
        pairs = BM25Retriever.retrieve(RetrieverSelf(), {"query": "annual fee"}, state)
        assert pairs == [("doc_a", 0.4)]
        assert RetrievalPipeline.get_document_content(pipe, "doc_a") == "SECTION ONLY"
        assert "FULL DOCUMENT" not in RetrievalPipeline.get_document_content(pipe, "doc_a")
    finally:
        restore_builtin_retrieval()
    assert RetrievalPipeline.get_document_content(pipe, "doc_a") == "FULL DOCUMENT"


