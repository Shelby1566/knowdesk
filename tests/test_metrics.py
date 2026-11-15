from eval.metrics import compute_action_recall, compute_online_doc_recall

def test_action_and_online_doc_recall():
    gold_actions = [
        {"name": "get_user_details", "arguments": {"user_id": "u1"}, "requestor": "assistant"},
        {
            "name": "call_discoverable_agent_tool",
            "arguments": {"agent_tool_name": "freeze_card", "arguments": '{"card_id": "c1"}'},
            "requestor": "assistant",
        },
    ]
    sim_msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "get_user_details", "arguments": {"user_id": "u1"}},
                {
                    "name": "call_discoverable_agent_tool",
                    "arguments": {"agent_tool_name": "freeze_card", "arguments": '{"card_id": "c_other"}'},
                },
            ],
        },
        {"role": "tool", "content": "Found document doc_A: Platinum Card Policy ..."},
    ]
    name_rec, exact_rec = compute_action_recall(sim_msgs, gold_actions)
    # 两个工具名都调对了 (name_rec = 1.0)，其中第1个参数完全匹配 (exact_rec = 0.5)
    assert abs(name_rec - 1.0) < 1e-9
    assert abs(exact_rec - 0.5) < 1e-9

    doc_rec = compute_online_doc_recall(
        sim_msgs, ["doc_A", "doc_B"], {"doc_A", "doc_B"}, {"Platinum Card Policy": "doc_A"}
    )
    assert abs(doc_rec - 0.5) < 1e-9
