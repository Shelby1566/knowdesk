---
name: knowdesk-banking-sop
version: "1.0"
description: "Standard Operating Procedure for Banking Ticket Handling via KnowDesk MCP v2 Server"
---
# KnowDesk Banking Ticket Skill
1. **Policy First (`KB_search`)**: Always query `KB_search` before quoting fee schedules, APRs, or dispute rules.
2. **Identity Verification (`log_verification`)**: Verify customer name, DOB, address, email, and phone, query `get_current_time`, and call `log_verification` prior to any write action.
3. **Discoverable Tool Protocol**:
   - Agent tools: Call `unlock_discoverable_agent_tool` before `call_discoverable_agent_tool`.
   - User tools: Call `give_discoverable_user_tool` when policy requires customer-side tool execution.
4. **High-Risk Write Confirmation**: All state-mutating operations (`freeze_*`, `close_*`, `transfer_*`, `update_*`, `dispute_*`) require supervisor approval at the `human_confirm` node.
