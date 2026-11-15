import sys

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "serve":
        import uvicorn
        uvicorn.run("service.app:app", host="0.0.0.0", port=8000)
    elif cmd == "search":
        from kb.hybrid_retriever import KnowDeskHybridRetriever
        q = " ".join(sys.argv[2:]) or "credit card annual fee"
        for h in KnowDeskHybridRetriever.get_instance().search(q, k=5):
            print(f"[{h['id']}] ({h['score']}) {h['title']}")
    else:
        print("Usage: knowdesk [serve | search <query>]")
