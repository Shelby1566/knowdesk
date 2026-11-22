import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "serve":
        import uvicorn

        uvicorn.run("knowdesk.service.app:app", host="0.0.0.0", port=8000)
    elif cmd == "search":
        from knowdesk.kb.hybrid_retriever import KnowDeskHybridRetriever

        query = " ".join(sys.argv[2:]) or "credit card annual fee"
        for hit in KnowDeskHybridRetriever.get_instance().search(query, k=5):
            print(f"[{hit['id']}] ({hit['score']}) {hit['title']}")
    else:
        print("Usage: knowdesk [serve | search <query>]")
