"""Command line.

    python -m lcsteps                      start the web interface (same as: serve)
    python -m lcsteps serve --port 9000 --no-open
    python -m lcsteps ask "when is my sourdough starter ready?"

Models come from two optional env vars (default: offline stand-ins, no key):
    LC_MODEL=anthropic:claude-opus-5-5        LC_EMBEDDINGS=openai:text-embedding-3-small
"""
from __future__ import annotations

import argparse

from .chain import make_embeddings, make_model
from .steps import DEFAULTS, Session, ask_events, ingest_events


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m lcsteps", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    p_serve = sub.add_parser("serve", help="start the web interface (the default)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8766)
    p_serve.add_argument("--no-open", action="store_true", help="do not open the browser")
    p_ask = sub.add_parser("ask", help="index a folder and answer one question in the terminal")
    p_ask.add_argument("question")
    p_ask.add_argument("--folder", default=DEFAULTS["folder"])
    p_ask.add_argument("-k", type=int, default=DEFAULTS["k"])
    args = parser.parse_args()

    session = Session(make_embeddings(), make_model())
    if args.command == "ask":
        for event in ingest_events(session, {"folder": args.folder}):
            if event["step"] == "done":
                print(f"indexed {event['chunks']} chunks from {event['documents']} documents\n")
        for event in ask_events(session, args.question, args.k):
            if event["step"] == "event" and event["phase"] in ("start", "end"):
                print(f"{event['ms']:8.1f} ms  {event['event']:<22} {event['name']}")
            elif event["step"] == "done":
                print(f"\n{event['answer']}")
        return

    from .server import serve
    serve(session, getattr(args, "host", "127.0.0.1"), getattr(args, "port", 8766),
          open_browser=not getattr(args, "no_open", False))


if __name__ == "__main__":
    main()
