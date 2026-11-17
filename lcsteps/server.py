"""Local web interface:  python -m lcsteps

A FastAPI app, started with uvicorn from inside this module.  It serves the
page in `lcsteps/webui/` (plain JavaScript, no build step) and a small API.

Long operations (ingest, ask) answer with newline-delimited JSON: one JSON
object per line, sent as soon as a step is finished (see `steps.py`).
"""
from __future__ import annotations

import json
import queue
import threading
import webbrowser
from pathlib import Path
from typing import Callable, Iterator

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .chain import FolderLoader, make_splitter
from .steps import Session, ask_events, describe_chunks, ingest_events, read_settings
from langchain_core.documents import Document

WEBUI = Path(__file__).parent / "webui"
_END = object()


def _ndjson(make_events: Callable[[], Iterator[dict]]) -> StreamingResponse:
    """Stream one JSON line per event; the work runs in its own thread."""
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            for event in make_events():
                events.put(event)
        except Exception as exc:                # report it in the stream
            events.put({"step": "error", "message": str(exc)})
        finally:
            events.put(_END)

    def lines():
        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not _END:
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="langchain-steps", docs_url="/api/docs", redoc_url=None)
    lock = threading.Lock()                     # one pipeline at a time

    def locked(make_events):
        def run():
            with lock:
                yield from make_events()
        return run

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(OSError)
    async def bad_request(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    def page(name: str, media_type: str | None = None):
        return FileResponse(WEBUI / name, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def index():
        return page("index.html")

    @app.get("/app.js", include_in_schema=False)
    def app_js():
        return page("app.js", "text/javascript")

    @app.get("/style.css", include_in_schema=False)
    def style_css():
        return page("style.css")

    @app.get("/api/state")
    def get_state():
        return session.state()

    @app.post("/api/files")
    def post_files(body: dict = Body(...)):
        """List the documents of a folder."""
        return {"files": [{"source": str(p), "title": p.stem.replace("_", " ").title(), "chars": p.stat().st_size}
                          for p in FolderLoader(body.get("folder", "")).paths()]}

    @app.post("/api/split")
    def post_split(body: dict = Body(...)):
        """Live preview: split one document with the given settings, store nothing."""
        chunk_size, overlap = read_settings(body)
        if "text" in body:
            text = str(body["text"])
        else:
            allowed = {str(p): p for p in FolderLoader(body.get("folder", "")).paths()}
            text = allowed[body["source"]].read_text(encoding="utf-8")
        chunks = make_splitter(chunk_size, overlap).split_documents([Document(page_content=text)])
        return {"chars": len(text), "chunks": describe_chunks(chunks)}

    @app.post("/api/ingest")
    def post_ingest(body: dict = Body(...)):
        """Load, split, embed and store; streams one event per step."""
        read_settings(body)
        FolderLoader(str(body.get("folder") or "demo_data")).paths()        # fail early on a bad folder
        return _ndjson(locked(lambda: ingest_events(session, body)))

    @app.post("/api/ask")
    def post_ask(body: dict = Body(...)):
        """Run the chain; streams every event the chain's runnables emit."""
        question = str(body.get("question", "")).strip()
        if not question:
            raise ValueError("the question is empty")
        k = max(1, int(body.get("k", 4)))
        return _ndjson(locked(lambda: ask_events(session, question, k)))

    @app.post("/api/reset")
    def post_reset():
        with lock:
            session.reset()
        return {"stats": session.stats()}

    return app


def serve(session: Session, host: str = "127.0.0.1", port: int = 8766, open_browser: bool = True) -> None:
    """Start uvicorn in this process and (optionally) open the page."""
    url = f"http://{host}:{port}/"
    print(f"langchain-steps web interface: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(session), host=host, port=port, log_level="warning", ws="none")
