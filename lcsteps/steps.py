"""Watch the LangChain objects of `chain.py` work, and describe each step.

Two generators, each yielding one small dict per finished step:

* `ingest_events`  load -> split -> embed -> store, by calling the loader,
  the splitter and the vector store one document at a time;
* `ask_events`     runs the chain with `chain.astream_events(...)`, LangChain's
  own event stream: every runnable in the chain reports when it starts, what
  it streams and what it returns.  Nothing is instrumented by hand.

The web interface (`server.py`) sends these dicts to the page as they appear.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Iterator

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue

from .chain import PROMPT, FolderLoader, build_chain, make_splitter, make_store
from .offline import HashEmbeddings

MAX_TOKENS_SHOWN = 48
DEFAULTS = {"folder": "demo_data", "chunk_size": 500, "overlap": 80, "k": 4}


class Session:
    """What the server keeps between requests: the models and the vector store."""

    def __init__(self, embeddings: Embeddings, model: BaseChatModel):
        self.embeddings = embeddings
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.store = make_store(self.embeddings)
        self.documents = 0

    def stats(self) -> dict:
        return {"documents": self.documents, "chunks": len(self.store.store)}

    def state(self) -> dict:
        return {
            "embeddings": type(self.embeddings).__name__,
            "model": getattr(self.model, "model_name", None) or getattr(self.model, "model", None)
                     or type(self.model).__name__,
            "store": type(self.store).__name__,
            "stats": self.stats(),
            "defaults": DEFAULTS,
            "template": [{"role": role, "template": message.prompt.template}
                         for role, message in zip(("system", "human"), PROMPT.messages)],
        }


# ---- describing things ---------------------------------------------------------
def read_settings(body: dict) -> tuple[int, int]:
    chunk_size = int(body.get("chunk_size", DEFAULTS["chunk_size"]))
    overlap = int(body.get("overlap", DEFAULTS["overlap"]))
    if chunk_size < 20:
        raise ValueError("chunk size must be at least 20")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be 0 or more and smaller than the chunk size")
    return chunk_size, overlap


def pasted_documents(body: dict) -> list[Document]:
    return [Document(page_content=str(d["text"]),
                     metadata={"source": str(d["source"]), "title": str(d.get("title") or d["source"])})
            for d in body.get("documents", [])]


def describe_chunks(chunks: list[Document]) -> list[dict]:
    """Each chunk with where it starts and how much it repeats of the one before."""
    out, previous_end = [], 0
    for position, chunk in enumerate(chunks):
        start = chunk.metadata.get("start_index", 0)
        text = chunk.page_content
        out.append({"position": position, "text": text, "chars": len(text), "start": start,
                    "overlap": max(0, min(len(text), previous_end - start)) if position else 0})
        previous_end = start + len(text)
    return out


def describe_vector(embeddings: Embeddings, text: str, vector: list[float]) -> dict:
    out: dict[str, Any] = {"dim": len(vector), "nonzero": sum(1 for v in vector if v),
                           "vector": [round(float(v), 4) for v in vector]}
    if isinstance(embeddings, HashEmbeddings):      # show how the hashing trick fills the vector
        tokens = embeddings.tokens(text)
        out["token_count"] = len(tokens)
        out["tokens"] = [{"token": t, "bucket": embeddings.bucket(t)} for t in tokens[:MAX_TOKENS_SHOWN]]
    return out


def text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)


def plain(value: Any) -> Any:
    """LangChain objects as JSON-ready values."""
    if isinstance(value, Document):
        return {"text": value.page_content, **{k: plain(v) for k, v in value.metadata.items()}}
    if isinstance(value, BaseMessage):
        return {"role": value.type, "content": text_of(value)}
    if isinstance(value, PromptValue):
        return [plain(m) for m in value.to_messages()]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---- A: build the index --------------------------------------------------------
def ingest_events(session: Session, body: dict) -> Iterator[dict]:
    chunk_size, overlap = read_settings(body)
    loader = FolderLoader(str(body.get("folder") or DEFAULTS["folder"]))
    pasted = pasted_documents(body)
    files = [{"source": str(p), "title": p.stem.replace("_", " ").title()} for p in loader.paths()]
    files += [{"source": d.metadata["source"], "title": d.metadata["title"], "pasted": True} for d in pasted]
    yield {"step": "plan", "files": files, "chunk_size": chunk_size, "overlap": overlap}

    started = time.perf_counter()
    session.reset()                                 # a rebuild: the store starts empty
    splitter = make_splitter(chunk_size, overlap)
    total = 0
    for index, doc in enumerate([*loader.lazy_load(), *pasted]):
        yield {"step": "load", "doc": index, "source": doc.metadata["source"], "chars": len(doc.page_content),
               "metadata": plain(doc.metadata)}

        chunks = splitter.split_documents([doc])
        yield {"step": "split", "doc": index, "chunks": describe_chunks(chunks)}

        # One LangChain call embeds the chunks and stores them; the vectors are read back to show them.
        ids = session.store.add_documents(chunks) if chunks else []
        for position, (chunk_id, chunk) in enumerate(zip(ids, chunks)):
            vector = session.store.store[chunk_id]["vector"]
            yield {"step": "embed", "doc": index, "position": position, "of": len(chunks),
                   **describe_vector(session.embeddings, chunk.page_content, vector)}
        session.documents += 1
        total += len(chunks)
        yield {"step": "store", "doc": index, "source": doc.metadata["source"], "rows": len(chunks),
               "first_id": ids[0] if ids else None, "stats": session.stats()}

    yield {"step": "done", "documents": session.documents, "chunks": total, "stats": session.stats(),
           "ms": round((time.perf_counter() - started) * 1000, 1)}


# ---- B: run the chain ----------------------------------------------------------
CARDS = {"retrieve": "retrieve", "format_docs": "format"}


def card_of(kind: str, name: str) -> str | None:
    """Which card of the page an event belongs to (None: only the event log shows it)."""
    if kind == "on_custom_event":
        return "embed_query" if name == "query_embedded" else None
    for prefix, card in (("on_chat_model", "model"), ("on_llm", "model"), ("on_prompt", "prompt"),
                         ("on_parser", "parser")):
        if kind.startswith(prefix):
            return card
    return CARDS.get(name)


def event_data(session: Session, card: str | None, phase: str, data: dict) -> dict:
    if card == "embed_query":
        return describe_vector(session.embeddings, data["text"], data["vector"])
    if card == "model":
        if phase == "start":
            return {"messages": plain(data.get("input", {}).get("messages", [[]])[0])}
        if phase == "stream":
            return {"token": text_of(data["chunk"])}
        output = data["output"]
        return {"content": text_of(output), "usage": plain(getattr(output, "usage_metadata", None))}
    if card == "parser" and phase == "stream":
        return {"chunk": plain(data["chunk"])}
    if card and phase == "end":
        return {"output": plain(data.get("output"))}
    return {}


def _sync(events: AsyncIterator[dict]) -> Iterator[dict]:
    """Read an async event stream from ordinary code (the event stream is async-only)."""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(events.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(events.aclose())
        loop.close()


def ask_events(session: Session, question: str, k: int) -> Iterator[dict]:
    rag = build_chain(session.store, session.model, k)
    started = time.perf_counter()
    count, answer = 0, ""
    for event in _sync(rag.astream_events(question, version="v2")):
        kind, name = event["event"], event["name"]
        phase = "end" if kind == "on_custom_event" else kind.rsplit("_", 1)[1]
        card = card_of(kind, name)
        count += 1
        if kind == "on_chain_end" and name == "rag_chain":
            answer = event["data"].get("output") or ""
        yield {"step": "event", "event": kind, "name": name, "card": card, "phase": phase,
               "ms": round((time.perf_counter() - started) * 1000, 1),
               "data": event_data(session, card, phase, event["data"])}
    yield {"step": "done", "answer": answer, "events": count,
           "ms": round((time.perf_counter() - started) * 1000, 1)}
