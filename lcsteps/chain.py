"""The whole LangChain program: a loader, a splitter, a vector store and one chain.

Everything here is ordinary LangChain.  `steps.py` only watches these objects
work and describes what they did.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .offline import ExtractiveChatModel, HashEmbeddings

PATTERNS = ("*.md", "*.txt")

SYSTEM = ("You answer questions using ONLY the provided context. Cite sources as [n] using the numbers "
          "in the context. If the context does not contain the answer, say you don't know.\n\n"
          "Context:\n{context}")
PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])


# ---- 1. load -------------------------------------------------------------------
class FolderLoader(BaseLoader):
    """Every .md and .txt file of a folder as one `Document` (text + metadata)."""

    def __init__(self, folder: str):
        self.folder = Path(folder)

    def paths(self) -> list[Path]:
        if not self.folder.is_dir():
            raise ValueError(f"not a folder: {self.folder}")
        return sorted(p for pattern in PATTERNS for p in self.folder.rglob(pattern))

    def lazy_load(self) -> Iterator[Document]:
        for path in self.paths():
            yield Document(page_content=path.read_text(encoding="utf-8"),
                           metadata={"source": str(path), "title": path.stem.replace("_", " ").title()})


# ---- 2. split ------------------------------------------------------------------
def make_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """Tries to cut at blank lines, then line ends, then spaces, then anywhere."""
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)


# ---- 3 + 4. embed and store ----------------------------------------------------
def make_embeddings() -> Embeddings:
    """LC_EMBEDDINGS=provider:model picks a real embedding model, e.g. openai:text-embedding-3-small."""
    name = os.environ.get("LC_EMBEDDINGS", "")
    if not name:
        return HashEmbeddings()
    from langchain.embeddings import init_embeddings
    return init_embeddings(name)


def make_model() -> BaseChatModel:
    """LC_MODEL=provider:model picks a real chat model, e.g. anthropic:claude-opus-5-5 or ollama:llama3.2."""
    name = os.environ.get("LC_MODEL", "")
    if not name:
        return ExtractiveChatModel()
    from langchain.chat_models import init_chat_model
    return init_chat_model(name)


def make_store(embeddings: Embeddings) -> InMemoryVectorStore:
    return InMemoryVectorStore(embeddings)


# ---- the chain -----------------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    """Number the retrieved chunks so the model can cite them as [n]."""
    return "\n\n".join(f"[{i}] ({doc.metadata.get('source', '?')})\n{doc.page_content}"
                       for i, doc in enumerate(docs, 1))


def build_chain(store: InMemoryVectorStore, model: BaseChatModel, k: int = 4) -> Runnable:
    """question (str)  ->  answer (str)."""

    @chain
    def retrieve(question: str) -> list[Document]:
        vector = store.embeddings.embed_query(question)
        # A custom event: anything a step wants to show appears in the event stream.
        dispatch_custom_event("query_embedded", {"text": question, "vector": vector})
        hits = store.similarity_search_with_score_by_vector(vector, k=k)
        return [Document(page_content=doc.page_content, metadata={**doc.metadata, "score": score})
                for doc, score in hits]

    rag = (
        {"context": retrieve | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | PROMPT
        | model
        | StrOutputParser()
    )
    return rag.with_config(run_name="rag_chain")
