"""Offline stand-ins, so the demo runs with no API key and no model download.

Both are real LangChain components: `HashEmbeddings` implements the
`Embeddings` interface and `ExtractiveChatModel` is a `BaseChatModel`, so the
chain treats them exactly like OpenAI, Anthropic or Ollama classes.  They are
deliberately simple: good enough to *see* the chain work, not to ship.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_TOKEN = re.compile(r"[a-z0-9]+")
# Very common words carry no topical signal; dropping them stops "is"/"the"
# from dominating the similarity score.
STOPWORDS = frozenset("""
a an and are as at be but by do does for from has have how i if in into is it its
my of on or our so that the their there these they this to was we what when where
which who why will with you your
""".split())


def words(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


class HashEmbeddings(Embeddings):
    """Bag-of-words embedding via the "hashing trick".

    Each word (and each word pair) is hashed to one of `dim` buckets; the
    vector counts hits per bucket and is L2-normalised.  It captures word
    overlap only, no meaning.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def tokens(self, text: str) -> list[str]:
        ws = words(text)
        return ws + [f"{a} {b}" for a, b in zip(ws, ws[1:])]

    def bucket(self, token: str) -> int:
        """The vector position a token lands in."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self.tokens(text):
                vec[self.bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_BLOCK = re.compile(r"^\[(\d+)\] \(.*?\)\n", re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveChatModel(BaseChatModel):
    """Not a language model: it quotes the context sentences that share the
    most words with the question, with their [n] source numbers.

    It reads the same messages a real chat model would get (the system message
    with the numbered context, then the question) and streams its reply word
    by word, so the page can show token streaming without an API key.
    """

    sentences: int = 2

    @property
    def _llm_type(self) -> str:
        return "extractive-offline"

    def _answer(self, messages: list[BaseMessage]) -> str:
        question = next((str(m.content) for m in reversed(messages) if m.type == "human"), "")
        context = "\n".join(str(m.content) for m in messages if m.type == "system")
        wanted = set(words(question))
        parts = _BLOCK.split(context)           # [intro, "1", text, "2", text, ...]
        sentences = [(number, sentence.strip())
                     for number, text in zip(parts[1::2], parts[2::2])
                     for sentence in _SENTENCE.split(text)
                     if sentence.strip() and not sentence.lstrip().startswith("#")]     # headings say little
        # A question word found in few sentences says more than one found in many.
        seen_in = {w: sum(1 for _, s in sentences if w in words(s)) for w in wanted}
        scored = []
        for number, sentence in sentences:
            score = sum(1 / seen_in[w] for w in wanted & set(words(sentence)))
            if score:
                scored.append((score, -len(scored), f"{sentence} [{number}]"))
        if not scored:
            return "I don't know: the context has nothing on this."
        best = sorted(scored, reverse=True)[: self.sentences]
        return " ".join(text for _, _, text in sorted(best, key=lambda item: -item[1]))

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._answer(messages)))])

    def _stream(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for piece in re.findall(r"\S+\s*", self._answer(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
