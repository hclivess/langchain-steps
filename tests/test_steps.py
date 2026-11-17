"""Run with:  python -m unittest discover -s tests"""
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from lcsteps.offline import ExtractiveChatModel, HashEmbeddings
from lcsteps.server import create_app
from lcsteps.steps import Session, ask_events, ingest_events

DEMO = str(Path(__file__).resolve().parent.parent / "demo_data")


def new_session() -> Session:
    return Session(HashEmbeddings(), ExtractiveChatModel())


def lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class StepsTest(unittest.TestCase):
    def test_ingest_reports_every_step(self):
        session = new_session()
        events = list(ingest_events(session, {"folder": DEMO, "chunk_size": 500, "overlap": 80}))
        steps = [e["step"] for e in events]
        self.assertEqual(steps[0], "plan")
        self.assertEqual(steps[-1], "done")
        self.assertEqual(steps.count("load"), 3)
        self.assertEqual(steps.count("store"), 3)
        self.assertEqual(steps.count("embed"), events[-1]["chunks"])
        self.assertEqual(session.stats(), {"documents": 3, "chunks": events[-1]["chunks"]})

    def test_chunks_fit_and_overlap(self):
        events = list(ingest_events(new_session(), {"folder": DEMO, "chunk_size": 300, "overlap": 60}))
        chunks = [c for e in events if e["step"] == "split" for c in e["chunks"]]
        self.assertTrue(all(c["chars"] <= 300 for c in chunks))
        self.assertTrue(any(c["overlap"] > 0 for c in chunks))
        self.assertTrue(all(c["overlap"] <= 60 for c in chunks))

    def test_chain_events_and_answer(self):
        session = new_session()
        list(ingest_events(session, {"folder": DEMO}))
        events = list(ask_events(session, "when is my sourdough starter ready?", 4))
        ended = [e["card"] for e in events if e["step"] == "event" and e["phase"] == "end" and e["card"]]
        self.assertEqual(ended, ["embed_query", "retrieve", "format", "prompt", "model", "parser"])
        hits = next(e for e in events if e.get("card") == "retrieve" and e["phase"] == "end")["data"]["output"]
        self.assertEqual(len(hits), 4)
        self.assertIn("sourdough", hits[0]["source"])
        self.assertTrue(any(e.get("card") == "model" and e["phase"] == "stream" for e in events))
        self.assertIn("5–7 days", events[-1]["answer"])

    def test_unknown_question(self):
        session = new_session()
        list(ingest_events(session, {"folder": DEMO}))
        events = list(ask_events(session, "zebra quantum telescope", 2))
        self.assertTrue(events[-1]["answer"].startswith("I don't know"))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(new_session()))

    def test_page_and_state(self):
        self.assertIn("langchain-steps", self.client.get("/").text)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        state = self.client.get("/api/state").json()
        self.assertEqual(state["embeddings"], "HashEmbeddings")
        self.assertEqual([m["role"] for m in state["template"]], ["system", "human"])

    def test_split_preview_stores_nothing(self):
        data = self.client.post("/api/split", json={"text": "one two three. " * 60, "chunk_size": 200,
                                                    "overlap": 40}).json()
        self.assertGreater(len(data["chunks"]), 1)
        self.assertEqual(self.client.get("/api/state").json()["stats"]["chunks"], 0)

    def test_ingest_then_ask_stream(self):
        events = lines(self.client.post("/api/ingest", json={"folder": DEMO}))
        self.assertEqual(events[-1]["step"], "done")
        events = lines(self.client.post("/api/ask", json={"question": "what water temperature for coffee", "k": 3}))
        self.assertEqual(events[-1]["step"], "done")
        self.assertIn("92", events[-1]["answer"])

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/api/ask", json={"question": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/ingest", json={"folder": "no-such-folder"}).status_code, 400)
        self.assertEqual(self.client.post("/api/split", json={"text": "x", "chunk_size": 100,
                                                              "overlap": 100}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
