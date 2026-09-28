"""Day 21 indexing tests. Embeddings are a deterministic fake, not a paid API."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.chunking import FixedSizeChunker, StructureChunker
from day21.documents import Document, load_documents
from day21.index_store import load_index, save_index


class FakeEmbedder:
    name = "fake-embedder"
    dimension = 4

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.append([float(len(text) % 7), 1.0, 0.0, float(len(text.split()) % 5)])
        return vectors


class Day21Tests(unittest.TestCase):
    def test_load_documents_reads_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("# Title\n\nHello knowledge base.\n", encoding="utf-8")
            (root / "agent.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / "skip.bin").write_text("nope", encoding="utf-8")
            docs = load_documents(root)
        sources = {doc.source for doc in docs}
        self.assertEqual(sources, {"notes.md", "agent.py"})
        note = next(doc for doc in docs if doc.source == "notes.md")
        self.assertEqual(note.title, "notes.md")
        self.assertIn("Hello", note.content)

    def test_fixed_size_does_not_cut_words_and_overlaps(self) -> None:
        words = " ".join(f"word{i:04d}" for i in range(80))
        doc = Document("demo.txt", "demo.txt", words, {"file_type": "txt"})
        chunks = FixedSizeChunker(size=100, overlap=20).chunk(doc)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertFalse(chunk.text.endswith("wor"))
            self.assertLessEqual(len(chunk.text), 100)
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))

    def test_structure_keeps_markdown_sections_and_python_defs(self) -> None:
        markdown = Document(
            "readme.md",
            "readme.md",
            "# Intro\n\nAlpha.\n\n## Setup\n\nBeta.\n",
            {"file_type": "md"},
        )
        sections = [chunk.section for chunk in StructureChunker().chunk(markdown)]
        self.assertEqual(sections, ["Intro", "Setup"])

        code = Document(
            "agent.py",
            "agent.py",
            "class Agent:\n    def run(self):\n        return 1\n\ndef helper():\n    return 2\n",
            {"file_type": "py"},
        )
        py_chunks = StructureChunker().chunk(code)
        self.assertEqual([chunk.section for chunk in py_chunks], ["Agent", "Agent.run", "helper"])
        run = next(chunk for chunk in py_chunks if chunk.section == "Agent.run")
        self.assertIn("def run", run.text)
        self.assertIn("return 1", run.text)

    def test_metadata_and_unique_ids(self) -> None:
        doc = Document("a.py", "a.py", "def one():\n    pass\n\ndef two():\n    pass\n", {"file_type": "py"})
        chunks = StructureChunker().chunk(doc)
        ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        for chunk in chunks:
            self.assertTrue(chunk.chunk_id)
            self.assertEqual(chunk.source, "a.py")
            self.assertEqual(chunk.title, "a.py")
            self.assertTrue(chunk.section)
            self.assertEqual(chunk.chunking_strategy, "structure")

    def test_index_roundtrip_matches_chunk_count(self) -> None:
        doc = Document("a.md", "a.md", "# One\n\n" + ("token " * 40) + "\n\n# Two\n\nend\n", {"file_type": "md"})
        chunks = FixedSizeChunker(size=50, overlap=10).chunk(doc)
        vectors = FakeEmbedder().embed_batch([chunk.text for chunk in chunks])
        self.assertEqual(len(vectors), len(chunks))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.sqlite"
            save_index(path, chunks, vectors, FakeEmbedder.name, FakeEmbedder.dimension)
            loaded = load_index(path)
        self.assertEqual(len(loaded.chunks), len(chunks))
        self.assertEqual(len(loaded.vectors), len(chunks))
        self.assertEqual(loaded.model, "fake-embedder")
        self.assertEqual(loaded.dimension, 4)
        self.assertEqual({chunk.chunk_id for chunk in loaded.chunks}, {chunk.chunk_id for chunk in chunks})
        self.assertTrue(all(len(vector) == 4 for vector in loaded.vectors))


if __name__ == "__main__":
    unittest.main()
