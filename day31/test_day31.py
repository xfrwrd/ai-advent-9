"""Day 31. The model is mocked. Git tests talk to the real MCP server."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.index_store import load_index
from day31.corpus import collect, load_documents
from day31.help import UNKNOWN, answer
from day31.index_docs import build_index
from day31.mcp_client import call_tool_sync
from day31.route import plan_for


def _git(repo: Path, *args: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Dev",
            "GIT_AUTHOR_EMAIL": "dev@example.com",
            "GIT_COMMITTER_NAME": "Dev",
            "GIT_COMMITTER_EMAIL": "dev@example.com",
        }
    )
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env)


def _repo() -> Path:
    path = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "init")
    _git(path, "checkout", "-b", "feature-docs")
    return path


def _docs_root() -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "README.md").write_text("# Project\n\nКорневое описание.\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "architecture.md").write_text(
        "# Архитектура\n\nDay 30 запускается командой python -m day30 и слушает 127.0.0.1:8030.\n",
        encoding="utf-8",
    )
    (root / "docs" / "openapi.yaml").write_text("openapi: 3.0.0\ninfo:\n  title: none\n", encoding="utf-8")
    (root / "day28").mkdir()
    (root / "day28" / "README.md").write_text("# Day 28\n\nЛокальный RAG и Ollama.\n", encoding="utf-8")
    (root / "day28" / "agent.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (root / ".env").write_text("DAY30_API_KEY=super-secret-value\n", encoding="utf-8")
    (root / "docs" / ".env").write_text("TOKEN=super-secret-value\n", encoding="utf-8")
    (root / "docs" / "key.pem").write_text("-----BEGIN\n", encoding="utf-8")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "README.md").write_text("# venv\nsuper-secret-value\n", encoding="utf-8")
    (root / "secrets").mkdir()
    (root / "secrets" / "id_rsa").write_text("-----BEGIN PRIVATE\n", encoding="utf-8")
    return root


class CorpusTests(unittest.TestCase):
    def test_readme_and_docs_are_loaded(self) -> None:
        root = _docs_root()
        relative = [path.relative_to(root).as_posix() for path in collect(root)]
        self.assertIn("README.md", relative)
        self.assertIn("docs/architecture.md", relative)
        self.assertIn("docs/openapi.yaml", relative)
        self.assertIn("day28/README.md", relative)
        documents = load_documents(root)
        joined = "\n".join(document.content for document in documents)
        self.assertIn("127.0.0.1:8030", joined)
        self.assertIn("Локальный RAG", joined)

    def test_secrets_and_source_are_excluded(self) -> None:
        root = _docs_root()
        relative = [path.relative_to(root).as_posix() for path in collect(root)]
        self.assertNotIn(".env", relative)
        self.assertNotIn("docs/.env", relative)
        self.assertNotIn("docs/key.pem", relative)
        self.assertFalse(any(".venv" in item or item.endswith(".py") or "id_rsa" in item for item in relative))
        joined = "\n".join(document.content for document in load_documents(root))
        self.assertNotIn("super-secret-value", joined)

    def test_reindex_does_not_duplicate(self) -> None:
        root = _docs_root()
        destination = root / "out" / "docs.sqlite"
        build_index(root, destination)
        first = load_index(destination)
        build_index(root, destination)
        second = load_index(destination)
        ids = [chunk.chunk_id for chunk in second.chunks]
        self.assertEqual([chunk.chunk_id for chunk in first.chunks], ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 0)
        self.assertTrue(any("lines" in chunk.section for chunk in second.chunks))

    def test_project_corpus_includes_real_docs(self) -> None:
        relative = [path.relative_to(PROJECT_ROOT).as_posix() for path in collect(PROJECT_ROOT)]
        self.assertIn("README.md", relative)
        self.assertIn("docs/architecture.md", relative)
        self.assertIn("day30/README.md", relative)
        self.assertNotIn("day31/README.md", relative)
        self.assertNotIn(".env", relative)
        self.assertFalse(any(item.endswith(".py") or item.startswith(".venv") for item in relative))


class RouteTests(unittest.TestCase):
    def test_questions(self) -> None:
        docs = plan_for("/help Как устроен RAG в этом проекте?")
        self.assertTrue(docs.use_docs)
        self.assertEqual(docs.tools, ())
        folder = plan_for("Что находится в папке day28?")
        self.assertTrue(folder.use_docs)
        self.assertEqual(folder.tools, ())
        branch = plan_for("/help Какая текущая Git-ветка?")
        self.assertFalse(branch.use_docs)
        self.assertEqual(branch.tools, ("git_current_branch",))
        both = plan_for("Какая ветка и как устроен Day 30?")
        self.assertTrue(both.use_docs)
        self.assertEqual(both.tools, ("git_current_branch",))


class HelpTests(unittest.TestCase):
    def _index(self) -> Path:
        root = _docs_root()
        path = root / "docs.sqlite"
        build_index(root, path)
        return path

    def test_docs_question_uses_sources_and_not_git(self) -> None:
        seen: list[str] = []

        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                self.prompt = prompt
                return "Day 30 слушает 127.0.0.1:8030."

        model = Model()
        result = answer(
            "/help Как запустить Day 30?",
            index_path=self._index(),
            llm=model,
            mcp=lambda name, arguments=None: seen.append(name) or {},
        )
        self.assertTrue(result.llm_called)
        self.assertEqual(seen, [])
        self.assertTrue(any(source["source"] == "docs/architecture.md" for source in result.sources))
        self.assertIn("8030", model.prompt)
        self.assertIn("127.0.0.1:8030", result.answer)

    def test_git_question_goes_through_mcp(self) -> None:
        seen: list[str] = []

        def mcp(name: str, arguments: dict | None = None) -> dict:
            seen.append(name)
            return {"detached": False, "branch": "feature-docs", "commit": "abc123"}

        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                self.prompt = prompt
                return "Текущая ветка feature-docs."

        model = Model()
        result = answer("/help Какая текущая Git-ветка?", llm=model, mcp=mcp)
        self.assertEqual(seen, ["git_current_branch"])
        self.assertEqual(result.mcp[0]["tool"], "git_current_branch")
        self.assertEqual(result.mcp[0]["result"]["branch"], "feature-docs")
        self.assertIn("feature-docs", model.prompt)
        self.assertNotIn("subprocess", (PROJECT_ROOT / "day31" / "help.py").read_text(encoding="utf-8"))
        self.assertNotIn("rev-parse", (PROJECT_ROOT / "day31" / "help.py").read_text(encoding="utf-8"))

    def test_unknown_docs_question_does_not_call_the_model(self) -> None:
        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                raise AssertionError("model should stay idle")

        result = answer(
            "/help Какой пароль от офисного Wi-Fi?",
            index_path=self._index(),
            llm=Model(),
        )
        self.assertFalse(result.llm_called)
        self.assertEqual(result.answer, UNKNOWN)
        self.assertEqual(result.sources, [])

    def test_missing_mcp_is_reported(self) -> None:
        def mcp(name: str, arguments: dict | None = None) -> dict:
            raise RuntimeError("stdio closed")

        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                raise AssertionError("model should stay idle")

        result = answer("/help Какая текущая Git-ветка?", llm=Model(), mcp=mcp)
        self.assertFalse(result.llm_called)
        self.assertIn("mcp_unavailable", str(result.mcp))
        self.assertIn("stdio closed", result.answer)

    def test_missing_ollama_is_reported(self) -> None:
        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                raise RuntimeError("Нет соединения с Ollama по адресу http://localhost:11434.")

        result = answer("/help Как запустить Day 30?", index_path=self._index(), llm=Model())
        self.assertFalse(result.llm_called)
        self.assertIn("Ollama", result.answer)
        self.assertTrue(result.sources)

    def test_missing_index_does_not_invent_an_answer(self) -> None:
        class Model:
            def ask(self, prompt: str, *, think: bool | None = None) -> str:
                raise AssertionError("model should stay idle")

        result = answer(
            "/help Как устроен RAG?",
            index_path=Path("/tmp/day31-missing-index.sqlite"),
            llm=Model(),
        )
        self.assertFalse(result.llm_called)
        self.assertIn("index_docs", result.answer)
        self.assertEqual(result.sources, [])


class McpGitTests(unittest.TestCase):
    def test_branch_comes_from_mcp(self) -> None:
        repo = _repo()
        payload = call_tool_sync("git_current_branch", repo=repo)
        self.assertEqual(payload["branch"], "feature-docs")
        self.assertFalse(payload["detached"])
        self.assertTrue(payload["commit"])

    def test_detached_head(self) -> None:
        repo = _repo()
        _git(repo, "checkout", "--detach")
        payload = call_tool_sync("git_current_branch", repo=repo)
        self.assertTrue(payload["detached"])
        self.assertIsNone(payload["branch"])
        self.assertIn("detached", payload["message"])

    def test_not_a_repository(self) -> None:
        empty = Path(tempfile.mkdtemp())
        payload = call_tool_sync("git_current_branch", repo=empty)
        self.assertEqual(payload["error"], "not_a_git_repo")
