"""
Run:  pip install pytest   then   pytest -q
No real API calls are made: the AI is replaced by a fake that returns canned JSON.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import badge  # noqa: E402
import llm as llm_mod  # noqa: E402
from courses import FREE_PRICES, LEVELS, load_catalog, rank_courses  # noqa: E402
from llm import LLM, LLMError, extract_json, validate_quiz, validate_roadmap  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


# ------------------------------------------------------------ catalog
def test_catalog_is_well_formed():
    cat = load_catalog()
    assert len(cat) >= 5
    for topic, block in cat.items():
        assert block["notes"] and block["labs"], topic
        for c in block["courses"]:
            for field in ("title", "platform", "url", "level", "price", "format", "hours", "why"):
                assert field in c, (topic, c.get("title"), field)
            assert c["level"] in LEVELS
            assert c["url"].startswith("https://")
        for item in block["notes"] + block["labs"]:
            assert item["url"].startswith("https://") and item["title"] and item["note"]


def test_every_topic_level_combo_has_a_course():
    cat = load_catalog()
    for topic, block in cat.items():
        for level in LEVELS:
            for free_only in (True, False):
                for fmt in ("Any", "Video", "Text", "Interactive"):
                    assert rank_courses(block["courses"], level, fmt, free_only), (topic, level, free_only, fmt)


def test_ranking_rules():
    cat = load_catalog()["Cyber Security"]["courses"]
    free = rank_courses(cat, "Beginner", "Any", True)
    assert all(c["price"] in FREE_PRICES for _, c in free)
    assert all(c["level"] != "Advanced" for _, c in free)
    scores = [s for s, _ in free]
    assert scores == sorted(scores, reverse=True)
    # asking for Interactive must lift an interactive course above a video one of the same level
    top = rank_courses(cat, "Intermediate", "Interactive", True)[0][1]
    assert top["format"] == "Interactive"
    # Beginner never sees Advanced
    py = load_catalog()["Machine Learning"]["courses"]
    assert all(c["level"] != "Advanced" for _, c in rank_courses(py, "Beginner", "Any", True))


# ------------------------------------------------------------ JSON + validators
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is:\n[{"a": 1}]\nHope it helps') == [{"a": 1}]
    assert extract_json('text {"a": [1, 2]} text') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_validate_roadmap():
    good = {"title": "T", "summary": "S", "modules": [
        {"title": f"M{i}", "goal": "g", "topics": ["a", "b"], "practice": "p", "hours": "4 hours" if i == 0 else i + 1}
        for i in range(5)]}
    out = validate_roadmap(good)
    assert len(out["modules"]) == 5 and out["modules"][0]["hours"] == 3  # bad hours -> default
    with pytest.raises(LLMError):
        validate_roadmap({"modules": [{"title": "only one"}]})
    with pytest.raises(LLMError):
        validate_roadmap("nonsense")


def test_validate_quiz_keeps_correct_answer_after_shuffle():
    raw = [{"question": f"Q{i}", "options": ["right", "w1", "w2", "w3"], "answer_index": 0, "explanation": "e"}
           for i in range(6)]
    raw += [{"question": "bad index", "options": ["a", "b"], "answer_index": 9},
            {"question": "", "options": ["a", "b"], "answer_index": 0},
            "not a dict"]
    quiz = validate_quiz(raw)
    assert len(quiz) == 6
    for q in quiz:
        assert q["options"][q["answer_index"]] == "right"
        assert len(q["options"]) == 4
    assert len({tuple(q["options"]) for q in quiz}) >= 2  # shuffled, not identical every time (very likely)
    assert len(validate_quiz({"questions": raw})) == 6
    with pytest.raises(LLMError):
        validate_quiz(raw[:2])


# ------------------------------------------------------------ badge
def test_badge_png_renders_even_with_long_text():
    png = badge.make_badge_png("A Very Long Name That Keeps Going And Going Ahmad-Khan", "Data Structures & Algorithms", 88, "18 Oct 2026")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000
    assert badge.badge_id("x", "y", "z") == badge.badge_id(" X ", "Y", "z")


# ------------------------------------------------------------ SDK call shape (mocked)
def test_claude_call_shape():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(content=[MagicMock(type="text", text="hello")])
    with patch("anthropic.Anthropic", return_value=fake_client) as ctor:
        out = LLM("Claude (Anthropic)", "claude-haiku-5-5", "k").chat("SYS", [{"role": "user", "content": "hi"}], 100)
    assert out == "hello"
    ctor.assert_called_once_with(api_key="k")
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5" and kwargs["system"] == "SYS" and kwargs["max_tokens"] == 100
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_groq_call_shape():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="yo"))])
    with patch("groq.Groq", return_value=fake_client):
        out = LLM("Groq", "llama-3.3-70b-versatile", "k").chat("SYS", [{"role": "user", "content": "hi"}], 100)
    assert out == "yo"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}


def test_errors_become_friendly_messages():
    class Boom(Exception):
        status_code = 401
    with patch("anthropic.Anthropic", side_effect=Boom("bad")):
        with pytest.raises(LLMError, match="key was rejected"):
            LLM("Claude (Anthropic)", "m", "k").chat("s", [{"role": "user", "content": "x"}])
    class Slow(Exception):
        status_code = 429
    with patch("groq.Groq", side_effect=Slow("slow")):
        with pytest.raises(LLMError, match="Rate limit"):
            LLM("Groq", "m", "k").chat("s", [{"role": "user", "content": "x"}])
    with pytest.raises(LLMError):
        LLM("Groq", "m", "")
    with pytest.raises(LLMError):
        LLM("Nope", "m", "k")


# ------------------------------------------------------------ full UI flow with a fake AI
ROADMAP = {"title": "Fake Roadmap", "summary": "Two simple sentences. Go.", "modules": [
    {"title": f"Module {i}", "goal": "Learn it.", "topics": ["x", "y", "z"], "practice": "Do a task.", "hours": 3}
    for i in range(1, 6)]}
QUIZ = [{"question": f"Question {i}?", "options": ["right", "wrong1", "wrong2", "wrong3"], "answer_index": 0,
         "explanation": "Because."} for i in range(1, 9)]


def fake_chat(self, system, messages, max_tokens=1500):
    prompt = messages[-1]["content"]
    if "Create a learning roadmap" in prompt:
        return "```json\n" + json.dumps(ROADMAP) + "\n```"
    if "multiple-choice quiz" in prompt:
        return "Here you go:\n" + json.dumps(QUIZ)
    if "study notes" in prompt:
        return "## Plain-English idea\nFake notes."
    if "best option" in prompt.lower() or "Pick the best" in prompt:
        return "Pick the first one. Downside: long."
    return f"Fake answer to: {prompt}"


def new_app(monkeypatch, with_key=True):
    for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    if with_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    return at.run()


def click(at, fragment):
    matches = [b for b in at.button if fragment in b.label]
    assert matches, f"no button containing {fragment!r}; have {[b.label for b in at.button]}"
    matches[0].click()
    return at.run()


def test_app_loads_without_key_and_warns(monkeypatch):
    at = new_app(monkeypatch, with_key=False)
    assert not at.exception
    at = click(at, "Generate roadmap")
    assert any("API key" in w.value for w in at.warning)
    assert at.session_state["roadmap"] is None


def test_full_happy_path_earns_badge(monkeypatch):
    with patch.object(llm_mod.LLM, "chat", fake_chat):
        at = new_app(monkeypatch)
        assert not at.exception
        # tab 1: ranked courses shown, choose one, ask AI
        assert any("Choose this course" in b.label for b in at.button)
        at = click(at, "Choose this course")
        assert at.session_state["picked"]["topic"] == at.selectbox(key="topic").value
        at = click(at, "Ask AI which one")
        assert any("Pick the first one" in i.value for i in at.info)
        # tab 2: roadmap
        at = click(at, "Generate roadmap")
        assert not at.exception and at.session_state["roadmap"]["title"] == "Fake Roadmap"
        at = click(at, "Write study notes")
        assert 0 in at.session_state["notes"]
        # badge locked at this point
        assert any("Finish both steps" in i.value for i in at.info)
        assert len(at.get("image")) == 0
        # tab 3: quiz, all correct
        at = click(at, "Generate quiz")
        assert len(at.session_state["quiz"]) == 8
        qid = at.session_state["qid"]
        for i, q in enumerate(at.session_state["quiz"]):
            at.radio(key=f"q_{qid}_{i}").set_value(q["options"][q["answer_index"]])
        at = click(at, "Submit answers")
        assert at.session_state["last_score"] == 100 and at.session_state["best_score"] == 100
        # still locked: modules not finished and no name
        assert any("Finish both steps" in i.value for i in at.info)
        # tick all modules
        for cb in at.checkbox:
            if cb.key and cb.key.startswith("mod_"):
                cb.check()
        at = at.run()
        # passed + finished, but no name yet -> asks for name
        assert any("Type your name" in w.value for w in at.warning)
        at.text_input(key="user_name").set_value("Ghufran")
        at = at.run()
        assert not at.exception
        assert len(at.get("image")) >= 1  # badge image is shown
        assert any(d.proto.label.startswith("⬇️ Download badge") for d in at.get("download_button"))
        assert at.session_state["celebrated"] is True


def test_failed_quiz_gives_no_badge(monkeypatch):
    with patch.object(llm_mod.LLM, "chat", fake_chat):
        at = new_app(monkeypatch)
        at = click(at, "Generate roadmap")
        at = click(at, "Generate quiz")
        qid = at.session_state["qid"]
        for i, q in enumerate(at.session_state["quiz"]):
            wrong = [o for j, o in enumerate(q["options"]) if j != q["answer_index"]][0]
            at.radio(key=f"q_{qid}_{i}").set_value(wrong)
        at = click(at, "Submit answers")
        assert at.session_state["last_score"] == 0
        for cb in at.checkbox:
            if cb.key and cb.key.startswith("mod_"):
                cb.check()
        at.text_input(key="user_name").set_value("Ghufran")
        at = at.run()
        assert at.session_state["celebrated"] is False
        assert len(at.get("image")) == 0
        assert any("Finish both steps" in i.value for i in at.info)


def test_unanswered_quiz_is_rejected(monkeypatch):
    with patch.object(llm_mod.LLM, "chat", fake_chat):
        at = new_app(monkeypatch)
        at = click(at, "Generate roadmap")
        at = click(at, "Generate quiz")
        at = click(at, "Submit answers")
        assert any("answer every question" in w.value for w in at.warning)
        assert at.session_state["last_score"] is None


def test_chat_assistant_replies_and_ai_failure_is_handled(monkeypatch):
    with patch.object(llm_mod.LLM, "chat", fake_chat):
        at = new_app(monkeypatch)
        at.chat_input[0].set_value("what is a pointer?").run()
        assert not at.exception
        assert [m["role"] for m in at.session_state["chat"]] == ["user", "assistant"]
        assert "what is a pointer?" in at.session_state["chat"][1]["content"]

    def broken(self, *a, **k):
        raise LLMError("Rate limit reached. Wait a minute.")
    with patch.object(llm_mod.LLM, "chat", broken):
        at = new_app(monkeypatch)
        at = click(at, "Generate roadmap")
        assert any("Rate limit" in e.value for e in at.error)
        assert at.session_state["roadmap"] is None
        at.chat_input[0].set_value("hello").run()
        assert any("Rate limit" in e.value for e in at.error)
        assert at.session_state["chat"] == []  # no dangling user message


def test_unusable_json_is_retried_once(monkeypatch):
    calls = {"n": 0}

    def flaky(self, system, messages, max_tokens=1500):
        calls["n"] += 1
        return "sorry, no json" if calls["n"] == 1 else json.dumps(ROADMAP)

    with patch.object(llm_mod.LLM, "chat", flaky):
        at = new_app(monkeypatch)
        at = click(at, "Generate roadmap")
        assert calls["n"] == 2 and at.session_state["roadmap"]["title"] == "Fake Roadmap"
