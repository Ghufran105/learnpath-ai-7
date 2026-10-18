"""
llm.py - talks to Claude (Anthropic) or Groq, and holds every AI task the app needs.

Each task = one clear prompt + one validator, so bad AI output shows a friendly
message instead of crashing the app.

Model names change over time. If a model is rejected, edit PROVIDERS below.
"""
from __future__ import annotations

import json
import random
import re

PROVIDERS = {
    "Claude (Anthropic)": {
        "key_name": "ANTHROPIC_API_KEY",
        "models": {
            "Claude Haiku 5.5 (fast, cheap)": "claude-haiku-5-5",
            "Claude Sonnet 5.5 (smarter)": "claude-sonnet-5-5",
        },
    },
    "Groq": {
        "key_name": "GROQ_API_KEY",
        "models": {
            "GPT-OSS 120B (best on Groq)": "openai/gpt-oss-120b",
            "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
        },
    },
}

SIMPLE_STYLE = (
    "You are a friendly teacher. Use very simple English and short sentences. "
    "Explain the idea in plain words first, and use one everyday analogy when it helps. "
    "Never invent facts, links or statistics. If you are not sure, say so."
)


class LLMError(Exception):
    """A friendly error message the UI can show directly."""


def _friendly(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if status == 401 or "authentication" in name or "invalid api key" in text:
        return "The API key was rejected. Check that you pasted the right key for this provider."
    if status == 429 or "ratelimit" in name or "rate limit" in text:
        return "Rate limit reached. Wait a minute and try again, or switch model/provider."
    if "decommission" in text or "deprecat" in text:
        return "The provider has retired this model. Pick another model in the sidebar (or update llm.py)."
    if status == 404 or "not_found" in text or ("model" in text and "not" in text and "exist" in text):
        return "This model name was not accepted. Pick another model in the sidebar (or update llm.py)."
    if "connection" in name or "timeout" in name:
        return "Could not reach the AI service. Check your internet and try again."
    return f"AI request failed: {exc}"


class LLM:
    """One tiny interface for both providers: llm.chat(system, messages) -> text."""

    def __init__(self, provider: str, model: str, api_key: str):
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown provider: {provider}")
        if not api_key:
            raise LLMError("No API key provided.")
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def chat(self, system: str, messages: list, max_tokens: int = 1500) -> str:
        try:
            if self.provider.startswith("Claude"):
                text = self._claude(system, messages, max_tokens)
            else:
                text = self._groq(system, messages, max_tokens)
        except LLMError:
            raise
        except Exception as exc:  # SDK errors -> friendly message
            raise LLMError(_friendly(exc)) from exc
        if not text.strip():
            raise LLMError("The AI returned an empty answer. Please try again.")
        return text

    def _claude(self, system, messages, max_tokens):
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _groq(self, system, messages, max_tokens):
        from groq import Groq

        client = Groq(api_key=self.api_key)
        extra = {}
        budget = max_tokens
        if self.model.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = "low"  # short thinking, faster and cheaper
            budget += 1000                      # thinking tokens count toward the limit
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=budget,
            temperature=0.4,
            **extra,
        )
        return resp.choices[0].message.content or ""


# ---------- JSON helpers ----------

def extract_json(text: str):
    """Parse JSON even if the AI wrapped it in ```json fences or added extra words."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pairs = [("{", "}"), ("[", "]")]
    # try the bracket that appears FIRST, so a list holding one object stays a list
    pairs.sort(key=lambda p: text.find(p[0]) if text.find(p[0]) != -1 else len(text) + 1)
    for open_c, close_c in pairs:
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("The AI did not return valid JSON. Please click the button again.")


def _ask_json(llm: LLM, system: str, prompt: str, validator, max_tokens: int = 1800):
    """Ask, parse, validate. One automatic retry if the answer is unusable."""
    last = None
    for _ in range(2):
        text = llm.chat(system, [{"role": "user", "content": prompt}], max_tokens)
        try:
            return validator(extract_json(text))
        except LLMError as exc:
            last = exc
            prompt += "\n\nIMPORTANT: your last answer was unusable. Return ONLY the JSON, in the exact shape."
    raise last


# ---------- Validators ----------

def validate_roadmap(data) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("modules"), list):
        raise LLMError("The AI gave an incomplete roadmap. Please try again.")
    modules = []
    for m in data["modules"]:
        if not isinstance(m, dict) or not str(m.get("title", "")).strip():
            continue
        raw_topics = m.get("topics") if isinstance(m.get("topics"), list) else []
        try:
            hours = max(1, int(m.get("hours", 3)))
        except (TypeError, ValueError):
            hours = 3
        modules.append(
            {
                "title": str(m["title"]).strip(),
                "goal": str(m.get("goal", "")).strip(),
                "topics": [str(t).strip() for t in raw_topics if str(t).strip()],
                "practice": str(m.get("practice", "")).strip(),
                "hours": hours,
            }
        )
    if len(modules) < 3:
        raise LLMError("The AI gave too few modules. Please try again.")
    return {
        "title": str(data.get("title") or "Your roadmap").strip(),
        "summary": str(data.get("summary", "")).strip(),
        "modules": modules[:8],
    }


def validate_quiz(data) -> list:
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise LLMError("The AI gave an incomplete quiz. Please try again.")
    quiz = []
    for q in data:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        raw = q.get("options")
        if not question or not isinstance(raw, list):
            continue
        options = [str(o).strip() for o in raw if str(o).strip()]
        try:
            idx = int(q.get("answer_index"))
        except (TypeError, ValueError):
            continue
        if len(options) < 2 or not 0 <= idx < len(options):
            continue
        correct = options[idx]
        unique = list(dict.fromkeys(options))[:4]
        if correct not in unique:
            unique[-1] = correct
        random.shuffle(unique)  # AI likes putting the answer first; shuffle it
        quiz.append(
            {
                "question": question,
                "options": unique,
                "answer_index": unique.index(correct),
                "explanation": str(q.get("explanation", "")).strip(),
            }
        )
    if len(quiz) < 3:
        raise LLMError("The AI gave too few usable questions. Please try again.")
    return quiz


# ---------- AI tasks ----------

def generate_roadmap(llm: LLM, topic: str, level: str) -> dict:
    prompt = (
        f"Create a learning roadmap.\nTopic: {topic}\nLearner level: {level}\n\n"
        "Return ONLY valid JSON, no extra text, in exactly this shape:\n"
        '{"title": "short title", "summary": "2 simple sentences", '
        '"modules": [{"title": "...", "goal": "one simple sentence", '
        '"topics": ["...", "..."], "practice": "one small hands-on task", "hours": 4}]}\n\n'
        "Rules: 5 or 6 modules, ordered from easiest to hardest. 3 to 5 topics per module. "
        "Simple words. hours must be a whole number."
    )
    return _ask_json(llm, SIMPLE_STYLE, prompt, validate_roadmap, max_tokens=1800)


def generate_quiz(llm: LLM, topic: str, level: str, module_titles: list, n: int = 8) -> list:
    prompt = (
        f"Write a multiple-choice quiz with exactly {n} questions.\n"
        f"Topic: {topic}\nLearner level: {level}\n"
        f"Cover these roadmap modules: {'; '.join(module_titles)}\n\n"
        "Return ONLY a valid JSON list, no extra text, in exactly this shape:\n"
        '[{"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer_index": 0, "explanation": "one simple sentence why"}]\n\n'
        "Rules: 4 options each, exactly one correct. answer_index is the position (0 to 3) of the "
        "correct option. Vary the position of the correct answer. Simple words. "
        "Questions must be factual and checkable, not opinions."
    )
    return _ask_json(llm, SIMPLE_STYLE, prompt, validate_quiz, max_tokens=2500)


def generate_notes(llm: LLM, topic: str, level: str, module: dict) -> str:
    prompt = (
        f"Write short study notes in Markdown.\nTopic: {topic}\nLearner level: {level}\n"
        f"Module: {module['title']}\nGoal: {module['goal']}\n"
        f"Subtopics: {', '.join(module['topics'])}\n\n"
        "Use these headings: Plain-English idea, Everyday analogy, Key points (bullets), "
        "Tiny example, Common mistakes, Quick self-check (3 questions). "
        "Keep it under 350 words."
    )
    return llm.chat(SIMPLE_STYLE, [{"role": "user", "content": prompt}], max_tokens=900).strip()


def explain_pick(llm: LLM, topic: str, level: str, courses: list) -> str:
    lines = "\n".join(
        f"- {c['title']} ({c['platform']}, {c['level']}, {c['price']}, ~{c['hours']}h, "
        f"{'has practice' if c.get('has_labs') else 'no built-in practice'})"
        for c in courses
    )
    prompt = (
        f"A {level} learner wants to study {topic}. Here are the options:\n{lines}\n\n"
        "Pick the best option from ONLY this list. Do not mention any other course. "
        "Explain in 3 simple sentences why it fits, and name one downside."
    )
    return llm.chat(SIMPLE_STYLE, [{"role": "user", "content": prompt}], max_tokens=400).strip()


def assistant_reply(llm: LLM, topic: str, level: str, course: str, module_titles: list, history: list) -> str:
    system = SIMPLE_STYLE + f" You are the study assistant inside a learning app. The learner studies {topic} at {level} level."
    if course:
        system += f" Their chosen course: {course}."
    if module_titles:
        system += " Roadmap modules: " + "; ".join(module_titles) + "."
    system += (
        " Keep answers under 200 words unless asked for more. If a question is far from the topic, "
        "answer briefly and steer back."
    )
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return llm.chat(system, msgs, max_tokens=700).strip()
