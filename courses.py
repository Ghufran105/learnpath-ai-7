"""
courses.py - loads the curated catalog and ranks courses.

Why a curated catalog? Coursera/Udemy/edX have no open "best course" API, and scraping
them breaks their rules (and your app). So the ranking here is plain, honest rules that
you can read and change - not an AI guess that could invent courses.
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "courses.json"

LEVELS = ["Beginner", "Intermediate", "Advanced"]
FORMATS = ["Any", "Video", "Text", "Interactive"]
FREE_PRICES = {"Free", "Free to audit"}


def load_catalog(path: Path = DATA_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def max_score(fmt: str) -> int:
    """Highest score a course can get with the current choices."""
    return 3 + (2 if fmt != "Any" else 0) + 1


def rank_courses(courses: list, level: str, fmt: str = "Any", free_only: bool = True) -> list:
    """
    Returns [(score, course), ...] best first.

    Score rules:
      +3 same level as the learner, +1 one level away (two levels away is hidden)
      +2 exact format match, +1 if the course is 'Mixed' and a format was chosen
      +1 if the course has built-in practice (labs / problem sets)
    Ties keep the order in courses.json, so the order there is your editorial choice.
    """
    ranked = []
    for order, course in enumerate(courses):
        if free_only and course["price"] not in FREE_PRICES:
            continue
        gap = abs(LEVELS.index(course["level"]) - LEVELS.index(level))
        if gap > 1:
            continue
        score = 3 if gap == 0 else 1
        if fmt != "Any":
            if course["format"] == fmt:
                score += 2
            elif course["format"] == "Mixed":
                score += 1
        if course.get("has_labs"):
            score += 1
        ranked.append((score, order, course))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    return [(score, course) for score, _, course in ranked]
