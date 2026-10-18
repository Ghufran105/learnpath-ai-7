"""
Checks every link in data/courses.json so you can fix dead ones.
Run:  python check_links.py
Websites change. Run this before you deploy and every few months after.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_FILE = Path(__file__).parent / "data" / "courses.json"


def collect(catalog):
    for topic, block in catalog.items():
        for kind in ("courses", "notes", "labs"):
            for item in block.get(kind, []):
                yield topic, kind, item["title"], item["url"]


def check(url, timeout=15):
    """Returns (status_code or None, label)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link-checker)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, "OK"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 429):
            return exc.code, "BLOCKED (site refuses scripts - open it in your browser)"
        if exc.code in (404, 410):
            return exc.code, "BROKEN"
        return exc.code, "CHECK MANUALLY"
    except Exception as exc:  # no internet, DNS, timeout...
        return None, f"CHECK MANUALLY ({type(exc).__name__})"


def main(path=DEFAULT_FILE):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    broken = 0
    for topic, kind, title, url in collect(catalog):
        code, label = check(url)
        broken += label == "BROKEN"
        print(f"[{label}] {topic} / {kind} / {title}\n    {url}")
    print(f"\nDone. Broken links: {broken}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE))
