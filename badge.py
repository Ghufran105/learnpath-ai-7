"""
badge.py - draws the completion badge as a PNG (easy to share on LinkedIn / WhatsApp).
Note: this is a self-awarded badge made by the app. It is not an official certificate.
"""
import hashlib
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

SIZE = 900
GOLD = (245, 179, 1)
NAVY = (15, 23, 42)
SLATE = (30, 41, 59)
WHITE = (248, 250, 252)
MUTED = (148, 163, 184)

_BOLD_FONTS = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"]
_REGULAR_FONTS = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]


def badge_id(name: str, topic: str, day: str) -> str:
    raw = f"{name.strip().lower()}|{topic.lower()}|{day}".encode("utf-8")
    return "LP-" + hashlib.sha1(raw).hexdigest()[:8].upper()


def _font(size: int, bold: bool = False):
    for fname in (_BOLD_FONTS if bold else _REGULAR_FONTS):
        try:
            return ImageFont.truetype(fname, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def _width(draw, text, font):
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _fit_font(draw, text, max_width, start, bold=False, minimum=22):
    size = start
    font = _font(size, bold)
    while _width(draw, text, font) > max_width and size > minimum:
        size -= 2
        font = _font(size, bold)
    return font


def _center(draw, y, text, font, fill):
    x = (SIZE - _width(draw, text, font)) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_badge_png(name: str, topic: str, score: int, day: str) -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), NAVY)
    d = ImageDraw.Draw(img)
    c = SIZE // 2

    d.ellipse((40, 40, SIZE - 40, SIZE - 40), fill=GOLD)
    d.ellipse((62, 62, SIZE - 62, SIZE - 62), fill=SLATE)
    d.ellipse((92, 92, SIZE - 92, SIZE - 92), outline=GOLD, width=5)

    _center(d, 170, "COURSE COMPLETED", _font(40, True), GOLD)

    topic_font = _fit_font(d, topic, 560, 76, bold=True)
    _center(d, 250, topic, topic_font, WHITE)

    d.line((c - 140, 380, c + 140, 380), fill=GOLD, width=4)

    _center(d, 410, "Awarded to", _font(32), MUTED)
    name_font = _fit_font(d, name, 560, 62, bold=True)
    _center(d, 460, name, name_font, WHITE)

    _center(d, 570, f"Final quiz score: {score}%", _font(36, True), GOLD)
    _center(d, 630, day, _font(30), MUTED)
    _center(d, 700, f"ID: {badge_id(name, topic, day)}", _font(26), MUTED)
    _center(d, 745, "LearnPath AI", _font(26, True), MUTED)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
