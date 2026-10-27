"""The web page's ids are unique: $(id) returns the first element with an id, so a second one silently takes its
place - LoRA Royale's Render epochs button stopped working when the Render with box was given its id (#241)."""
import os
import re
from collections import Counter

PAGE = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "index.html")


def test_no_id_is_used_twice():
    with open(PAGE, encoding="utf-8") as f:
        ids = Counter(re.findall(r'\bid="([^"]+)"', f.read()))
    assert {k: n for k, n in ids.items() if n > 1} == {}
