"""Load General Conference talks from scriptures.byu.edu."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

from app.config import load_config

logger = logging.getLogger(__name__)

BYU_BASE = "https://scriptures.byu.edu/content/talks_ajax"

# Every layout carries "<year>–<A|O>:<page>, <speaker>, <title>" in #talklabel.
LABEL_RE = re.compile(r"^(?P<year>\d{4})\s*[–\-]\s*(?P<session>[AO])\s*:\s*(?P<page>\d+)\s*,\s*(?P<rest>.+)$")
SESSION_MONTH = {"A": "April", "O": "October"}
NAME_SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV"}

# Footnote markers and the notes footer are BYU chrome, not talk text.
STRIP_SELECTORS = ["sup.noteMarker", "footer.notes", "div.notes", "script", "style"]
BLOCK_TAGS = ["p", "h2", "h3", "blockquote", "li"]

# The ID space has scattered gaps; a long run of misses means we're past the end.
MISS_STREAK_LIMIT = 200
HARD_ID_CAP = 12000


def parse_talk_label(text: str) -> dict:
    """Parse the #talklabel text into year, session, page, speaker, title, and date."""
    m = LABEL_RE.match(text.replace("\xa0", " ").strip())
    if not m:
        return {}
    parts = [p.strip() for p in m["rest"].split(",")]
    speaker = parts[0]
    i = 1
    while i < len(parts) and parts[i] in NAME_SUFFIXES:
        speaker += ", " + parts[i]
        i += 1
    title = ", ".join(parts[i:]).strip() or None
    return {
        "year": int(m["year"]),
        "session": m["session"],
        "page": int(m["page"]),
        "speaker": speaker or None,
        "title": title,
        "date": f"{SESSION_MONTH[m['session']]} {m['year']}",
    }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _in_header(el) -> bool:
    return el.find_parent(id="details") is not None or el.find_parent(
        class_=lambda c: c in ("byline", "author", "gchead")
    ) is not None


def extract_body(doc: BeautifulSoup) -> str:
    """Return talk paragraphs joined by blank lines, across all BYU page layouts."""
    for sel in STRIP_SELECTORS:
        for el in doc.select(sel):
            el.decompose()

    container = (
        doc.find("div", class_="gcbody")
        or doc.find(id="primary")
        or doc.find("div", class_="body")
        or doc.find(id="talkwrapper")
    )
    if container is None:
        return ""

    paras = []
    for el in container.find_all(BLOCK_TAGS):
        if el.find(["p", "li"]) or _in_header(el):
            continue
        text = _clean(el.get_text(" ", strip=True))
        if text:
            paras.append(text)
    return "\n\n".join(paras)


@dataclass
class GeneralConferenceLoader:
    loader_config: dict = field(default_factory=dict)

    def __post_init__(self):
        config = load_config()
        cfg = config.get("ingestion", {}).get("sources", {}).get("general_conference", {})
        self.delay = cfg.get("per_request_delay", 0.5)
        self.max_id = cfg.get("max_ids", -1)

    def load(self, resume_from: int = 0) -> Iterator[dict]:
        """Yield talk dicts one at a time, starting after `resume_from`."""
        auto = self.max_id is None or self.max_id <= 0
        last_id = HARD_ID_CAP if auto else self.max_id
        start_id = max(1, resume_from + 1)
        logger.info(f"Fetching general conference talks: IDs {start_id}..{'auto' if auto else last_id}")

        fetched = skipped = miss_streak = 0
        start_time = time.time()

        for i in range(start_id, last_id + 1):
            talk = self._fetch_one(i, retries=3)
            if talk is None:
                skipped += 1
                miss_streak += 1
                if auto and miss_streak >= MISS_STREAK_LIMIT:
                    logger.info(f"{MISS_STREAK_LIMIT} consecutive misses after ID {i - MISS_STREAK_LIMIT}; stopping scan")
                    break
                continue

            miss_streak = 0
            yield talk
            fetched += 1

            if fetched % 50 == 0 or fetched == 1:
                elapsed = time.time() - start_time
                rate = fetched / elapsed if elapsed > 0 else 0
                logger.info(f"  [ID {i}] {fetched} fetched, {skipped} skipped (rate: {rate:.1f}/sec)")

            if i % 25 == 0:
                time.sleep(self.delay)

        elapsed = time.time() - start_time
        rate = fetched / elapsed if elapsed > 0 else 0
        logger.info(f"General Conference complete: {fetched} fetched, {skipped} skipped (rate: {rate:.1f}/sec, {elapsed:.0f}s total)")

    def _get(self, talk_id: int, retries: int) -> Optional[str]:
        url = f"{BYU_BASE}/{talk_id}"
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=15)
            except requests.RequestException as e:
                logger.warning(f"Request error for ID {talk_id} (attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code != 200 or len(resp.text) < 300:
                logger.debug(f"Skipping ID {talk_id}: HTTP {resp.status_code}, size={len(resp.text)}")
                return None
            return resp.text
        logger.error(f"Gave up on ID {talk_id} after {retries} attempts")
        return None

    def _fetch_one(self, talk_id: int, retries: int = 3) -> Optional[dict]:
        """Fetch and parse a single talk. Returns dict or None when skipped."""
        html = self._get(talk_id, retries)
        if html is None:
            return None
        return self.parse(talk_id, html)

    def parse(self, talk_id: int, html: str) -> Optional[dict]:
        doc = BeautifulSoup(html, "html.parser")

        label_el = doc.find(id="talklabel")
        label = parse_talk_label(label_el.get_text(" ", strip=True)) if label_el else {}

        title = self._first_text(doc, "p.gctitle", "h1") or label.get("title")
        speaker = label.get("speaker") or self._first_text(doc, "p.gcspeaker", "p.author-name", "h2.author")
        if speaker:
            speaker = re.sub(r"^By\s+", "", speaker)
        citation = self._first_text(doc, "p.gcbib")

        content = extract_body(doc)
        if len(content) < 80:
            logger.debug(f"Skipping ID {talk_id}: short body ({len(content)} chars)")
            return None

        if not title:
            title = f"Unknown Title (ID {talk_id})"

        return {
            "content": content,
            "source": f"General Conference - {title}",
            "metadata": {
                "source_type": "general_conference",
                "id": talk_id,
                "title": title,
                "speaker": speaker,
                "date": label.get("date") or citation,
                "year": label.get("year"),
                "conference": f"{label['year']}-{label['session']}" if label else None,
                "citation": citation,
                "url": f"{BYU_BASE}/{talk_id}",
            },
        }

    @staticmethod
    def _first_text(doc: BeautifulSoup, *selectors: str) -> Optional[str]:
        for sel in selectors:
            el = doc.select_one(sel)
            if el is not None:
                text = _clean(el.get_text(" ", strip=True))
                if text:
                    return text
        return None
