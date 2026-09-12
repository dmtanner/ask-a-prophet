"""Load Journal of Discourses from scriptures.byu.edu."""

import logging
import re
import time
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Journal of Discourse sermon ID ranges in BYU's talks database
VOLUME_RANGES = {
    1: (10001, 10054),  # 54 sermons
    2: (20001, 20056),  # 56 sermons
    3: (30001, 30051),  # 51 sermons
    4: (40001, 40070),  # 70 sermons
}

BYU_JOD_BASE = "https://scriptures.byu.edu/content/talks_ajax"


def _text(el) -> Optional[str]:
    if el is None:
        return None
    return re.sub(r"[ \t\xa0]+", " ", el.get_text(" ", strip=True)).strip() or None


def extract_body(doc: BeautifulSoup) -> str:
    """Return sermon paragraphs joined by blank lines, minus BYU page-break and citation chrome."""
    for el in doc.select("div.break, span.citation, div.hyphen"):
        el.decompose()
    body = doc.find("div", class_="discourseBody")
    if body is None:
        return ""
    paras = []
    for p in body.find_all("div", class_="paragraph"):
        text = re.sub(r"\s+", " ", p.get_text("")).strip()
        if text:
            paras.append(text)
    if not paras:
        return _text(body) or ""
    return "\n\n".join(paras)


class JournalOfDiscoursesLoader:

    def load(self, resume_from: int = 0, skip_completed: set | None = None) -> Iterator[dict]:
        """Yield sermons from all 4 volumes of the Journal of Discourses.

        :param resume_from: ID to resume from. If > 1, skips all IDs up to and including it.
        :param skip_completed: Set of volume numbers already fetched; those volumes are skipped.
        """
        completed = set(skip_completed) if skip_completed else set()
        
        # Build list of valid volume ranges as (start, stop) pairs
        vol_list = sorted(VOLUME_RANGES.items())
        
        filtered_vols = []
        for vid_num, (lo, hi) in vol_list:
            # Skip volumes already fetched
            if vid_num in completed:
                logger.info(f"Skipping JoD Volume {vid_num} (already completed)")
                continue
            
            # Apply resume_from filter within range
            effective_lo = max(lo, resume_from + 1) if resume_from > 0 else lo
            
            if effective_lo > hi:
                continue
                
            filtered_vols.append((vid_num, effective_lo, hi))

        logger.info(f"Fetching Journal of Discourses sermons across {len(filtered_vols)} volume ranges")

        fetched = 0
        skipped = 0
        start_time = time.time()
        seen_ids = set()

        for vid_num, lo, hi in filtered_vols:
            total_range_size = hi - lo + 1
            
            for tid in range(lo, hi + 1):
                sermon = self._fetch_one(tid, retries=3)
                if sermon is None:
                    skipped += 1
                    continue

                # Dedup check — BYU may return same talk on some IDs
                title = sermon.get("metadata", {}).get("title") or f"<no-title:{tid}>"
                if (tid, title) in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add((tid, title))

                yield sermon
                fetched += 1

                # Progress logging every serial_number
                if fetched % 25 == 0 or fetched == 1:
                    elapsed = time.time() - start_time
                    rate = fetched / elapsed if elapsed > 0 else 0
                    eta_items = total_range_size - (tid - lo + 1)
                    eta_str = f", ETA ~{int(eta_items/rate/60)}m" if rate > 0 and eta_items else ""
                    logger.info(f"  [{lo}-{hi}] fetched {fetched}, skipped {skipped} (rate: {rate:.1f}/sec){eta_str}")

                # Politeness delay every 25 requests
                if tid % 25 == 0:
                    time.sleep(1.0)

        elapsed = time.time() - start_time
        rate = fetched / elapsed if elapsed > 0 else 0
        logger.info(f"Journal of Discourses complete: {fetched} fetched, {skipped} skipped (rate: {rate:.1f}/sec, {elapsed:.0f}s total)")

    def _fetch_one(self, talk_id: int, retries: int = 3) -> Optional[dict]:
        """Fetch and parse a single JoD sermon. Returns dict or None on failure."""
        url = f"{BYU_JOD_BASE}/{talk_id}"

        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=10)
            except requests.RequestException as e:
                logger.warning(f"Request error for JoD ID {talk_id} (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    logger.error(f"Gave up on JoD ID {talk_id} after {retries} attempts")
                time.sleep(1 * (attempt + 1))
                continue

            if resp.status_code != 200 or len(resp.text) < 300:
                logger.debug(f"Skipping JoD ID {talk_id}: HTTP {resp.status_code}, size={len(resp.text)}")
                return None

            break
        else:
            logger.warning(f"All retries exhausted for JoD ID {talk_id}")
            return None

        return self.parse(talk_id, resp.text)

    def parse(self, talk_id: int, html: str) -> Optional[dict]:
        url = f"{BYU_JOD_BASE}/{talk_id}"
        doc = BeautifulSoup(html, "html.parser")

        title = _text(doc.select_one("div.discourseHeader div.title"))
        speaker = _text(doc.select_one("div.discourseInfo div.speaker"))
        date_str = _text(doc.select_one("div.discourseInfo div.date"))
        speaker = re.sub(r"^Speaker:\s*", "", speaker) if speaker else None
        date_str = re.sub(r"^Date:\s*", "", date_str) if date_str else None

        # Fallback: the nav label reads "JD 1:34, Heber C. Kimball, Title".
        label = _text(doc.find(id="talklabel"))
        if label and (not speaker or not title):
            parts = [x.strip() for x in label.split(",")]
            if len(parts) >= 3:
                speaker = speaker or parts[1]
                title = title or ", ".join(parts[2:])

        volume = sermon_num = None
        for vol_num, (lo, hi) in VOLUME_RANGES.items():
            if lo <= talk_id <= hi:
                volume = vol_num
                sermon_num = talk_id - lo + 1
                break

        content = extract_body(doc)
        if len(content) < 80:
            logger.debug(f"Skipping JoD ID {talk_id}: short body ({len(content)} chars)")
            return None

        if not title:
            title = f"Unknown (ID {talk_id})"

        return {
            "content": content,
            "source": f"Journal of Discourses Vol {volume}, Sermon {sermon_num} — {title}",
            "metadata": {
                "source_type": "journal_of_discourses",
                "id": talk_id,
                "volume": volume,
                "sermon_number": sermon_num,
                "title": title,
                "speaker": speaker,
                "date": date_str,
                "url": url,
            },
        }
