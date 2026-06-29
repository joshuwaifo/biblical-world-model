"""
Download and parse canonical Bible texts into SQLite.

Sources (all public domain):
  - World English Bible British Edition (WEB+DC): ebible.org USFX XML
  - 1 Enoch (R.H. Charles 1913): GitHub mirror
  - Book of Jubilees (R.H. Charles 1902): GitHub mirror

Run:  python data/ingest.py
"""

import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from db.models import Book, Chapter, Verse
from db.session import SessionLocal, init_db


# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
WEB_USFX_URL = "https://eBible.org/Scriptures/eng-webbe_usfx.zip"

# Public-domain plain-text for Ethiopian additions
# Using GitHub mirrors to avoid sacred-texts.com rate limiting
ENOCH_URL = "https://www.gutenberg.org/cache/epub/77935/pg77935.txt"
# Jubilees: R.H. Charles 1902 text not yet on Gutenberg; TODO: add when sourced


# ---------------------------------------------------------------------------
# Testament assignment for USFX 3-letter book codes
# ---------------------------------------------------------------------------
_OT = {
    "GEN","EXO","LEV","NUM","DEU","JOS","JDG","RUT","1SA","2SA",
    "1KI","2KI","1CH","2CH","EZR","NEH","EST","JOB","PSA","PRO",
    "ECC","SNG","ISA","JER","LAM","EZK","DAN","HOS","JOL","AMO",
    "OBA","JON","MIC","NAM","HAB","ZEP","HAG","ZEC","MAL",
}
_DC = {
    "TOB","JDT","1MA","2MA","WIS","SIR","BAR","LJE",
    "S3Y","SUS","BEL","1ES","2ES","MAN","PS2",
}
_NT = {
    "MAT","MRK","LUK","JHN","ACT","ROM","1CO","2CO","GAL","EPH",
    "PHP","COL","1TH","2TH","1TI","2TI","TIT","PHM","HEB","JAS",
    "1PE","2PE","1JN","2JN","3JN","JUD","REV",
}

def _testament(code: str) -> str:
    if code in _OT: return "OT"
    if code in _DC: return "Deuterocanon"
    if code in _NT: return "NT"
    return "Ethiopian"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  cached  {dest.name}")
        return dest
    print(f"  GET     {url}")
    r = httpx.get(url, follow_redirects=True, timeout=120,
                  headers={"User-Agent": "biblical-world-model/0.1 (research)"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# ---------------------------------------------------------------------------
# USFX parser — regex approach
#
# USFX uses *milestone* markers, not containers:
#   <v id="1" bcv="GEN.1.1" />   ← self-closing; starts verse
#   ... text mixed with <w>, <f>, etc. ...
#   <ve />                        ← self-closing; ends verse
#
# The bcv attribute gives book/chapter/verse directly.
# We strip footnotes (<f>…</f>) and XML tags, leaving plain verse text.
# ---------------------------------------------------------------------------
_FOOTNOTE_RE = re.compile(r"<f\b[^>]*>.*?</f>", re.DOTALL)
_CROSSREF_RE = re.compile(r"<x\b[^>]*>.*?</x>", re.DOTALL)
_NOTE_RE     = re.compile(r"<note\b[^>]*>.*?</note>", re.DOTALL)
_TAG_RE      = re.compile(r"<[^>]+>")
_VERSE_RE    = re.compile(
    r'<v\s[^>]*\bbcv="([A-Z0-9]+)\.(\d+)\.(\d+)"[^>]*/>'  # verse milestone
    r'(.*?)'                                                   # verse content
    r'<ve\s*/>',                                              # verse end
    re.DOTALL,
)


def _parse_usfx(xml_bytes: bytes) -> list[tuple]:
    text = xml_bytes.decode("utf-8", errors="replace")
    # Remove sub-trees that aren't verse content
    text = _FOOTNOTE_RE.sub(" ", text)
    text = _CROSSREF_RE.sub(" ", text)
    text = _NOTE_RE.sub(" ", text)

    rows: list[tuple] = []
    for m in _VERSE_RE.finditer(text):
        book_code = m.group(1)
        chapter   = int(m.group(2))
        verse_n   = int(m.group(3))
        content   = _TAG_RE.sub(" ", m.group(4))
        content   = " ".join(content.split())
        if content:
            rows.append((book_code, chapter, verse_n, content))
    return rows


# ---------------------------------------------------------------------------
# Plain-text parser for Charles translations (Enoch, Jubilees)
#
# Format varies, but common patterns:
#   "CHAPTER I" or "Chapter 1" → new chapter
#   "1. Text here" or "1 Text here" → verse
# ---------------------------------------------------------------------------
_ROMAN_MAP = {
    "I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,
    "XI":11,"XII":12,"XIII":13,"XIV":14,"XV":15,"XVI":16,"XVII":17,"XVIII":18,
    "XIX":19,"XX":20,"XXI":21,"XXII":22,"XXIII":23,"XXIV":24,"XXV":25,
    "XXVI":26,"XXVII":27,"XXVIII":28,"XXIX":29,"XXX":30,"XXXI":31,"XXXII":32,
    "XXXIII":33,"XXXIV":34,"XXXV":35,"XXXVI":36,"XXXVII":37,"XXXVIII":38,
    "XXXIX":39,"XL":40,"XLI":41,"XLII":42,"XLIII":43,"XLIV":44,"XLV":45,
    "XLVI":46,"XLVII":47,"XLVIII":48,"XLIX":49,"L":50,"LI":51,"LII":52,
    "LIII":53,"LIV":54,"LV":55,"LVI":56,"LVII":57,"LVIII":58,"LIX":59,
    "LX":60,"LXI":61,"LXII":62,"LXIII":63,"LXIV":64,"LXV":65,"LXVI":66,
    "LXVII":67,"LXVIII":68,"LXIX":69,"LXX":70,"LXXI":71,"LXXII":72,
    "LXXIII":73,"LXXIV":74,"LXXV":75,"LXXVI":76,"LXXVII":77,"LXXVIII":78,
    "LXXIX":79,"LXXX":80,"LXXXI":81,"LXXXII":82,"LXXXIII":83,"LXXXIV":84,
    "LXXXV":85,"LXXXVI":86,"LXXXVII":87,"LXXXVIII":88,"LXXXIX":89,"XC":90,
    "XCI":91,"XCII":92,"XCIII":93,"XCIV":94,"XCV":95,"XCVI":96,"XCVII":97,
    "XCVIII":98,"XCIX":99,"C":100,"CI":101,"CII":102,"CIII":103,"CIV":104,
    "CV":105,"CVI":106,"CVII":107,"CVIII":108,
}

# Matches the chapter-opening line: "I. 1. text..." or "II. 3. text..."
_CHAP_OPEN_RE = re.compile(
    r"^([IVXLC]+)\.\s+(\d+)\.\s+(.*)", re.DOTALL
)
# Matches an inline verse boundary: " N. " within flowing text
_INLINE_VERSE_RE = re.compile(r"\s+(\d+)\.\s+")
# Annotation brackets used by Charles — strip for clean verse text
_ANNOT_RE = re.compile(r"[⌜⌝〚〛‹›\[\]()†]")


def _clean(text: str) -> str:
    text = _ANNOT_RE.sub("", text)
    return " ".join(text.split())


def _parse_enoch_gutenberg(raw: str, book_abbr: str) -> list[tuple]:
    """
    Parse R.H. Charles 1 Enoch (Gutenberg pg77935.txt).

    Format: the translation text begins at the first line matching
    "^ROMAN. DIGIT. text" (e.g. "I. 1. The words of...").
    Within each chapter block, verse numbers appear inline in the
    flowing text ("... are removed. 2. And he took up...").
    """
    # Locate the start of the actual translation (skip introduction)
    start = raw.find("\nI. 1.")
    if start == -1:
        return []
    text = raw[start + 1:]   # drop the leading newline

    # Split into chapter blocks on lines that start "ROMAN. DIGIT."
    # We split on "\nROMAN. " pattern
    chap_split_re = re.compile(r"\n(?=[IVXLC]+\. \d+\.)")
    blocks = chap_split_re.split(text)

    rows: list[tuple] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = _CHAP_OPEN_RE.match(block)
        if not m:
            continue
        roman = m.group(1)
        chapter = _ROMAN_MAP.get(roman)
        if chapter is None:
            continue
        first_verse_num = int(m.group(2))
        rest = m.group(3)

        # Now split the rest by inline verse numbers
        # Build a list of (verse_num, raw_text) by scanning inline markers
        parts: list[tuple[int, str]] = [(first_verse_num, "")]
        cursor = 0
        for vm in _INLINE_VERSE_RE.finditer(rest):
            # append text so far to current verse
            parts[-1] = (parts[-1][0], parts[-1][1] + rest[cursor:vm.start()])
            next_v = int(vm.group(1))
            parts.append((next_v, ""))
            cursor = vm.end()
        parts[-1] = (parts[-1][0], parts[-1][1] + rest[cursor:])

        for vnum, vtext in parts:
            vtext = _clean(vtext)
            if vtext:
                rows.append((book_abbr, chapter, vnum, vtext))

    return rows


# ---------------------------------------------------------------------------
# Book metadata parser — reads BookNames.xml from the WEB zip
# ---------------------------------------------------------------------------
def _parse_book_names(xml_bytes: bytes) -> dict[str, str]:
    """Return {usfx_code: short_name}."""
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace").lstrip("﻿"))
    return {
        elem.get("code", ""): elem.get("short", elem.get("abbr", ""))
        for elem in root
        if elem.get("code")
    }


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------
def _get_or_create_book(db: Session, code: str, name: str, canonical_order: int) -> Book:
    book = db.query(Book).filter_by(abbreviation=code).first()
    if not book:
        book = Book(
            name=name,
            abbreviation=code,
            testament=_testament(code),
            canonical_order=canonical_order,
        )
        db.add(book)
        db.flush()
    return book


def _write_verses(db: Session, rows: list[tuple], book_names: dict[str, str]):
    books: dict[str, Book] = {}
    chapters: dict[tuple, Chapter] = {}
    order_counter: dict[str, int] = {}   # assign stable order on first encounter

    for book_code, chap_n, verse_n, text in rows:
        if book_code not in books:
            if book_code not in order_counter:
                order_counter[book_code] = len(order_counter) + 1
            name = book_names.get(book_code, book_code)
            books[book_code] = _get_or_create_book(
                db, book_code, name, order_counter[book_code]
            )

        chap_key = (book_code, chap_n)
        if chap_key not in chapters:
            c = db.query(Chapter).filter_by(
                book_id=books[book_code].id, number=chap_n
            ).first()
            if not c:
                c = Chapter(book_id=books[book_code].id, number=chap_n)
                db.add(c)
                db.flush()
            chapters[chap_key] = c

        existing = db.query(Verse).filter_by(
            chapter_id=chapters[chap_key].id, number=verse_n
        ).first()
        if not existing:
            db.add(Verse(chapter_id=chapters[chap_key].id, number=verse_n, text=text))

    db.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def ingest_all():
    init_db()
    db: Session = SessionLocal()

    # 1. WEB British Edition (66 books + Deuterocanon)
    print("WEB British Edition + Deuterocanon...")
    web_zip_path = _download(WEB_USFX_URL, DATA_DIR / "web_usfx.zip")
    with zipfile.ZipFile(BytesIO(web_zip_path.read_bytes())) as z:
        names = z.namelist()
        usfx_name  = next(n for n in names if n.endswith("_usfx.xml"))
        books_name = next((n for n in names if n == "BookNames.xml"), None)

        book_names = _parse_book_names(z.read(books_name)) if books_name else {}
        rows = _parse_usfx(z.read(usfx_name))

    print(f"  {len(rows):,} verses parsed")
    _write_verses(db, rows, book_names)

    # 2. 1 Enoch (R.H. Charles, Gutenberg #77935)
    print("1 Enoch (R.H. Charles 1917 ed.)...")
    enoch_path = _download(ENOCH_URL, DATA_DIR / "enoch.txt")
    rows = _parse_enoch_gutenberg(enoch_path.read_text(errors="replace"), "1EN")
    print(f"  {len(rows):,} verses parsed")
    _write_verses(db, rows, {"1EN": "1 Enoch"})
    # Jubilees (R.H. Charles 1902) — no clean public-domain plain-text found yet; TODO

    total = db.query(Verse).count()
    print(f"\nTotal verses in DB: {total:,}")
    db.close()


if __name__ == "__main__":
    ingest_all()
