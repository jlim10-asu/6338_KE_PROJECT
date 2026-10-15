"""Pure helpers for ingestion (no database, network or LLM): press-release cleaning, paragraph and sentence numbering,
abstract parsing from PMC / PubMed / Crossref XML, abstract section labels, and abstract quality checks.
Used by s03a_fetch_abstracts.py, s03_ingest.py, s04_extract.py and tests/test_ingest.py."""
import re
import xml.etree.ElementTree as ET

# ---------- sentences ----------
_ABBREV = "".join(rf"(?<!\b{a}\.)" for a in ("Dr", "Mr", "Ms", "Mrs", "Prof", "St", "vs", "al", "Fig", "Figs", "No", "U.S", "e.g", "i.e",
                                              "approx", "Jan", "Feb", "Aug", "Sept", "Oct", "Nov", "Dec", "Inc", "Ltd", "Co", "p", "pp", "vol", "ref", "refs", "Eq"))
# A sentence ends at . ! ? (plus closing quotes/brackets) followed by whitespace and an upper-case letter, digit, or opening quote.
_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])([\"”’)\]]*)\s+(?=[A-Z0-9\"'“‘(\[])")


def split_sentences(text):
    out, start = [], 0
    for m in _SPLIT.finditer(text or ""):
        out.append(text[start:m.start() + len(m.group(1))])
        start = m.end()
    out.append((text or "")[start:])
    return [" ".join(s.split()) for s in out if s.strip()]


def words(text):
    return len((text or "").split())


# ---------- press releases ----------
_END_MARK = re.compile(r"(?m)^\s*#\s*#\s*#\s*$|\s#{3}(\s|$)")      # EurekAlert ends the release body with "###"
_DROP_PARA = re.compile(
    r"^(media contacts?|contacts?|media|press contacts?|for more information|disclaimer|about (the |us\b)|journal\b|doi\b|"
    r"method of research|subject of research|article title|article publication date|coi statement|funding|keywords?|"
    r"(this|the) (research|study|work|project) was (supported|funded)|research reported in this publication was supported)",
    re.I)
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d ()-]{7,}\d")


def clean_press(text):
    """Cut the release at the EurekAlert end mark and drop contact, funding and metadata paragraphs.
    Returns (paragraphs, removed_chars)."""
    text = (text or "").replace("\r\n", "\n")
    m = _END_MARK.search(text)
    body = text[:m.start()] if m else text
    paras = re.split(r"\n\s*\n", body) if re.search(r"\n\s*\n", body) else body.split("\n")
    keep = []
    for p in (" ".join(x.split()) for x in paras):
        if not p:
            continue
        if _DROP_PARA.match(p) or (len(_CONTACT.findall(p)) and words(p) < 40):
            continue
        if keep and words(p) <= 6 and not re.search(r"[.!?\"”’)]$", p):   # a subheading ('AUTHOR COMMENTS'); the first paragraph may be a headline
            continue
        keep.append(p)
    return keep, len(text) - sum(len(p) for p in keep)


def number_press(paragraphs):
    """[(para_no, sent_no, sid, text)] with IDs P{p}.S{s}, 1-based."""
    return [(p, s, f"P{p}.S{s}", sent) for p, para in enumerate(paragraphs, 1) for s, sent in enumerate(split_sentences(para), 1)]


def press_windows(numbered, max_words):
    """Group paragraphs into windows of at most max_words. Each window owns its paragraphs and sees the previous
    paragraph as context only. Returns [(owned_paragraph_numbers, context_paragraph_number or None)]."""
    paras = sorted({p for p, _, _, _ in numbered})
    size = {p: sum(words(t) for q, _, _, t in numbered if q == p) for p in paras}
    if sum(size.values()) <= max_words:
        return [(paras, None)]
    windows, cur, n = [], [], 0
    for p in paras:
        if cur and n + size[p] > max_words:
            windows.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += size[p]
    windows.append(cur)
    return [(w, (w[0] - 1) if i else None) for i, w in enumerate(windows)]


# ---------- abstract sections ----------
SECTION_OF = {}
for cat, labels in {
    "BACKGROUND": ["background", "introduction", "importance", "context", "purpose", "aim", "aims", "objective", "objectives",
                   "rationale", "background and aims", "background and purpose", "background and objectives", "aims and objectives",
                   "abstract objective"],
    "METHODS": ["methods", "method", "design", "setting", "participants", "patients", "materials and methods", "patients and methods",
                "methods and materials", "measures", "measurements", "exposure", "exposures", "intervention", "interventions",
                "main outcomes and measures", "main outcome measures", "study design", "data sources", "design and setting",
                "design, setting, and participants", "design, setting and participants", "methods & results"],
    "RESULTS": ["results", "findings", "main results", "principal findings", "methods and results", "methods & results",
                "methods and findings", "methodology/principal findings", "results and discussion"],
    "CONCLUSIONS": ["conclusion", "conclusions", "interpretation", "conclusions and relevance", "discussion", "implications",
                    "conclusions and implications", "conclusions/significance", "significance"],
    "UNLABELED": ["abstract", "summary"],
    "OTHER": ["funding", "registration", "trial registration", "clinical trial registration", "keywords"],
}.items():
    for lab in labels:
        SECTION_OF.setdefault(lab, cat)
SECTION_OF["methods & results"] = "RESULTS"
NLM_CATEGORY = {"BACKGROUND": "BACKGROUND", "OBJECTIVE": "BACKGROUND", "METHODS": "METHODS", "RESULTS": "RESULTS",
                "CONCLUSIONS": "CONCLUSIONS", "UNASSIGNED": "UNLABELED"}
CITABLE_SECTIONS = {"RESULTS", "CONCLUSIONS", "UNLABELED"}
_LABELS = sorted(SECTION_OF, key=len, reverse=True)
# Label words match in any case; the next word must start upper-case unless a colon follows ('Results indicated' is prose).
_HEADER = re.compile(r"^((?i:" + "|".join(re.escape(l) for l in _LABELS) + r"))(\s*:\s*|\s+(?=[A-Z0-9]))")


def section_category(label):
    lab = " ".join((label or "").replace(":", " ").split()).lower()
    if lab.upper() in NLM_CATEGORY:
        return NLM_CATEGORY[lab.upper()]
    return SECTION_OF.get(lab, "UNLABELED" if not lab else "OTHER")


def _header(sentence):
    """(category, rest) when the sentence starts with a section label: 'RESULTS: The ...', 'Results Of the 140 ...',
    'Background Little is known ...'. A label followed by a lower-case word ('Results indicated ...') is prose, not a header."""
    m = _HEADER.match(sentence)
    if not m:
        return None
    label, sep = m.group(1), m.group(2)
    if ":" not in sep and not (label.isupper() or label[:1].isupper()):
        return None
    return SECTION_OF[label.lower()], sentence[m.end():].strip()


def label_text(text):
    """Section-label an abstract given as plain text. Returns [(category, sentence)] with header words removed.
    An abstract with no recognised headers is all UNLABELED."""
    out, current = [], "UNLABELED"
    for sent in split_sentences(text):
        h = _header(sent)
        while h:                       # 'Abstract Objective To ...' carries two labels
            current, sent = h
            h = _header(sent) if sent else None
        if sent:
            out.append((current, sent))
    return out


def label_sections(sections):
    """[(raw_label, text)] from a structured source -> [(category, sentence)]. Unlabeled blocks fall back to header detection."""
    out = []
    for raw, text in sections:
        cat = section_category(raw)
        if cat == "UNLABELED":
            out += label_text(text)
        else:
            out += [(cat, s) for s in split_sentences(text)]
    return out


def number_abstract(labeled):
    """[(sent_no, sid, category, text)] with IDs S{n}, 1-based."""
    return [(n, f"S{n}", cat, s) for n, (cat, s) in enumerate(labeled, 1)]


# ---------- abstract quality ----------
_FRONT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\bcorrespondence\b|\bkeywords?\s*:|\breceived\s*:|\baccepted\s*:|\bpublished\s*:|"
                    r"\bcopyright\b|©|\baffiliations?\b", re.I)
BLOCKING_FLAGS = {"front_matter", "truncated_start", "too_short", "empty"}


def abstract_flags(text, labeled, min_words=80, max_words=600):
    flags = []
    n = words(text)
    if not n:
        return ["empty"]
    if n < min_words:
        flags.append("too_short")
    if n > max_words:
        flags.append("too_long")
    if _FRONT.search(text):
        flags.append("front_matter")
    first = labeled[0][1] if labeled else text
    if first[:1].islower() or words(first) < 4:
        flags.append("truncated_start")
    if not re.search(r"[.!?)\]\d”\"]\s*$", text.strip()):
        flags.append("truncated_end")
    return flags


def token_jaccard(a, b):
    wa, wb = set(re.findall(r"\w+", (a or "").lower())), set(re.findall(r"\w+", (b or "").lower()))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


# ---------- XML parsers (PMC JATS, PubMed, Crossref JATS) ----------
def _text(el):
    return " ".join("".join(el.itertext()).split())


def _local(tag):
    return tag.rsplit("}", 1)[-1].split(":")[-1]


_SKIP_ABSTRACT_TYPES = {"graphical", "teaser", "summary", "author-summary", "editor-summary", "precis", "toc", "web-summary",
                        "short", "lay", "plain-language-summary", "key-points"}


def _jats_abstract(abstract_el):
    """[(label, text)] from a JATS <abstract>: one block per <sec>, or one unlabeled block of its paragraphs."""
    blocks = []
    secs = [c for c in abstract_el if _local(c.tag) == "sec"]
    if secs:
        for sec in secs:
            title = next((_text(c) for c in sec if _local(c.tag) == "title"), "")
            body = " ".join(_text(c) for c in sec if _local(c.tag) != "title")
            if body:
                blocks.append((title, body))
    else:
        body = " ".join(_text(c) for c in abstract_el if _local(c.tag) != "title")
        if body:
            blocks.append(("", body))
    return blocks


def parse_pmc(xml_text):
    """PMC efetch XML -> [(label, text)] for the article's main abstract (graphical, teaser and summary abstracts skipped)."""
    root = ET.fromstring(xml_text)
    metas = [el for el in root.iter() if _local(el.tag) == "article-meta"]
    cands = [a for m in metas[:1] for a in m if _local(a.tag) == "abstract"] or \
            [el for el in root.iter() if _local(el.tag) == "abstract"]
    plain = [el for el in cands if (el.get("abstract-type") or "").lower() not in _SKIP_ABSTRACT_TYPES]
    return _jats_abstract(plain[0]) if plain else []


def parse_pubmed(xml_text):
    root = ET.fromstring(xml_text)
    out = []
    for el in root.iter("AbstractText"):
        t = _text(el)
        if t:
            out.append((el.get("NlmCategory") or el.get("Label") or "", t))
    return out


def parse_crossref(jats):
    """Crossref 'abstract' field: a JATS fragment with jats: prefixes."""
    s = re.sub(r"<(/?)jats:", r"<\1", jats or "")
    try:
        root = ET.fromstring(f"<abstract>{s}</abstract>")
        return _jats_abstract(root)
    except ET.ParseError:
        t = " ".join(re.sub(r"<[^>]+>", " ", s).split())
        return [("", t)] if t else []


def join_sections(sections):
    return " ".join(t for _, t in sections)


# ---------- corpus records (shared by s03a and s03 so both see the same sample) ----------
REQUIRED = ["euid", "doi", "release_date", "title", "institution", "press_text", "abstract"]
OPTIONAL = ["ror", "paper_title", "journal", "publication_date", "cited_by_count", "is_retracted",
            "study_design", "author_names", "author_affiliations", "openalex_field",
            "domain", "abstract_source", "issn_l", "pmcid", "year"]


def _authors(s):
    import json
    try:
        return json.loads(s) if isinstance(s, str) else []
    except json.JSONDecodeError:
        return []


def _first_inst(auths):
    return next((i for a in auths for i in a.get("institutions") or [] if i.get("name")), {})


def _truthy(v):
    try:
        return float(v) != 0
    except (TypeError, ValueError):
        return str(v).strip().lower() in ("true", "yes")


def load_records(corpus_csv, dev_limit=0, seed=42):
    """corpus.csv -> the records schema used downstream, filtered to rows with every required field, deduplicated by euid,
    and sampled to dev_limit rows (seed 42). Returns (records DataFrame, rows before filtering)."""
    import pandas as pd
    df = pd.read_csv(corpus_csv, dtype=str)
    auths = df["authors_json"].map(_authors) if "authors_json" in df else pd.Series([[]] * len(df))
    df["press_text"] = df["body_text"]
    df["title"] = df["paper_title"]                     # corpus has no press-release headline
    df["release_date"] = df["publication_date"].fillna(df["year"].dropna() + "-01-01")
    df["institution"] = auths.map(lambda a: _first_inst(a).get("name"))
    df["ror"] = auths.map(lambda a: _first_inst(a).get("ror"))
    df["author_names"] = auths.map(lambda a: "; ".join(x["name"] for x in a if x.get("name")) or None)
    df["author_affiliations"] = auths.map(
        lambda a: "; ".join(dict.fromkeys(i["name"] for x in a for i in x.get("institutions") or [] if i.get("name"))) or None)
    df["is_retracted"] = (df["oa_is_retracted"].map(_truthy) | df["cr_retracted"].map(_truthy)).map(str).str.lower()
    df["openalex_field"] = df["primary_field"]
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    before = len(df)
    df = df.dropna(subset=REQUIRED)
    df = df[df[REQUIRED].apply(lambda col: col.str.strip().str.len() > 0).all(axis=1)]
    df["doi"] = df["doi"].str.lower().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    df = df.drop_duplicates("euid")
    if dev_limit:
        df = df.sample(n=min(dev_limit, len(df)), random_state=seed)
    return df[REQUIRED + OPTIONAL].copy(), before
