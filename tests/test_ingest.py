"""Tests for ingest.py and the v4 FindingRef schema. No database, network or LLM needed: python tests/test_ingest.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from ingest import (abstract_flags, clean_press, label_sections, label_text, number_abstract, number_press, parse_crossref,
                    parse_pmc, parse_pubmed, press_windows, split_sentences)
from schemas import FindingRef

# 1. Press cleaning: cut at ###, drop contact / funding paragraphs, keep the body
release = """Coffee lowers diabetes risk, a new study finds. It followed 900 adults.

"Our results are striking," said Dr. Smith.

Media contact: Jane Doe, jane@uni.edu, +1 555 123 4567

###

About the University: founded in 1900."""
paras, removed = clean_press(release)
assert paras == ["Coffee lowers diabetes risk, a new study finds. It followed 900 adults.", '"Our results are striking," said Dr. Smith.'], paras
assert removed > 0
print("1 clean_press cuts at ### and drops contacts OK")

# 2. Numbering and windows: IDs P{p}.S{s}; a long release is split into paragraph windows with one context paragraph
num = number_press(paras)
assert [x[2] for x in num] == ["P1.S1", "P1.S2", "P2.S1"], num
long = number_press([f"Sentence {i} has five words." for i in range(10)])
wins = press_windows(long, 12)
assert wins[0] == ([1, 2], None) and wins[1] == ([3, 4], 2), wins
assert press_windows(num, 2000) == [([1, 2], None)]
print("2 numbering and windows OK")

# 3. Sentence splitting: digits and abbreviations
assert split_sentences("Risk fell (p=0·03). 132 patients had SSI. See Metzger, p. 14. Values were 8.5 vs. 5.0 overall.") == [
    "Risk fell (p=0·03).", "132 patients had SSI.", "See Metzger, p. 14.", "Values were 8.5 vs. 5.0 overall."]
print("3 sentence splitting OK")

# 4. Abstract section labels from header words, including headers without colons; prose 'Results indicated' is not a header
lab = label_text("Abstract Objective To determine the effect. Study Design Retrospective, case-control study. "
                 "Results Of the 140 smokers, 61 quit. Conclusion This is the first study.")
assert [c for c, _ in lab] == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"] and lab[2][1] == "Of the 140 smokers, 61 quit.", lab
assert {c for c, _ in label_text("Teeth grow. Results indicated a correlation. These results indicate growth.")} == {"UNLABELED"}
assert [c for c, _ in label_text("Background Little is known. Methods and Results This study included 110 742 patients.")] == ["BACKGROUND", "RESULTS"]
assert number_abstract(lab)[2][:3] == (3, "S3", "RESULTS")
print("4 abstract section labels OK")

# 5. Quality flags: front matter, truncated start, clean
front = "// Margot Bucau 1 , Nathalie Gault 2 Correspondence to: margot.bucau@aphp.fr Keywords: anal dysplasia " + "word " * 90 + "end."
assert "front_matter" in abstract_flags(front, label_text(front))
trunc = "INTRODUCTION: genes. " + "Many clinical studies have elucidated the correlation. " * 15
assert "truncated_start" in abstract_flags(trunc, label_text(trunc))
clean = "Coffee intake was studied in adults. " * 20
assert abstract_flags(clean, label_text(clean)) == []
print("5 abstract quality flags OK")

# 6. Parsers: PubMed AbstractText categories, PMC JATS sections (graphical abstract skipped), Crossref JATS fragment
pubmed = """<PubmedArticleSet><PubmedArticle><MedlineCitation><Article><Abstract>
<AbstractText Label="IMPORTANCE" NlmCategory="BACKGROUND">Preterm infants matter.</AbstractText>
<AbstractText Label="RESULTS" NlmCategory="RESULTS">Survival rose to <i>78%</i>.</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
assert parse_pubmed(pubmed) == [("BACKGROUND", "Preterm infants matter."), ("RESULTS", "Survival rose to 78%.")]
pmc = """<pmc-articleset><article><front><article-meta>
<abstract abstract-type="graphical"><p>Picture.</p></abstract>
<abstract><sec><title>Background</title><p>Stigma harms health.</p></sec><sec><title>Results</title><p>Women reported more shame.</p></sec></abstract>
</article-meta></front></article></pmc-articleset>"""
assert parse_pmc(pmc) == [("Background", "Stigma harms health."), ("Results", "Women reported more shame.")]
cr = "<jats:title>Abstract</jats:title><jats:sec><jats:title>Conclusions</jats:title><jats:p>Quitting helps.</jats:p></jats:sec>"
assert parse_crossref(cr) == [("Conclusions", "Quitting helps.")]
assert [c for c, _ in label_sections(parse_pubmed(pubmed))] == ["BACKGROUND", "RESULTS"]
print("6 PubMed / PMC / Crossref parsers OK")

# 7. FindingRef: IDs must be shown and citable
f = dict(text="x", exposure="coffee", outcome="diabetes", direction="decrease", claim_strength=1, evidence_ids=["S3"], confidence_score=0.9)
assert FindingRef.model_validate(f, context={"known": {"S1", "S3"}, "citable": {"S3"}}).evidence_ids == ["S3"]
for ctx in ({"known": {"S1"}, "citable": {"S1"}}, {"known": {"S3"}, "citable": set()}):
    try:
        FindingRef.model_validate(f, context=ctx)
        raise AssertionError("should have been rejected")
    except ValidationError:
        pass
print("7 FindingRef ID validation OK\nALL TESTS PASSED")
