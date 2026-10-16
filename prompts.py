PROMPT_NOTE = ("v4: whole-document extraction from numbered sentences; findings cite sentence IDs and code builds evidence_span "
               "(EXTRACT_DOC_*). v3 chunk prompts (EXTRACT_SYSTEM, EXTRACT_USER, REPAIR_USER) are kept for eval_resolver.py R1.")

EXTRACT_SYSTEM = """You extract research findings from medical press releases and journal abstracts. Your output populates a knowledge graph that compares how strongly a press release states a finding with how strongly the source paper states it. For that comparison to work, the wording-based fields must describe the text as written, not what the study design would justify.

<definitions>
A finding is a result of the study being reported that relates an exposure to an outcome.
- exposure: the intervention, drug, behavior, or risk factor (e.g., "coffee consumption", "metformin", "night-shift work").
- outcome: the disease, symptom, measurement, or health state it affects (e.g., "type 2 diabetes", "systolic blood pressure").
- direction: the effect of the exposure on the outcome: increase, decrease, no_effect (the study reports no association or no difference), or unclear.
- claim_strength: how strongly the wording asserts the relationship (labels from Wright & Augenstein, 2021):
  0 = no relationship (a null result, e.g., "was not associated with")
  1 = correlational ("associated with", "linked to", "higher risk among")
  2 = conditional causal ("may reduce", "could lead to", "might protect against")
  3 = direct causal ("reduces", "causes", "prevents", "protects against", "leads to")
</definitions>

<rules>
1. Extract findings only from <text>. <look_ahead> is the first sentence of the next chunk, which is extracted separately; use it only to understand a sentence cut off at the end of <text>.
2. Extract results of the study being reported, including results stated in researcher quotes. Skip background on prior work, study aims, methods, funding, and contact details, because they are not this study's findings.
3. Create one finding per exposure–outcome pair per statement. Split compound statements ("coffee and tea lowered the risk of stroke and dementia") into one finding per pair. If the same pair is restated with a different claim strength, as when a quote goes further than the summary, extract each statement; if it is restated with the same strength, extract it once.
4. Rate claim_strength from the wording alone. A press release saying an observational study "prevents" a disease is rated 3; that gap between wording and evidence is what the system measures.
5. Write exposure and outcome as short noun phrases using the standard medical name: expand abbreviations ("T2D" becomes "type 2 diabetes") and keep doses, percentages, and populations out of the entity. Standard names let entity linking match them to the Disease Ontology.
6. Copy evidence_span character for character from <text> as complete sentences: from the first word of the sentence that states the finding through its final punctuation, with the same spelling, casing, and punctuation, and no ellipses. Include the preceding sentence too when the finding's subject is only named there (the sentence starts with "It", "This", "These", or similar). Whole sentences keep the subject and the hedge words that the strength comparison is read from. When one sentence reports several findings, each of those findings uses the same sentence. Code checks that evidence_span is an exact substring of <text> that starts and ends at sentence boundaries, and rejects the finding otherwise.
7. Write text as one standalone sentence that keeps the original hedge or causal verb, so the restatement carries the same claim strength as the evidence.
8. Set population to who was studied when <text> states it (e.g., "adults over 65", "mice"); otherwise null.
9. Set confidence_score to your confidence that the finding is extracted correctly: about 0.9 for an explicit, unambiguous result; 0.5 to 0.7 when the exposure, outcome, or direction had to be inferred; below 0.5 when unsure the sentence reports a finding at all.
10. When <text> reports no finding, return an empty findings list.
</rules>

<examples>
<example>
<source_type>abstract</source_type>
<text>We followed 41,000 adults for 12 years. Participants who drank three or more cups of coffee daily had a 20% lower risk of type 2 diabetes (HR 0.80, 95% CI 0.72-0.89). Coffee intake was not associated with stroke.</text>
<look_ahead></look_ahead>
<output>{"findings": [{"text": "Drinking three or more cups of coffee daily is associated with a lower risk of type 2 diabetes in adults.", "exposure": "coffee consumption", "outcome": "type 2 diabetes", "population": "adults", "direction": "decrease", "claim_strength": 1, "evidence_span": "Participants who drank three or more cups of coffee daily had a 20% lower risk of type 2 diabetes (HR 0.80, 95% CI 0.72-0.89).", "confidence_score": 0.9}, {"text": "Coffee consumption was not associated with stroke in adults.", "exposure": "coffee consumption", "outcome": "stroke", "population": "adults", "direction": "no_effect", "claim_strength": 0, "evidence_span": "Coffee intake was not associated with stroke.", "confidence_score": 0.9}]}</output>
</example>
<example>
<source_type>press</source_type>
<text>Eating blueberries may slow cognitive decline in older adults, according to new research. "Our results show that blueberries slow cognitive decline," said lead author Dr. Jane Smith.</text>
<look_ahead>The team now plans a larger trial.</look_ahead>
<output>{"findings": [{"text": "Eating blueberries may slow cognitive decline in older adults.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 2, "evidence_span": "Eating blueberries may slow cognitive decline in older adults, according to new research.", "confidence_score": 0.9}, {"text": "Blueberries slow cognitive decline in older adults, according to the lead author.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 3, "evidence_span": "\\"Our results show that blueberries slow cognitive decline,\\" said lead author Dr. Jane Smith.", "confidence_score": 0.85}]}</output>
</example>
<example>
<source_type>press</source_type>
<text>The study was funded by the National Institutes of Health. Previous research has linked obesity to several cancers.</text>
<look_ahead>The researchers analyzed records from 2 million patients.</look_ahead>
<output>{"findings": []}</output>
</example>
<example>
<source_type>abstract</source_type>
<text>In this randomized trial, metformin reduced fasting glucose and HbA1c compared with placebo.</text>
<look_ahead>Patients receiving metformin also lost more weight.</look_ahead>
<output>{"findings": [{"text": "Metformin reduced fasting blood glucose compared with placebo.", "exposure": "metformin", "outcome": "fasting blood glucose", "population": null, "direction": "decrease", "claim_strength": 3, "evidence_span": "In this randomized trial, metformin reduced fasting glucose and HbA1c compared with placebo.", "confidence_score": 0.9}, {"text": "Metformin reduced glycated hemoglobin compared with placebo.", "exposure": "metformin", "outcome": "glycated hemoglobin", "population": null, "direction": "decrease", "claim_strength": 3, "evidence_span": "In this randomized trial, metformin reduced fasting glucose and HbA1c compared with placebo.", "confidence_score": 0.9}]}</output>
</example>
</examples>"""

EXTRACT_USER = """<source_type>{source_type}</source_type>
<text>
{text}
</text>
<look_ahead>
{look_ahead}
</look_ahead>
Extract the findings reported in <text>."""

REPAIR_USER = """<source_type>{source_type}</source_type>
<text>
{text}
</text>
<look_ahead>
{look_ahead}
</look_ahead>
<failed_findings>
{failed}
</failed_findings>
<validation_errors>
{errors}
</validation_errors>
The findings in <failed_findings> were extracted from <text> and failed automated validation for the reasons in <validation_errors>. Return corrected versions of only these findings:
- "not a verbatim substring": copy evidence_span again exactly as it appears in <text>.
- "not complete sentences": extend evidence_span to the whole sentence(s) in <text>, from the first word through the final punctuation.
- "must be non-empty" or "must be different entities": identify both the exposure and the outcome from <text>, or drop the finding if <text> names only one.
- A value outside its allowed range: use a value inside the range.
Drop any finding that <text> does not support, and return an empty findings list if none can be corrected."""

NED_SYSTEM = """You link a medical entity mention, taken from a press release or journal abstract, to one entry in a reference vocabulary (the Human Disease Ontology or Hetionet). Mentions linked to the same entry are merged into one graph node, and a later step compares findings across papers by those nodes. A wrong link therefore merges unrelated findings, while NIL only leaves a mention unmerged, so return NIL whenever no candidate clearly names the same concept.

<candidate_format>
Each line in <candidates> is: id | name | kind | synonyms (the synonyms are also separated by |).
</candidate_format>

<rules>
1. Select the candidate that names the same concept as the mention, using <context> to resolve abbreviations and ambiguous terms.
2. A synonym match counts as the same concept ("heart attack" and "myocardial infarction").
3. Return NIL when the closest candidate is broader, narrower, or only related to the mention, and when the mention is a lifestyle factor, measurement, or procedure that the candidates do not cover.
4. Set selected_id to a candidate id copied exactly, or to NIL.
5. Set confidence to about 0.9 for an exact name or synonym match, and 0.6 to 0.8 when the context was needed to decide. For NIL, give your confidence that no candidate fits.
6. Write brief_explanation as one short sentence naming the evidence that decided the link.
</rules>

<examples>
<example>
<mention>T2D</mention>
<context>Participants who drank coffee daily had a lower risk of T2D</context>
<candidates>
DOID:9352 | type 2 diabetes mellitus | Disease | NIDDM | non-insulin-dependent diabetes mellitus | type 2 diabetes
DOID:9744 | type 1 diabetes mellitus | Disease | IDDM | insulin-dependent diabetes mellitus
DOID:9351 | diabetes mellitus | Disease | diabetes
</candidates>
<output>{"selected_id": "DOID:9352", "confidence": 0.95, "brief_explanation": "T2D abbreviates type 2 diabetes, an exact synonym of DOID:9352."}</output>
</example>
<example>
<mention>MS</mention>
<context>Patients with MS who took vitamin D had fewer relapses</context>
<candidates>
DOID:2377 | multiple sclerosis | Disease | insular sclerosis
DOID:1754 | mitral valve stenosis | Disease | Mitral stenosis
</candidates>
<output>{"selected_id": "DOID:2377", "confidence": 0.8, "brief_explanation": "Relapses in the context indicate multiple sclerosis rather than mitral stenosis."}</output>
</example>
<example>
<mention>sleep duration</mention>
<context>Shorter sleep duration was associated with higher blood pressure</context>
<candidates>
DOID:535 | sleep disorder | Disease | Non-organic sleep disorder
</candidates>
<output>{"selected_id": "NIL", "confidence": 0.9, "brief_explanation": "Sleep duration is a behavior measure, not the disease sleep disorder."}</output>
</example>
</examples>"""

NED_USER = """<mention>{mention}</mention>
<context>{context}</context>
<candidates>
{candidates}
</candidates>
Link the mention to one candidate id, or return NIL."""
ADJUDICATE_SYSTEM = """You compare one statement from a medical press release with findings extracted from the abstract of the paper that press release announces. Your output decides two things in a knowledge graph: which abstract finding, if any, the press statement is reporting, and whether the press release states that finding more strongly, equally, or less strongly than the paper. All candidates come from the same paper, so most share its topic; the task is to find the one that reports the same result, not the one on the same subject.

<definitions>
- Same finding: the press statement and the candidate relate the same exposure to the same outcome in the same study. Lay wording counts as the same ("heart attack" and "myocardial infarction", "kids" and "children"), and so does dropping detail such as a dose or a subgroup. A different outcome, a different exposure, or a different comparison is a different finding even when the topic matches.
- claim_strength: how strongly the wording asserts the relationship (labels from Wright & Augenstein, 2021):
  0 = no relationship (a null result, e.g., "was not associated with", "did not differ")
  1 = correlational ("associated with", "linked to", "higher risk among")
  2 = conditional causal ("may reduce", "could lead to", "might protect against")
  3 = direct causal ("reduces", "causes", "prevents", "protects against", "leads to")
- direction: increase, decrease, no_effect (no association or no difference), or unclear.
</definitions>

<rules>
1. Pick the candidate that reports the same finding as the press statement. When several do, pick the closest in exposure, outcome, and population. Return match 0 when none does, because a wrong match produces a false exaggeration label while 0 only leaves the statement unscored.
2. Rate statement_strength and claim_strength from <original_wording>, the verbatim source text, not from <restated>, which is a paraphrase that may have drifted. When the original wording covers several results, rate the wording for the specific finding being compared.
3. Apply the scale identically to both sides, from the wording alone, ignoring the study design. Identical or near-identical wording must receive the same strength; the comparison measures differences in wording, so any difference you assign must point to a specific word or phrase.
4. Rate the press statement even when match is 0, and set claim_strength and claim_direction to null in that case.
5. Set null_reframed to true only when the matched claim reports no effect or no difference and the press statement presents that result as a benefit or a safety assurance (e.g., "did not differ in complications" becoming "is safe"). Rate both strengths by the scale as usual; the flag records the reframing that the scale cannot express.
6. Write reasoning first, in one or two sentences: the deciding difference or match between the statement and the candidates, then the words that set each strength.
</rules>

<examples>
<example>
<press_statement>
<restated>Higher screen time is associated with poorer sleep quality in adolescents.</restated>
<original_wording>Higher screen time was associated with poorer sleep quality among adolescents</original_wording>
</press_statement>
<candidates>
<candidate number="1">
<restated>Higher daily screen time was associated with poorer sleep quality in adolescents.</restated>
<original_wording>higher daily screen time was associated with poorer sleep quality</original_wording>
</candidate>
<candidate number="2">
<restated>Screen time was not associated with body mass index in adolescents.</restated>
<original_wording>Screen time was not associated with BMI.</original_wording>
</candidate>
</candidates>
<output>{"reasoning": "Candidate 1 reports the same screen time and sleep quality result; candidate 2 is a different outcome. Both wordings say 'was associated with', so both are correlational.", "match": 1, "statement_strength": 1, "statement_direction": "decrease", "claim_strength": 1, "claim_direction": "decrease", "null_reframed": false}</output>
</example>
<example>
<press_statement>
<restated>Daily walking prevents dementia, according to the researchers.</restated>
<original_wording>"Walking every day prevents dementia," said the lead author.</original_wording>
</press_statement>
<candidates>
<candidate number="1">
<restated>Walking at least 30 minutes daily was associated with a lower risk of dementia in older adults.</restated>
<original_wording>walking at least 30 minutes daily was associated with a 25% lower risk of dementia</original_wording>
</candidate>
<candidate number="2">
<restated>Walking at least 30 minutes daily was associated with a lower risk of stroke in older adults.</restated>
<original_wording>Daily walking was also associated with lower stroke risk.</original_wording>
</candidate>
</candidates>
<output>{"reasoning": "Candidate 1 is the walking and dementia finding; candidate 2 is stroke. The press quote says 'prevents' (direct causal) while the abstract says 'was associated with' (correlational).", "match": 1, "statement_strength": 3, "statement_direction": "decrease", "claim_strength": 1, "claim_direction": "decrease", "null_reframed": false}</output>
</example>
<example>
<press_statement>
<restated>The new drug cut hospital stays by two days.</restated>
<original_wording>The new drug cut hospital stays by two days</original_wording>
</press_statement>
<candidates>
<candidate number="1">
<restated>The drug reduced 30-day mortality compared with placebo.</restated>
<original_wording>The drug reduced 30-day mortality compared with placebo</original_wording>
</candidate>
<candidate number="2">
<restated>Adverse event rates did not differ between the drug and placebo groups.</restated>
<original_wording>Adverse event rates were similar between groups.</original_wording>
</candidate>
</candidates>
<output>{"reasoning": "The statement is about length of hospital stay; the candidates report mortality and adverse events, which are different outcomes. 'Cut' is direct causal wording.", "match": 0, "statement_strength": 3, "statement_direction": "decrease", "claim_strength": null, "claim_direction": null, "null_reframed": false}</output>
</example>
<example>
<press_statement>
<restated>The procedure is safe for patients over 75.</restated>
<original_wording>the procedure is safe for patients over 75</original_wording>
</press_statement>
<candidates>
<candidate number="1">
<restated>Complication rates did not differ between patients over 75 and younger patients.</restated>
<original_wording>Complication rates did not differ between patients aged over 75 years and younger patients</original_wording>
</candidate>
</candidates>
<output>{"reasoning": "Candidate 1 is the complication result the safety statement reports. Both describe no harmful effect of age, so both are rated 0, but the press release turns 'did not differ' into an assurance that the procedure is safe.", "match": 1, "statement_strength": 0, "statement_direction": "no_effect", "claim_strength": 0, "claim_direction": "no_effect", "null_reframed": true}</output>
</example>
</examples>"""

ADJUDICATE_USER = """<press_statement>
<restated>{text}</restated>
<original_wording>{span}</original_wording>
</press_statement>
<candidates>
{candidates}
</candidates>
Choose the candidate that reports the same finding as the press statement, or 0 if none does, then rate both original wordings on the same scale."""

CANDIDATE = """<candidate number="{n}">
<restated>{text}</restated>
<original_wording>{span}</original_wording>
</candidate>"""


# ---------------- v4: whole-document extraction from numbered sentences (s04) ----------------
EXTRACT_DOC_SYSTEM = """You extract research findings from medical press releases and journal abstracts. Your output populates a knowledge graph that compares how strongly a press release states a finding with how strongly the source paper states it. For that comparison to work, the wording-based fields must describe the text as written, not what the study design would justify.

<input_format>
<text> holds the whole document, one numbered sentence per line:
- Press release: "[P2.S3] sentence" (paragraph 2, sentence 3).
- Abstract: "[S7 | RESULTS] sentence", where the section is BACKGROUND, METHODS, RESULTS, CONCLUSIONS, OTHER or UNLABELED.
- "[ctx] sentence" lines are context from the preceding part of a long release; they carry no ID and cannot be cited.
</input_format>

<definitions>
A finding is a result of the study being reported that relates an exposure to an outcome.
- exposure: the intervention, drug, behavior, or risk factor (e.g., "coffee consumption", "metformin", "night-shift work").
- outcome: the disease, symptom, measurement, or health state it affects (e.g., "type 2 diabetes", "systolic blood pressure").
- direction: the effect of the exposure on the outcome: increase, decrease, no_effect (the study reports no association or no difference), or unclear.
- claim_strength: how strongly the wording asserts the relationship (labels from Wright & Augenstein, 2021):
  0 = no relationship (a null result, e.g., "was not associated with")
  1 = correlational ("associated with", "linked to", "higher risk among", "less likely than")
  2 = conditional causal ("may reduce", "could lead to", "might protect against", "can be")
  3 = direct causal ("reduces", "causes", "prevents", "protects against", "leads to")
</definitions>

<rules>
1. Read the whole document before extracting, so a sentence whose subject is named earlier ("It", "These mice", "During that time") is understood.
2. Extract results of the study being reported, including results stated in researcher quotes. Skip background on prior work, study aims, methods, funding, and contact details, because they are not this study's findings. In abstracts, cite only RESULTS, CONCLUSIONS or UNLABELED sentences; BACKGROUND and METHODS sentences are context.
3. A finding needs a stated relationship between a real exposure and an outcome. Skip these, because they are not exposure-outcome findings: a test's or tool's accuracy (sensitivity, prediction), comparisons between calendar periods, lists of rates that never state a relationship, checks that a model or method works, and implications the study did not measure ("may cause confusion").
4. Create one finding per exposure-outcome pair per statement. Split compound statements ("coffee and tea lowered the risk of stroke and dementia") into one finding per pair. If the same pair is restated with a different claim strength, as when a conclusion or a quote goes further than the results, extract each statement; if it is restated with the same strength, extract it once.
5. Rate claim_strength from the wording alone, not the effect size or p-value. A press release saying an observational study "prevents" a disease is rated 3; that gap between wording and evidence is what the system measures.
6. Write exposure and outcome as short noun phrases using the standard medical name: expand abbreviations ("T2D" becomes "type 2 diabetes") and keep doses, percentages, time windows, and populations out of the entity. Standard names let entity linking match them to the Disease Ontology.
7. Set evidence_ids to the ID(s) of the sentence(s) that state the finding, copied exactly as shown (e.g., "P2.S3" or "S7"). Add the preceding sentence's ID when the finding's subject is named only there. Code rebuilds the evidence text from these IDs and rejects IDs that are not shown or not citable.
8. Write text as one standalone sentence that keeps the original hedge or causal verb, so the restatement carries the same claim strength as the evidence.
9. Set population to who was studied when the document states it (e.g., "adults over 65", "mice"); otherwise null.
10. Set confidence_score to your confidence that the finding is extracted correctly: about 0.9 for an explicit, unambiguous result; 0.5 to 0.7 when the exposure, outcome, or direction had to be inferred; below 0.5 when unsure the sentence reports a finding at all.
11. When the document reports no finding, return an empty findings list.
</rules>

<examples>
<example>
<source_type>abstract</source_type>
<text>
[S1 | BACKGROUND] Coffee has been linked to lower diabetes risk in earlier cohorts.
[S2 | METHODS] We followed 41,000 adults for 12 years.
[S3 | RESULTS] Participants who drank three or more cups of coffee daily had a 20% lower risk of type 2 diabetes (HR 0.80, 95% CI 0.72-0.89).
[S4 | RESULTS] Coffee intake was not associated with stroke.
[S5 | CONCLUSIONS] Coffee consumption may protect against type 2 diabetes.
</text>
<output>{"findings": [{"text": "Drinking three or more cups of coffee daily is associated with a lower risk of type 2 diabetes in adults.", "exposure": "coffee consumption", "outcome": "type 2 diabetes", "population": "adults", "direction": "decrease", "claim_strength": 1, "evidence_ids": ["S3"], "confidence_score": 0.9}, {"text": "Coffee consumption was not associated with stroke in adults.", "exposure": "coffee consumption", "outcome": "stroke", "population": "adults", "direction": "no_effect", "claim_strength": 0, "evidence_ids": ["S4"], "confidence_score": 0.9}, {"text": "Coffee consumption may protect against type 2 diabetes.", "exposure": "coffee consumption", "outcome": "type 2 diabetes", "population": "adults", "direction": "decrease", "claim_strength": 2, "evidence_ids": ["S5"], "confidence_score": 0.9}]}</output>
</example>
<example>
<source_type>press</source_type>
<text>
[P1.S1] Eating blueberries may slow cognitive decline in older adults, according to new research.
[P1.S2] The study followed 900 people for six years.
[P2.S1] "Our results show that blueberries slow cognitive decline," said lead author Dr. Jane Smith.
[P3.S1] The study was funded by the National Institutes of Health.
</text>
<output>{"findings": [{"text": "Eating blueberries may slow cognitive decline in older adults.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 2, "evidence_ids": ["P1.S1"], "confidence_score": 0.9}, {"text": "Blueberries slow cognitive decline in older adults, according to the lead author.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 3, "evidence_ids": ["P2.S1"], "confidence_score": 0.85}]}</output>
</example>
<example>
<source_type>press</source_type>
<text>
[P1.S1] Researchers gave mice a high-fat diet for 12 weeks.
[P1.S2] During that time, the mice's hearts showed markers of stress.
[P2.S1] Their new blood test detected heart disease with 85% sensitivity.
</text>
<output>{"findings": [{"text": "A high-fat diet was followed by markers of heart stress in mice.", "exposure": "high-fat diet", "outcome": "cardiac stress markers", "population": "mice", "direction": "increase", "claim_strength": 1, "evidence_ids": ["P1.S1", "P1.S2"], "confidence_score": 0.7}]}</output>
</example>
</examples>"""

EXTRACT_DOC_USER = """<source_type>{source_type}</source_type>
<text>
{text}
</text>
Extract the findings reported in <text>."""

REPAIR_DOC_USER = """<source_type>{source_type}</source_type>
<text>
{text}
</text>
<failed_findings>
{failed}
</failed_findings>
<validation_errors>
{errors}
</validation_errors>
The findings in <failed_findings> were extracted from <text> and failed automated validation for the reasons in <validation_errors>. Return corrected versions of only these findings:
- "not in the numbered text": copy the sentence IDs exactly as they appear in <text>.
- "cannot be evidence": cite the RESULTS, CONCLUSIONS, UNLABELED or numbered press sentence that states the finding, or drop the finding if only a context, background or methods sentence states it.
- "must be non-empty" or "must be different entities": identify both the exposure and the outcome from <text>, or drop the finding if <text> names only one.
- A value outside its allowed range: use a value inside the range.
Drop any finding that <text> does not support, and return an empty findings list if none can be corrected."""


# ---------------- s10: community summaries (Edge et al. 2024, community reports) ----------------
COMMUNITY_SYSTEM = """You write a short report on one community of a medical knowledge graph. A community is a group of findings from journal abstracts that share medical entities (exposures or outcomes), found by the Leiden algorithm. The report is shown to users and is searched to answer broad questions ("what do studies say about X?"), so it must state only what the listed claims say.

<input_format>
<papers>: one line per paper, "[P1] title | journal | date | doi", with "RETRACTED" when the paper is retracted.
<claims>: one line per finding, "[C1] P1 | exposure -> outcome | direction | strength | press: ... | wording".
- strength is how strongly the paper's wording asserts the relationship: 0 no relationship, 1 correlational ("associated with"), 2 conditional causal ("may reduce"), 3 direct causal ("reduces").
- press lists how press releases about that paper stated the same finding, as computed by an earlier step: overstated, aligned or understated, with the press and paper strengths. "none" means no press statement was matched to it.
- wording is the paper's own sentence.
<stats>: counts computed by code; use them as given.
</input_format>

<rules>
1. Use only the listed claims and stats. Add no outside knowledge, numbers, or conclusions the claims do not state.
2. Describe each relationship with the paper's strength: strength 0 as "no association found", 1 as "associated with", 2 as "may", 3 as a direct causal verb. Never state a relationship more strongly than its strength.
3. title: at most 12 words naming the shared exposures and outcomes.
4. summary: 3 to 5 sentences covering what the findings study, where papers agree, where they conflict (opposite directions, or a null result beside a positive one), and how press coverage compares, using only the press labels given. Name retracted papers when you use their claims.
5. key_findings: 1 to 5 findings, most widely reported first. Each is one sentence and cites in claim_refs every claim ID that states it. A conflict between papers can be a key finding.
6. Cite claims only by the IDs shown (C1, C2, ...)."""

COMMUNITY_USER = """<papers>
{papers}
</papers>
<claims>
{claims}
</claims>
<stats>
{stats}
</stats>
Write the community report."""

COMMUNITY_REPAIR = """{user}

Your previous report failed validation: {error}
Write the report again, fixing that problem."""


# ---------------- GraphRAG engine: router (FR-5 to FR-7) ----------------
# Versioned through its own cache tag (graphrag.py: router|ROUTER_VERSION|effort), not PROMPT_VERSION, so editing this
# prompt does not invalidate the cached s04-s10 responses. Bump ROUTER_VERSION whenever ROUTER_SYSTEM changes.
ROUTER_VERSION = "r1"

ROUTER_SYSTEM = """You route user questions for a question-answering system over a medical knowledge graph. The graph holds press releases, the statements they make, the journal papers they announce, findings (claims) extracted from those papers' abstracts, and a label for each matched pair saying whether the press release states the finding more strongly than the paper (overstated), equally (aligned), or less strongly (understated). Your output picks the type of question and the predefined database query that answers it, and extracts the query's parameters. You do not answer the question.

<path_types>
local: about one press statement, one paper finding, or one press release.
  - statement_check: whether a press-release statement is supported by its source paper ("the release says X; does the paper back it up?").
  - claim_check: what a paper found, and how press releases reported that finding.
  - release_check: every statement in one press release checked against its paper. Use when a release ID or DOI is given and the question asks about the release as a whole.
cross_paper: compares a finding with other papers that study the same exposure or outcome.
  - cross_paper: "do other studies agree?", "how many other papers report this?"
aggregate: counts, rates, rankings, or trends across many records.
  - agg_rate_by_year: overstatement rates per release year, optionally within a year range.
  - agg_institution_rank: institutions ranked by how often their releases overstate.
  - agg_by_entity: how often releases about a named exposure or outcome are overstated, aligned, or understated.
global: overall themes across the collection, with no single statement or paper in view.
  - topic_overview: "what do studies say about X overall?", "what are the main topics?"
unsupported: not about these press releases, papers, or their claims, including requests for personal medical advice. template_id is null.
</path_types>

<params>
- text: for statement_check, claim_check and cross_paper, the statement or finding restated as one plain sentence without question words ("coffee lowers the risk of liver cancer"). Otherwise null.
- entities: the medical exposures, outcomes, drugs or conditions the question names, as short noun phrases. Required for agg_by_entity; include them for every other type when present.
- euid, doi: copy exactly as written; never invent one.
- year_from, year_to, min_releases: only when the question states them.
</params>

<examples>
Q: The press release says coffee lowers the risk of liver cancer. Does the study back that up?
A: {"reasoning": "Asks whether one press statement is supported by its paper.", "path_type": "local", "template_id": "statement_check", "params": {"text": "coffee lowers the risk of liver cancer", "entities": ["coffee", "liver cancer"]}}
Q: What did the paper 10.1000/example.2020.001 actually find, and how was it reported?
A: {"reasoning": "Asks what one paper found and how the press reported it.", "path_type": "local", "template_id": "claim_check", "params": {"doi": "10.1000/example.2020.001"}}
Q: Check every claim in press release 912345.
A: {"reasoning": "Asks about one release as a whole.", "path_type": "local", "template_id": "release_check", "params": {"euid": "912345"}}
Q: Do other studies agree that air pollution increases dementia risk?
A: {"reasoning": "Compares a finding across other papers.", "path_type": "cross_paper", "template_id": "cross_paper", "params": {"text": "air pollution increases dementia risk", "entities": ["air pollution", "dementia"]}}
Q: Has the rate of overstated press releases changed between 2019 and 2024?
A: {"reasoning": "Asks for a rate trend by year.", "path_type": "aggregate", "template_id": "agg_rate_by_year", "params": {"year_from": 2019, "year_to": 2024}}
Q: Among institutions with at least 5 releases, which overstate most often?
A: {"reasoning": "Asks for a ranking of institutions.", "path_type": "aggregate", "template_id": "agg_institution_rank", "params": {"min_releases": 5}}
Q: How often are press releases about vitamin D overstated?
A: {"reasoning": "Asks for label counts for one topic.", "path_type": "aggregate", "template_id": "agg_by_entity", "params": {"entities": ["vitamin D"]}}
Q: What do the studies say overall about the gut microbiome and mental health?
A: {"reasoning": "Asks for an overview of a theme across papers.", "path_type": "global", "template_id": "topic_overview", "params": {"entities": ["gut microbiome", "mental health"]}}
Q: Should I stop taking my statins?
A: {"reasoning": "Asks for personal medical advice.", "path_type": "unsupported", "template_id": null, "params": {}}
</examples>"""

ROUTER_USER = """<question>{question}</question>
Route this question."""

ROUTER_REPAIR = """{user}

Your previous routing failed validation: {error}
Route the question again, fixing that problem."""


# ---------------- GraphRAG engine: generator (FR-5, FR-8) ----------------
# Versioned through its own cache tag (graphrag.py: generator|GENERATOR_VERSION|effort). Bump when GENERATOR_SYSTEM changes.
GENERATOR_VERSION = "g2"  # g2: label is about wording only; no derived numbers; strengths in words

GENERATOR_SYSTEM = """You answer questions about medical press releases and the journal papers they announce, using only rows retrieved from a knowledge graph. Each answer sentence is checked by code: every ID it cites must appear in the rows, so cite only IDs you can see.

<fields>
- press_text: a statement's wording in the press release. paper_text: a finding's wording in the paper's abstract.
- press_strength, paper_strength: how strongly the wording asserts the relationship: 0 no relationship, 1 correlational ("associated with"), 2 conditional causal ("may reduce"), 3 direct causal ("reduces").
- label: the press statement compared with its matched paper finding: overstated (press wording stronger), aligned, or understated. explanation: the reason recorded when the label was assigned.
- resolution_status / unresolved_reason: whether the statement was matched to a paper finding (no_finding, below_threshold, ambiguous).
- direction_conflict: press and paper report opposite directions. null_reframed: the paper found no effect and the press presents a benefit or safety assurance.
- paper_retracted: the paper is retracted.
- seed_cos: how close a row is to the question (cosine similarity, -1 to 1). Rows with much lower values than the best row are usually off-topic.
- cross_paper rows: other papers that share an exposure or outcome with the starting finding. match_type says what is shared. same_direction is set only when both exposure and outcome are shared; otherwise the papers study different relationships and must not be described as agreeing or disagreeing.
- aggregate rows: scored = statements matched to a paper finding (the denominator for rates); overstated_rate = overstated / scored; releases_with_overstatement counts releases with at least one overstated statement; overstated_euids lists example releases.
- topic_overview rows: a community of related paper findings with title, summary, key_findings (citing claim IDs), and key_claims mapping each claim_id to its paper's doi.
</fields>

<rules>
1. Use only the rows. Leave out rows that do not bear on the question. Add no outside knowledge.
2. Cite papers as doi:<doi value> and press releases as release:<euid value>, copied exactly from the rows. Every finding sentence cites at least one ID. For topic_overview, cite the doi that key_claims gives for the claim IDs a finding uses.
3. Describe relationships with the strength recorded: never state a finding more strongly than its strength. Write strengths in words for a general reader ("reported an association", "says it may", "states it causes"), not as numbers.
4. A label compares wording only. "overstated" means the press release words the finding more strongly than the paper; it does not mean the paper lacks the finding. For "does the paper support this?" questions, answer both parts: whether the paper reports the finding, and whether the press release states it as strongly as the paper does.
5. Use numbers only as they appear in the rows. Do not convert, combine, or derive figures (for example, hazard ratios to percentages). If a figure the question mentions is not in the rows, say it cannot be checked from the rows; do not call it unsupported or wrong.
6. Start with a one-sentence direct answer, then give the evidence in order of relevance. Do not contradict the direct answer later.
7. For counts and rates, give the numerator and denominator (for example, 30 of 181 scored statements). Do not call a difference significant or a trend real; the data were not tested for that. Cite example releases from overstated_euids.
8. When the rows do not contain what the question asks, set insufficient_context to true and add a limitation sentence saying what is missing.
9. Mention a retracted paper whenever you use it. Give no medical advice.
10. Keep the answer under 150 words unless the question asks for a list."""

GENERATOR_USER = """<question>{question}</question>
<question_type>{path_type} / {template_id}</question_type>
<rows>
{rows}
</rows>
Answer the question from these rows."""

GENERATOR_REPAIR = """{user}

Your previous answer had citation problems: {error}
Answer again, citing only IDs that appear in the rows."""
