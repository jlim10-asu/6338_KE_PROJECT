PROMPT_NOTE = "v2: XML-structured, rules carry their reasons, 4 diverse examples, task restated after the data"

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
6. Copy evidence_span character for character from <text>: one contiguous passage with the same spelling, casing, and punctuation, and no ellipses. Code checks it as an exact substring and rejects the finding otherwise.
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
<output>{"findings": [{"text": "Drinking three or more cups of coffee daily is associated with a lower risk of type 2 diabetes in adults.", "exposure": "coffee consumption", "outcome": "type 2 diabetes", "population": "adults", "direction": "decrease", "claim_strength": 1, "evidence_span": "Participants who drank three or more cups of coffee daily had a 20% lower risk of type 2 diabetes", "confidence_score": 0.9}, {"text": "Coffee consumption was not associated with stroke in adults.", "exposure": "coffee consumption", "outcome": "stroke", "population": "adults", "direction": "no_effect", "claim_strength": 0, "evidence_span": "Coffee intake was not associated with stroke.", "confidence_score": 0.9}]}</output>
</example>
<example>
<source_type>press</source_type>
<text>Eating blueberries may slow cognitive decline in older adults, according to new research. "Our results show that blueberries slow cognitive decline," said lead author Dr. Jane Smith.</text>
<look_ahead>The team now plans a larger trial.</look_ahead>
<output>{"findings": [{"text": "Eating blueberries may slow cognitive decline in older adults.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 2, "evidence_span": "Eating blueberries may slow cognitive decline in older adults", "confidence_score": 0.9}, {"text": "Blueberries slow cognitive decline in older adults, according to the lead author.", "exposure": "blueberry consumption", "outcome": "cognitive decline", "population": "older adults", "direction": "decrease", "claim_strength": 3, "evidence_span": "Our results show that blueberries slow cognitive decline", "confidence_score": 0.85}]}</output>
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
<output>{"findings": [{"text": "Metformin reduced fasting blood glucose compared with placebo.", "exposure": "metformin", "outcome": "fasting blood glucose", "population": null, "direction": "decrease", "claim_strength": 3, "evidence_span": "metformin reduced fasting glucose and HbA1c compared with placebo", "confidence_score": 0.9}, {"text": "Metformin reduced glycated hemoglobin compared with placebo.", "exposure": "metformin", "outcome": "glycated hemoglobin", "population": null, "direction": "decrease", "claim_strength": 3, "evidence_span": "metformin reduced fasting glucose and HbA1c compared with placebo", "confidence_score": 0.9}]}</output>
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
