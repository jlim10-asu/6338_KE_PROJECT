R1 on copenlu train (100 pairs), macro-F1 x 100. This project uses few-shot prompting with no task-specific training. Reference column: models trained on the same dataset by Wright & Augenstein (2021), shown as supervised/PET/MT-PET; pair-level = their T1, sentence-level = their T2, trained on 200 or 4500 sentences.

| Method | Measure | This project | Reference: Wright & Augenstein (2021) |
|---|---|---|---|
| Adjudicated, forced (rates every pair) | Label | 71.1 | Pair-level models: 29.05/39.12/47.35 |
|  | Press strength | 70.0 | Sentence-level models: 200 ex. 54.78/63.56/64.72 · 4500 ex. 63.26/64.20/63.93 |
|  | Abstract strength | 71.8 | Sentence-level models: 200 ex. 59.41/62.76/63.27 · 4500 ex. 67.26/64.92/64.69 |
| Adjudicated (may decline a pair) | Label | 51.7 (82.5 on 50 rated) | – (reference models always label) |
| Independent (each sentence alone) | Label | 55.4 | Sentence-level models: 200 ex. 49.03/56.57/57.44 · 4500 ex. 58.66/60.45/61.11 |
|  | Press strength | 70.9 | Sentence-level models: 200 ex. 54.78/63.56/64.72 · 4500 ex. 63.26/64.20/63.93 |
|  | Abstract strength | 59.9 | Sentence-level models: 200 ex. 59.41/62.76/63.27 · 4500 ex. 67.26/64.92/64.69 |
| Majority baseline (always aligned) | Label | 25.3 | – |

Forced adjudication still declined 0 pairs (counted as wrong).

Note: the reference results are on the test split; this run uses the training split, so it is context, not a like-for-like comparison.
