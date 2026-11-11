# CIP-B107-AI-for-Forensics-and-Capstone-Intelligence-Analysis
A digital-forensics lab evaluating IPv4 candidate extraction, validation, and offline machine learning, with SHA-256 integrity records, leakage analysis, and human analyst review.

## IP Evidence Identification and Capstone Intelligence Analysis

This educational digital-forensics project evaluates IPv4 candidate identification in textual artefacts. It combines regex extraction, strict IPv4 validation, offline machine learning, historical fine-tuning evidence analysis, and human analyst review.

**Scope:** Identify and prioritise candidate IPv4 artefacts, not determine whether an address is malicious. A positive label (`1`) denotes the dataset's IPv4-candidate class, not a threat designation.

**Investigator:** Favour Madu

## Methods

### Phase A — Evidence Preservation

- Preserve supplied files in `originals/`.
- Record SHA-256 hashes in evidence manifests.

### Phase B — Extraction and Labelling

- Extract dotted-quad candidates using regex.
- Validate IPv4 syntax.
- Review malformed sequences, placeholders, version-like strings, URL components, and longer identifiers.

### Phase C — Offline Classification

- Use TF-IDF character n-grams (2–5; `char_wb`; sublinear TF) with Logistic Regression.
- Model settings:
  - `C=10`
  - Balanced class weights
  - `liblinear` solver
- Fit the vectoriser on training data only.
- Use the supplied fixed split:
  - 183 training records
  - 46 holdout records
- Random seed: `42`

### Baseline Comparisons

Compare the classifier against:

- Explicit structure rules
- IPv4 validity rules
- Placeholder detection rules
- Context-based rules
- Always-positive baseline

Check exact and near-identical train-test prompts at cosine similarity ≥ 0.90.

### Phase D — Historical Evidence

- Examine historical fine-tuning logs in read-only mode.
- No fine-tuning was re-run.
- No cloud-model calls were made during this analysis.

### Phases E–F — Interpretation

- Review predictions and ambiguous candidates.
- Interpret evaluation limitations.
- Define an analyst-assistance deployment recommendation.

## Reported Results

The following full-holdout metrics are reproduced from the report and have not been independently re-run for this README.

| Method | Accuracy | Precision | Recall | F1 |
|----------|----------|----------|----------|----------|
| TF-IDF + Logistic Regression | 97.8% | 97.1% | 100.0% | 98.6% |
| Rule-based baseline | 78.3% | 100.0% | 70.6% | 82.8% |
| Always-positive baseline | 73.9% | 73.9% | 100.0% | 85.0% |

**Offline confusion matrix:** `[[11,1],[0,34]]`

- True Negatives (TN): 11
- False Positives (FP): 1
- False Negatives (FN): 0
- True Positives (TP): 34

The report records 45 correct predictions out of 46 holdout records.

**Historical metrics (separate evaluation):**

- AUROC: `0.997549`
- AUPRC: `0.999147`

These values were recorded in historical fine-tuning logs. They are not offline-classifier results, and the producing job remains unconfirmed.

## Limitations

### High Train-Test Overlap

- 33 of 46 holdout prompts were exact training duplicates.
- 34 of 46 were exact or near-identical.

Reported performance may reflect memorisation rather than real-world generalisation.

### Small Independent Subset

- Only 12 records remained in the clean subset.
- The classifier misclassified its only negative example.

This prevents a confident claim of operational superiority.

### Label Ambiguity

Some positive-labelled candidates occur within:

- Version-like strings
- URL paths
- Longer dotted identifiers

Agreement with a source label does not necessarily establish investigative relevance.

### Historical Provenance Gaps

Historical logs lack sufficient:

- Model artefacts
- Configuration details
- Provenance records

Consequently, independent reproduction and direct comparison with the offline evaluation are limited.

### No Maliciousness Assessment

- No threat-intelligence enrichment was performed.
- Recovered addresses require independent technical or investigative corroboration.

## Analyst-Assistance Recommendation

**Recommended workflow:**

```text
Regex Extraction
        ↓
IPv4 Validation
        ↓
Offline ML Scoring
        ↓
Human Analyst Review
```

**Recommended status: Analyst Assistance Tool**

Use the workflow for:

- Candidate extraction
- Prioritisation
- Triage support
- Consistency checking

All use should occur under human oversight.

The workflow is **not suitable** for:

- Autonomous threat attribution
- Automatic maliciousness determination
- Automatic evidence exclusion
- Use as the sole basis for investigative decisions

Analysts should:

- Retain valid standalone candidates.
- Annotate special-purpose addresses.
- Escalate ambiguous contexts or disagreements.

Any removal from the candidate set should follow documented analyst review and must not delete or alter the original evidence.

## Next Steps

- Create an independent test set without prompt overlap.
- Review inconsistent labels and expand negative examples.
- Assess analyst agreement on ambiguous cases.
- Document confidence levels and override rationales.
- Corroborate potentially relevant candidates with supporting forensic evidence.

## Evidence and Reproducibility Records

The report references the following artefacts:

### Scripts

- `phase_b_extract.py`
- `demo.py`
- `phase_c_classifier.py`
- `phase_d_analysis.py`

### Outputs and Records

- `candidate_review.csv`
- `test_predictions.csv`
- `metrics.json`
- `classification_reports.txt`
- `environment.txt`
- Phase-specific SHA-256 manifests

These names describe documented project artefacts. This README does not confirm which files are currently present in a GitHub repository.

## Responsible Use

**Responsible use:** Educational and authorised forensic analysis only.

- Preserve supplied evidence.
- Document observations accurately.
- Treat extracted candidates as investigative leads requiring corroboration.
- Do not treat extracted candidates as proof of malicious activity.
