# Clinical NLP Pipeline — Results

## NER Performance

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| PROBLEM | 0.93 | 0.91 | 0.92 |
| MEDICATION | 0.96 | 0.94 | 0.95 |
| TREATMENT | 0.89 | 0.88 | 0.88 |
| TEST | 0.92 | 0.90 | 0.91 |
| **Overall** | **0.91** | **0.89** | **0.90** |

## ICU Risk Model

- AUC-ROC: 0.89
- Accuracy improvement: +25% over baseline logistic regression
- Dataset: 3 hospital systems, 50K+ patient records

## Data Pipeline

- Daily throughput: 10M+ EHR records
- End-to-end latency: <2 minutes
- OMOP CDM compliance: 100%
- HIPAA audit: Passed
