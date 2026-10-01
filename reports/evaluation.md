### Synthetic-data evaluation

These results measure this small, authored dataset only. They are not production accuracy, time savings, or customer satisfaction.

| Split | Examples | Raw accuracy | Macro F1 | Routing coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|
| validation | 16 | 100.0% | 1.000 | 87.5% | 100.0% |
| test | 24 | 100.0% | 1.000 | 58.3% | 100.0% |

Raw accuracy includes every top prediction, before abstention. Routing coverage is the fraction assigned an intent rather than manual review. Accepted accuracy is measured only on that subset.

Review challenge recall: **70.0%** across 10 unsupported or ambiguous cases.

Review challenge cases incorrectly given a specific route:

- Can you help me plan a vacation? → `billing`
- I need a refund and cannot reset my password. → `account_access`
- Please add dark mode and fix the crashing upload. → `feature_request`

The thresholds were fixed before evaluating the holdout. No test or validation text is fitted by TF-IDF. See evaluation.json for per-class metrics, the confusion matrix, and individual errors.

A real pilot needs consented, anonymized inquiries, independent labels, a larger untouched test set, language checks, and human review. No business improvement has been measured.
