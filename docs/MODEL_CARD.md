# Model card

## Intended use

An educational English support-message classifier for small software teams. It recommends one of four administrative queues: billing, account access, bug report, or feature request. All responses need human review. It does not resolve tickets, authenticate people, issue refunds, or send messages.

## Data provenance

All 130 examples were authored as synthetic fixtures during AI-assisted development: 80 train, 16 validation, 24 test, and 10 review challenges. No customer records, private correspondence, or third-party datasets were used. Train, validation, and test are balanced across the four classes. The challenge set includes unrelated, short, non-English, and multi-intent messages.

All splits share an author and vocabulary. Exact duplicates across train/validation/test are checked, but semantic and stylistic similarity remains. Results are insufficient evidence of real-world generalization. The separate demo inbox deliberately reuses training examples; it is not a test split. Dataset contents use the repository MIT license.

## Algorithm and operating point

`TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words='english')` and `LogisticRegression(C=4, max_iter=1000, random_state=42)`. Only train.csv is fitted. The app trains locally from CSV at startup; no binary model download is required.

Manual review is suggested if fewer than 2 distinct informative unigrams are present; ASCII share of letters is below 85%; training-vocabulary coverage is below 35%; top score is below 0.45; or the top-two score margin is below 0.15. These heuristics were fixed before evaluating the datasets. There is no test-set optimization or calibration. ASCII checks are not true language detection.

## Evaluation

Run `python -m support_ai.evaluate`. Reports include accuracy, macro F1, per-class metrics, confusion matrix, coverage, accepted-only accuracy, and challenge-set misses. Validation is reported separately. Confidence intervals, subgroup analysis, calibration, and real customer evaluation have not been established.

## Limitations

Negation, mixed intents, short text, new vocabulary, and unsupported languages can be misclassified with high scores. English stopword removal can lose negation cues. Categories exclude sales, general product questions, security incidents, legal requests, and many other types of messages. Thresholds cannot reliably detect all out-of-scope input. The model is single-label, so it cannot fully represent messages containing several requests.

## Data handling

Inference is local. The app does not explicitly persist input or send it to a model vendor. Streamlit telemetry is disabled. Input remains in browser/server session memory; downloads contain original messages. On a hosted instance, the operator can access input. This demo has no authentication, audit trail, retention management, or ticket-system integration. Use fictional messages in public demonstrations.

## Responses and explanations

The model learns classification; replies are fixed templates selected by route. They request context but do not promise refunds, resolution dates, or feature releases. Feature contributions describe the positive TF-IDF × coefficient terms for the top class, not causal explanations. Review the original message before using any output.
