# Support Triage AI — Project brief

## Problem

Small software teams receive a mixture of billing questions, account-access problems, bug reports, and feature suggestions. Manually sorting the inbox and asking for missing context consumes attention before the actual issue can be handled.

## Solution

A local ML classifier suggests a queue and an editable first response. It exposes influential text features, flags uncertain inputs, and supports small CSV batches. Every message remains subject to human review.

## Design choices

TF-IDF and logistic regression provide a fast, interpretable baseline without API costs or GPU requirements. Streamlit keeps the interface small enough for a beginner to understand. Authored templates make the first response inspectable and avoid inventing product policies.

## Demonstrated engineering

- Supervised text classification with separated training and evaluation data.
- Explanations, score thresholds, and manual-review fallback.
- Bounded CSV imports, per-row errors, and spreadsheet-safe exports.
- Model and UI tests, reproducible reports, and CI configuration.

## What remains unproven

All examples are synthetic. The tool has no real customer usage, measured time savings, or deployed ticket-system integration. Scores on small authored datasets do not establish production quality.

## Next iteration

Collect consented, anonymized tickets; define labels with reviewers; keep an independent holdout; compare handling time and routing errors with manual triage. Consider an other/support label and multi-intent handling before introducing larger models.

## Short portfolio description

A Python customer-support triage assistant using TF-IDF and logistic regression, with interpretable predictions, editable reply templates, CSV workflows, and reproducible synthetic-data evaluation.

Describe personal development contributions and AI assistance accurately. Do not imply historical usage, organizational adoption, or business results that have not occurred.
