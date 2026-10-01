# Local verification

- Python: 3.12.14, macOS arm64.
- Dependency versions: requirements-lock.txt.
- `python -m pytest -q`: 26 passed in 2.97 seconds.
- `python -m support_ai.evaluate`: completed using the new support-message datasets.
- Synthetic holdout: 24 messages, 100% raw accuracy, 58.3% routing coverage.
- Review challenge: 7/10 correctly deferred. A vacation request and two mixed-intent messages were incorrectly assigned specific queues; see evaluation.json.
- Streamlit AppTest: billing classification, five-row sample batch, and manual review passed.
- Browser verification: billing classification, explanation chart, and editable template rendered successfully on the local server.
- GitHub Actions is configured to test pushes and pull requests. See the repository Actions tab for the latest remote result. No hosted application deployment is provided.

The new domain was evaluated using unchanged classifier hyperparameters and review thresholds. The demo remains a synthetic-data prototype, with human review required for every draft.
