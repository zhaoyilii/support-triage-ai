"""Train only on train.csv; uncertain predictions always get a review queue."""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'billing': 'Billing & subscriptions',
    'account_access': 'Account access',
    'bug_report': 'Bug report',
    'feature_request': 'Feature request',
}
MAX_TEXT_LENGTH = 4000
# Fixed demo operating point; not calibrated or optimized on the test set.
MIN_SCORE = 0.45
MIN_MARGIN = 0.15
MIN_COVERAGE = 0.35
TEMPLATES = {
    'billing': ('Your billing inquiry',
        'Thank you for contacting us about billing. Please share the invoice reference '
        'and a brief description of the charge or subscription change you want reviewed. '
        'Our team will check the details before confirming any next steps. '
        'Please do not include full payment card or bank account details.'),
    'account_access': ('Help with account access',
        'Thank you for reporting an account access issue. Please describe the sign-in '
        'step that fails and the exact error text, if any. Use the official account '
        'recovery flow when available. Never share passwords, recovery codes, or '
        'one-time authentication codes in a support message.'),
    'bug_report': ('Details for your bug report',
        'Thank you for reporting this issue. Could you share the steps to reproduce it, '
        'what you expected, what happened instead, and your app or browser version? '
        'If you include a screenshot, please remove private information first. '
        'These details will help our team investigate.'),
    'feature_request': ('Your product suggestion',
        'Thank you for sharing your suggestion. Could you describe the task you are '
        'trying to accomplish and how the proposed feature would help? '
        'Our team can record that context for product review. '
        'This acknowledgment does not confirm a release date or roadmap commitment.'),
    'manual_review': ('Thanks for contacting support',
        'Thank you for reaching out. Could you briefly describe the problem or '
        'request you would like help with? A team member will review your message.'),
}


def read_dataset(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def train_model(path=ROOT / 'data/train.csv'):
    rows = read_dataset(path)
    if not rows or any(not r['text'].strip() or r['label'] not in LABELS for r in rows):
        raise ValueError('Training rows need nonempty text and supported labels.')
    if {r['label'] for r in rows} != set(LABELS):
        raise ValueError('Training data must contain all four intent labels.')
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
        ('classifier', LogisticRegression(C=4.0, max_iter=1000, random_state=42)),
    ])
    model.fit([r['text'] for r in rows], [r['label'] for r in rows])
    return model


@dataclass
class Prediction:
    suggested_label: str
    route: str
    model_score: float
    margin: float
    vocabulary_coverage: float
    reason: str
    evidence: list
    scores: dict
    subject: str
    draft: str

    def to_dict(self):
        return asdict(self)


def classify(model, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Please enter a nonempty English inquiry.')
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f'Inquiry must be at most {MAX_TEXT_LENGTH} characters.')
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    score = float(probabilities[top])
    margin = score - float(probabilities[order[1]])
    label = str(classifier.classes_[top])
    # Coverage of distinct non-stopword unigrams, not bigram counts.
    tokens = {t for t in vectorizer.build_analyzer()(text) if ' ' not in t}
    coverage = sum(t in vectorizer.vocabulary_ for t in tokens) / max(len(tokens), 1)
    letters = [c for c in text if c.isalpha()]
    ascii_ratio = sum(c.isascii() for c in letters) / max(len(letters), 1)
    reasons = []
    if len(tokens) < 2:
        reasons.append('Too little informative text')
    if ascii_ratio < 0.85:
        reasons.append('Possible unsupported language')
    if coverage < MIN_COVERAGE:
        reasons.append('Too many unfamiliar words')
    if score < MIN_SCORE:
        reasons.append('Low model score')
    if margin < MIN_MARGIN:
        reasons.append('Competing intent predictions')
    route = 'manual_review' if reasons else label
    names = vectorizer.get_feature_names_out()
    contributions = features.toarray()[0] * classifier.coef_[top]
    evidence = [
        {'term': str(names[i]), 'contribution': round(float(contributions[i]), 4)}
        for i in np.argsort(contributions)[::-1][:5] if contributions[i] > 0
    ]
    subject, draft = TEMPLATES[route]
    return Prediction(label, route, score, margin, coverage,
        '; '.join(reasons) if reasons else 'Passes demo routing thresholds; human approval still required',
        evidence, {str(k): float(v) for k, v in zip(classifier.classes_, probabilities)},
        subject, draft)
