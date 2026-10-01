"""Reproducible evaluation with no test-set training or threshold optimization."""
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .model import ROOT, LABELS, MIN_SCORE, MIN_MARGIN, MIN_COVERAGE, classify, read_dataset, train_model


def evaluate_split(model, path):
    rows = read_dataset(path)
    truth = [r['label'] for r in rows]
    predictions = [classify(model, r['text']) for r in rows]
    raw = [p.suggested_label for p in predictions]
    accepted = [i for i, p in enumerate(predictions) if p.route != 'manual_review']
    return {
        'examples': len(rows),
        'raw_accuracy': accuracy_score(truth, raw),
        'raw_macro_f1': f1_score(truth, raw, average='macro', zero_division=0),
        'routing_coverage': len(accepted) / len(rows),
        'accepted_accuracy': sum(predictions[i].route == truth[i] for i in accepted) / len(accepted) if accepted else None,
        'labels': list(LABELS),
        'confusion_matrix': confusion_matrix(truth, raw, labels=list(LABELS)).tolist(),
        'classification_report': classification_report(truth, raw, zero_division=0, output_dict=True),
        'review_or_error_cases': [
            {'text': r['text'], 'expected': r['label'], 'prediction': p.to_dict()}
            for r, p in zip(rows, predictions) if p.route != r['label']
        ],
    }


def main():
    model = train_model()
    result = {'data_type': 'Hand-authored synthetic examples; not real-world performance',
        'training_examples': len(read_dataset(ROOT / 'data/train.csv')),
        'thresholds': {'min_score': MIN_SCORE, 'min_margin': MIN_MARGIN, 'min_coverage': MIN_COVERAGE}}
    for split in ('validation', 'test'):
        result[split] = evaluate_split(model, ROOT / f'data/{split}.csv')
    review = read_dataset(ROOT / 'data/review_cases.csv')
    misses = [{'text': r['text'], 'prediction': p.to_dict()} for r in review
        if (p := classify(model, r['text'])).route != 'manual_review']
    result['review_challenge'] = {'examples': len(review),
        'manual_review_recall': (len(review) - len(misses)) / len(review), 'missed_cases': misses}
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['### Synthetic-data evaluation', '',
        'These results measure this small, authored dataset only. They are not production accuracy, time savings, or customer satisfaction.', '',
        '| Split | Examples | Raw accuracy | Macro F1 | Routing coverage | Accepted accuracy |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('validation', 'test'):
        s = result[split]
        accepted = f"{s['accepted_accuracy']:.1%}" if s['accepted_accuracy'] is not None else 'N/A'
        lines.append(f"| {split} | {s['examples']} | {s['raw_accuracy']:.1%} | {s['raw_macro_f1']:.3f} | {s['routing_coverage']:.1%} | {accepted} |")
    lines += ['', 'Raw accuracy includes every top prediction, before abstention. Routing coverage is the fraction assigned an intent rather than manual review. Accepted accuracy is measured only on that subset.', '',
        f"Review challenge recall: **{result['review_challenge']['manual_review_recall']:.1%}** across {len(review)} unsupported or ambiguous cases.", '',
        'Review challenge cases incorrectly given a specific route:', '']
    lines += [f"- {m['text']} → `{m['prediction']['route']}`" for m in misses] or ['- None in this small challenge set.']
    lines += ['', 'The thresholds were fixed before evaluating the holdout. No test or validation text is fitted by TF-IDF. See evaluation.json for per-class metrics, the confusion matrix, and individual errors.', '',
        'A real pilot needs consented, anonymized inquiries, independent labels, a larger untouched test set, language checks, and human review. No business improvement has been measured.']
    (output / 'evaluation.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
