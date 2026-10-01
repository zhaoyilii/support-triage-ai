"""Bounded CSV import/export. Exports are drafts, never messages sent."""
import csv
import io
from .model import classify

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 200


def parse_csv(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError('CSV must be no larger than 2 MB.')
    try:
        source = payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Please upload a UTF-8 CSV file.') from exc
    try:
        reader = csv.DictReader(io.StringIO(source), strict=True)
        if reader.fieldnames is None or 'text' not in reader.fieldnames:
            raise ValueError('CSV must include a column named text.')
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('CSV column names must be unique.')
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('CSV rows must match the header column count.')
            rows.append(row['text'])
            if len(rows) > MAX_ROWS:
                raise ValueError(f'Upload at most {MAX_ROWS} inquiries per batch.')
        if not rows:
            raise ValueError('CSV contains no inquiries.')
        return rows
    except csv.Error as exc:
        raise ValueError('Unable to parse CSV. Check quoting and field lengths.') from exc


def process_batch(model, texts):
    results = []
    for index, text in enumerate(texts, start=1):
        try:
            p = classify(model, text)
            results.append({'row': index, 'text': text, 'route': p.route,
                'suggested_label': p.suggested_label, 'model_score': round(p.model_score, 4),
                'reason': p.reason, 'subject': p.subject, 'draft': p.draft,
                'status': 'needs_human_approval'})
        except ValueError as exc:
            results.append({'row': index, 'text': text, 'route': 'invalid_input',
                'suggested_label': '', 'model_score': '', 'reason': str(exc),
                'subject': '', 'draft': '', 'status': 'invalid_input'})
    return results


def safe_cell(value):
    # Quoting alone does not prevent spreadsheet formula execution.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def export_csv(rows):
    if not rows:
        raise ValueError('No results to export.')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: safe_cell(v) for k, v in row.items()})
    return stream.getvalue().encode('utf-8-sig')
