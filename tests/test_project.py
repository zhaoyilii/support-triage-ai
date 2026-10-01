import csv
import io
import pytest
from support_ai.model import ROOT, LABELS, classify, read_dataset, train_model
from support_ai.batch import parse_csv, process_batch, export_csv


@pytest.fixture(scope='module')
def model():
    return train_model()


@pytest.mark.parametrize(('text','label'), [('I was charged twice for my monthly subscription.', 'billing'), ('I forgot my password and cannot log in.', 'account_access'), ('The app crashes every time I upload a file.', 'bug_report'), ('Could you add a dark mode to the application?', 'feature_request')])
def test_core_demo_routes(model, text, label):
    result = classify(model, text)
    assert result.route == label
    assert result.evidence
    assert abs(sum(result.scores.values()) - 1) < 1e-8


@pytest.mark.parametrize('text', ['I need help.', '火星旅行计划', 'zxqvbnm plkjhg', 'Hi'])
def test_unfamiliar_inquiries_abstain(model, text):
    result = classify(model, text)
    assert result.route == 'manual_review'
    assert 'Could you briefly describe' in result.draft


@pytest.mark.parametrize('text', ['', '  ', None, 'a' * 4001])
def test_invalid_single_input(model, text):
    with pytest.raises(ValueError):
        classify(model, text)


def test_data_split_integrity_and_no_test_vocabulary_leak(model):
    splits = [read_dataset(ROOT / f'data/{s}.csv') for s in ['train','validation','test']]
    texts = [r['text'].strip().lower() for rows in splits for r in rows]
    assert len(texts) == len(set(texts))
    for rows in splits:
        assert set(r['label'] for r in rows) == set(LABELS)
    analyzer = model.named_steps['tfidf'].build_analyzer()
    train_features = {t for r in splits[0] for t in analyzer(r['text'])}
    assert set(model.named_steps['tfidf'].vocabulary_) <= train_features


def test_csv_round_trip_and_formula_escaping(model):
    rows = process_batch(model, ['=HYPERLINK("bad")', 'Please send my latest invoice.', '', 'x' * 4001])
    output = list(csv.DictReader(io.StringIO(export_csv(rows).decode('utf-8-sig'))))
    assert output[0]['text'].startswith("'=")
    assert output[1]['status'] == 'needs_human_approval'
    assert output[2]['route'] == output[3]['route'] == 'invalid_input'


@pytest.mark.parametrize('payload', [b'', b'wrong\nvalue', b'text\n', b'text,text\na,b',
    b'text\na,b', b'text,extra\na', b'\xff', b'text\n"unclosed', b'text\n' + b'a\n' * 201,
    b'text\n' + b'x' * (2*1024*1024)])
def test_malformed_or_oversize_csv(payload):
    with pytest.raises(ValueError):
        parse_csv(payload)


def test_utf8_bom_and_quoted_multiline_csv():
    assert parse_csv(b'\xef\xbb\xbftext,ignored\n"hello,\nworld",foo\n') == ['hello,\nworld']


def test_streamlit_workflows():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.success[0].value == LABELS['billing']
    sample_button = next(b for b in app.button if b.label == 'Analyze sample inbox')
    sample_button.click().run()
    assert not app.exception
    assert len(app.session_state['batch_result']) == 5
    app.selectbox[0].select('Unclear inquiry').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['single_result'][1].route == 'manual_review'
