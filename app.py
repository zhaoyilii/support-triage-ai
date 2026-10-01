"""Run from the repository root: streamlit run app.py"""
import hashlib
import pandas as pd
import streamlit as st
from support_ai.model import ROOT, LABELS, classify, train_model
from support_ai.batch import parse_csv, process_batch, export_csv

st.set_page_config(page_title='Support Triage AI | Customer inbox assistant', page_icon='🌱', layout='wide')


@st.cache_resource
def get_model(training_digest):
    return train_model()


model = get_model(hashlib.sha256((ROOT / 'data/train.csv').read_bytes()).hexdigest())
st.caption('SUPPORT TRIAGE AI  /  LOCAL MACHINE LEARNING')
st.title('Turn inquiries into a clear next step.')
st.write('Classify customer support inquiries, inspect the model’s reasoning, and prepare a follow-up draft.')
st.info('Independent demo using synthetic data. English input only. Every draft requires human approval; nothing is sent.')
with st.sidebar:
    st.header('How it works')
    st.write('1. Enter an inquiry\n2. Inspect its predicted intent\n3. Review and edit the template\n4. Export for human follow-up')
    st.divider()
    st.caption('TF-IDF + logistic regression • 80 training examples • 4 intent categories')
    st.caption('No API key. Local inference. This app does not save inquiry files to disk. Use fictional examples in public demos.')
    st.caption('Model scores are uncalibrated; they are not verified probabilities of correctness or customer value.')

single, batch, about = st.tabs(['Single inquiry', 'Batch workspace', 'Model & limits'])
with single:
    examples = {
        'Billing': 'I was charged twice for my monthly subscription.',
        'Account access': 'I forgot my password and cannot log in.',
        'Bug report': 'The app crashes every time I upload a file.',
        'Feature request': 'Could you add a dark mode to the application?',
        'Unclear inquiry': 'I need help.',
    }
    choice = st.selectbox('Try an example', list(examples))
    with st.form('inquiry'):
        text = st.text_area('Inquiry in English', value=examples[choice], height=140, max_chars=4000)
        submitted = st.form_submit_button('Analyze inquiry', type='primary')
    if submitted:
        try:
            st.session_state['single_result'] = (text, classify(model, text))
            st.session_state.pop('draft_editor', None)
        except ValueError as exc:
            st.session_state.pop('single_result', None)
            st.error(str(exc))
    if 'single_result' in st.session_state:
        analyzed_text, prediction = st.session_state['single_result']
        st.caption(f'Analyzed inquiry: {analyzed_text}')
        left, right = st.columns([1, 1])
        with left:
            st.subheader('Routing recommendation')
            if prediction.route == 'manual_review':
                st.warning('Manual review needed')
            else:
                st.success(LABELS[prediction.route])
            a, b = st.columns(2)
            a.metric('Top model score', f'{prediction.model_score:.1%}')
            b.metric('Vocabulary coverage', f'{prediction.vocabulary_coverage:.1%}')
            st.caption(prediction.reason)
            scores = pd.DataFrame({'Intent': [LABELS[k] for k in prediction.scores], 'Score': prediction.scores.values()})
            st.bar_chart(scores.set_index('Intent'), horizontal=True)
            if prediction.evidence:
                st.caption('Terms with the largest positive contribution to the top class logit (not causal explanations).')
                st.dataframe(prediction.evidence, hide_index=True, use_container_width=True)
        with right:
            st.subheader('Suggested follow-up')
            st.caption('Template selected by the model’s route. This is not LLM-generated content.')
            draft = st.text_area('Edit before using', value=f'Subject: {prediction.subject}\n\n{prediction.draft}', height=280, key='draft_editor')
            st.download_button('Download edited draft', draft, 'follow_up_draft.txt', 'text/plain')
with batch:
    st.subheader('Review a small support inbox')
    st.write('Upload a UTF-8 CSV with a `text` column. Maximum 200 rows / 2 MB. Additional columns are ignored.')
    st.download_button('Download sample CSV', (ROOT / 'data/sample_inquiries.csv').read_bytes(), 'sample_inquiries.csv', 'text/csv')
    uploaded = st.file_uploader('Inquiry CSV', type=['csv'])
    use_sample = st.button('Analyze sample inbox')
    run_upload = st.button('Analyze uploaded CSV', disabled=uploaded is None)
    if use_sample or run_upload:
        try:
            payload = (ROOT / 'data/sample_inquiries.csv').read_bytes() if use_sample else uploaded.getvalue()
            st.session_state['batch_result'] = process_batch(model, parse_csv(payload))
        except ValueError as exc:
            st.session_state.pop('batch_result', None)
            st.error(str(exc))
    if 'batch_result' in st.session_state:
        rows = st.session_state['batch_result']
        a, b, c = st.columns(3)
        a.metric('Inquiries', len(rows))
        b.metric('Manual review', sum(r['route'] == 'manual_review' for r in rows))
        c.metric('Invalid rows', sum(r['route'] == 'invalid_input' for r in rows))
        st.caption('Most recent analyzed batch. Click Analyze again after changing the upload. All valid rows still require human approval.')
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.download_button('Export review queue', export_csv(rows), 'support_review_queue.csv', 'text/csv')
with about:
    st.subheader('A small, inspectable ML baseline')
    st.write('TF-IDF turns words and two-word phrases into features. Logistic regression learns weights from labeled inquiries. A score, margin, and vocabulary check route uncertain text to manual review.')
    st.write('Training: 80 authored examples. Validation: 16. Held-out test: 24. Review challenge set: 10. All are synthetic, from one author, and do not demonstrate real-world generalization.')
    st.write('The app retrains from the small training CSV at startup. Evaluation never fits on validation or test data. Run `python -m support_ai.evaluate` to regenerate reports.')
    st.warning('Known limits: unfamiliar wording, mixed intents, negation, non-English text, and unfamiliar domains can still be misclassified. Review rules are heuristics, not a safety guarantee. This tool does not authenticate users, issue refunds, resolve incidents, or send messages.')
    report = ROOT / 'reports/evaluation.md'
    if report.exists():
        st.markdown(report.read_text())
