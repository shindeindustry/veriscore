import os
import re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='VeriScore', layout='centered')

st.markdown(
    """
    <style>
    .main > div.block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    .title-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 24px;
        color: white;
        padding: 2rem 2.5rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }
    .title-box h1 {
        margin: 0;
        font-size: 3rem;
        letter-spacing: -0.04em;
    }
    .title-box p {
        margin: 0.5rem 0 0;
        color: #cbd5e1;
        font-size: 1.05rem;
    }
    .section-box {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.5rem;
    }
    .reasons-box {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
    }
    .section-label {
        margin: 0 0 0.75rem;
        font-weight: 600;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='title-box'><h1>VeriScore</h1><p>Know what's real.</p></div>",
    unsafe_allow_html=True,
)

review_text = st.text_area('Review text', height=240, placeholder='Paste the review here...')

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    api_key = st.text_input('OpenAI API Key', type='password')

if not api_key:
    st.warning('Enter your OpenAI API key or set the OPENAI_API_KEY environment variable.')
    st.stop()

client = OpenAI(api_key=api_key)

PROMPT_TEMPLATE = '''You are an expert AI system trained to detect fake or AI-generated product reviews.

Analyze the review and provide:

1. Fake Probability (0–100%)
2. Verdict: (Likely Fake / Possibly Fake / Likely Genuine)
3. Key Reasons (bullet points)
4. Writing Patterns Observed

Review:
"""
{review}
"""'''


def analyze_review(review: str) -> str:
    prompt = PROMPT_TEMPLATE.format(review=review.strip())
    response = client.responses.create(
        model='gpt-4.1-mini',
        input=prompt,
        temperature=0.2,
        max_tokens=450,
    )
    return getattr(response, 'output_text', None) or ''.join(
        part.get('text', '')
        for item in response.output
        for part in item.get('content', [])
        if part.get('type') == 'output_text'
    )


def parse_analysis(text: str) -> dict:
    result = {
        'fake_probability': '',
        'verdict': '',
        'key_reasons': [],
        'writing_patterns': '',
        'raw': text,
    }

    def find(pattern: str) -> str:
        match = re.search(pattern, text, re.I | re.S)
        return match.group(1).strip() if match else ''

    result['fake_probability'] = find(r'fake probability\s*[:\-]?\s*([^\n]+)')
    result['verdict'] = find(r'verdict\s*[:\-]?\s*([^\n]+)')

    reasons_block = find(r'key reasons\s*[:\-]?\s*(.+?)(?:writing patterns|$)')
    if reasons_block:
        bullets = re.findall(r'[-*•]\s*(.+)', reasons_block)
        if bullets:
            result['key_reasons'] = [item.strip() for item in bullets if item.strip()]
        else:
            result['key_reasons'] = [line.strip() for line in reasons_block.splitlines() if line.strip()]

    result['writing_patterns'] = find(r'writing patterns(?: observed)?\s*[:\-]?\s*(.+)')
    return result


def parse_progress(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', value)
    if not match:
        return None
    progress = int(float(match.group(1)))
    return max(0, min(progress, 100))


def verdict_style(verdict: str) -> tuple[str, str]:
    verdict_lower = verdict.lower()
    if 'likely genuine' in verdict_lower:
        return 'success', 'Likely Genuine'
    if 'possibly fake' in verdict_lower:
        return 'warning', 'Possibly Fake'
    if 'likely fake' in verdict_lower or 'fake' in verdict_lower:
        return 'error', 'Likely Fake'
    return 'info', verdict or 'Unknown'


if st.button('Analyze'):
    if not review_text.strip():
        st.warning('Please enter a review before analyzing.')
    else:
        with st.spinner('Analyzing review...'):
            try:
                analysis = analyze_review(review_text)
                parsed = parse_analysis(analysis)
                progress_value = parse_progress(parsed['fake_probability'])

                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.markdown('#### Analysis Result')
                st.divider()

                top_cols = st.columns([2, 1])
                with top_cols[0]:
                    if parsed['fake_probability']:
                        st.metric('Fake Probability', parsed['fake_probability'])
                    if parsed['verdict']:
                        verdict_level, verdict_label = verdict_style(parsed['verdict'])
                        if verdict_level == 'success':
                            st.success(verdict_label)
                        elif verdict_level == 'warning':
                            st.warning(verdict_label)
                        elif verdict_level == 'error':
                            st.error(verdict_label)
                        else:
                            st.info(verdict_label)

                with top_cols[1]:
                    if progress_value is not None:
                        st.markdown('**Confidence Gauge**')
                        st.progress(progress_value)
                        st.caption('Higher score means more likely to be fake.')

                st.divider()

                if parsed['key_reasons']:
                    st.markdown('**Key Reasons**')
                    st.markdown('<div class="reasons-box">', unsafe_allow_html=True)
                    for reason in parsed['key_reasons']:
                        st.markdown(f'- {reason}')
                    st.markdown('</div>', unsafe_allow_html=True)

                if parsed['writing_patterns']:
                    st.markdown('**Writing Patterns Observed**')
                    st.write(parsed['writing_patterns'])

                if not (parsed['key_reasons'] or parsed['writing_patterns']):
                    st.markdown('**Full Response**')
                    st.code(parsed['raw'], language='text')

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as exc:
                st.error(f'Error calling OpenAI: {exc}')
