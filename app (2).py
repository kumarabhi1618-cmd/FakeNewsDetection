"""
Fake News Detection — Streamlit App (BERT, loaded from Hugging Face Hub)

Model repo: abhishekkumar7/FakeNewsDetectorBERT

Run locally:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    streamlit run app.py
"""

import re
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_REPO = "abhishekkumar7/FakeNewsDetectorBERT"
MAX_LENGTH = 512

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1a1a2e; }
    .subtitle    { color:#555; font-size:1rem; margin-bottom:1.5rem; }
    .fake-label  { background:#fde8e8; color:#c0392b; border-radius:8px;
                   padding:0.6rem 1.2rem; font-size:1.4rem; font-weight:700; }
    .real-label  { background:#e8f8f2; color:#1a7a50; border-radius:8px;
                   padding:0.6rem 1.2rem; font-size:1.4rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING (must match training preprocessing for BERT)
# ══════════════════════════════════════════════════════════════════════════════
DATELINE_RE   = re.compile(r'^[A-Z][A-Z\s,\.]+\([^)]+\)\s*[-\u2013\u2014]\s*', re.MULTILINE)
URL_RE        = re.compile(r'https?://\S+|www\.\S+')
HTML_RE       = re.compile(r'<[^>]+')
SPECIAL_RE    = re.compile(r'[^a-z\s]')
WHITESPACE_RE = re.compile(r'\s+')


@st.cache_resource(show_spinner=False)
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text: str, nlp) -> str:
    """Same preprocessing used for BERT training — stopwords kept."""
    text = str(text)
    text = DATELINE_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.lower()
    text = SPECIAL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (from Hugging Face Hub, cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_REPO)
    model     = BertForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device, nlp):
    import torch
    clean = preprocess(text, nlp)
    enc = tokenizer(
        clean, return_tensors="pt",
        padding="max_length", truncation=True, max_length=MAX_LENGTH
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    label = "Real" if pred == 1 else "Fake"
    return label, float(probs[1]), float(probs[0])  # label, P(Real), P(Fake)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🔍 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste a news article below and the fine-tuned BERT model '
    'will classify it as <b>Real</b> or <b>Fake</b>.</p>',
    unsafe_allow_html=True,
)

col_input, col_result = st.columns([3, 2], gap="large")

with col_input:
    article_text = st.text_area(
        "📰 Paste article text here",
        height=320,
        placeholder="Enter the full body of the news article...",
    )
    run_btn = st.button("🚀 Classify Article", type="primary", use_container_width=True)

with col_result:
    result_placeholder = st.empty()
    result_placeholder.info("Prediction will appear here after you click **Classify Article**.")

# ── Run inference ─────────────────────────────────────────────────────────────
if run_btn:
    if not article_text.strip():
        st.warning("⚠️ Please paste some article text before classifying.")
        st.stop()

    with st.spinner("Loading resources (first run downloads the model, ~440 MB)..."):
        nlp = load_spacy()
        try:
            tokenizer, model, device = load_model()
        except Exception as e:
            st.error(f"❌ Failed to load model `{MODEL_REPO}`:\n\n{e}")
            st.stop()

    with st.spinner("Running inference..."):
        label, p_real, p_fake = predict(article_text, tokenizer, model, device, nlp)

    with col_result:
        result_placeholder.empty()

        if label == "Fake":
            st.markdown(
                '<div class="fake-label">🚨 FAKE NEWS — High probability this article is fabricated.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="real-label">✅ REAL NEWS — This article appears legitimate.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 1.6))
        colors = ["#c0392b", "#1a7a50"]
        bars = ax.barh(["Fake", "Real"], [p_fake * 100, p_real * 100],
                        color=colors, height=0.5)
        for bar, val in zip(bars, [p_fake * 100, p_real * 100]):
            ax.text(min(val + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=11, fontweight="bold",
                    color="white" if val > 15 else "#333")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        confidence = max(p_real, p_fake) * 100
        if confidence >= 90:
            conf_label = "🟢 High confidence"
        elif confidence >= 70:
            conf_label = "🟡 Moderate confidence"
        else:
            conf_label = "🔴 Low confidence — treat with caution"

        st.caption(f"{conf_label} ({confidence:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE ARTICLES
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🧪 Sample articles for testing", expanded=False):
    st.markdown("**Real-style sample:**")
    st.code(
        "Scientists have confirmed a new vaccine candidate shows over 90 percent efficacy "
        "in phase 3 trials, according to peer-reviewed results published in the Lancet. "
        "The trial involved 30,000 participants across 12 countries and was funded by a "
        "coalition of public health organizations. Regulatory submission is expected within weeks."
    )
    st.markdown("**Fake-style sample:**")
    st.code(
        "SHOCKING: Government ADMITS secret chemtrail program poisoning water supply!! "
        "Insiders reveal FEMA camps are ready. Share this before they DELETE it. "
        "Big Pharma and globalist elites have been hiding the truth for decades. "
        "Your doctor WON'T tell you this one simple trick that CURES everything!!"
    )

st.markdown("---")
st.caption(f"Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) — BERT fine-tuned for fake news detection")
