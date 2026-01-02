import streamlit as st
import joblib
import re
import torch
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# -------------------------------------------------------------------
# Session state for main textarea
# -------------------------------------------------------------------
if "email_input" not in st.session_state:
    st.session_state["email_input"] = ""

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DE/EN Spam Classifier",
    page_icon="📧",
    layout="wide",
)

# -------------------------------------------------------------------
# Light preprocessing (same as training)
# -------------------------------------------------------------------
def replace_patterns(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " <URL> ", text)   # URLs
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " <EMAIL> ", text)  # emails
    text = re.sub(r"\b\d+\b", " <NUM> ", text)            # numbers
    return text


def light_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = replace_patterns(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------------------
# Load vectorizers, models, and translation model
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # TF-IDF vectorizers
        tfidf_de = joblib.load("tfidf_de.pkl")
        tfidf_en = joblib.load("tfidf_en.pkl")

        # Map: model name -> (german_model, english_model)
        models = {}

        # Logistic Regression (required)
        models["Logistic Regression"] = (
            joblib.load("model_lr_de.pkl"),
            joblib.load("model_lr_en.pkl"),
        )

        # Linear SVC (optional)
        try:
            models["Linear SVC"] = (
                joblib.load("model_svm_de.pkl"),
                joblib.load("model_svm_en.pkl"),
            )
        except FileNotFoundError:
            pass

        # Multinomial NB (optional)
        try:
            models["Multinomial NB"] = (
                joblib.load("model_nb_de.pkl"),
                joblib.load("model_nb_en.pkl"),
            )
        except FileNotFoundError:
            pass

        if not models:
            st.error("No models loaded. Ensure at least Logistic Regression models exist.")
            st.stop()

        # Translation model (German -> English)
        MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        mt_model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

        return tfidf_de, tfidf_en, models, tokenizer, mt_model, device

    except FileNotFoundError as e:
        st.error(f"❌ Model or vectorizer file not found: {e.filename}")
        st.error("Please ensure at least these files are in the app directory:")
        st.code(
            "- tfidf_de.pkl\n"
            "- tfidf_en.pkl\n"
            "- model_lr_de.pkl\n"
            "- model_lr_en.pkl"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


tfidf_de, tfidf_en, MODELS, mt_tokenizer, mt_model, device = load_artifacts()


# -------------------------------------------------------------------
# Helpers: language detection, translation, model prediction
# -------------------------------------------------------------------
def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_de_to_en(text: str) -> str:
    """Translate a single German text to English using MarianMT."""
    if not text.strip():
        return text

    encoded = mt_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = mt_model.generate(**encoded, max_length=256)

    translation = mt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation


def predict_with_model(model, X):
    """
    Run prediction and safely compute spam probability P(label=1).
    Returns (pred_label, spam_prob).
    """
    pred = int(model.predict(X)[0])

    spam_prob = None
    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(X)[0]  # e.g. [0.9, 0.1] or [0.1, 0.9]
        classes = list(model.classes_)         # e.g. [0, 1] or [1, 0]
        if 1 in classes:
            spam_index = classes.index(1)
            spam_prob = float(proba_vec[spam_index])

    return pred, spam_prob


def run_both_pipelines(email_text: str, model_pair):
    """
    Core logic:
    - Detect language
    - If German: run German model on original AND English model on translated text
    - If English: run English model only
    - Otherwise: fallback to English model only

    model_pair: (german_model, english_model)
    """
    model_de, model_en = model_pair
    detected = detect_lang(email_text)

    result = {
        "detected_lang": detected,
        "german": None,          # (pred, prob)
        "english": None,         # (pred, prob)
        "translated_text": None, # translation if used
    }

    if detected == "de":
        # 1) German model on original text
        cleaned_de = light_clean(email_text)
        X_de = tfidf_de.transform([cleaned_de])
        pred_de, prob_de = predict_with_model(model_de, X_de)
        result["german"] = (pred_de, prob_de)

        # 2) English model on translated text
        translated = translate_de_to_en(email_text)
        result["translated_text"] = translated

        cleaned_en = light_clean(translated)
        X_en = tfidf_en.transform([cleaned_en])
        pred_en, prob_en = predict_with_model(model_en, X_en)
        result["english"] = (pred_en, prob_en)

    elif detected == "en":
        # English input -> only English model
        cleaned_en = light_clean(email_text)
        X_en = tfidf_en.transform([cleaned_en])
        pred_en, prob_en = predict_with_model(model_en, X_en)
        result["english"] = (pred_en, prob_en)

    else:
        # Unknown / other language -> fallback to English model
        cleaned_en = light_clean(email_text)
        X_en = tfidf_en.transform([cleaned_en])
        pred_en, prob_en = predict_with_model(model_en, X_en)
        result["english"] = (pred_en, prob_en)

    return result


# -------------------------------------------------------------------
# Global CSS styling (only visuals)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fb 0%, #fdfdfd 60%);
    }

    .main-card {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 1.75rem 2rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-top: 1rem;
    }

    .app-header {
        padding: 0rem 0rem 0.75rem 0rem !important;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #4b5563;
        font-size: 0.95rem;
    }

    .result-card {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .ham {
        background-color: #e6f6e6;
        border: 1px solid #89c98a;
        color: #145214;
    }
    .spam {
        background-color: #ffe6e6;
        border: 1px solid #ff8a8a;
        color: #7f1111;
    }
    .prob {
        font-weight: 500;
        margin-top: 0.35rem;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f3f4f6;
        border-right: 1px solid #e5e7eb;
    }

    /* Make selectbox look like a pure dropdown (not searchable) */
    [data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    [data-baseweb="select"] input {
        cursor: pointer !important;
        caret-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📧 Multilingual Spam Classifier</div>
        <div class="app-subtitle">
            Demo using <b>TF-IDF</b> plus different classifiers (LogReg, SVC, NB) on a German spam dataset
            and its <b>English-translated</b> version.<br>
            For German emails, both <b>German</b> and <b>English (translated)</b> models are applied.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown(
        """
        - Language is auto-detected using `langdetect`.
        - If the email is **German (de)**:
          - 🇩🇪 German model runs on the original text.
          - 🇬🇧 English model runs on the English translation.
        - If the email is **English (en)**:
          - Only the English model is applied.
        """
    )

    st.markdown("---")
    st.subheader("🧠 Choose classifier")

    model_choice = st.selectbox(
        "Base model:",
        list(MODELS.keys()),
        index=0,  # default: first (typically Logistic Regression)
    )

    st.caption(f"Currently using: **{model_choice}** for both German and English pipelines.")

    st.markdown("---")
    st.subheader("📊 Test performance (from Colab, LogReg)")
    st.markdown("- German (LogReg) – F1 ≈ `0.955`")
    st.markdown("- English (LogReg) – F1 ≈ `0.964`")
    st.markdown("---")
    st.caption("Use the examples to see where models agree or differ.")

# -------------------------------------------------------------------
# Main layout
# -------------------------------------------------------------------
left_col, right_col = st.columns([2.3, 1.2], gap="large")

# ----- RIGHT COLUMN: EXAMPLE EMAILS -----
with right_col:
    with st.container():
        st.subheader("🧪 Quick examples")

        examples = {
            "Ham – Freundliche Erinnerung": """Hey,

ich wollte nur kurz fragen, ob wir uns morgen wie geplant fürs Lernen treffen.
Sag mir einfach, welche Uhrzeit dir passt.

Liebe Grüße  
Anna""",

            "Spam – Gewinnspiel / Preis": """Herzlichen Glückwunsch!

Sie wurden zufällig ausgewählt, einen 1.000€ Gutschein zu gewinnen. Klicken Sie jetzt auf den folgenden Link, um Ihren Preis zu sichern: http://super-gewinn.cc

Dieses Angebot ist nur heute gültig!""",

            "Spam – Konto gesperrt (Phishing)": """Sehr geehrter Kunde,

Ihr Konto wurde aus Sicherheitsgründen vorübergehend gesperrt. Um es wieder zu aktivieren, melden Sie sich bitte über den folgenden Link an und bestätigen Sie Ihre Daten:
www.bank-login-security.com

Wenn Sie dies nicht innerhalb von 24 Stunden tun, wird Ihr Konto dauerhaft geschlossen."""
        }

        selected_example = st.selectbox(
            "Choose an example email:",
            list(examples.keys())
        )

        if st.button("Fill with example"):
            st.session_state["email_input"] = examples[selected_example]

        st.caption("Click an example to quickly fill the textbox on the left.")

# ----- LEFT COLUMN: INPUT + RESULT -----
with left_col:
    with st.container():
        st.subheader("✉️ Paste email text")

        example_placeholder = (
            "Fügen Sie hier eine deutsche E-Mail ein …\n"
            "oder paste an English email here …"
        )

        email_text = st.text_area(
            "Email content",
            height=250,
            placeholder=example_placeholder,
            label_visibility="collapsed",
            key="email_input",
        )

        classify_button = st.button("🔍 Classify", disabled=False)

        if classify_button:
            if not email_text.strip():
                st.warning("Please paste some email text first.")
            else:
                model_pair = MODELS[model_choice]

                with st.spinner("🔄 Processing..."):
                    status = st.empty()
                    status.info("🔍 Detecting language & running selected model(s)...")
                    result = run_both_pipelines(email_text, model_pair)
                    status.empty()

                detected = result["detected_lang"]
                st.markdown(f"Detected language: `{detected}`")

                # Layout for results
                if detected == "de" and result["german"] is not None:
                    col_de, col_en = st.columns(2)

                    # ----- German model card -----
                    with col_de:
                        pred_de, prob_de = result["german"]
                        css_class = "spam" if pred_de == 1 else "ham"
                        headline = (
                            f"🇩🇪 German model ({model_choice}): 🚨 SPAM"
                            if pred_de == 1
                            else f"🇩🇪 German model ({model_choice}): ✅ HAM"
                        )

                        if prob_de is not None:
                            spam_p = prob_de
                            ham_p = 1 - prob_de
                            prob_text = (
                                f"<div class='prob'>"
                                f"Spam: <b>{spam_p:.2%}</b> | "
                                f"Ham: <b>{ham_p:.2%}</b>"
                                f"</div>"
                            )
                        else:
                            prob_text = ""

                        st.markdown(
                            f"""
                            <div class='result-card {css_class}'>
                                {headline}
                                {prob_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ----- English model card -----
                    with col_en:
                        pred_en, prob_en = result["english"]
                        css_class = "spam" if pred_en == 1 else "ham"
                        headline = (
                            f"🇬🇧 English model (translated, {model_choice}): 🚨 SPAM"
                            if pred_en == 1
                            else f"🇬🇧 English model (translated, {model_choice}): ✅ HAM"
                        )

                        if prob_en is not None:
                            spam_p = prob_en
                            ham_p = 1 - prob_en
                            prob_text = (
                                f"<div class='prob'>"
                                f"Spam: <b>{spam_p:.2%}</b> | "
                                f"Ham: <b>{ham_p:.2%}</b>"
                                f"</div>"
                            )
                        else:
                            prob_text = ""

                        st.markdown(
                            f"""
                            <div class='result-card {css_class}'>
                                {headline}
                                {prob_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if result["translated_text"]:
                            with st.expander("🔁 See English translation used for classification"):
                                st.write(result["translated_text"])

                else:
                    # Non-German input -> only English model
                    pred_en, prob_en = result["english"]
                    css_class = "spam" if pred_en == 1 else "ham"
                    headline = (
                        f"🇬🇧 English model ({model_choice}): 🚨 SPAM"
                        if pred_en == 1
                        else f"🇬🇧 English model ({model_choice}): ✅ HAM"
                    )

                    if prob_en is not None:
                        spam_p = prob_en
                        ham_p = 1 - prob_en
                        prob_text = (
                            f"<div class='prob'>"
                            f"Spam: <b>{spam_p:.2%}</b> | "
                            f"Ham: <b>{ham_p:.2%}</b>"
                            f"</div>"
                        )
                    else:
                        prob_text = ""

                    st.markdown(
                        f"""
                        <div class='result-card {css_class}'>
                            {headline}
                            {prob_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
