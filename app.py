import os
import streamlit as st
from predict import classify_gold_ornament, CONFIDENCE_THRESHOLD
from compare_models import classify_with_openai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Gold Ornament Classifier",
    page_icon="✨",
    layout="wide",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #c9a227;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6b6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #faf8f0 0%, #f5f0e1 100%);
        border: 1px solid #e8e0c8;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(201, 162, 39, 0.1);
    }
    .result-label {
        font-weight: 600;
        color: #8b7355;
        font-size: 0.9rem;
    }
    .result-value {
        font-size: 1.25rem;
        color: #2d2d2d;
    }
    .confidence-badge {
        display: inline-block;
        background: #c9a227;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .confidence-badge.below {
        background: #888;
        color: white;
    }
    .threshold-note {
        background: #fff8e7;
        border-left: 4px solid #c9a227;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .compare-header {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #6d5516;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
    }
    .compare-sub {
        text-align: center;
        color: #7a7a7a;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #c9a227 0%, #a88622 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #a88622 0%, #8b7319 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">✨ Gold Ornament Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload an image to get Ornament Type, Estimated Weight & Wastage</p>',
    unsafe_allow_html=True,
)

gemini_api_key = os.environ.get("GEMINI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
confidence_threshold = CONFIDENCE_THRESHOLD
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def render_result_cards(result: dict, title: str, threshold: float) -> None:
    all_vals = result.get("all_values", {})
    st.markdown(f"### {title}")
    for key in ("ornament_type", "weight_grams", "wastage_percent"):
        item = all_vals.get(key, {})
        val = item.get("value", "—")
        conf = item.get("confidence", 0)
        label = key.replace("_", " ").title()
        if key == "weight_grams" and isinstance(val, (int, float)):
            disp = f"{val} g"
        elif key == "wastage_percent" and isinstance(val, (int, float)):
            disp = f"{val}%"
        else:
            disp = str(val)
        meets = conf >= threshold
        badge_class = "" if meets else " below"
        badge_text = f"{int(conf*100)}%" if meets else f"{int(conf*100)}% (below threshold)"
        st.markdown(
            f'<div class="result-card">'
            f'<span class="result-label">{label}</span><br>'
            f'<span class="result-value">{disp}</span>'
            f'<span class="confidence-badge{badge_class}">{badge_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

uploaded = st.file_uploader(
    "Upload gold ornament image",
    type=["jpg", "jpeg", "png", "gif", "webp"],
    help="Upload a clear photo of the gold ornament.",
)

if uploaded:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded, use_container_width=True, caption="Uploaded image")
    with col2:
        if st.button("🔍 Classify image"):
            if not gemini_api_key:
                st.error("Please set your Gemini API key in a `.env` file (GEMINI_API_KEY).")
            elif not openai_api_key:
                st.error("Please set your OpenAI API key in a `.env` file (OPENAI_API_KEY).")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        path = f"uploads/{uploaded.name}"
                        os.makedirs("uploads", exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(uploaded.getvalue())

                        gemini_result = classify_gold_ornament(
                            path,
                            api_key=gemini_api_key,
                            confidence_threshold=confidence_threshold,
                        )
                        openai_result = classify_with_openai(
                            path,
                            model=DEFAULT_OPENAI_MODEL,
                            confidence_threshold=confidence_threshold,
                        )

                        os.remove(path)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.stop()

                st.markdown(
                    f'<p class="threshold-note">Predictions below {int(confidence_threshold*100)}% are marked as below threshold.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown('<p class="compare-header">Side-by-Side Comparison</p>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="compare-sub"></p>',
                    unsafe_allow_html=True,
                )
                gemini_col, openai_col = st.columns(2)
                with gemini_col:
                    render_result_cards(gemini_result, "Gemini", confidence_threshold)
                with openai_col:
                    render_result_cards(openai_result, "OpenAI", confidence_threshold)

else:
    st.info("👆 Upload an image to get started.")
