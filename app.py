import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="LingoSoft AI", page_icon="☁️", layout="centered")

# Custom CSS for Pastel Theme
st.markdown("""
    <style>
    /* Soft Pastel Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        color: #4a4a4a;
    }
    
    /* Elegant Pastel Title */
    .main-title {
        text-align: center;
        font-family: 'Quicksand', sans-serif;
        color: #88a1c8;
        font-size: 3rem;
        font-weight: 700;
        margin-top: -20px;
    }
    
    .sub-title {
        text-align: center;
        color: #a1a1a1;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* Pastel Input Box */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #4a4a4a !important;
        border: 2px solid #daeaf6 !important;
        border-radius: 20px !important;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.05);
    }

    /* Soft Blue Button */
    .stButton>button {
        background-color: #daeaf6;
        color: #6a89cc;
        border: none;
        padding: 10px 20px;
        border-radius: 15px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #c5def0;
        color: #4a69bd;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. App Header
st.markdown('<h1 class="main-title">LingoSoft AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A gentle way to discover the world\'s languages.</p>', unsafe_allow_html=True)

# Language Mapping (Expand this as needed)
LANGUAGE_MAP = {
    'en': ('English', '🇺🇸'), 'es': ('Spanish', '🇪🇸'), 'fr': ('French', '🇫🇷'),
    'de': ('German', '🇩🇪'), 'hi': ('Hindi', '🇮🇳'), 'ur': ('Urdu', '🇵🇰'),
    'ar': ('Arabic', '🇸🇦'), 'zh': ('Chinese', '🇨🇳'), 'ja': ('Japanese', '🇯🇵')
}

@st.cache_resource
def load_nlp():
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# 3. Interactive Section
user_text = st.text_area("", placeholder="Paste your text here...", height=150)

if st.button("✨ Identify Language"):
    if user_text.strip():
        with st.spinner("Whispering to the AI..."):
            classifier = load_nlp()
            res = classifier(user_text)[0]
            
            name, flag = LANGUAGE_MAP.get(res['label'], ("Unknown", "🌐"))
            confidence = res['score']
            
            st.markdown("---")
            # Result Cards in Pastel Colors
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {flag} {name}")
                st.caption("Language Detected")
            with col2:
                st.markdown(f"### {confidence:.1%}")
                st.caption("AI Confidence")
            
            st.progress(confidence)
    else:
        st.info("The text box is empty. Type something!")