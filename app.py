import streamlit as st
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GGITS Official AI Assistant", layout="centered", initial_sidebar_state="expanded")

# 2. GOOGLE SHEET LINK (Yahan apna CSV link paste karein)
SHEET_URL = "APNA_CSV_LINK_YAHAN_PASTE_KAREIN"

# 3. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "info_box" not in st.session_state:
    st.session_state.info_box = None

# 4. DESIGN (Wahi purana look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    [data-testid="stVerticalBlock"] > div:first-child { position: sticky; top: 0; z-index: 1000; background: #f8f9fc; padding-bottom: 10px; }
    .title-text { text-align: center; background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 52px; font-weight: 800; margin-top: -20px; margin-bottom: 20px; }
    div.stButton > button { height: 150px; width: 100%; border-radius: 24px; border: 1px solid rgba(26, 42, 108, 0.1) !important; background: rgba(255, 255, 255, 0.9) !important; color: #000000 !important; font-size: 19px !important; font-weight: 800 !important; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.03) !important; }
    div.stButton > button:hover { transform: translateY(-10px); background: #1a2a6c !important; color: white !important; }
    .info-display { background: #ffffff; color: #000000; padding: 25px; border-radius: 24px; margin-bottom: 20px; border-left: 8px solid #1a2a6c; box-shadow: 0 15px 35px rgba(0,0,0,0.06); font-weight: 700; }
    .sidebar-heading { color: #000000 !important; text-align: center; font-weight: 900; font-size: 28px; background: white; padding: 10px; border-radius: 12px; margin-bottom: 20px; }
    [data-testid="stSidebar"] { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# 5. SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-heading">GGITS HUB</div>', unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.messages = []
        st.session_state.info_box = None
        st.rerun()
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)

# 6. HEADER
st.markdown('<div class="title-text">GGITS AI ASSISTANT</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🎓\nADMISSION"): st.session_state.info_box = "🎓 **ADMISSION 2024:** Enrollment open for B.Tech, MBA, & B.Pharm. 📍 Admin Block, Window 1."
with c2:
    if st.button("💼\nPLACEMENTS"): st.session_state.info_box = "💼 **PLACEMENTS:** Highest Package ₹12.5 LPA. Top Recruiters: TCS, Cisco, Amdocs."
with c3:
    if st.button("💰\nFEES INFO"): st.session_state.info_box = "💰 **FEES:** B.Tech Tuition ~₹78,000/yr. Medhavi & Post-Metric Scholarship available."
with c4:
    if st.button("🏛️\nINFRA"): st.session_state.info_box = "🏛️ **CAMPUS:** Smart Labs, Lifts, and 25-Acre Wi-Fi enabled campus."

if st.session_state.info_box:
    st.markdown(f'<div class="info-display">{st.session_state.info_box}</div>', unsafe_allow_html=True)

# 7. AI LOGIC (Google Sheet Integration)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        df = pd.read_csv(SHEET_URL)
        questions = df['Question'].astype(str).tolist()
        answers = df['Answer'].astype(str).tolist()
        embeddings = model.encode(questions)
        return model, embeddings, answers
    except Exception as e:
        st.error("Sheet Load Error: CSV Link sahi nahi hai!")
        return model, None, None

model, embeddings, answers = load_resources()

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🔵" if msg["role"]=="assistant" else "🔴"):
        st.markdown(msg["content"])

# 8. INTERACTIVE CHAT
if prompt := st.chat_input("Poochiye (e.g. Parking kahan hai, HOD list)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if embeddings is not None:
        u_vec = model.encode([prompt.lower()])
        sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
        
        if max(sims) > 0.35: 
            response = answers[np.argmax(sims)]
        else:
            response = "🏢 **GGITS Office:** Specific data not found in Sheet. Please contact 0761-2673654. 😊"
    else:
        response = "Data Error: Please check Google Sheet connection."
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
