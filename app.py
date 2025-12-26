import streamlit as st
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GGITS AI Assistant", layout="wide", initial_sidebar_state="expanded")

# --- GOOGLE SHEET CONNECTION (Updated with bypass) ---
# Maine link ko clean kiya hai taaki Google authentication error na de
SHEET_URL = "https://docs.google.com/spreadsheets/d/1Rwe3CrCrXM8l3zNwHJWijbWRVjOvrMCFtJdiSUj9QGk/export?format=csv"

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "info_box" not in st.session_state:
    st.session_state.info_box = None

# 3. CSS (Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    .title-text { text-align: center; font-weight: 800; font-size: 32px; background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }
    [data-testid="stHorizontalBlock"] { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; overflow-x: auto !important; gap: 8px !important; }
    div.stButton > button { height: 70px !important; min-width: 90px !important; border-radius: 12px !important; background: white !important; color: #1a2a6c !important; font-weight: 700 !important; font-size: 11px !important; border: 1px solid #ddd !important; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .contact-card { background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; color: white; font-size: 13px; border-left: 4px solid #ef4444; margin-top: 20px; }
    .sidebar-heading { color: #000 !important; text-align: center; font-weight: 900; font-size: 20px; background: white; padding: 10px; border-radius: 10px; }
    [data-testid="stSidebar"] { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-heading">GGITS HUB</div>', unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CHAT"):
        st.session_state.messages = []
        st.session_state.info_box = None
        st.rerun()
    st.markdown("""<div class="contact-card"><b>📞 Contact:</b><br>0761-2673654<br>info@ggits.org<br><br><b>📍 Location:</b><br>Jabalpur, MP</div>""", unsafe_allow_html=True)
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)

# 5. HEADER (Horizontal Buttons)
st.markdown('<div class="title-text">GGITS ASSISTANT</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🎓\nADMISSION"): st.session_state.info_box = "🎓 **Admission:** Open for 2024. Visit Admin Block."
with c2:
    if st.button("💼\nPLACEMENT"): st.session_state.info_box = "💼 **Placement:** Highest 12.5 LPA. TCS, Cisco, Amdocs."
with c3:
    if st.button("💰\nFEES"): st.session_state.info_box = "💰 **Fees:** B.Tech ~₹78k/yr. Scholarship available."
with c4:
    if st.button("🏛️\nINFRA"): st.session_state.info_box = "🏛️ **Campus:** 25-Acre Wi-Fi, Smart Labs, Lift."

if st.session_state.info_box:
    st.info(st.session_state.info_box)

# 6. AI BRAIN (Loading from Google Sheet with Cache Fix)
@st.cache_resource(ttl=600) # 10 minutes cache
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        # Sheet ko public access ke saath load karna
        df = pd.read_csv(SHEET_URL)
        # Column names clean karna (spaces hatana)
        df.columns = df.columns.str.strip()
        questions = df['Question'].astype(str).tolist()
        answers = df['Answer'].astype(str).tolist()
        embeddings = model.encode(questions)
        return model, embeddings, questions, answers
    except Exception as e:
        # Agar sheet fail ho toh ye backup data dikhayega
        st.warning("Sheet connecting... Using Backup Data.")
        backup_q = ["hi", "fees", "admission"]
        backup_a = ["Hello!", "B.Tech fees is 78k.", "Admission is via DTE."]
        return model, model.encode(backup_q), backup_q, backup_a

model, embeddings, questions, answers = load_ai()

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. CHAT INPUT
if prompt := st.chat_input("Poochiye (e.g., Fees kya hai?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if embeddings is not None:
        u_vec = model.encode([prompt.lower()])
        sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
        
        if max(sims) > 0.35:
            ans = answers[np.argmax(sims)]
        else:
            ans = "🏢 **GGITS Office:** Iska jawab sheet mein nahi mila. Please call 0761-2673654."
    else:
        ans = "System error: Connecting to database..."

    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
