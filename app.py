import streamlit as st
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GGITS AI Assistant", layout="wide", initial_sidebar_state="expanded")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "info_box" not in st.session_state:
    st.session_state.info_box = None

# 3. CSS (Forced Horizontal Buttons & Sidebar Contact)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    
    .title-text { 
        text-align: center; font-weight: 800; font-size: 32px;
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px; 
    }

    /* Magic Code for Horizontal Buttons on Mobile */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 8px !important;
    }

    div.stButton > button { 
        height: 70px !important; min-width: 90px !important; border-radius: 12px !important; 
        background: white !important; color: #1a2a6c !important; 
        font-weight: 700 !important; font-size: 11px !important;
        border: 1px solid #ddd !important; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }

    .contact-card {
        background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; 
        color: white; font-size: 13px; border-left: 4px solid #ef4444; margin-top: 20px;
    }
    
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

# 6. AI BRAIN (Yahan data add kiya gaya hai)
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb = [
        {"q": "hi hello hey greetings", "a": "Hello! I am your GGITS Assistant. How can I help you today? 😊"},
        {"q": "fees structure btech mba bpharm cost money", "a": "💰 **Fees:** B.Tech is approx ₹78,000 per year. We offer Medhavi and Post-Metric scholarships."},
        {"q": "admission process criteria how to join", "a": "🎓 **Admission:** You can apply via MP DTE counselling. For direct inquiry, call 0761-2673654."},
        {"q": "placement package company highest salary job", "a": "💼 **Placements:** Excellent records! Highest package ₹12.5 LPA. Companies: TCS, Reliance, Cisco, etc."},
        {"q": "parking bike car area", "a": "🅿️ **Parking:** Dedicated safe parking near the college main gate."},
        {"q": "hod list faculty department heads", "a": "👨‍🏫 **HODs:** CSE: Dr. Ashok Verma, IT: Prof. R.K. Rawat, EC: Dr. Shailja Shukla."}
    ]
    questions = [i['q'] for i in kb]
    embeddings = model.encode(questions)
    return model, embeddings, kb

model, embeddings, kb = load_ai()

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. CHAT INPUT
if prompt := st.chat_input("Poochiye (e.g., Fees kya hai?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Bot logic
    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    
    if max(sims) > 0.30: # Sensitivity adjusted
        ans = kb[np.argmax(sims)]['a']
    else:
        ans = "🏢 **GGITS Office:** I'm still learning. Please call 0761-2673654 for this specific info."

    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
