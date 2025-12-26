import streamlit as st
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GGITS Official AI Assistant", layout="wide", initial_sidebar_state="expanded")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "info_box" not in st.session_state:
    st.session_state.info_box = None

# 3. CSS (Forced Horizontal Buttons & Sidebar Styling)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    
    .title-text { 
        text-align: center; 
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
        font-size: 35px;
        margin-bottom: 20px; 
    }

    /* Buttons ko Horizontal rakhne ka magic code */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }

    div.stButton > button { 
        height: 75px !important; 
        width: 100% !important; 
        border-radius: 12px !important; 
        border: 1px solid rgba(26, 42, 108, 0.1) !important; 
        background: white !important; 
        color: #1a2a6c !important; 
        font-weight: 700 !important; 
        font-size: 11px !important;
        padding: 5px !important;
    }

    .contact-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-size: 13px;
        border-left: 4px solid #ef4444;
        margin-top: 20px;
    }
    
    .sidebar-heading { color: #000000 !important; text-align: center; font-weight: 900; font-size: 22px; background: white; padding: 10px; border-radius: 12px; margin-bottom: 15px; }
    [data-testid="stSidebar"] { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR (With Contact Details Below Clear Button)
with st.sidebar:
    st.markdown('<div class="sidebar-heading">GGITS HUB</div>', unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.messages = []
        st.session_state.info_box = None
        st.rerun()
    
    # Yahan Contact Details hain
    st.markdown("""
    <div class="contact-card">
        <b>📞 Contact Us:</b><br>
        Admission: 0761-2673654<br>
        Email: info@ggits.org<br><br>
        <b>📍 Location:</b><br>
        Jabalpur, Madhya Pradesh
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)

# 5. HEADER (4 Buttons in One Line)
st.markdown('<div class="title-text">GGITS AI ASSISTANT</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🎓\nADMISSION"): st.session_state.info_box = "🎓 **ADMISSION:** B.Tech, MBA, & B.Pharm admission open."
with c2:
    if st.button("💼\nPLACEMENT"): st.session_state.info_box = "💼 **PLACEMENTS:** Highest Package ₹12.5 LPA."
with c3:
    if st.button("💰\nFEES"): st.session_state.info_box = "💰 **FEES:** B.Tech ~₹78k. Scholarships available."
with c4:
    if st.button("🏛️\nINFRA"): st.session_state.info_box = "🏛️ **CAMPUS:** Smart Labs, 25-Acre Wi-Fi campus."

if st.session_state.info_box:
    st.markdown(f'<div style="background:white; padding:15px; border-radius:15px; border-left:5px solid #1a2a6c; margin-bottom:20px; font-weight:600;">{st.session_state.info_box}</div>', unsafe_allow_html=True)

# 6. AI LOGIC
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Aapka data yahan aayega
    pats = ["hi", "hello", "parking"]
    embeddings = model.encode(pats)
    return model, embeddings, pats

model, embeddings, pats = load_resources()

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Poochiye..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
