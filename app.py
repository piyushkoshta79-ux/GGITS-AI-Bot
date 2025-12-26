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

# 3. RESPONSIVE DESIGN & SIDEBAR CONTACT
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    
    /* Responsive Title */
    .title-text { 
        text-align: center; 
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
        margin-bottom: 20px; 
    }
    @media (max-width: 768px) { .title-text { font-size: 28px !important; margin-top: 0px; } }
    @media (min-width: 769px) { .title-text { font-size: 52px !important; } }

    /* Buttons Style */
    div.stButton > button { 
        height: 100px; width: 100%; border-radius: 20px; 
        border: 1px solid rgba(26, 42, 108, 0.1) !important; 
        background: white !important; color: #1a2a6c !important; 
        font-weight: 800 !important; font-size: 14px !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05) !important; 
    }

    /* Sidebar Contact Details */
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

# 4. SIDEBAR (With Contact Details)
with st.sidebar:
    st.markdown('<div class="sidebar-heading">GGITS HUB</div>', unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.messages = []
        st.session_state.info_box = None
        st.rerun()
    
    # College Contact Info
    st.markdown("""
    <div class="contact-card">
        <b>📞 Contact Us:</b><br>
        Admission: 0761-2673654<br>
        Email: info@ggits.org<br><br>
        <b>📍 Location:</b><br>
        Jabalpur, Madhya Pradesh
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)

# 5. HEADER (Aaju-Baju Buttons)
st.markdown('<div class="title-text">GGITS AI ASSISTANT</div>', unsafe_allow_html=True)

# Ise 2-2 columns mein divide kiya hai taaki mobile mein aaju-baju rahein
col1, col2 = st.columns(2)
with col1:
    if st.button("🎓\nADMISSION"): st.session_state.info_box = "🎓 **ADMISSION 2024:** Enrollment open for B.Tech, MBA, & B.Pharm."
with col2:
    if st.button("💼\nPLACEMENT"): st.session_state.info_box = "💼 **PLACEMENTS:** Highest Package ₹12.5 LPA. Recruiters: TCS, Cisco."

col3, col4 = st.columns(2)
with col3:
    if st.button("💰\nFEES"): st.session_state.info_box = "💰 **FEES:** B.Tech ~₹78,000/yr. Scholarships available."
with col4:
    if st.button("🏛️\nINFRA"): st.session_state.info_box = "🏛️ **CAMPUS:** Smart Labs, 25-Acre Wi-Fi enabled campus."

if st.session_state.info_box:
    st.markdown(f'<div class="info-display" style="background:white; padding:20px; border-radius:20px; border-left:8px solid #1a2a6c; margin-bottom:20px;">{st.session_state.info_box}</div>', unsafe_allow_html=True)

# 6. AI LOGIC
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Dummy data (Aap apna poora data yahan rakh sakte hain)
    master_data = {"intents": [{"tag": "hi", "patterns": ["hi"], "responses": ["Hello!"]}]}
    pats, maps = [], []
    for intent in master_data['intents']:
        for p in intent['patterns']:
            pats.append(p.lower()); maps.append(intent)
    return model, model.encode(pats), maps

model, embeddings, intent_map = load_resources()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🔵" if msg["role"]=="assistant" else "🔴"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Poochiye..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    response = "🏢 **GGITS Office:** Specific data not found." if max(sims) < 0.35 else "Jawab mil gaya!"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
