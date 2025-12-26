import streamlit as st
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

# 3. CSS (Horizontal Buttons + Sidebar Contact)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%) !important; font-family: 'Inter', sans-serif; }
    
    .title-text { 
        text-align: center; 
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; font-size: 32px; margin-bottom: 20px; 
    }

    /* Magic Code for Horizontal Buttons on Mobile */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 5px !important;
    }

    div.stButton > button { 
        height: 80px !important; width: 100% !important; border-radius: 12px !important; 
        background: white !important; color: #1a2a6c !important; 
        font-weight: 700 !important; font-size: 10px !important;
        border: 1px solid #ddd !important;
    }

    .contact-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px; border-radius: 12px; color: white;
        font-size: 13px; border-left: 4px solid #ef4444; margin-top: 20px;
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
    
    st.markdown("""
    <div class="contact-card">
        <b>📞 Contact Details:</b><br>
        Admission: 0761-2673654<br>
        Email: info@ggits.org<br><br>
        <b>📍 Location:</b><br>
        Jabalpur, MP
    </div>
    """, unsafe_allow_html=True)
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)

# 5. HEADER (Horizontal Buttons)
st.markdown('<div class="title-text">GGITS AI ASSISTANT</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🎓\nADMISSION"): st.session_state.info_box = "🎓 **Admission 2024:** Enrollment open for B.Tech & MBA."
with c2:
    if st.button("💼\nPLACEMENT"): st.session_state.info_box = "💼 **Placements:** Highest Package ₹12.5 LPA."
with c3:
    if st.button("💰\nFEES"): st.session_state.info_box = "💰 **Fees:** B.Tech Tuition ~₹78,000/yr."
with c4:
    if st.button("🏛️\nINFRA"): st.session_state.info_box = "🏛️ **Infra:** 25-Acre Wi-Fi enabled campus."

if st.session_state.info_box:
    st.info(st.session_state.info_box)

# 6. AI LOGIC (Information System)
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Aapka Knowledge Base
    data = {
        "intents": [
            {"tag": "hi", "patterns": ["hi", "hello", "hey"], "responses": ["Hello! How can I help you today? 😊"]},
            {"tag": "parking", "patterns": ["parking", "bike", "car"], "responses": ["🅿️ **Parking:** College parking is available near the main entrance."]},
            {"tag": "hod", "patterns": ["hod", "list", "heads"], "responses": ["👨‍🏫 **HODs:** CSE: Dr. Ashok Verma, IT: Prof. R.K. Rawat."]},
            {"tag": "admission", "patterns": ["admission", "process", "join"], "responses": ["🎓 **Admission:** Visit the Admin Block or call 0761-2673654."]}
        ]
    }
    pats, maps = [], []
    for i in data['intents']:
        for p in i['patterns']:
            pats.append(p.lower())
            maps.append(i)
    return model, model.encode(pats), maps

model, embeddings, intent_map = load_ai()

# Chat Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. CHAT INPUT (Bot Response Logic)
if prompt := st.chat_input("Poochiye (e.g., Parking kahan hai?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Similarity Check
    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    
    if max(sims) > 0.38:
        ans = random.choice(intent_map[np.argmax(sims)]['responses'])
    else:
        ans = "🏢 **GGITS Support:** Is baare mein jaankari nahi mili. Please 0761-2673654 par call karein."

    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
