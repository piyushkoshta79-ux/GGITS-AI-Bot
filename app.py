import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GG Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. PREMIUM CSS (Design as it is + Sidebar Enhancements)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    .stApp { background-color: #f0f2f6; font-family: 'Poppins', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 100%);
        padding: 35px; border-radius: 20px; text-align: center; color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-bottom: 30px;
    }
    .main-header h1 { font-weight: 800; font-size: 40px; margin: 0; letter-spacing: 1px; }

    /* Sidebar Attractive Styling */
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 2px solid #eee; }
    
    .sidebar-card {
        background: #f8f9fa; padding: 15px; border-radius: 12px;
        border-left: 5px solid #b21f1f; margin-bottom: 20px; font-size: 13px;
    }

    /* Delete Button - Extra Attractive */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(90deg, #ff4b2b, #ff416c) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        height: 45px !important; width: 100% !important;
        transition: 0.3s; box-shadow: 0 4px 10px rgba(255, 75, 43, 0.3);
    }
    .stButton > button[kind="secondary"]:hover {
        transform: scale(1.02); box-shadow: 0 6px 15px rgba(255, 75, 43, 0.4);
    }

    /* Quick Buttons Styling */
    div.stButton > button:not([kind="secondary"]) {
        border-radius: 15px !important; height: 90px !important; width: 100% !important;
        background: white !important; color: #1a2a6c !important; font-weight: 700 !important;
        border: none !important; box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR (Attractive Version)
with st.sidebar:
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Attractive Clear Chat Section
    st.markdown("### ⚙️ Actions")
    if st.button("🗑️ CLEAR HISTORY", kind="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Sidebar Info Card
    st.markdown("""
    <div class="sidebar-card">
        <b>💡 Tip:</b><br>
        Aap Admission, Fees ya Placement ke baare mein sawal puch sakte hain!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("📞 **Helpline:**<br>0761-2673654", unsafe_allow_html=True)

# 5. HEADER
st.markdown("""
    <div class="main-header">
        <h1>GG ASSISTANT</h1>
        <p>Gyan Ganga Institute of Technology & Sciences</p>
    </div>
""", unsafe_allow_html=True)

# 6. QUICK ACTION BUTTONS
col1, col2, col3, col4 = st.columns(4)
actions = {
    "🎓\nADMISSION": "Admission 2024 is open! You can apply through MP DTE Counselling or visit the campus.",
    "💼\nPLACEMENTS": "GGITS Placements are top-notch. Highest package: ₹12.5 LPA. Top companies: TCS, Cisco, Amdocs.",
    "💰\nFEES INFO": "B.Tech Fees: ~₹78,000 per year. Scholarships like Medhavi & Post-Metric are available.",
    "🏛️\nINFRA": "25-Acre Smart Campus with advanced labs, library, and Wi-Fi facilities."
}

with col1:
    if st.button("🎓\nADMISSION"): st.session_state.messages.append({"role": "assistant", "content": actions["🎓\nADMISSION"]})
with col2:
    if st.button("💼\nPLACEMENTS"): st.session_state.messages.append({"role": "assistant", "content": actions["💼\nPLACEMENTS"]})
with col3:
    if st.button("💰\nFEES INFO"): st.session_state.messages.append({"role": "assistant", "content": actions["💰\nFEES INFO"]})
with col4:
    if st.button("🏛️\nINFRASTRUCTURE"): st.session_state.messages.append({"role": "assistant", "content": actions["🏛️\nINFRA"]})

# 7. AI BRAIN
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb = [
        {"q": "hi hello hey greetings", "a": "Hello! I am your GG Assistant. How can I help you today? 😊"},
        {"q": "fees structure btech mba bpharm", "a": "💰 **Fees:** B.Tech: ~78k/yr, MBA: ~60k/yr. We offer various scholarships."},
        {"q": "placement records highest package company", "a": "💼 **Placements:** Highest 12.5 LPA. Recruiters: TCS, Reliance, Cisco, etc."},
        {"q": "location address where is college", "a": "📍 **Location:** Bargi Hills, Jabalpur, MP 482003."},
    ]
    questions = [i['q'] for i in kb]
    embeddings = model.encode(questions)
    return model, embeddings, kb

model, embeddings, kb = load_ai()

# --- RENDER CHAT ---
for msg in st.session_state.messages:
    avatar_icon = "🤖" if msg["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# 8. CHAT INPUT
if prompt := st.chat_input("Poochiye GGITS ke baare mein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"): 
        st.markdown(prompt)

    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    
    if max(sims) > 0.35:
        ans = kb[np.argmax(sims)]['a']
    else:
        ans = "🏢 **GG Assistant:** I'm not sure about that. Please contact the college office at 0761-2673654."

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(ans)
    
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
