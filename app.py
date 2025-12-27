import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GG Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. ADVANCED RED-BLACK-WHITE CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    
    /* Background */
    .stApp { background-color: #fcfcfc; font-family: 'Poppins', sans-serif; }
    
    /* Main Header - Matches PPT Gradient */
    .main-header {
        background: linear-gradient(135deg, #b21f1f 0%, #1a1a1a 100%);
        padding: 35px; border-radius: 20px; text-align: center; color: white;
        box-shadow: 0 10px 30px rgba(178, 31, 31, 0.2); margin-bottom: 30px;
        border-bottom: 4px solid #b21f1f;
    }
    .main-header h1 { font-weight: 800; font-size: 42px; margin: 0; letter-spacing: 2px; }

    /* SIDEBAR STYLING - Sleek Black & Red */
    section[data-testid="stSidebar"] {
        background-color: #0f0f0f !important;
        border-right: 3px solid #b21f1f;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    .sidebar-info-card {
        background: #1a1a1a;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        border-left: 5px solid #b21f1f;
        margin-bottom: 20px;
    }

    /* PIYUSH KOSHTA DEVELOPER CARD - Matches PPT Intro */
    .dev-card {
        background: linear-gradient(135deg, #b21f1f 0%, #000000 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* ATTRACTIVE BUTTONS */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    /* Horizontal Quick Buttons - Clean White & Red */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        height: 100px !important;
        background: white !important;
        color: #b21f1f !important;
        border: 1px solid #eeeeee !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        font-size: 16px !important;
    }
    
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background: #b21f1f !important;
        color: white !important;
        border: 1px solid #b21f1f !important;
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(178, 31, 31, 0.3) !important;
    }

    /* Chat Bubbles Styling */
    .stChatMessage { background-color: #ffffff !important; border: 1px solid #f0f0f0; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# 4. ATTRACTIVE SIDEBAR
with st.sidebar:
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- PIYUSH KOSHTA DEVELOPER CARD ---
    st.markdown("""
    <div class="dev-card">
        <p style="margin:0; font-size:11px; opacity:0.7; text-transform:uppercase; letter-spacing:2px;">Project Lead</p>
        <h3 style="margin:5px 0; font-size:22px; font-weight:800; color:#ffffff;">PIYUSH KOSHTA</h3>
        <div style="width:40px; height:3px; background:#b21f1f; margin: 12px auto;"></div>
        <p style="margin:0; font-size:13px; color:#cccccc;">AI & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Info Card
    st.markdown("""
    <div class="sidebar-info-card">
        <h4 style="margin:0; color:#ff4b4b;">❤️ Campus Support</h4>
        <p style="font-size:12px; color:#bbbbbb; margin-top:5px;">
            Harnessing NLP to transform campus interactions at GGITS.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear Chat History Button (Black/Red Style)
    if st.button("🗑️ RESET CONVERSATION", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("<p style='color:#888888; font-size:12px;'>Official Helpline:<br><b>0761-2673654</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888888; font-size:12px;'>Institutional Web:<br><a href='https://ggits.org' style='color:#b21f1f; text-decoration:none;'>www.ggits.org</a></p>", unsafe_allow_html=True)

# 5. HEADER
st.markdown("""
    <div class="main-header">
        <h1>GG ASSISTANT</h1>
        <p style="opacity:0.9; font-weight:400; letter-spacing:1px;">Intelligent Campus Concierge • AI-Powered</p>
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
if prompt := st.chat_input("Ask GG Assistant..."):
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
