import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP (Professional Look)
st.set_page_config(page_title="GGITS Assistant", page_icon="🎓", layout="wide")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. EXTRA RICH CSS (Modern UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    
    .stApp { background-color: #f0f2f6; font-family: 'Poppins', sans-serif; }
    
    /* Elegant Header */
    .main-header {
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .main-header h1 { font-weight: 800; font-size: 45px; margin: 0; letter-spacing: 2px; }
    .main-header p { font-size: 16px; opacity: 0.9; margin-top: 10px; }

    /* Modern Buttons */
    [data-testid="stHorizontalBlock"] { gap: 15px !important; }
    
    div.stButton > button {
        border-radius: 15px !important;
        height: 100px !important;
        width: 100% !important;
        background: white !important;
        color: #1a2a6c !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
    }
    
    div.stButton > button:hover {
        background: #1a2a6c !important;
        color: white !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 25px rgba(26, 42, 108, 0.2) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #eee; }
    .sidebar-content { text-align: center; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR (Clean & White)
with st.sidebar:
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.write("📍 **Location:** Bargi Hills, Jabalpur")
    st.write("📞 **Support:** 0761-2673654")

# 5. PREMIUM HEADER
st.markdown("""
    <div class="main-header">
        <h1>GG ASSISTANT</h1>
        <p>Your Gyan Ganga Institute of Technology & Sciences</p>
    </div>
""", unsafe_allow_html=True)

# 6. QUICK ACTION BUTTONS
col1, col2, col3, col4 = st.columns(4)
actions = {
    "🎓\nADMISSION": "Our admission process is simple! Apply through MP DTE Counselling or visit the campus for direct inquiry.",
    "💼\nPLACEMENTS": "GGITS has a great record! Highest package: ₹12.5 LPA. Top companies include TCS, Cisco, and Amdocs.",
    "💰\nFEES INFO": "B.Tech fees are approx ₹78,000/year. Various scholarships are available for eligible students.",
    "🏛️\nINFRASTRUCTURE": "A 25-acre lush green campus with smart classrooms, advanced labs, and an amazing library."
}

with col1:
    if st.button("🎓\nADMISSION"): st.session_state.messages.append({"role": "assistant", "content": actions["🎓\nADMISSION"]})
with col2:
    if st.button("💼\nPLACEMENTS"): st.session_state.messages.append({"role": "assistant", "content": actions["💼\nPLACEMENTS"]})
with col3:
    if st.button("💰\nFEES INFO"): st.session_state.messages.append({"role": "assistant", "content": actions["💰\nFEES INFO"]})
with col4:
    if st.button("🏛️\nINFRASTRUCTURE"): st.session_state.messages.append({"role": "assistant", "content": actions["🏛️\nINFRASTRUCTURE"]})

# 7. AI BRAIN (Stable Data)
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb = [
        {"q": "hi hello hey greetings sup", "a": "Hello! I am your GG Assistant. How can I help you today? 😊"},
        {"q": "fees structure btech mba bpharm cost", "a": "💰 **Fees:** B.Tech: ~₹78k/yr, MBA: ~₹60k/yr. Hostel: ~₹60k/yr."},
        {"q": "placement records highest package companies", "a": "💼 **Placements:** Highest 12.5 LPA. Average 4.5 LPA. Companies: TCS, Cisco, Persistent."},
        {"q": "location address where is college", "a": "📍 **Location:** Bargi Hills, Jabalpur, MP. Near the beautiful Narmada river valley."},
        {"q": "hod faculty names heads", "a": "👨‍🏫 **HODs:** CSE: Dr. Ashok Verma, IT: Prof. R.K. Rawat, EC: Dr. Shailja Shukla."}
    ]
    questions = [i['q'] for i in kb]
    embeddings = model.encode(questions)
    return model, embeddings, kb

model, embeddings, kb = load_ai()

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. CHAT INPUT
if prompt := st.chat_input("Poochiye GGITS ke baare mein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Simple Match Logic
    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    
    if max(sims) > 0.35:
        ans = kb[np.argmax(sims)]['a']
    else:
        ans = "🏢 **GG Assistant:** I'm not sure about that. Please contact our helpline at 0761-2673654."

    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()

