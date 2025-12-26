import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP (Stable Design)
st.set_page_config(page_title="GGITS AI Assistant", layout="wide", initial_sidebar_state="expanded")

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. PREMIUM CSS (Wapas wahi pehle wala look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: #f8f9fc; font-family: 'Inter', sans-serif; }
    
    .title-text { 
        text-align: center; font-weight: 800; font-size: 35px;
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 30px; 
    }

    /* Horizontal Buttons Styling */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        overflow-x: auto !important;
        gap: 10px !important;
    }

    div.stButton > button { 
        height: 80px !important; width: 100% !important; border-radius: 15px !important; 
        background: white !important; color: #1a2a6c !important; 
        font-weight: 700 !important; border: 1px solid #e0e0e0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: 0.3s;
    }
    
    div.stButton > button:hover { border-color: #1a2a6c !important; transform: translateY(-2px); }

    [data-testid="stSidebar"] { background: #0f172a !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR
with st.sidebar:
    st.image("https://ggits.org/wp-content/uploads/2021/03/ggits-logo.png", use_container_width=True)
    st.markdown("---")
    if st.button("🗑️ CLEAR CHAT"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("### 📞 Contact Support")
    st.info("0761-2673654\ninfo@ggits.org")

# 5. HEADER & QUICK ACTION BUTTONS
st.markdown('<div class="title-text">GGITS AI HUB</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🎓\nADMISSION"): st.session_state.messages.append({"role": "assistant", "content": "🎓 **Admission 2024:** Admissions are open via MP DTE Counselling. Need help? Call 0761-2673654."})
with col2:
    if st.button("💼\nPLACEMENT"): st.session_state.messages.append({"role": "assistant", "content": "💼 **Placements:** Highest Package 12.5 LPA. Top recruiters: TCS, Cisco, Reliance."})
with col3:
    if st.button("💰\nFEES"): st.session_state.messages.append({"role": "assistant", "content": "💰 **Fees:** B.Tech ~₹78,000/year. Scholarships like Medhavi and Post-Metric are available."})
with col4:
    if st.button("🏛️\nINFRA"): st.session_state.messages.append({"role": "assistant", "content": "🏛️ **Campus:** 25-acre lush green campus, smart labs, and high-speed Wi-Fi."})

# 6. AI BRAIN (Hardcoded Data - No Google Sheet needed)
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Aap niche isi list mein aur sawal-jawab add kar sakte hain
    kb = [
        {"q": "hi hello hey greetings", "a": "Hello! Welcome to GGITS. How can I assist you today? 😊"},
        {"q": "fees structure btech mba bpharm cost", "a": "💰 **Fees:** B.Tech is approx ₹78,000/year, MBA is approx ₹60,000/year. Hostel extra."},
        {"q": "admission process how to join", "a": "🎓 **Admission:** Register on MP DTE portal and choose GGITS (Jabalpur) during choice filling."},
        {"q": "placement records highest package", "a": "💼 **Placement:** Our highest package is ₹12.5 LPA. Average is around ₹4.5 LPA."},
        {"q": "location address where is college", "a": "📍 **Location:** Bargi Hills, Jabalpur, Madhya Pradesh 482003."},
        {"q": "hod names faculty list", "a": "👨‍🏫 **HODs:** CSE: Dr. Ashok Verma, IT: Prof. R.K. Rawat, EC: Dr. Shailja Shukla."},
        {"q": "hostel facility fees", "a": "🏠 **Hostel:** We have separate hostels for boys and girls with mess facility. Fees ~₹60,000/year."}
    ]
    questions = [i['q'] for i in kb]
    embeddings = model.encode(questions)
    return model, embeddings, kb

model, embeddings, kb = load_ai()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. CHAT INPUT
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Logic
    u_vec = model.encode([prompt.lower()])
    sims = [1 - cosine(u_vec[0], p_vec) for p_vec in embeddings]
    
    if max(sims) > 0.35:
        ans = kb[np.argmax(sims)]['a']
    else:
        ans = "🏢 **GGITS Support:** I don't have this specific info yet. Please contact the college office at 0761-2673654."

    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
