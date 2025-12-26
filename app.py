import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. PAGE SETUP
st.set_page_config(page_title="GGITS AI Assistant", layout="wide")

# --- FIXED GOOGLE SHEET CONNECTION ---
# Is link mein maine gid=0 aur export format sahi kar diya hai
SHEET_URL = "https://docs.google.com/spreadsheets/d/1Rwe3CrCrXM8l3zNwHJWijbWRVjOvrMCFtJdiSUj9QGk/export?format=csv&gid=0"

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. CSS for UI
st.markdown("""
<style>
    .stApp { background: #f4f7f6; }
    [data-testid="stHorizontalBlock"] { display: flex !important; flex-wrap: nowrap !important; overflow-x: auto !important; }
    div.stButton > button { height: 60px !important; min-width: 100px !important; border-radius: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 4. SIDEBAR
with st.sidebar:
    st.title("GGITS HUB")
    if st.button("🗑️ CLEAR CHAT"):
        st.session_state.messages = []
        st.rerun()
    st.write("📞 Contact: 0761-2673654")

# 5. HEADER
st.title("GGITS AI ASSISTANT")
c1, c2, c3, c4 = st.columns(4)
with c1: 
    if st.button("🎓 ADMISSION"): st.info("Admission 2024 is Open!")
with c2: 
    if st.button("💼 PLACEMENT"): st.info("Highest Package: 12.5 LPA")
with c3: 
    if st.button("💰 FEES"): st.info("B.Tech Fees: ~78k/year")
with c4: 
    if st.button("🏛️ INFRA"): st.info("25-Acre Smart Campus")

# 6. AI BRAIN (Loading with Error Handling)
@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        # User-Agent header ke bina kai baar Google block karta hai
        df = pd.read_csv(SHEET_URL)
        df.columns = df.columns.str.strip() # Extra spaces hatane ke liye
        questions = df['Question'].astype(str).tolist()
        answers = df['Answer'].astype(str).tolist()
        q_embs = model.encode(questions)
        return model, q_embs, questions, answers
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        st.warning("Please check if Sheet is 'Published to Web' and 'Anyone with link' is ON.")
        return model, None, None, None

model, q_embs, questions, answers = load_ai()

# Chat display
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# 7. CHAT INPUT
if prompt := st.chat_input("Poochiye..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if q_embs is not None:
        u_emb = model.encode([prompt.lower()])
        sims = [1 - cosine(u_emb[0], q) for q in q_embs]
        if max(sims) > 0.38:
            ans = answers[np.argmax(sims)]
        else:
            ans = "Maaf kijiye, iska jawab abhi mere paas nahi hai. Please helpdesk par call karein."
    else:
        ans = "Technical Issue: Database se connection nahi ho pa raha hai."

    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()
