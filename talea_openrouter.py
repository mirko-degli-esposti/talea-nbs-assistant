import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="TALEA NBS", page_icon="🌱", layout="wide")

# Config
MODELS = {
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4 Turbo": "openai/gpt-4-turbo",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct"
}

FIELDS = {
    "project_name": {"label": "Nome progetto", "q": "Nome del progetto?"},
    "city": {"label": "Città", "q": "Città?"},
    "country": {"label": "Paese", "q": "Paese?"},
    "year": {"label": "Anno", "q": "Anno di completamento?"},
    "scale": {"label": "Scala", "q": "Scala?", 
              "opts": ["District", "Urban", "Metropolitan", "Building"]},
    "location": {"label": "Localizzazione", "q": "Dove si trova?",
                 "opts": ["Historical centre", "Consolidated city", "Suburbs", "Rural"]},
    "typology": {"label": "Tipologia", "q": "Tipo di spazio?",
                 "opts": ["Building", "Open space", "Infrastructure"]},
    "property": {"label": "Proprietà", "q": "Proprietà?",
                 "opts": ["Public", "Private", "Mixed"]},
    "management": {"label": "Gestione", "q": "Gestione?",
                   "opts": ["Public", "Private", "Community"]},
    "uses": {"label": "Usi", "q": "Usi principali?",
             "opts": ["Recreational", "Educational", "Cultural", "Commercial"]}
}

# Init
if 'msgs' not in st.session_state:
    st.session_state.msgs = []
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'curr' not in st.session_state:
    st.session_state.curr = None
if 'done' not in st.session_state:
    st.session_state.done = []
if 'model' not in st.session_state:
    st.session_state.model = MODELS["Claude 3.5 Sonnet"]

# Functions
def call_llm(msg, ctx=""):
    msgs = [{"role": "system", "content": "Assistente TALEA NBS. Fai UNA domanda concisa. Italiano formale."}]
    msgs.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs])
    msgs.append({"role": "user", "content": msg + ctx})
    
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}"},
            json={"model": st.session_state.model, "messages": msgs, "max_tokens": 300}
        )
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ {e}"

def next_field():
    for k in FIELDS:
        if k not in st.session_state.done:
            return k

# UI
st.title("🌱 TALEA NBS Assistant")

# Sidebar
with st.sidebar:
    st.selectbox("Modello", list(MODELS.keys()), 
                 key='model_name',
                 on_change=lambda: setattr(st.session_state, 'model', MODELS[st.session_state.model_name]))
    
    st.progress(len(st.session_state.done) / len(FIELDS))
    st.metric("Progresso", f"{len(st.session_state.done)}/{len(FIELDS)}")
    
    if st.session_state.curr:
        st.info(f"🎯 {FIELDS[st.session_state.curr]['label']}")
    
    st.markdown("---")
    
    if st.session_state.data:
        with st.expander("📋 Dati"):
            for k, v in st.session_state.data.items():
                st.caption(f"**{FIELDS[k]['label']}:** {v}")
    
    if len(st.session_state.done) == len(FIELDS):
        df = pd.DataFrame([st.session_state.data])
        st.download_button(
            "📥 Excel",
            df.to_csv(index=False).encode('utf-8'),
            f"talea_{datetime.now():%Y%m%d}.csv"
        )
    
    if st.button("🔄 Reset"):
        st.session_state.msgs = []
        st.session_state.data = {}
        st.session_state.done = []
        st.session_state.curr = None
        st.rerun()

# Chat
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if p := st.chat_input("Risposta..."):
    st.session_state.msgs.append({"role": "user", "content": p})
    with st.chat_message("user"):
        st.markdown(p)
    
    if st.session_state.curr:
        st.session_state.data[st.session_state.curr] = p
        st.session_state.done.append(st.session_state.curr)
    
    nxt = next_field()
    
    if not nxt:
        r = "✅ Completo! Scarica Excel dalla sidebar."
        st.session_state.msgs.append({"role": "assistant", "content": r})
        with st.chat_message("assistant"):
            st.markdown(r)
        st.balloons()
    else:
        f = FIELDS[nxt]
        c = f"\n\nCAMPO: {f['label']}\nDOMANDA: {f['q']}"
        if 'opts' in f:
            c += f"\nOPZIONI: {', '.join(f['opts'])}"
        
        with st.spinner("💭"):
            r = call_llm(p, c)
        
        st.session_state.curr = nxt
        st.session_state.msgs.append({"role": "assistant", "content": r})
        with st.chat_message("assistant"):
            st.markdown(r)
    
    st.rerun()

# Welcome
if not st.session_state.msgs:
    w = "Ciao! Compiliamo la Sezione A TALEA.\n\nNome del progetto?"
    st.session_state.msgs.append({"role": "assistant", "content": w})
    st.session_state.curr = "project_name"
    with st.chat_message("assistant"):
        st.markdown(w)
