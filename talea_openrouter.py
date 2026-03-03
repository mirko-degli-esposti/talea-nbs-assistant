import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="TALEA NBS", page_icon="🌱", layout="wide")

# ── CONFIG ────────────────────────────────────────────────────

MODELS = {
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "GPT-4o":            "openai/gpt-4o",
    "GPT-4 Turbo":       "openai/gpt-4-turbo",
    "Llama 3.1 70B":     "meta-llama/llama-3.1-70b-instruct",
}

FIELDS = {
    "project_name": {"label": "Nome progetto",   "q": "Come si chiama il progetto NBS?"},
    "city":         {"label": "Città",            "q": "In quale città si trova?"},
    "country":      {"label": "Paese",            "q": "In quale paese?"},
    "year":         {"label": "Anno",             "q": "Anno di completamento?"},
    "scale":        {"label": "Scala",            "q": "A quale scala opera il progetto?",
                     "opts": ["District", "Urban", "Metropolitan", "Building"]},
    "location":     {"label": "Localizzazione",   "q": "Dove si trova nell'area urbana?",
                     "opts": ["Historical centre", "Consolidated city", "Suburbs", "Rural"]},
    "typology":     {"label": "Tipologia",        "q": "Che tipo di spazio è?",
                     "opts": ["Building", "Open space", "Infrastructure"]},
    "property":     {"label": "Proprietà",        "q": "Di chi è la proprietà?",
                     "opts": ["Public", "Private", "Mixed"]},
    "management":   {"label": "Gestione",         "q": "Come è gestito?",
                     "opts": ["Public", "Private", "Community"]},
    "uses":         {"label": "Usi principali",   "q": "Quali sono gli usi principali?",
                     "opts": ["Recreational", "Educational", "Cultural", "Commercial"]},
}

# ── LOAD CASES ────────────────────────────────────────────────

@st.cache_resource
def load_cases():
    """Carica cases.json dalla root del repo."""
    p = Path("cases.json")
    if not p.exists():
        return None, None
    with open(p, encoding="utf-8") as f:
        cases = json.load(f)
    embeddings = np.array([c["embedding"] for c in cases], dtype=np.float32)
    return cases, embeddings

CASES, EMBEDDINGS = load_cases()
RAG_AVAILABLE = CASES is not None

# ── RAG HELPERS ───────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity: a (1,D) vs b (N,D)."""
    return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-9)


def embed_query(text: str) -> np.ndarray:
    """Chiama l'API di embedding OpenRouter (usa il modello text-embedding-3-small)."""
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}"},
            json={"model": "openai/text-embedding-3-small", "input": text},
            timeout=10,
        )
        data = r.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def search_cases(query: str, top_k: int = 2):
    """Restituisce i top_k casi più simili alla query."""
    if not RAG_AVAILABLE:
        return []
    q_emb = embed_query(query)
    if q_emb is None:
        return []
    # Normalize
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims = cosine_sim(q_emb.reshape(1, -1), EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        c = CASES[i]
        results.append({
            "name":        c["name"],
            "city":        c["city"],
            "country":     c["country"],
            "year":        c.get("year", ""),
            "description": (c.get("description") or "")[:300] + "…",
            "score":       float(sims[i]),
        })
    return results


def format_cases_for_llm(cases: list) -> str:
    """Formatta i casi come contesto per il prompt."""
    if not cases:
        return ""
    lines = ["\n\n📚 ESEMPI SIMILI DAL DATABASE TALEA:"]
    for c in cases:
        lines.append(
            f"• {c['name']} ({c['city']}, {c['country']}, {c['year']}): "
            f"{c['description']}"
        )
    return "\n".join(lines)

# ── SESSION STATE ─────────────────────────────────────────────

defaults = {
    "msgs":       [],
    "data":       {},
    "curr":       None,
    "done":       [],
    "model":      MODELS["Claude 3.5 Sonnet"],
    "last_cases": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LLM CALL ─────────────────────────────────────────────────

def call_llm(user_input: str, field_ctx: str = "", case_ctx: str = "") -> str:
    system = (
        "Sei un assistente esperto di Nature-Based Solutions (NBS) e del framework TALEA. "
        "Il tuo compito è guidare la compilazione della Sezione A della scheda TALEA. "
        "Fai UNA sola domanda concisa per volta. "
        "Quando ti vengono forniti esempi simili, citali brevemente per arricchire il dialogo. "
        "Usa italiano formale."
    )
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]
    user_msg = user_input + field_ctx + case_ctx

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}"},
            json={
                "model":      st.session_state.model,
                "messages":   [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}],
                "max_tokens": 400,
            },
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Errore: {e}"

# ── HELPERS ───────────────────────────────────────────────────

def next_field():
    for k in FIELDS:
        if k not in st.session_state.done:
            return k
    return None

# ── SIDEBAR ───────────────────────────────────────────────────

with st.sidebar:
    st.image("https://em-content.zobj.net/source/twitter/376/seedling_1f331.png", width=48)
    st.title("TALEA NBS")

    if RAG_AVAILABLE:
        st.success(f"📚 {len(CASES)} casi studio caricati")
    else:
        st.warning("⚠️ `cases.json` non trovato — RAG disabilitato")

    st.divider()

    # Model selector
    sel = st.selectbox("Modello AI", list(MODELS.keys()))
    st.session_state.model = MODELS[sel]

    # Progress
    n_done = len(st.session_state.done)
    n_tot  = len(FIELDS)
    st.progress(n_done / n_tot)
    st.metric("Progresso sezione A", f"{n_done} / {n_tot}")

    if st.session_state.curr:
        st.info(f"🎯 Campo corrente: **{FIELDS[st.session_state.curr]['label']}**")

    st.divider()

    # Collected data
    if st.session_state.data:
        with st.expander("📋 Dati raccolti", expanded=True):
            for k, v in st.session_state.data.items():
                st.caption(f"**{FIELDS[k]['label']}:** {v}")

    # Similar cases in sidebar
    if st.session_state.last_cases:
        with st.expander("🔍 Esempi simili", expanded=True):
            for c in st.session_state.last_cases:
                st.markdown(f"**{c['name']}**  \n{c['city']}, {c['country']} — {c['year']}")
                st.caption(c["description"])
                st.caption(f"Similarità: {c['score']:.0%}")
                st.divider()

    # Download
    if n_done == n_tot:
        df = pd.DataFrame([st.session_state.data])
        st.download_button(
            "📥 Scarica CSV",
            df.to_csv(index=False).encode("utf-8"),
            f"talea_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("🔄 Ricomincia"):
        for k in ["msgs", "data", "done", "last_cases"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else {}
        st.session_state.curr = None
        st.rerun()

# ── MAIN CHAT ────────────────────────────────────────────────

st.header("🌱 TALEA NBS Assistant — Sezione A")

# Render history
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Welcome message (only on first load)
if not st.session_state.msgs:
    welcome = (
        "Ciao! Sono l'assistente TALEA NBS. "
        "Ti guiderò nella compilazione della **Sezione A** — Caratteristiche fisiche.\n\n"
        "Per iniziare: **come si chiama il progetto NBS che vuoi documentare?**"
    )
    st.session_state.msgs.append({"role": "assistant", "content": welcome})
    st.session_state.curr = "project_name"
    with st.chat_message("assistant"):
        st.markdown(welcome)

# Chat input
if user_input := st.chat_input("Scrivi qui la tua risposta…"):

    # Show user message
    st.session_state.msgs.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save current field value
    if st.session_state.curr:
        st.session_state.data[st.session_state.curr] = user_input
        st.session_state.done.append(st.session_state.curr)

    nxt = next_field()

    if not nxt:
        # All fields done
        reply = (
            "✅ **Sezione A completata!**\n\n"
            "Tutti i campi sono stati raccolti. "
            "Puoi scaricare il CSV dalla sidebar."
        )
        st.session_state.msgs.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.balloons()

    else:
        f = FIELDS[nxt]

        # Build field context
        field_ctx = f"\n\n[CAMPO DA COMPILARE: {f['label']} | Domanda suggerita: {f['q']}"
        if "opts" in f:
            field_ctx += f" | Opzioni valide: {', '.join(f['opts'])}"
        field_ctx += "]"

        # RAG: build query from current data + user input
        rag_query = user_input
        if st.session_state.data:
            ctx_parts = [f"{FIELDS[k]['label']}: {v}" for k, v in st.session_state.data.items()]
            rag_query = "; ".join(ctx_parts) + "; " + user_input

        similar = search_cases(rag_query, top_k=2)
        st.session_state.last_cases = similar

        case_ctx = format_cases_for_llm(similar) if similar else ""

        # Call LLM
        with st.spinner("💭 Elaborazione…"):
            reply = call_llm(user_input, field_ctx, case_ctx)

        st.session_state.curr = nxt
        st.session_state.msgs.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.rerun()
