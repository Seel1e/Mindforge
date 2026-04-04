"""
app/app.py
───────────
MindForge — Streamlit chat interface.

Plain English:
  This is the visual interface (the webpage) that users interact with.
  It has:
  - A chat window (like ChatGPT)
  - A sidebar where you can enter your profile info (age, stress level etc.)
    to get a personalised risk assessment
  - Coloured risk badges (Low=green, Medium=yellow, High=red)

Run with:
  streamlit run app/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from loguru import logger

# ── Page config (must be FIRST Streamlit call) ────────────────
st.set_page_config(
    page_title="MindForge — Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .risk-badge-low    { background:#22c55e; color:#fff; padding:4px 12px; border-radius:12px; font-weight:600; }
  .risk-badge-medium { background:#f59e0b; color:#fff; padding:4px 12px; border-radius:12px; font-weight:600; }
  .risk-badge-high   { background:#ef4444; color:#fff; padding:4px 12px; border-radius:12px; font-weight:600; }
  .context-box       { background:#1e293b; color:#94a3b8; padding:10px; border-radius:8px; font-size:0.8rem; }
  .latency-text      { color:#64748b; font-size:0.75rem; }
  .model-info        { background:#0f172a; border-left:3px solid #6366f1; padding:8px 14px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MindForge AI (this takes ~30s the first time) …")
def load_pipeline():
    from src.inference.pipeline import MindForgePipeline
    return MindForgePipeline(use_rag=True, use_risk_model=True)


def risk_badge(level: str) -> str:
    level_lower = level.lower()
    css_class = f"risk-badge-{level_lower}"
    return f'<span class="{css_class}">⚠ Risk: {level}</span>'


# ── Sidebar — User Profile ────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x80/6366f1/ffffff?text=🧠+MindForge", use_column_width=True)
    st.markdown("---")
    st.subheader("Your Profile (optional)")
    st.caption("Fill this in for a personalised risk assessment alongside your chat.")

    profile_data = {}
    with st.expander("Demographics", expanded=False):
        profile_data["age"] = st.slider("Age", 13, 90, 25)
        profile_data["gender"] = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        profile_data["employment_status"] = st.selectbox("Employment", ["Employed", "Student", "Unemployed", "Self-employed"])
        profile_data["work_environment"] = st.selectbox("Work environment", ["On-site", "Remote", "Hybrid"])

    with st.expander("Mental Health Indicators", expanded=False):
        profile_data["stress_level"] = st.slider("Stress level (0=none, 10=extreme)", 0, 10, 5)
        profile_data["sleep_hours"] = st.slider("Sleep hours per night", 2.0, 12.0, 7.0, step=0.5)
        profile_data["physical_activity_days"] = st.slider("Exercise days per week", 0, 7, 3)
        profile_data["depression_score"] = st.slider("Depression score (PHQ-9 style, 0–27)", 0, 27, 5)
        profile_data["anxiety_score"] = st.slider("Anxiety score (GAD-7 style, 0–21)", 0, 21, 5)
        profile_data["social_support_score"] = st.slider("Social support (0=none, 10=strong)", 0, 10, 6)
        profile_data["productivity_score"] = st.slider("Productivity (0=very low, 10=very high)", 0, 10, 6)
        profile_data["mental_health_history"] = st.selectbox("Past mental health history?", ["No", "Yes"])
        profile_data["seeks_treatment"] = st.selectbox("Currently seeking treatment?", ["No", "Yes"])

    use_profile = st.checkbox("Use profile for risk assessment", value=True)

    st.markdown("---")
    st.markdown('<div class="model-info">🤖 Powered by <b>MindForge LLM</b><br>Fine-tuned Mistral-7B + RAG</div>', unsafe_allow_html=True)
    st.caption("⚠️ Not a substitute for professional medical advice.")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────
st.title("🧠 MindForge — Mental Health AI")
st.caption("An AI companion for mental health education and support. Always consult a professional for clinical needs.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm MindForge, your mental health AI companion. "
                "I'm here to listen, provide information, and support you. "
                "How are you feeling today?"
            ),
        }
    ]

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "risk_level" in msg and msg["risk_level"]:
            st.markdown(risk_badge(msg["risk_level"]), unsafe_allow_html=True)
        if "latency_ms" in msg:
            st.markdown(f'<span class="latency-text">⏱ {msg["latency_ms"]}ms</span>', unsafe_allow_html=True)
        if "context" in msg and msg["context"]:
            with st.expander("📚 Retrieved context (RAG)", expanded=False):
                st.markdown(f'<div class="context-box">{msg["context"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message …"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build profile
    from src.inference.pipeline import UserProfile
    profile = None
    if use_profile and profile_data.get("age"):
        profile = UserProfile(**profile_data)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("MindForge is thinking …"):
            try:
                pipeline = load_pipeline()
                response = pipeline.chat(prompt, profile=profile)

                st.markdown(response.answer)

                if response.risk_level:
                    st.markdown(risk_badge(response.risk_level), unsafe_allow_html=True)
                    if response.risk_probabilities:
                        cols = st.columns(3)
                        for i, (label, prob) in enumerate(response.risk_probabilities.items()):
                            cols[i].metric(label, f"{prob*100:.1f}%")

                st.markdown(f'<span class="latency-text">⏱ {response.latency_ms}ms</span>', unsafe_allow_html=True)

                if response.retrieved_context:
                    with st.expander("📚 Retrieved context (RAG)", expanded=False):
                        st.markdown(f'<div class="context-box">{response.retrieved_context}</div>', unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(
                    f"Model not found: {e}\n\n"
                    "Please run the training pipeline first:\n"
                    "`python -m src.training.finetune_llm`"
                )
                response = type("R", (), {
                    "answer": "Model not yet trained.", "risk_level": None,
                    "retrieved_context": None, "latency_ms": 0,
                })()
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                st.error(f"An error occurred: {e}")
                response = type("R", (), {
                    "answer": str(e), "risk_level": None,
                    "retrieved_context": None, "latency_ms": 0,
                })()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "risk_level": getattr(response, "risk_level", None),
        "latency_ms": getattr(response, "latency_ms", 0),
        "context": getattr(response, "retrieved_context", None),
    })
