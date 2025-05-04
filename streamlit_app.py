# streamlit_app.py

import streamlit as st
import pandas as pd
import json
from streamlit import session_state as ss
from streamlit.errors import StreamlitAPIException
from app.agent import chat_node

if "chat_input" not in ss:
    ss.chat_input = ""

if "chat_history" not in ss:
    ss.chat_history = []

# Callback to handle sending messages
def send_message():
    user_msg = ss.chat_input
    ss.chat_history.append({"role": "user", "content": user_msg})
    state_chat = {
        "tweets_dashboard": "tweets_dashboard.csv",
        "support_dashboard": "support_dashboard.csv",
        "chat": ss.chat_history,
    }
    response = chat_node(state_chat).get("chat_response", "Lo siento, no tengo respuesta.")
    ss.chat_history.append({"role": "assistant", "content": response})
    # Clear the input field
    ss.chat_input = ""

# 1) Configura la página en modo “wide”
st.set_page_config(
    page_title="OmniSentimentAI Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .chat-messages {
        max-height: 60vh;
        overflow-y: auto;
        padding-bottom: 70px; /* space for input */
    }
    /* Ensure chat messages have space at bottom */
    .chat-messages {
        padding-bottom: 80px;
    }
       .chat-window {
           position: fixed;
           bottom: 20px;
           right: 20px;
           width: 350px;
           height: 450px;
           background: white;
           border: 1px solid #ccc;
           border-radius: 8px;
           box-shadow: 0 4px 8px rgba(0,0,0,0.1);
           display: flex;
           flex-direction: column;
           z-index: 999;
       }
       .chat-header {
           padding: 8px;
           background: #f0f0f0;
           border-bottom: 1px solid #ddd;
           font-weight: bold;
       }
       .chat-body {
           flex: 1;
           padding: 8px;
           overflow-y: auto;
       }
       .chat-input {
           padding: 8px;
           border-top: 1px solid #ddd;
       }
       .chat-input .stTextInput {
           width: 100% !important;
       }
    </style>
    """,
    unsafe_allow_html=True,
)

# 2) Función para cargar datos (sin cache)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Limpiar registros pero conservar todos los campos originales del CSV
    def clean_row(row):
        # Conservar todos los campos originales
        row = row.copy()
        sent = row.get("sentiment")
        if isinstance(sent, str) and sent.strip().startswith("{"):
            try:
                j = json.loads(sent)
                row["sentiment"] = j.get("sentiment")
                row["score"]     = j.get("score")
            except:
                pass
        # Parse keywords JSON-list if present, pero evita prompts falsos
        kw = row.get("keywords")
        if isinstance(kw, str) and kw.strip().startswith("["):
            try:
                parsed_kw = json.loads(kw)
                if isinstance(parsed_kw, list) and not any(
                    isinstance(k, str) and "key phrases" in k.lower() for k in parsed_kw
                ):
                    row["keywords"] = parsed_kw
                else:
                    row["keywords"] = []
            except:
                row["keywords"] = []
        return row

    return df.apply(clean_row, axis=1)

# 3) Carga los dos CSV generados por tus pipelines
df_tweets  = load_data("tweets_dashboard.csv")
df_support = load_data("support_dashboard.csv")

# Truncate the 'id' column to the first 4 characters for display
df_tweets["id"] = df_tweets["id"].astype(str).str[:4]
df_support["id"] = df_support["id"].astype(str).str[:4]

if st.button("🔄 Reload Data"):
    st.cache_data.clear()
    df_tweets  = load_data("tweets_dashboard.csv")
    df_support = load_data("support_dashboard.csv")
    try:
        st.experimental_rerun()
    except AttributeError:
        st.warning("Streamlit no soporta st.experimental_rerun en esta versión. Por favor, recarga la página manualmente para ver los datos actualizados.")

# Explode keywords columns into separate rows for counting
if "keywords" in df_tweets.columns:
    df_tweets["keywords_list"] = df_tweets["keywords"]
if "keywords" in df_support.columns:
    df_support["keywords_list"] = df_support["keywords"]

# 4) Título y secciones por canal
st.title("OmniSentimentAI Dashboard")

tweets_col, support_col = st.columns(2)

with tweets_col:
    st.header("Tweets Dashboard")
    st.dataframe(df_tweets, use_container_width=True)
    if "sentiment" in df_tweets.columns:
        st.subheader("Sentiment Distribution (Tweets)")
        st.bar_chart(df_tweets["sentiment"].value_counts())
    else:
        st.warning("No se encontró la columna 'sentiment' en tweets_dashboard.csv")


    # -- Language distribution --
    if "language" in df_tweets.columns:
        st.subheader("Language Distribution")
        st.bar_chart(df_tweets["language"].value_counts())

    # -- Emotion distribution --
    if "emotion" in df_tweets.columns:
        st.subheader("Emotion Distribution")
        st.bar_chart(df_tweets["emotion"].value_counts())

    # -- Urgency distribution --
    if "urgency" in df_tweets.columns:
        st.subheader("Urgency Distribution")
        st.bar_chart(df_tweets["urgency"].value_counts())

    # -- Top Keywords --
    if "support_area" in df_tweets.columns:
        st.subheader("Top Support Area")
        top_kw = df_tweets["support_area"].explode().value_counts().head(10)
        st.bar_chart(top_kw)


with support_col:
    st.header("Support Dashboard")
    st.dataframe(df_support, use_container_width=True)
    if "sentiment" in df_support.columns:
        st.subheader("Sentiment Distribution (Support)")
        st.bar_chart(df_support["sentiment"].value_counts())
    else:
        st.warning("No se encontró la columna 'sentiment' en support_dashboard.csv")


    # -- Language distribution --
    if "language" in df_support.columns:
        st.subheader("Language Distribution (Support)")
        st.bar_chart(df_support["language"].value_counts())

    # -- Emotion distribution --
    if "emotion" in df_support.columns:
        st.subheader("Emotion Distribution (Support)")
        st.bar_chart(df_support["emotion"].value_counts())

    # -- Urgency distribution --
    if "urgency" in df_support.columns:
        st.subheader("Urgency Distribution (Support)")
        st.bar_chart(df_support["urgency"].value_counts())

    # -- Top Keywords --
    if "support_area" in df_support.columns:
        st.subheader("Top Support Area (Support)")
        top_kw_s = df_support["support_area"].explode().value_counts().head(10)
        st.bar_chart(top_kw_s)



# --- Chat en la barra lateral ---
st.sidebar.header("Chat")
# Mostrar historial
for msg in ss.chat_history:
    if msg["role"] == "user":
        st.sidebar.chat_message("user").write(msg["content"])
    else:
        st.sidebar.chat_message("assistant").write(msg["content"])

st.sidebar.text_input(
    "Mensaje:", key="chat_input",
    placeholder="Escribe tu mensaje...",
    label_visibility="hidden",
    on_change=send_message
)