# streamlit_app.py

import streamlit as st
import pandas as pd
import json

# 1) Configura la página en modo “wide”
st.set_page_config(
    page_title="OmniSentimentAI Dashboard",
    layout="wide",
)

# 2) Función cache para cargar datos
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Si alguna celda de 'sentiment' está como string JSON, lo parseamos
    def clean_row(row):
        sent = row.get("sentiment")
        if isinstance(sent, str) and sent.strip().startswith("{"):
            try:
                j = json.loads(sent)
                row["sentiment"] = j.get("sentiment")
                row["score"]     = j.get("score")
            except:
                pass
        return row

    return df.apply(clean_row, axis=1)

# 3) Carga los dos CSV generados por tus pipelines
df_tweets  = load_data("tweets_dashboard.csv")
df_support = load_data("support_dashboard.csv")

# 4) Título y secciones por canal
st.title("OmniSentimentAI Dashboard")

tweets_col, support_col = st.columns(2)

with tweets_col:
    st.header("Tweets Dashboard")
    st.dataframe(df_tweets, use_container_width=True)
    st.subheader("Sentiment Distribution (Tweets)")
    st.bar_chart(df_tweets["sentiment"].value_counts())
    st.subheader("Sample Tweet Records")
    st.write(df_tweets.sample(5, random_state=42))

with support_col:
    st.header("Support Dashboard")
    st.dataframe(df_support, use_container_width=True)
    st.subheader("Sentiment Distribution (Support)")
    st.bar_chart(df_support["sentiment"].value_counts())
    st.subheader("Sample Support Records")
    st.write(df_support.sample(5, random_state=42))