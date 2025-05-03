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


# 3) Carga el CSV generado por tu DashboardTool
df = load_data("dashboard_data.csv")

# 4) Título y sección de datos crudos
st.title("OmniSentimentAI Dashboard")
st.subheader("Raw Data")
st.dataframe(df, use_container_width=True)

# 5) Distribuciones lado a lado
sent_col, intent_col = st.columns(2)

with sent_col:
    st.subheader("Sentiment Distribution")
    sent_counts = df["sentiment"].value_counts()
    st.bar_chart(sent_counts)

with intent_col:
    st.subheader("Intent Distribution")
    int_counts = df["intent"].value_counts()
    st.bar_chart(int_counts)

# 6) Muestra algunas métricas o gráficos adicionales
st.subheader("Score over Time (Sample)")
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df_ts = df.set_index("timestamp").resample("H")["score"].mean().dropna()
    st.line_chart(df_ts)

# 7) Tabla de muestra
st.subheader("Sample Records")
st.write(df.sample(5, random_state=42))