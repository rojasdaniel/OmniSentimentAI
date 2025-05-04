# app/agent.py

import json
import nltk
import os
import pandas as pd
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

from app.tool import (
    KaggleIngestionTool,
    SupportIngestionTool,
    PreprocessingTool,
    TopicToolEn,
    SentimentToolEn,
    IntentToolEn,
    AlertTool,
    DashboardTool,
    LanguageDetectionTool,
    UrgencyAssessmentTool,
    EmotionAnalysisTool,
    KeywordExtractorTool,
    DuplicateDetectionTool,
    ResponseSuggestionTool,
)

# Descargar stopwords una sola vez
nltk.download("stopwords", quiet=True)
SW_EN = set(nltk.corpus.stopwords.words("english"))

# Definición del estado que atraviesa el grafo
class OmniState(TypedDict, total=False):
    tweets_path: str
    support_path: str
    tweets_sample_size: int    # número máximo de tuits a procesar
    support_sample_size: int   # número máximo de tickets a procesar
    tweets: List[Dict[str, Any]]
    tweets_records: List[Dict[str, Any]]
    tweets_dashboard: str
    support: List[Dict[str, Any]]
    support_records: List[Dict[str, Any]]
    support_dashboard: str

def chat_node(state: OmniState) -> Dict[str, Any]:
    """
    Nodo de chat: usa state['chat'] y las rutas
    state['tweets_dashboard'], state['support_dashboard'] para
    cargar KB, luego responde en 'chat_response'.
    """
    history = state.get("chat", [])
    if not history:
        return {}

    # Carga KB de ambos dashboards
    tweets_csv  = state.get("tweets_dashboard", "tweets_dashboard.csv")
    support_csv = state.get("support_dashboard", "support_dashboard.csv")
    try:
        df_t = pd.read_csv(tweets_csv)
        df_s = pd.read_csv(support_csv)
        df_kb = pd.concat([df_t, df_s], ignore_index=True)
        kb_excerpt = df_kb.head(20).to_csv(index=False)
    except Exception as e:
        kb_excerpt = f"Error loading dashboards: {e}"

    # Último mensaje del usuario
    last_user = [m for m in history if m.get("role") == "user"][-1]["content"]
    prompt = (
        "Eres un asistente que responde consultando estos datos CSV:\n"
        f"{kb_excerpt}\n\n"
        f"Pregunta: {last_user}\n"
        "Responde basándote en esos datos."
    )

    llm_chat = ChatOpenAI(temperature=0)
    response = llm_chat([HumanMessage(content=prompt)])
    return {"chat_response": response.content}


# ─── Tweets pipeline ────────────────────────────────────────
def ingest_tweets(state: OmniState) -> Dict[str, Any]:
    print(">> [ingest_tweets] start, state keys:", list(state.keys()))
    tweets = KaggleIngestionTool(sample_size=10).run(state["tweets_path"])
    print(f">> [ingest_tweets] loaded {len(tweets)} tweets (limit=10)")
    return {"tweets": tweets}

def preprocess_tweets(state: OmniState) -> Dict[str, Any]:
    pre = PreprocessingTool(stopwords=SW_EN)
    cleaned = [{**d, "texto": pre.run(d["texto"])} for d in state["tweets"]]
    print(f">> [preprocess_tweets] cleaned {len(cleaned)} tweets")
    return {"tweets": cleaned}

def analyze_tweets(state: OmniState) -> Dict[str, Any]:
    out = []
    for d in state["tweets"]:
        s = SentimentToolEn().run(d["texto"])
        i = IntentToolEn().run(d["texto"])
        out.append({"id": d["id"], "canal": d["canal"], **s, **i})
    print(f">> [analyze_tweets] created {len(out)} tweet records")
    return {"tweets_records": out}

# ─── Soporte pipeline ───────────────────────────────────────
def ingest_support(state: OmniState) -> Dict[str, Any]:
    print(">> [ingest_support] start, state keys:", list(state.keys()))
    support = SupportIngestionTool(sample_size=10).run(state["support_path"])
    print(f">> [ingest_support] loaded {len(support)} support records (limit=10)")
    return {"support": support}

def preprocess_support(state: OmniState) -> Dict[str, Any]:
    pre = PreprocessingTool(stopwords=SW_EN)
    cleaned = [{**d, "texto": pre.run(d["texto"])} for d in state["support"]]
    print(f">> [preprocess_support] cleaned {len(cleaned)} support records")
    return {"support": cleaned}

def topic_support(state: OmniState) -> Dict[str, Any]:
    with_topic = [{**d, "topic": TopicToolEn().run(d["texto"])["topic"]} for d in state["support"]]
    return {"support": with_topic}

def analyze_support(state: OmniState) -> Dict[str, Any]:
    out = []
    for d in state["support"]:
        s = SentimentToolEn().run(d["texto"])
        i = IntentToolEn().run(d["texto"])
        rec = {"id": d["id"], "canal": d["canal"], "topic": d["topic"], **s, **i}
        out.append(rec)
    print(f">> [analyze_support] created {len(out)} support records")
    return {"support_records": out}

def alert_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Dispara AlertTool para cada tweet negativo + queja.
    """
    for rec in state.get("tweets_records", []):
        AlertTool().run(json.dumps(rec, ensure_ascii=False))
    return {}

def dashboard_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Genera tweets_dashboard.csv con los registros de tweets.
    """
    df = pd.DataFrame(state.get("tweets_records", []))
    path = "tweets_dashboard.csv"
    df.to_csv(path, index=False)
    return {"tweets_dashboard": path}

def alert_support(state: OmniState) -> Dict[str, Any]:
    """
    Dispara AlertTool para cada ticket de soporte negativo + queja.
    """
    for rec in state.get("support_records", []):
        AlertTool().run(json.dumps(rec, ensure_ascii=False))
    return {}

def dashboard_support(state: OmniState) -> Dict[str, Any]:
    """
    Genera support_dashboard.csv con los registros de soporte.
    """
    df = pd.DataFrame(state.get("support_records", []))
    path = "support_dashboard.csv"
    df.to_csv(path, index=False)
    return {"support_dashboard": path}

def detect_language_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Detecta el idioma de cada tweet y lo anota en state['tweets'].
    """
    tweets = state.get("tweets", [])
    for d in tweets:
        lang = LanguageDetectionTool().run(d.get("texto", ""))["language"]
        d["language"] = lang
    print(f">> [detect_language_tweets] detected languages for {len(tweets)} tweets")
    return {"tweets": tweets}

def detect_language_support(state: OmniState) -> Dict[str, Any]:
    """
    Detecta el idioma de cada registro de soporte y lo anota en state['support'].
    """
    support = state.get("support", [])
    for d in support:
        lang = LanguageDetectionTool().run(d.get("texto", ""))["language"]
        d["language"] = lang
    print(f">> [detect_language_support] detected languages for {len(support)} support records")
    return {"support": support}

# ─── Enrichment nodes ──────────────────────────────────────
def enrich_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Aplica detección de idioma, emoción, urgencia, extracción de keywords
    y sugerencia de respuesta a cada registro de tweets.
    """
    recs = state.get("tweets_records", [])
    enriched = []
    for r in recs:
        text = r.get("texto", "")
        r2 = r.copy()
        r2["language"]   = LanguageDetectionTool().run(text)["language"]
        r2["emotion"]    = EmotionAnalysisTool().run(text)["emotion"]
        r2["urgency"]    = UrgencyAssessmentTool().run(text)["urgency"]
        r2["keywords"]   = KeywordExtractorTool().run(text)["keywords"]
        r2["suggestion"] = ResponseSuggestionTool().run(json.dumps(r, ensure_ascii=False))["response_draft"]
        enriched.append(r2)
    print(f">> [enrich_tweets] enriched {len(enriched)} tweets")
    return {"tweets_records": enriched}

def enrich_support(state: OmniState) -> Dict[str, Any]:
    """
    Aplica detección de idioma, emoción, urgencia, extracción de keywords
    y sugerencia de respuesta a cada ticket de soporte.
    """
    recs = state.get("support_records", [])
    enriched = []
    for r in recs:
        text = r.get("texto", "")
        r2 = r.copy()
        r2["language"]   = LanguageDetectionTool().run(text)["language"]
        r2["emotion"]    = EmotionAnalysisTool().run(text)["emotion"]
        r2["urgency"]    = UrgencyAssessmentTool().run(text)["urgency"]
        r2["keywords"]   = KeywordExtractorTool().run(text)["keywords"]
        r2["suggestion"] = ResponseSuggestionTool().run(json.dumps(r, ensure_ascii=False))["response_draft"]
        enriched.append(r2)
    print(f">> [enrich_support] enriched {len(enriched)} support records")
    return {"support_records": enriched}

def merge_end(state: OmniState) -> Dict[str, Any]:
    """
    Nodo de fusión: espera ambas ramas de dashboard y luego termina.
    """
    print(">> [merge_end] both pipelines completed")
    return {}

# ─── Construcción del grafo ─────────────────────────────────
builder = StateGraph(OmniState)

# Tweets pipeline
builder.add_node("ingest_tweets",    ingest_tweets)
builder.add_node("detect_language_tweets", detect_language_tweets)
builder.add_node("pre_tweets",       preprocess_tweets)
builder.add_node("an_tweets",        analyze_tweets)
builder.add_node("enrich_tweets", enrich_tweets)
builder.add_node("alert_tweets",     alert_tweets)
builder.add_node("dashboard_tweets", dashboard_tweets)

# Support pipeline
builder.add_node("ingest_support",    ingest_support)
builder.add_node("detect_language_support", detect_language_support)
builder.add_node("pre_support",       preprocess_support)
builder.add_node("topic_support",     topic_support)
builder.add_node("an_support",        analyze_support)
builder.add_node("enrich_support", enrich_support)
builder.add_node("alert_support",     alert_support)
builder.add_node("dashboard_support", dashboard_support)
# Graph construction section
builder.add_node("chat", chat_node)
# Merge node
builder.add_node("merge_end", merge_end)

# Define transitions: branch from START to both pipelines
builder.add_edge(START,         "ingest_tweets")
builder.add_edge(START,         "ingest_support")

# Tweets flow
builder.add_edge("ingest_tweets",         "detect_language_tweets")
builder.add_edge("detect_language_tweets","pre_tweets")
builder.add_edge("pre_tweets",     "an_tweets")
builder.add_edge("an_tweets",      "enrich_tweets")
builder.add_edge("enrich_tweets",  "alert_tweets")
builder.add_edge("alert_tweets",   "dashboard_tweets")
builder.add_edge("dashboard_tweets", "merge_end")

# Support flow
builder.add_edge("ingest_support",          "detect_language_support")
builder.add_edge("detect_language_support","pre_support")
builder.add_edge("pre_support",      "topic_support")
builder.add_edge("topic_support",    "an_support")
builder.add_edge("an_support",       "enrich_support")
builder.add_edge("enrich_support", "alert_support")
builder.add_edge("alert_support",  "dashboard_support")
builder.add_edge("dashboard_support", "merge_end")

builder.add_edge("dashboard_tweets", "merge_end")
builder.add_edge("dashboard_support", "merge_end")
builder.add_edge("merge_end", "chat")
builder.add_edge("chat", END)

print("🏗️ Construyendo grafo…")
graph = builder.compile()
print("✅ Grafo listo.")

# Prueba CLI
if __name__ == "__main__":
    state0 = {
        "tweets_path":  "training.1600000.processed.noemoticon.csv",
        "support_path": "twcs.csv",
    }
    final = graph.invoke(state0)
    print("➡️ Tweets dashboard:", final.get("tweets_dashboard"))
    print("➡️ Support dashboard:", final.get("support_dashboard"))