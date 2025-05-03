# app/agent.py

import json
import nltk
import os
import pandas as pd
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END

from app.tool import (
    KaggleIngestionTool,
    SupportIngestionTool,
    PreprocessingTool,
    TopicToolEn,
    SentimentToolEn,
    IntentToolEn,
    AlertTool,
    DashboardTool,
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

# ─── Tweets pipeline ────────────────────────────────────────
def ingest_tweets(state: OmniState) -> Dict[str, Any]:
    print(">> [ingest_tweets] start, state keys:", list(state.keys()))
    n = state.get("tweets_sample_size", 10)
    print(f">> [ingest_tweets] sample_size from state: {n}")
    all_tweets = KaggleIngestionTool(sample_size=n).run(state["tweets_path"])
    truncated = all_tweets[:n]
    print(f">> [ingest_tweets] loaded {len(all_tweets)} tweets, truncated to {len(truncated)}")
    return {"tweets": truncated}

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

def alert_tweets(state: OmniState) -> Dict[str, Any]:
    for rec in state["tweets_records"]:
        AlertTool().run(json.dumps(rec, ensure_ascii=False))
    return {}

def dashboard_tweets(state: OmniState) -> Dict[str, Any]:
    # Generar CSV específico para tweets
    rows = state["tweets_records"]
    df = pd.DataFrame(rows)
    filename = "tweets_dashboard.csv"
    df.to_csv(filename, index=False)
    print(f">> [dashboard_tweets] wrote {len(df)} rows to {filename}")
    return {"tweets_dashboard": filename}


# ─── Soporte pipeline ───────────────────────────────────────
def ingest_support(state: OmniState) -> Dict[str, Any]:
    print(">> [ingest_support] start, state keys:", list(state.keys()))
    n = state.get("support_sample_size", 10)
    print(f">> [ingest_support] sample_size from state: {n}")
    all_support = SupportIngestionTool(sample_size=n).run(state["support_path"])
    truncated = all_support[:n]
    print(f">> [ingest_support] loaded {len(all_support)} support records, truncated to {len(truncated)}")
    return {"support": truncated}

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

def alert_support(state: OmniState) -> Dict[str, Any]:
    for rec in state["support_records"]:
        AlertTool().run(json.dumps(rec, ensure_ascii=False))
    return {}

def dashboard_support(state: OmniState) -> Dict[str, Any]:
    # Generar CSV específico para soporte
    rows = state["support_records"]
    df = pd.DataFrame(rows)
    filename = "support_dashboard.csv"
    df.to_csv(filename, index=False)
    print(f">> [dashboard_support] wrote {len(df)} rows to {filename}")
    return {"support_dashboard": filename}


# ─── Construcción del grafo ─────────────────────────────────
builder = StateGraph(OmniState)

# Tweets branch
builder.add_node("ingest_tweets",   ingest_tweets)
builder.add_node("pre_tweets",      preprocess_tweets)
builder.add_node("an_tweets",       analyze_tweets)
builder.add_node("alert_tweets",    alert_tweets)
builder.add_node("dash_tweets",     dashboard_tweets)
builder.add_edge(START,             "ingest_tweets")
builder.add_edge("ingest_tweets",   "pre_tweets")
builder.add_edge("pre_tweets",      "an_tweets")
builder.add_edge("an_tweets",       "alert_tweets")
builder.add_edge("alert_tweets",    "dash_tweets")
builder.add_edge("dash_tweets",     END)

# Support branch
builder.add_node("ingest_support",  ingest_support)
builder.add_node("pre_support",     preprocess_support)
builder.add_node("topic_support",   topic_support)
builder.add_node("an_support",      analyze_support)
builder.add_node("alert_support",   alert_support)
builder.add_node("dash_support",    dashboard_support)
builder.add_edge(START,             "ingest_support")
builder.add_edge("ingest_support",  "pre_support")
builder.add_edge("pre_support",     "topic_support")
builder.add_edge("topic_support",   "an_support")
builder.add_edge("an_support",      "alert_support")
builder.add_edge("alert_support",   "dash_support")
builder.add_edge("dash_support",    END)

# Compilar grafo una sola vez
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
    print("➡️ Dashboard guardado en:", final.get("dashboard_path"))