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
from collections import Counter
from dateutil import parser

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
    LanguageDetectionTool,
    UrgencyAssessmentTool,
    EmotionAnalysisTool,
    SupportAreaAssignmentTool,
    DuplicateDetectionTool,
    ResponseSuggestionTool,
    SarcasmDetectionTool,
    SummaryTool,
    EntityRecognitionTool,
)

# Descargar stopwords una sola vez
nltk.download("stopwords", quiet=True)
SW_EN = set(nltk.corpus.stopwords.words("english"))
print("USANDO AGENT.PY ACTUALIZADO")
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
    size = state.get("tweets_sample_size", 10)
    tweets = KaggleIngestionTool(sample_size=size).run(state["tweets_path"])
    print(f">> [ingest_tweets] loaded {len(tweets)} tweets (limit={size})")
    return {"tweets": tweets}

def preprocess_tweets(state: OmniState) -> Dict[str, Any]:
    pre = PreprocessingTool(stopwords=SW_EN)
    cleaned = [{**d, "texto": pre.run(d["texto"])} for d in state["tweets"]]
    print(f">> [preprocess_tweets] cleaned {len(cleaned)} tweets")
    return {"tweets": cleaned}

def analyze_tweets(state: OmniState) -> Dict[str, Any]:
    out = []
    for d in state["tweets"]:
        # Run sentiment and intent with caching
        s = SentimentToolEn().run(d["texto"], record_id=d["id"])
        i = IntentToolEn().run(d["texto"], record_id=d["id"])
        # Assemble full record with all fields
        # Run emotion, urgency, support area and suggestion
        e = EmotionAnalysisTool().run(d["texto"], record_id=d["id"])
        u = UrgencyAssessmentTool().run(d["texto"], record_id=d["id"])
        sa = SupportAreaAssignmentTool().run(d["texto"], record_id=d["id"])
        sug = ResponseSuggestionTool().run(
            json.dumps(d, ensure_ascii=False),
            record_id=d["id"],
            user=d.get("user", ""),
            language=d.get("language", "")
        )
        # Assemble full record including all fields
        out.append({
            "id": d["id"],
            "canal": d.get("canal"),
            "user": d.get("user"),
            "timestamp": d.get("timestamp"),
            "query": d.get("query"),
            "texto": d.get("texto"),
            "language": d.get("language"),
            "sarcasm": d.get("sarcasm", False),
            "sentiment": s.get("sentiment"),
            "score": s.get("score"),
            "intent": i.get("intent"),
            "emotion": e.get("emotion"),
            "urgency": u.get("urgency"),
            "support_area": sa.get("support_area", "unknown"),
            "suggestion": sug.get("response_draft")
        })
    print(f">> [analyze_tweets] created {len(out)} tweet records")
    return {"tweets_records": out}

def dedupe_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Elimina registros duplicados usando DuplicateDetectionTool.
    """
    recs = state.get("tweets_records", [])
    deduper = DuplicateDetectionTool()
    unique_recs = []
    seen_ids = set()
    for rec in recs:
        if rec["id"] in seen_ids:
            continue
        # Assuming DuplicateDetectionTool returns similarity score or boolean
        # Here we just add all unique ids, real logic could be more complex
        unique_recs.append(rec)
        seen_ids.add(rec["id"])
    print(f">> [dedupe_tweets] filtered {len(recs)} to {len(unique_recs)} unique tweets")
    return {"tweets_records": unique_recs}

# ─── Soporte pipeline ───────────────────────────────────────
def ingest_support(state: OmniState) -> Dict[str, Any]:
    print(">> [ingest_support] start, state keys:", list(state.keys()))
    size = state.get("support_sample_size", 10)
    support = SupportIngestionTool(sample_size=size).run(state["support_path"])
    print(f">> [ingest_support] loaded {len(support)} support records (limit={size})")
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
        s = SentimentToolEn().run(d["texto"], record_id=d["id"])
        i = IntentToolEn().run(d["texto"], record_id=d["id"])
        rec = {
            "id": d["id"],
            "canal": d["canal"],
            "author_id": d.get("author_id"),
            "timestamp": d.get("timestamp"),
            "inbound": d.get("inbound"),
            "response_tweet_id": d.get("response_tweet_id"),
            "in_response_to_tweet_id": d.get("in_response_to_tweet_id"),
            "language": d.get("language"),
            "sarcasm": d.get("sarcasm", False),
            "topic": d["topic"],
            "texto": d.get("texto"),
            "sentiment": s.get("sentiment"),
            "score": s.get("score"),
            "intent": i.get("intent"),
        }
        out.append(rec)
    print(f">> [analyze_support] created {len(out)} support records")
    return {"support_records": out}

def dedupe_support(state: OmniState) -> Dict[str, Any]:
    """
    Elimina registros duplicados usando DuplicateDetectionTool.
    """
    recs = state.get("support_records", [])
    deduper = DuplicateDetectionTool()
    unique_recs = []
    seen_ids = set()
    for rec in recs:
        if rec["id"] in seen_ids:
            continue
        # Assuming DuplicateDetectionTool returns similarity score or boolean
        # Here we just add all unique ids, real logic could be more complex
        unique_recs.append(rec)
        seen_ids.add(rec["id"])
    print(f">> [dedupe_support] filtered {len(recs)} to {len(unique_recs)} unique support records")
    return {"support_records": unique_recs}

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
    # Ensure the cleaned text field is present
    if "texto" not in df.columns and "texto" in state.get("support", [{}])[0]:
        df["texto"] = pd.DataFrame(state.get("support", []))["texto"]
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

def detect_sarcasm_tweets(state: OmniState) -> Dict[str, Any]:
    tweets = state.get("tweets", [])
    for d in tweets:
        d["sarcasm"] = SarcasmDetectionTool().run(d.get("texto", ""))["sarcasm"]
    print(f">> [detect_sarcasm_tweets] detected sarcasm for {len(tweets)} tweets")
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

def detect_sarcasm_support(state: OmniState) -> Dict[str, Any]:
    support = state.get("support", [])
    for d in support:
        d["sarcasm"] = SarcasmDetectionTool().run(d.get("texto", ""))["sarcasm"]
    print(f">> [detect_sarcasm_support] detected sarcasm for {len(support)} support records")
    return {"support": support}


# --- New functions ---
def sentiment_trend_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Calcula la proporción de sentimientos en el lote de tweets y alerta si más de 50% son negativos.
    """
    recs = state.get("tweets_records", [])
    counts = Counter(r.get("sentiment") for r in recs)
    total = len(recs) or 1
    trend = {sent: counts.get(sent, 0) / total for sent in ("positive", "neutral", "negative")}
    alert = trend.get("negative", 0) > 0.5
    print(f">> [sentiment_trend_tweets] trend={trend}, alert={alert}")
    return {"sentiment_trend": trend, "sentiment_alert": alert}

def measure_response_time_support(state: OmniState) -> Dict[str, Any]:
    """
    Calcula el tiempo de respuesta en minutos para cada ticket de soporte.
    """
    raw = state.get("support", [])
    time_map = {d["id"]: parser.parse(d["timestamp"]) for d in raw if d.get("timestamp")}
    recs = state.get("support_records", [])
    for rec in recs:
        req_id = rec.get("in_response_to_tweet_id")
        if req_id and req_id in time_map and rec.get("timestamp"):
            t_req = time_map[req_id]
            t_resp = parser.parse(rec["timestamp"])
            rec["response_time_minutes"] = (t_resp - t_req).total_seconds() / 60
        else:
            rec["response_time_minutes"] = None
    print(f">> [measure_response_time_support] computed response times for {len(recs)} records")
    return {"support_records": recs}

# --- summarize/extract entities functions ---
def summarize_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Genera un resumen breve (1-2 frases) para cada tweet.
    """
    recs = state.get("tweets_records", [])
    for r in recs:
        r["summary"] = SummaryTool().run(r.get("texto", ""))["summary"]
    print(f">> [summarize_tweets] summarized {len(recs)} tweets")
    return {"tweets_records": recs}

def extract_entities_tweets(state: OmniState) -> Dict[str, Any]:
    """
    Extrae entidades nombradas de cada tweet.
    """
    recs = state.get("tweets_records", [])
    for r in recs:
        r["entities"] = EntityRecognitionTool().run(r.get("texto", ""))["entities"]
    print(f">> [extract_entities_tweets] extracted entities for {len(recs)} tweets")
    return {"tweets_records": recs}

def summarize_support(state: OmniState) -> Dict[str, Any]:
    """
    Genera un resumen breve (1-2 frases) para cada ticket de soporte.
    """
    recs = state.get("support_records", [])
    for r in recs:
        r["summary"] = SummaryTool().run(r.get("texto", ""))["summary"]
    print(f">> [summarize_support] summarized {len(recs)} support records")
    return {"support_records": recs}

def extract_entities_support(state: OmniState) -> Dict[str, Any]:
    """
    Extrae entidades nombradas de cada ticket de soporte.
    """
    recs = state.get("support_records", [])
    for r in recs:
        r["entities"] = EntityRecognitionTool().run(r.get("texto", ""))["entities"]
    print(f">> [extract_entities_support] extracted entities for {len(recs)} support records")
    return {"support_records": recs}

def load_cache(path):
    """
    Carga un JSONL cacheando por id, pero mergeando entradas
    sucesivas para el mismo id en lugar de sobrescribirlas.
    """
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_id = rec.get("id")
            if not rec_id:
                continue
            if rec_id in cache:
                # añade/actualiza campos, sin borrar los previos
                cache[rec_id].update(rec)
            else:
                cache[rec_id] = rec
    return cache

def append_to_cache(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def enrich_tweets(state: OmniState) -> Dict[str, Any]:
    cache_path = "cache/processed_tweets.jsonl"
    cache = load_cache(cache_path)
    recs = state.get("tweets_records", [])
    enriched = []
    for r in recs:
        if r["id"] in cache:
            enriched.append(cache[r["id"]])
            continue
        text = r.get("texto", "")
        r2 = r.copy()
        # Preserve original text for dashboard
        r2["texto"] = text
        r2["emotion"]    = EmotionAnalysisTool().run(text, record_id=r["id"])["emotion"]
        r2["urgency"]    = UrgencyAssessmentTool().run(text, record_id=r["id"])["urgency"]
        support_area_result = SupportAreaAssignmentTool().run(text, record_id=r["id"])
        if isinstance(support_area_result, dict):
            r2["support_area"] = support_area_result.get("support_area", "unknown")
        else:
            r2["support_area"] = "unknown"
        r2["suggestion"] = ResponseSuggestionTool().run(
            json.dumps(r, ensure_ascii=False),
            record_id=r["id"],
            user=r.get("user", ""),
            language=r.get("language", "")
        )["response_draft"]
        enriched.append(r2)
        append_to_cache(cache_path, r2)
    print(f">> [enrich_tweets] enriched {len(enriched)} tweets")
    return {"tweets_records": enriched}

def enrich_support(state: OmniState) -> Dict[str, Any]:
    cache_path = "cache/processed_support.jsonl"
    cache = load_cache(cache_path)
    recs = state.get("support_records", [])
    enriched = []
    for r in recs:
        if r["id"] in cache:
            enriched.append(cache[r["id"]])
            continue
        text = r.get("texto", "")
        r2 = r.copy()
        # Preserve original text for dashboard
        r2["texto"] = text
        r2["emotion"]    = EmotionAnalysisTool().run(text, record_id=r["id"])["emotion"]
        r2["urgency"]    = UrgencyAssessmentTool().run(text, record_id=r["id"])["urgency"]
        r2["support_area"] = SupportAreaAssignmentTool().run(text, record_id=r["id"]).get("support_area", "unknown")
        r2["suggestion"] = ResponseSuggestionTool().run(
            json.dumps(r, ensure_ascii=False),
            record_id=r["id"],
            user=r.get("author_id", ""),
            language=r.get("language", "")
        )["response_draft"]
        enriched.append(r2)
        append_to_cache(cache_path, r2)
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

#
# Tweets pipeline
builder.add_node("ingest_tweets",    ingest_tweets)
builder.add_node("detect_language_tweets", detect_language_tweets)
builder.add_node("detect_sarcasm_tweets", detect_sarcasm_tweets)
builder.add_edge("detect_language_tweets", "detect_sarcasm_tweets")
builder.add_edge("detect_sarcasm_tweets", "pre_tweets")
builder.add_node("pre_tweets",       preprocess_tweets)
builder.add_node("an_tweets",        analyze_tweets)
builder.add_node("dedupe_tweets",    dedupe_tweets)
builder.add_node("enrich_tweets", enrich_tweets)
# --- Insert sentiment_trend_tweets node after enrich_tweets ---
builder.add_node("sentiment_trend_tweets", sentiment_trend_tweets)
builder.add_edge("enrich_tweets", "sentiment_trend_tweets")
builder.add_edge("sentiment_trend_tweets", "alert_tweets")
builder.add_node("alert_tweets",     alert_tweets)
builder.add_node("summarize_tweets", summarize_tweets)
builder.add_edge("alert_tweets", "summarize_tweets")
builder.add_node("extract_entities_tweets", extract_entities_tweets)
builder.add_edge("summarize_tweets", "extract_entities_tweets")
builder.add_edge("extract_entities_tweets", "dashboard_tweets")
builder.add_node("dashboard_tweets", dashboard_tweets)

#
# Support pipeline
builder.add_node("ingest_support",    ingest_support)
builder.add_node("detect_language_support", detect_language_support)
builder.add_node("detect_sarcasm_support", detect_sarcasm_support)
builder.add_edge("detect_language_support", "detect_sarcasm_support")
builder.add_edge("detect_sarcasm_support", "pre_support")
builder.add_node("pre_support",       preprocess_support)
builder.add_node("topic_support",     topic_support)
builder.add_node("an_support",        analyze_support)
builder.add_node("dedupe_support",    dedupe_support)
builder.add_node("enrich_support", enrich_support)
# --- Insert measure_response_time_support node after enrich_support ---
builder.add_node("measure_response_time_support", measure_response_time_support)
builder.add_edge("enrich_support", "measure_response_time_support")
builder.add_edge("measure_response_time_support", "alert_support")
builder.add_node("alert_support",     alert_support)
builder.add_node("summarize_support", summarize_support)
builder.add_edge("alert_support", "summarize_support")
builder.add_node("extract_entities_support", extract_entities_support)
builder.add_edge("summarize_support", "extract_entities_support")
builder.add_edge("extract_entities_support", "dashboard_support")
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
builder.add_edge("pre_tweets",     "an_tweets")
builder.add_edge("an_tweets",      "dedupe_tweets")
builder.add_edge("dedupe_tweets",  "enrich_tweets")
# builder.add_edge("enrich_tweets",  "alert_tweets")  # replaced by sentiment_trend_tweets
builder.add_edge("alert_tweets",   "dashboard_tweets")
builder.add_edge("dashboard_tweets", "merge_end")

# Support flow
builder.add_edge("ingest_support",          "detect_language_support")
builder.add_edge("pre_support",      "topic_support")
builder.add_edge("topic_support",    "an_support")
builder.add_edge("an_support",       "dedupe_support")
builder.add_edge("dedupe_support", "enrich_support")
# builder.add_edge("enrich_support", "alert_support")  # replaced by measure_response_time_support
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