# app/agent.py

import json
import nltk
from typing_extensions import TypedDict
from typing import List, Dict, Any

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from app.tool import (
    KaggleIngestionTool,
    PreprocessingTool,
    SentimentToolEn,
    IntentToolEn,
    AlertTool,
    DashboardTool,
)

# Descargar stopwords (solo una vez)
nltk.download("stopwords", quiet=True)
SW_EN = set(nltk.corpus.stopwords.words("english"))


class OmniState(TypedDict, total=False):
    docs: List[Dict[str, Any]]
    texto: str
    info: Dict[str, Any]
    batch_json: str
    dashboard_path: str


def ingest_node(file_path: str) -> OmniState:
    """
    Lee los primeros 10 tweets del CSV y los guarda en state['docs'].
    """
    docs = KaggleIngestionTool(sample_size=10).run(file_path)
    return {"docs": docs}


def preprocess_node(state: OmniState) -> OmniState:
    """
    Toma state['docs'], limpia URLs, emojis y stopwords; actualiza state['docs'].
    """
    cleaned = []
    preproc = PreprocessingTool(stopwords=SW_EN)
    for doc in state["docs"]:
        doc["texto"] = preproc.run(doc["texto"])
        cleaned.append(doc)
    return {"docs": cleaned}


def analyze_node(state: OmniState) -> OmniState:
    """
    Clasifica sentimiento e intención de cada texto y construye state['batch_json'].
    """
    batch = []
    s_tool = SentimentToolEn()
    i_tool = IntentToolEn()
    for doc in state["docs"]:
        s = s_tool.run(doc["texto"])
        i = i_tool.run(doc["texto"])
        record = {"id": doc["id"], "canal": doc["canal"], **s, **i}
        batch.append(json.dumps(record, ensure_ascii=False))
    return {"batch_json": "\n".join(batch)}


def alert_node(state: OmniState) -> OmniState:
    """
    Dispara AlertTool para cada línea de state['batch_json'] si aplica.
    """
    for rec in state["batch_json"].splitlines():
        AlertTool().run(rec)
    return {}


def dashboard_node(state: OmniState) -> OmniState:
    """
    Genera dashboard_data.csv a partir de state['batch_json'] y devuelve la ruta.
    """
    path = DashboardTool().run(state["batch_json"])
    return {"dashboard_path": path}


# --- Construcción del grafo ---

graph = StateGraph(OmniState)

graph.add_node("ingest",     ToolNode([ingest_node]))
graph.add_node("preprocess", ToolNode([preprocess_node]))
graph.add_node("analyze",    ToolNode([analyze_node]))
graph.add_node("alert",      ToolNode([alert_node]))
graph.add_node("dashboard",  ToolNode([dashboard_node]))

graph.add_edge(START,        "ingest")
graph.add_edge("ingest",     "preprocess")
graph.add_edge("preprocess", "analyze")
graph.add_edge("analyze",    "alert")
graph.add_edge("alert",      "dashboard")
graph.add_edge("dashboard",  END)

if __name__ == "__main__":
    result = graph.compile().invoke({
        "file_path": "training.1600000.processed.noemoticon.csv"
    })
    print("✅ Pipeline completo. Dashboard en:", result["dashboard_path"])