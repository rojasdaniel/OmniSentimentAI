# app/tool.py

import warnings

# ───────────────────────────────────────────────────────────────────────────────
# Ignorar todas las LangChainDeprecationWarning y mensajes “deprecated”
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    module="langchain.*",
)
warnings.filterwarnings(
    "ignore",
    module="langchain_community.*",
)
# ───────────────────────────────────────────────────────────────────────────────

from langchain_openai import OpenAI

import os
import csv
import json
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Falta la variable OPENAI_API_KEY en el .env")

print("🔑 OPENAI_API_KEY is", "FOUND" if OPENAI_API_KEY else "MISSING")
# LLM global
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr
from langchain.prompts import PromptTemplate



class KaggleIngestionTool(BaseTool):
    name: str = "ingest_kaggle"
    description: str = "Lee el CSV de Sentiment140 y devuelve los primeros registros."
    sample_size: int = 10

    def __init__(self, sample_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "sample_size", sample_size)

    def _run(self, file_path: str) -> List[Dict[str, Any]]:
        registros = []
        with open(file_path, encoding="latin-1") as f:
            reader = csv.reader(f)
            for _, tid, date, *_ , text in reader:
                registros.append({
                    "id":        tid,
                    "canal":     "twitter",
                    "texto":     text,
                    "timestamp": date,
                })
        return registros

    async def _arun(self, file_path: str) -> List[Dict[str, Any]]:
        return self._run(file_path)

class TopicToolEn(BaseTool):
    name: str = "topic_tool_en"
    description: str = (
        "Clasifica el tema de un ticket de soporte. "
        "Opciones: facturación, conectividad, cuenta, otro. "
        "Devuelve SÓLO la palabra correspondiente."
    )
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any         = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Dado este ticket de soporte, responde con SOLO UNA PALABRA "
                "(facturación, conectividad, cuenta u otro):\n\n"
                "{text}"
            )
        )
        # creamos el RunnableSequence sin usar LLMChain
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        topic = self._pipeline.invoke({"text": text}).strip().lower()
        # aseguramos que esté en las opciones
        if topic not in {"facturación","conectividad","cuenta","otro"}:
            topic = "otro"
        return {"topic": topic}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)
    
class PreprocessingTool(BaseTool):
    name: str = "preprocess_text"
    description: str = "Limpia URLs, emojis, minúsculas y elimina stopwords."
    _stopwords: set = PrivateAttr()

    def __init__(self, stopwords: set, **kwargs):
        super().__init__(**kwargs)
        self._stopwords = stopwords

    def _run(self, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^\wáéíóúñÑ ]", " ", text)
        tokens = text.lower().split()
        tokens = [t for t in tokens if t not in self._stopwords]
        return " ".join(tokens)

    async def _arun(self, text: str) -> str:
        return self._run(text)




class SentimentToolEn(BaseTool):
    name: str = "sentiment_tool_en"
    description: str = "Clasifica el sentimiento de un texto en inglés y devuelve JSON."
    _prompt:    PromptTemplate = PrivateAttr()
    _pipeline:  Any            = PrivateAttr()  # RunnableSequence debajo

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Classify the sentiment of the following text as JSON:\n"
                "{{\"sentiment\": <positive|neutral|negative>, \"score\": <0-1>}}\n\n"
                "Text: {text}"
            ),
        )
        # Componemos PromptTemplate | llm en un RunnableSequence
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        # Invocamos la secuencia con invoke()
        raw = self._pipeline.invoke({"text": text})
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sentiment": raw.strip(), "score": None}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)


class IntentToolEn(BaseTool):
    name: str = "intent_tool_en"
    description: str = "Clasifica la intención de un texto en inglés."
    _prompt:    PromptTemplate = PrivateAttr()
    _pipeline:  Any            = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Given the following text, respond with ONLY one word:\n"
                "(question, complaint, suggestion or other)\n\n"
                "{text}"
            ),
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        intent = self._pipeline.invoke({"text": text}).strip().lower()
        return {"intent": intent}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)


class AlertTool(BaseTool):
    name: str = "alert_tool"
    description: str = "Dispara alerta si JSON tiene sentimiento negativo + queja."

    def _run(self, info_json: str) -> str:
        data = json.loads(info_json)
        if data.get("sentiment") == "negative" and data.get("intent") == "complaint":
            msg = f"🚨 ALERTA crítica: id={data['id']} score={data['score']}"
            print(msg)
            return msg
        return "OK"

    async def _arun(self, info_json: str) -> str:
        return self._run(info_json)


class DashboardTool(BaseTool):
    name: str = "dashboard_tool"
    description: str = "Construye un DataFrame con resultados y lo vuelca en CSV."

    def _run(self, batch_json: str) -> str:
        rows = [json.loads(rec) for rec in batch_json.splitlines()]
        df = pd.DataFrame(rows)
        df.to_csv("dashboard_data.csv", index=False)
        return "dashboard_data.csv"

    async def _arun(self, batch_json: str) -> str:
        return self._run(batch_json)


from typing import List, Dict, Any

class SupportIngestionTool(BaseTool):
    name: str = "ingest_support"
    description: str = (
        "Lee el CSV de soporte al cliente (twcs.csv) y devuelve registros con:\n"
        "- id (tweet_id)\n"
        "- canal='soporte'\n"
        "- texto (campo text)\n"
        "- timestamp (campo created_at)"
    )

    def _run(self, file_path: str) -> List[Dict[str, Any]]:
        registros = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                registros.append({
                    "id":        fila["tweet_id"],
                    "canal":     "soporte",
                    "texto":     fila["text"],
                    "timestamp": fila["created_at"],
                })
        return registros

    async def _arun(self, file_path: str) -> List[Dict[str, Any]]:
        return self._run(file_path)