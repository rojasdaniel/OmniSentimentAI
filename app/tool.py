# app/tool.py

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Falta la variable OPENAI_API_KEY en el .env")

import csv
import json
import pandas as pd
import re
from typing import List, Dict, Any

from langchain.tools import BaseTool
from pydantic import PrivateAttr
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Instancia global del LLM
llm: OpenAI = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)


class KaggleIngestionTool(BaseTool):
    name: str = "ingest_kaggle"
    description: str = "Lee el CSV de Sentiment140 y devuelve registros."
    sample_size: int = 1000

    def __init__(self, sample_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "sample_size", sample_size)

    def _run(self, file_path: str) -> List[Dict[str, Any]]:
        registros: List[Dict[str, Any]] = []
        with open(file_path, encoding="latin-1") as f:
            reader = csv.reader(f)
            for i, fila in enumerate(reader):
                if i >= self.sample_size:
                    break
                _, tid, date, *_, text = fila
                registros.append({
                    "id":        tid,
                    "canal":     "twitter",
                    "texto":     text,
                    "timestamp": date,
                })
        return registros

    async def _arun(self, file_path: str) -> List[Dict[str, Any]]:
        return self._run(file_path)


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
    description: str = "Clasifica sentimiento EN (positive|neutral|negative) → JSON."
    _prompt: PromptTemplate = PrivateAttr()
    _chain:  LLMChain        = PrivateAttr()

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
        self._chain = LLMChain(llm=llm, prompt=self._prompt)

    def _run(self, text: str) -> Dict[str, Any]:
        raw = self._chain.run(text=text)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sentiment": raw.strip(), "score": None}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)


class IntentToolEn(BaseTool):
    name: str = "intent_tool_en"
    description: str = "Clasifica intención EN (question|complaint|suggestion|other)."
    _prompt: PromptTemplate = PrivateAttr()
    _chain:  LLMChain        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Given the following text, respond with ONLY one word "
                "(question, complaint, suggestion or other):\n\n{text}"
            ),
        )
        self._chain = LLMChain(llm=llm, prompt=self._prompt)

    def _run(self, text: str) -> Dict[str, Any]:
        intent = self._chain.run(text=text).strip().lower()
        return {"intent": intent}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)


class AlertTool(BaseTool):
    name: str = "alert_tool"
    description: str = "Dispara alerta si JSON tiene sentimiento negativo + queja."

    def _run(self, info_json: str) -> str:
        data = json.loads(info_json)
        if data.get("sentiment") == "negative" and data.get("intent") == "complaint":
            alerta = f"🚨 ALERTA crítica: id={data['id']} score={data['score']}"
            print(alerta)
            return alerta
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