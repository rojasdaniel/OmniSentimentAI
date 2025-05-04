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
from typing import Any, List, Dict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Falta la variable OPENAI_API_KEY en el .env")

print("🔑 OPENAI_API_KEY is", "FOUND" if OPENAI_API_KEY else "MISSING")
# LLM global
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
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
            for i, fila in enumerate(reader):
                if i >= self.sample_size:
                    break
                # Sentiment140 schema: [0]=sentiment, [1]=id, [2]=date, [3]=query, [4]=user, [5]=text
                _, tid, date, query, user, text = fila
                registros.append({
                    "id":        tid,
                    "canal":     "twitter",
                    "texto":     text,
                    "timestamp": date,
                    "query":     query,
                    "user":      user,
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
        raw = self._pipeline.invoke({"text": text})
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract "Sentiment: X Score: Y"
            m = re.search(r"Sentiment[:]?\s*([A-Za-z]+).*?Score[:]?\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
            if m:
                sentiment = m.group(1).lower()
                score = float(m.group(2))
            else:
                # fallback single label
                sentiment = raw.strip().lower()
                score = None
            return {"sentiment": sentiment, "score": score}

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


class SupportIngestionTool(BaseTool):
    name: str = "ingest_support"
    description: str = (
        "Lee el CSV de soporte al cliente (twcs.csv) y devuelve registros con:\n"
        "- id (tweet_id)\n"
        "- canal='soporte'\n"
        "- texto (campo text)\n"
        "- timestamp (campo created_at)"
    )
    sample_size: int = 10

    def __init__(self, sample_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "sample_size", sample_size)

    def _run(self, file_path: str) -> List[Dict[str, Any]]:
        registros = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, fila in enumerate(reader):
                if i >= self.sample_size:
                    break
                registros.append({
                    "id":                     fila["tweet_id"],
                    "canal":                  "soporte",
                    "texto":                  fila["text"],
                    "timestamp":              fila["created_at"],
                    "author_id":              fila.get("author_id"),
                    "inbound":                fila.get("inbound"),
                    "response_tweet_id":      fila.get("response_tweet_id"),
                    "in_response_to_tweet_id":fila.get("in_response_to_tweet_id"),
                })
        return registros

    async def _arun(self, file_path: str) -> List[Dict[str, Any]]:
        return self._run(file_path)


# 1) Language detection
class LanguageDetectionTool(BaseTool):
    name: str = "detect_language"
    description: str = "Detecta el idioma de un texto y devuelve su código ISO."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template="Detect the language of the following text and respond with ONLY the ISO 639-1 code (e.g., 'en', 'es'):\n\n{text}"
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        raw = self._pipeline.invoke({"text": text}).strip()
        match = re.search(r"\b[a-z]{2}\b", raw.lower())
        code = match.group(0) if match else raw.lower().split()[0]
        return {"language": code}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)

# 2) Urgency assessment
class UrgencyAssessmentTool(BaseTool):
    name: str = "assess_urgency"
    description: str = "Evalúa la urgencia de un texto. Devuelve JSON {'urgency':'alta|media|baja'}."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Based on the following text, assess its urgency level. "
                "Return only JSON: {{\"urgency\":\"alta|media|baja\"}}.\n\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        raw = self._pipeline.invoke({"text": text})
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback when model returns only the label
            urgency = raw.strip().strip('"')
            return {"urgency": urgency}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)

# 3) Emotion analysis
class EmotionAnalysisTool(BaseTool):
    name: str = "emotion_analysis"
    description: str = "Clasifica la emoción principal: anger, joy, sadness o surprise."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Classify the primary emotion of this text. "
                "Return only one of: anger, joy, sadness, surprise.\n\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        emotion = self._pipeline.invoke({"text": text}).strip().lower()
        return {"emotion": emotion}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)

# 5) Keyword extraction
class KeywordExtractorTool(BaseTool):
    name: str = "extract_keywords"
    description: str = "Extrae las palabras clave más importantes de un texto."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Given the following text, extract the most relevant key phrases or keywords "
                "that summarize its main content. "
                "Return ONLY a JSON array of strings (e.g., [\"phrase1\", \"phrase2\"]).\n\n"
                "Text:\n\n"
                "{text}"
            )
        )
        # Model should not echo instructions, only output JSON list
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        raw = self._pipeline.invoke({"text": text})
        try:
            kws = json.loads(raw)
        except json.JSONDecodeError:
            # Try fixing single quotes
            try:
                kws = json.loads(raw.replace("'", '"'))
            except Exception:
                # As last resort, split on commas/brackets
                kws = [w.strip() for w in re.split(r"[,\[\]]+", raw) if w.strip()]
        return {"keywords": kws}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)

# 7) Duplicate detection
class DuplicateDetectionTool(BaseTool):
    name: str = "detect_duplicate"
    description: str = (
        "Detecta si un texto es duplicado de un listado previo. "
        "Recibe 'batch_json|||text' y devuelve {'duplicate': True|False}."
    )

    def _run(self, args: str) -> Dict[str, Any]:
        batch_json, text = args.split("|||", 1)
        records = [json.loads(line) for line in batch_json.splitlines()]
        texts = [r.get("texto","") for r in records]
        return {"duplicate": text in texts}

    async def _arun(self, args: str) -> Dict[str, Any]:
        return self._run(args)

# 10) Response suggestion
class ResponseSuggestionTool(BaseTool):
    name: str = "suggest_response"
    description: str = "Genera un borrador de respuesta para un registro dado."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template="Generate a polite and concise response to the following message:\n\n{text}"
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        draft = self._pipeline.invoke({"text": text}).strip()
        return {"response_draft": draft}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)