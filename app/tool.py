# app/tool.py

import warnings
import hashlib

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
                "You are an expert customer support topic classifier. "
                "Given the following customer support message, choose exactly one topic from "
                "['facturación', 'conectividad', 'cuenta', 'otro'] and return it in JSON format as "
                "{{\"topic\": \"chosen_topic\"}}.\n\n"
                "Message:\n{text}"
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





from typing import ClassVar

class SentimentToolEn(BaseTool):
    name: str = "sentiment_tool_en"
    description: str = "Clasifica el sentimiento de un texto en inglés y devuelve JSON."
    _prompt:    PromptTemplate = PrivateAttr()
    _pipeline:  Any            = PrivateAttr()  # RunnableSequence debajo
    CACHE_PATH: ClassVar[str] = "cache/processed_tweets.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are a sentiment analysis assistant. Analyze the sentiment of the following text "
                "and return a JSON object with two fields:\n"
                "- sentiment: one of 'positive', 'neutral', or 'negative'\n"
                "- score: a confidence score between 0.0 and 1.0\n\n"
                "Example:\n"
                "{{\"sentiment\": \"negative\", \"score\": 0.23}}\n\n"
                "Text:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None) -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Check cache if record_id provided
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "sentiment" in record:
                                # Clean sentiment from cache as well
                                sentiment_clean = (
                                    record["sentiment"].strip().lower().replace("sentiment:", "").strip()
                                    if isinstance(record["sentiment"], str) else record["sentiment"]
                                )
                                return {"sentiment": sentiment_clean, "score": record.get("score")}
                        except Exception:
                            continue
        raw = self._pipeline.invoke({"text": text})
        try:
            result = json.loads(raw)
            # Clean sentiment value if present
            if "sentiment" in result and isinstance(result["sentiment"], str):
                result["sentiment"] = result["sentiment"].strip().lower().replace("sentiment:", "").strip()
                # Remove extra newlines or words
                result["sentiment"] = result["sentiment"].split("\n")[0].strip()
        except Exception:
            sentiment, score = None, None
            for line in raw.strip().splitlines():
                lower = line.lower()
                if lower.startswith("sentiment"):
                    parts = line.split(":", 1)
                    # Remove prefix and clean
                    sentiment = (
                        parts[1].strip().lower().replace("sentiment:", "").strip()
                        if len(parts) > 1 else None
                    )
                    if sentiment:
                        sentiment = sentiment.split("\n")[0].strip()
                if lower.startswith("score"):
                    parts = line.split(":", 1)
                    try:
                        score = float(parts[1].strip())
                    except Exception:
                        score = None
            if sentiment is not None:
                result = {"sentiment": sentiment, "score": score}
            else:
                # Clean the entire raw as fallback, remove prefix and newlines
                sentiment = raw.strip().lower().replace("sentiment:", "").strip()
                sentiment = sentiment.split("\n")[0].strip()
                result = {"sentiment": sentiment, "score": None}
        if record_id:
            result["id"] = record_id
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None) -> Dict[str, Any]:
        return self._run(text, record_id=record_id)



class IntentToolEn(BaseTool):
    name: str = "intent_tool_en"
    description: str = "Clasifica la intención de un texto en inglés."
    _prompt:    PromptTemplate = PrivateAttr()
    _pipeline:  Any            = PrivateAttr()
    CACHE_PATH: ClassVar[str] = "cache/processed_tweets.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are an intent classification assistant. For the following text, select exactly one intent "
                "from ['question', 'complaint', 'request', 'feedback', 'other'] and return it in JSON format as "
                "{{\"intent\": \"chosen_intent\"}}.\n\n"
                "Text:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None) -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "intent" in record:
                                return {"intent": record["intent"]}
                        except Exception:
                            continue
        raw = self._pipeline.invoke({"text": text}).strip()
        try:
            parsed = json.loads(raw)
            intent_value = parsed.get("intent", "").strip().lower()
        except Exception:
            intent_value = raw.lower()
        result = {"intent": intent_value}
        if record_id:
            to_store = {"id": record_id, "intent": intent_value}
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_store, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None) -> Dict[str, Any]:
        return self._run(text, record_id=record_id)


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
            template=(
                "You are a language detection system. Identify the language of the following text and return a JSON object "
                "with the ISO 639-1 code as {{{{\"language\": \"<code>\"}}}}.\n\n"
                "Text:\n{text}"
            )
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
    CACHE_PATH: ClassVar[str] = "cache/processed_support.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are an urgency assessment model. Determine the urgency level of the following message "
                "as one of ['baja', 'media', 'alta', 'crítica'] and return JSON as {{\"urgency\": \"level\"}}.\n\n"
                "Message:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None) -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "urgency" in record:
                                return {"urgency": record["urgency"]}
                        except Exception:
                            continue
        raw = self._pipeline.invoke({"text": text})
        try:
            result = json.loads(raw)
        except Exception:
            m = re.search(r"(alta|media|baja)", raw.lower())
            urgency = m.group(1) if m else raw.strip().lower()
            result = {"urgency": urgency}
        if record_id:
            to_store = {"id": record_id, "urgency": result["urgency"]}
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_store, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None) -> Dict[str, Any]:
        return self._run(text, record_id=record_id)

# 3) Emotion analysis

class EmotionAnalysisTool(BaseTool):
    name: str = "emotion_analysis"
    description: str = "Clasifica la emoción principal: anger, joy, sadness o surprise."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()
    CACHE_PATH: ClassVar[str] = "cache/processed_support.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are an emotion detection assistant. Identify the primary emotion in the following message "
                "from the set ['anger', 'joy', 'sadness', 'surprise', 'fear', 'disgust', 'neutral'] and return JSON "
                "as {{\"emotion\": \"chosen_emotion\"}}.\n\n"
                "Message:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None) -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "emotion" in record:
                                return {"emotion": record["emotion"]}
                        except Exception:
                            continue
        raw = self._pipeline.invoke({"text": text}).strip()
        try:
            parsed = json.loads(raw)
            emotion_value = parsed.get("emotion", "").strip().lower()
        except Exception:
            emotion_value = raw.lower()
        result = {"emotion": emotion_value}
        if record_id:
            to_store = {"id": record_id, "emotion": emotion_value}
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_store, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None) -> Dict[str, Any]:
        return self._run(text, record_id=record_id)


# 5) Support area assignment (reemplaza extracción de keywords)

class SupportAreaAssignmentTool(BaseTool):
    name: str = "assign_support_area"
    description: str = "Asigna el área de soporte responsable del texto: facturación, técnico, ventas, atención al cliente, otro."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any = PrivateAttr()
    CACHE_PATH: ClassVar[str] = "cache/processed_support.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are a customer support routing assistant. Based on the following message, choose the correct support area "
                "from ['facturación', 'técnico', 'ventas', 'atención al cliente', 'otro'] and return JSON "
                "as {{\"support_area\": \"area\"}}.\n\n"
                "Message:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None) -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "support_area" in record:
                                return {"support_area": record["support_area"]}
                        except Exception:
                            continue
        area = self._pipeline.invoke({"text": text}).strip().lower()
        valid = {"facturación", "técnico", "ventas", "atención al cliente", "otro"}
        area_final = area if area in valid else "otro"
        result = {"support_area": area_final}
        if record_id:
            to_store = {"id": record_id, "support_area": area_final}
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_store, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None) -> Dict[str, Any]:
        return self._run(text, record_id=record_id)

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

# Sarcasm detection tool
class SarcasmDetectionTool(BaseTool):
    name: str = "detect_sarcasm"
    description: str = "Detecta si un texto contiene sarcasmo o ironía. Devuelve JSON {'sarcasm': true|false}."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any         = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are a sarcasm detection assistant. Analyze the following text and determine whether it "
                "contains sarcasm or irony. Return a JSON object with a boolean field 'sarcasm'. "
                "Example: {{\"sarcasm\": true}} if sarcastic, {{\"sarcasm\": false}} otherwise.\n\n"
                "Text:\n{text}"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str) -> Dict[str, Any]:
        raw = self._pipeline.invoke({"text": text}).strip()
        try:
            parsed = json.loads(raw)
            sarcasm_flag = bool(parsed.get("sarcasm", False))
        except Exception:
            # fallback: simple keyword check
            sarcasm_flag = "sure" in text.lower() or "yeah right" in text.lower()
        return {"sarcasm": sarcasm_flag}

    async def _arun(self, text: str) -> Dict[str, Any]:
        return self._run(text)

# 10) Response suggestion

class ResponseSuggestionTool(BaseTool):
    name: str = "suggest_response"
    description: str = "Genera un borrador de respuesta para un registro dado."
    _prompt: PromptTemplate = PrivateAttr()
    _pipeline: Any        = PrivateAttr()
    CACHE_PATH: ClassVar[str] = "cache/processed_support.jsonl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = PromptTemplate(
            input_variables=["text", "user", "language"],
            template=(
                "You are a customer support assistant. "
                "Generate a polite and concise response in the same language as the original message, which is {language}. "
                "Address user {user} directly and provide helpful information.\n\n"
                "Original message:\n{text}\n\n"
                "Response:"
            )
        )
        self._pipeline = self._prompt | llm

    def _run(self, text: str, record_id: str = None, user: str = "", language: str = "") -> Dict[str, Any]:
        # Use text hash as record_id if none provided
        if record_id is None:
            record_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record_id:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("id") == record_id and "response_draft" in record:
                                return {"response_draft": record["response_draft"]}
                        except Exception:
                            continue
        draft = self._pipeline.invoke({"text": text, "user": user, "language": language}).strip()
        result = {"response_draft": draft}
        if record_id:
            to_store = {"id": record_id, "response_draft": draft}
            with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_store, ensure_ascii=False) + "\n")
        return result

    async def _arun(self, text: str, record_id: str = None, user: str = "", language: str = "") -> Dict[str, Any]:
        return self._run(text, record_id=record_id, user=user, language=language)