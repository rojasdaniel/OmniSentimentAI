# OmniSentimentAI v1.0.0

Un sistema multiagente para análisis de sentimiento y atención al cliente sobre datos de Twitter y soporte.

---

## 📘 Overview

OmniSentimentAI es una aplicación de prueba de concepto (PoC) que ingiere datos de dos fuentes principales (tweets y tickets de soporte), los procesa mediante un grafo de agentes y genera dashboards interactivos. Cada agente (tool) realiza una tarea específica —desde limpieza de texto hasta generación de respuestas— y el resultado se almacena en CSVs que pueden consultarse vía API o Streamlit.

---

## 🚀 Features

### Ingestión de datos  
- **Sentiment140 (Kaggle)**: carga tweets con campos `id`, `timestamp`, `query`, `user`, `texto`.  
- **Customer Support Tweets (TWCS)**: carga tickets con `tweet_id` (→`id`), `created_at` (→`timestamp`), `author_id`, `inbound`, `text` (→`texto`), `response_tweet_id`, `in_response_to_tweet_id`.

### Preprocesado  
- Limpieza de URLs, emojis y caracteres especiales.  
- Eliminación de stopwords en inglés.

### Detección de idioma  
- Reconoce el idioma de cada mensaje y añade `language` (ISO 639-1).

### Análisis básico  
- **SentimentToolEn**: determina `sentiment` (positive/neutral/negative) + `score`.  
- **IntentToolEn**: clasifica `intent` (question/complaint/request/feedback/other).  
- **EmotionAnalysisTool**: extrae `emotion` (anger/joy/sadness/surprise/fear/disgust/neutral).  
- **UrgencyAssessmentTool**: estima `urgency` (baja/media/alta/crítica).

### Enriquecimiento  
- **SupportAreaAssignmentTool**: asigna `support_area` (facturación/técnico/ventas/atención al cliente/otro).  
- **SarcasmDetectionTool**: flag `sarcasm` si el texto es sarcástico.  
- **SummaryTool**: genera `summary` (1–2 frases).  
- **EntityRecognitionTool**: lista `entities` (nombres, fechas, IDs).  
- **DuplicateDetectionTool**: elimina duplicados.

### Métricas adicionales  
- **SentimentTrend**: proporción de sentimientos por batch + alerta si >50% negativos.  
- **ResponseTime**: tiempo de respuesta (`response_time_minutes`) en tickets de soporte.

### Sugerencias automatizadas  
- **ResponseSuggestionTool**: genera un borrador de respuesta en el mismo idioma.

### Output  
- **tweets_dashboard.csv** y **support_dashboard.csv** incluyen todas las columnas generadas.

---

## 🛠️ Installation

### Requisitos  
- Python ≥ 3.11  
- Conda (opcional) o Poetry  
- Git

### Usando Conda + Poetry  
```bash
conda create -n agents python=3.11
conda activate agents
conda install -c conda-forge poetry
poetry install
```

### Usando environment.yml + pip  
```bash
conda env create -f environment.yml
conda activate agents
pip install -r requirements.txt
```

---

## 🏃‍♂️ Quickstart

### 1. Ejecutar el pipeline localmente  
```bash
python3 << 'EOF'
from app.agent import graph
state0 = {
  "tweets_path": "training.1600000.processed.noemoticon.csv",
  "support_path": "twcs.csv",
  "tweets_sample_size": 10,
  "support_sample_size": 10
}
result = graph.invoke(state0)
print(result)
EOF
```

### 2. Levantar la API  
```bash
uvicorn app.api:app --reload
```

### 3. Iniciar el dashboard  
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
.
├─ app/
│  ├─ agent.py             # definición del grafo y nodos
│  ├─ tool.py              # implementación de todas las herramientas
│  └─ api.py               # endpoints FastAPI (opcional)
├─ cache/                  # almacén de resultados LLM (JSONL)
├─ environment.yml         # entorno Conda
├─ requirements.txt        # dependencias pip
├─ pyproject.toml          # versión, metadata (Poetry)
├─ README.md               # documentación (esta)
├─ tweets_dashboard.csv    # salida tweets
├─ support_dashboard.csv   # salida soporte
├─ training.1600000.processed.noemoticon.csv  # Sentiment140
└─ twcs.csv                # Customer Support Tickets
```

---

## 🔖 Versioning & Changelog

Este repositorio sigue **SemVer**.  
- Versión actual: **v1.0.0**  

Ver [CHANGELOG.md](CHANGELOG.md) para detalles de esta release.

---

## 📈 Next Steps (v2.0)

- Migrar a la SDK de OpenAI Agents para un **PlannerAgent** autónomo.  
- Incorporar **guardrails**, **handoffs** y **autoreflexión** entre agentes.  
- Exponer métricas en tiempo real y mejorar la orquestación.

---

> ¡Gracias por usar OmniSentimentAI!  
