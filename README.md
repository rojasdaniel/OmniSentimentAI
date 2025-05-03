# OmniSentimentAI

Un sistema multiagente para análisis de sentimiento y atención al cliente sobre datos de Twitter y soporte.  
Utiliza **LangChain** y **LangGraph** para orquestar flujos de ingesta, preprocesado, análisis, alertas y generación de dashboards.  

## Características

- **Ingestión** de datos de Sentiment140 (Kaggle) y tickets de soporte (CSV).  
- **Preprocesado** de texto: limpieza de URLs, emojis, stopwords.  
- **Análisis** de sentimiento e intención con modelos de OpenAI.  
- **Alertas** automáticas para quejas críticas (sentimiento negativo + intención de queja).  
- **Dashboards** separados (`tweets_dashboard.csv` y `support_dashboard.csv`) para visualización.  
- **API** FastAPI para invocar el pipeline de forma remota.  
- **Front-end** en Streamlit para explorar resultados interactivos.

## Requisitos

- Python ≥3.11  
- Conda (opcional) o Poetry  
- Git LFS para datos grandes  

## Instalación

```bash
# Usando Conda
conda create -n agents python=3.11
conda activate agents
conda install -c conda-forge poetry

# Instalar dependencias
poetry install
# o si usas environment.yml
conda env create -f environment.yml
```

## Uso rápido

### Ejecutar el pipeline en local

```bash
python3 << 'EOF'
from app.agent import graph
# Configura las rutas y el tamaño de muestra
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

### Iniciar la API con FastAPI

```bash
uvicorn app.api:app --reload
```

### Levantar el dashboard con Streamlit

```bash
streamlit run streamlit_app.py
```

---

### Create environment from scratch

```bash
conda create -n agents python=3.11
conda activate agents
conda install -c conda-forge poetry
conda env export --from-history > environment.yml
```

### Create environment from file
```bash
conda env create -f environment.yml
```

### Run agent
```bash
langgraph dev
```

### Run fastapi server
```bash
fastapi dev app/api.py
```
