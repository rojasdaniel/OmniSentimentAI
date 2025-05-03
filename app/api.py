# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import graph
import requests
import io
import pandas as pd

class InvokeRequest(BaseModel):
    file_path: str

app = FastAPI(
    title="OmniSentimentAI API",
    version="0.1.0",
)

@app.get("/health", operation_id="health_check")
async def health():
    return {"status": "ok"}

@app.get("/run", operation_id="run_pipeline")
async def run_pipeline():
    # Aquí invocas el grafo usando la ruta real del CSV
    return graph.invoke({"file_path": "training.1600000.processed.noemoticon.csv"})

@app.get("/", operation_id="root_redirect")
async def root():
    # Si quieres que “/” haga lo mismo que “/run”, simplemente:
    return run_pipeline()

@app.post("/invoke", operation_id="invoke_pipeline")
async def invoke_pipeline(req: InvokeRequest):
    # Para depurar mejor, podemos quitar el try/except temporalmente
    # y así verás el traceback completo en la consola y en la respuesta 500.
    result = graph.invoke(req.dict())
    return result


