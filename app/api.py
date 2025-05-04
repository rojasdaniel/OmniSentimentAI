# app/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import graph

class InvokeRequest(BaseModel):
    tweets_path: str
    support_path: str
    tweets_sample_size: int | None = None
    support_sample_size: int | None = None

class InvokeResponse(BaseModel):
    tweets_dashboard: str
    support_dashboard: str

app = FastAPI(
    title="OmniSentimentAI API",
    description="API para invocar el pipeline multi‑agente de OmniSentimentAI",
    version="0.1.0",
)

# El grafo compilado se exporta como `graph` en agent.py
runner = graph

@app.post("/invoke", response_model=InvokeResponse)
async def invoke_pipeline(req: InvokeRequest):
    """
    Invoca el pipeline con los paths de tweets y soporte,
    y devuelve las rutas de ambos dashboards.
    """
    try:
        # Construir inputs para el grafo
        inputs = {
            "tweets_path": req.tweets_path,
            "support_path": req.support_path,
        }
        if req.tweets_sample_size is not None:
            inputs["tweets_sample_size"] = req.tweets_sample_size
        if req.support_sample_size is not None:
            inputs["support_sample_size"] = req.support_sample_size

        result = runner.invoke(inputs)
        return InvokeResponse(
            tweets_dashboard=result["tweets_dashboard"],
            support_dashboard=result["support_dashboard"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))