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
    version="0.1.0",
)

# Precompilamos el grafo una sola vez
runner = graph

@app.post("/invoke", response_model=InvokeResponse)
async def invoke_pipeline(req: InvokeRequest):
    """
    Invoca el pipeline y devuelve las rutas de los dashboards.
    """
    try:
        result = runner.invoke({
            "tweets_path":         req.tweets_path,
            "support_path":        req.support_path,
            **({"tweets_sample_size":  req.tweets_sample_size}  if req.tweets_sample_size  is not None else {}),
            **({"support_sample_size": req.support_sample_size} if req.support_sample_size is not None else {}),
        })
        return InvokeResponse(
            tweets_dashboard=result["tweets_dashboard"],
            support_dashboard=result["support_dashboard"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))