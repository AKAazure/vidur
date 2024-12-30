from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pickle
from vidur.entities.batch import Batch
from vidur.entities.execution_time import ExecutionTime
from vidur.entities.request import Request
from vidur.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor
from vidur.utils.model_persistence import get_model_path

app = FastAPI()

# Define the request schema
class RequestItem(BaseModel):
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int

class BatchPredictionRequest(BaseModel):
    replica_id: int
    requests: List[RequestItem]
    model_name: str
    tp_num : int
    mem_margin:float = 0.9

class BatchPredictionResponse(BaseModel):
    execution_times: List[float]

# Load predictor based on replica configuration
def load_predictor(replica_config_path: str) -> BaseExecutionTimePredictor:
    try:
        with open(replica_config_path, 'rb') as file:
            predictor = pickle.load(file)
        return predictor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictor: {str(e)}")

@app.post("/prediction", response_model=BatchPredictionResponse)
async def predict_execution_time(request: BatchPredictionRequest):
    try:

        replica_config_path = get_model_path(request.model_name,request.tp_num,request.mem_margin)
        # Load predictor from the specified config path
        predictor = load_predictor(replica_config_path)
        
        # Create Request and Batch instances
        requests = [
            Request(
                arrived_at=req.arrived_at,
                num_prefill_tokens=req.num_prefill_tokens,
                num_decode_tokens=req.num_decode_tokens
            ) for req in request.requests
        ]

        batch = Batch(
            replica_id=request.replica_id,
            requests=requests,
            num_tokens=[req.num_prefill_tokens + req.num_decode_tokens for req in requests]
        )

        # Predict execution time
        execution_time: ExecutionTime = predictor.get_execution_time(batch, 0)
        
        return BatchPredictionResponse(
            execution_times=[execution_time.model_time for _ in requests]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn fastapi_vidur_api:app --reload
