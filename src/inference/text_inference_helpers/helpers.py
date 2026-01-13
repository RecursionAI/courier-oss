from fastapi.responses import JSONResponse
from src.db.schemas import CourierModel
from src.inference.models import InferenceRequest
from src.model_manager.model_manager import model_manager

# Simplified create_text_response using model manager
async def create_text_response(request: InferenceRequest, model: CourierModel):
    # Prepare payload for vLLM
    payload = request.model_dump()
    
    # Use model manager for all inference - it handles flex vs static internally
    result = await model_manager.inference(model, payload)

    # if request.stream:
    #     return result

    if isinstance(result, dict) and result.get("status_code", 200) >= 400:
        return JSONResponse(result, status_code=result.get("status_code", 500))

    # Ensure we strip any prompt echo if present (vLLM usually doesn't echo by default, but good to have)
    content = result.get("content", "").strip()
    
    # Return formatted result for endpoints.py
    return {"content": content, "result": result}
