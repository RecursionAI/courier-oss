import os
import uuid
import json
from datetime import datetime

from fastapi import HTTPException

import uvicorn

from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from datetime import date

from flowdb.client import FlowDB

from src.db.schemas import CourierModel, CourierUser, CourierMembership, Analytics, AnalyticsRequest, TrendRequest

from src.inference.helpers import create_audio_response, create_vision_response, api_valid, get_active_mem_gb
from src.inference.models import InferenceRequest, NewModelRequest, NewLibModelRequest, DeleteModelRequest, \
    DeleteLibModelRequest
from src.inference.text_inference_helpers.helpers import create_text_response
from src.lora.lora_train import lora_train
from src.lora.models import LoraRequest, EvalRequest
from typing import List, Optional, Any, Union
from src.lora.helpers import evaluate, create_formatted_dataset, delete_dataset
import asyncio
from dotenv import load_dotenv

from src.model_manager.model_manager import model_manager
from fastapi import FastAPI, Depends, Security, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import time
import threading

load_dotenv()

uc_header = {"Authorization": "uce_asdfawea_94923949"}
# base_url = "https://launchco.uc.r.appspot.com/api/uce"
base_url = "https://launchco.uc.r.appspot.com/api/courier"
uce_key = "uce_asdfawea_94923949"

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

db = FlowDB(api_key=os.getenv("FLOWDB_API_KEY"))

library = db.collection("library", CourierModel)
workbench = db.collection("workbench", CourierModel)
courier_users = db.collection("courier_users", CourierUser)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key == uce_key:
        return api_key
    valid = api_valid(api_key, courier_users)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


async def verify_admin_key(api_key: str = Security(api_key_header)):
    if api_key != uce_key:
        raise HTTPException(status_code=401, detail="Invalid Admin API key")
    return api_key


async def record_analytics(api_key: str, model_name: str, prompt_tokens: int, generation_tokens: int,
                           peak_memory: float, start_time: datetime, end_time: datetime):
    try:
        analytics_coll = db.collection(f"Analytics-{api_key}", Analytics)
        active_mem = get_active_mem_gb()
        active_mem_gb = f"{active_mem:.2f} GB"

        analytic_request = AnalyticsRequest(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            peak_memory=peak_memory,
            start_time=str(start_time),
            end_time=str(end_time),
            system_active_memory=active_mem_gb
        )

        today = date.today()
        formatted_date = today.strftime("%m-%d-%Y")
        today_analytics = analytics_coll.read(key=f"{formatted_date}")
        if not today_analytics:
            today_analytics = Analytics(id=f"{formatted_date}", requests=[analytic_request])
        else:
            today_analytics.requests.append(analytic_request)

        analytics_coll.upsert(today_analytics)
    except Exception as e:
        print(f"Error recording analytics: {e}")


def get_models_from_db() -> Optional[List[CourierModel]]:
    limit = 100
    skip = 0
    workbench_models: List[CourierModel] = []
    try:
        while True:
            batch = workbench.list(limit=limit, skip=skip)

            if not batch:
                break

            workbench_models.extend(batch)
            skip += limit

        return workbench_models

    except Exception as e:
        print(f"Error fetching models from DB: {e}")


async def periodic_cleanup():
    while True:
        await model_manager.cleanup_expired_models()
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    try:
        workbench_models = get_models_from_db()
        for cm in workbench_models:
            if cm.api_type == "static":
                await model_manager.ensure_model_loaded(cm)
    except Exception as e:
        print(f"Error loading model modules: {e}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/lora/')
async def lora(request: LoraRequest, api_key: str = Depends(verify_api_key)):
    await lora_train(request, base_url)
    return JSONResponse({"detail": "Training started successfully"}, status_code=200)


@app.post('/add-workbench-model/')
async def add_workbench_model(request: NewModelRequest, api_key: str = Depends(verify_api_key)):
    lib_model = library.read(key=f"{request.name}")
    try:
        model_uuid = uuid.uuid4()
        courier_model = CourierModel(name=lib_model.name, context_window=request.context_window, id=f"{model_uuid}",
                                     model_id=f"{model_uuid}", model_type=lib_model.model_type,
                                     adapter_path=request.adapter_path, file_path=lib_model.file_path,
                                     api_type=request.api_type, instances=request.instances, memberships=[
                CourierMembership(id=f"{api_key}", api_key=f"{api_key}", role="admin")],
                                     family=lib_model.family, nickname=request.nickname,
                                     weights_gb=lib_model.weights_gb)

        if request.api_type == "static":
            await model_manager.ensure_model_loaded(courier_model)

        workbench.upsert(courier_model)

        return JSONResponse({"detail": f"{request.name} added to workbench successfully"}, status_code=201)

    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=500)


@app.post('/add-library-model/')
def add_library_model(request: NewLibModelRequest, api_key: str = Depends(verify_admin_key)):
    try:
        model_uuid = uuid.uuid4()
        library.upsert(CourierModel(name=request.name, context_window=request.context_window, id=f"{request.name}",
                                    model_id=f"{model_uuid}", model_type=request.model_type,
                                    adapter_path=request.adapter_path, file_path=request.file_path, api_type="flex",
                                    memberships=[],
                                    family=request.family, nickname=None, weights_gb=request.weights_gb))
        return JSONResponse({"detail": f"{request.name} added to library successfully"}, status_code=201)
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=500)


@app.post('/remove-library-model/')
def remove_library_model(request: DeleteLibModelRequest, api_key: str = Depends(verify_admin_key)):
    try:
        library.delete(key=f"{request.name}")
        return JSONResponse({"detail": f"{request.name} deleted successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=500)


@app.post('/remove-workbench-model/')
async def remove_workbench_model(request: DeleteModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        model = workbench.read(key=f"{request.model_id}")
        if model is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)

        permission = False
        for mem in model.memberships:
            if mem.api_key == f"{api_key}" and mem.role == "admin":
                permission = True

        if not permission and api_key != uce_key:
            return JSONResponse({"error": "You do not have permission to delete this model"}, status_code=401)

        workbench.delete(key=f"{model.id}")

        # Model manager handles unloading
        await model_manager._unload_model(model.name)

        return JSONResponse({"detail": f"{request.name} deleted successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found and/or error: {e}")


@app.get('/get-all-workbench-models/')
async def get_all_workbench_models(api_key: str = Depends(verify_admin_key)):
    workbench_models = get_models_from_db()
    return JSONResponse({"models": [wm.model_dump() for wm in workbench_models]}, status_code=200)


@app.get("/get-workbench-models/")
def get_workbench_models(api_key: str = Depends(verify_api_key)):
    try:
        models_for_user: List[CourierModel] = []
        workbench_models = get_models_from_db()

        for wm in workbench_models:
            for mem in wm.memberships:
                if f"{mem.api_key}" == f"{api_key}":
                    models_for_user.append(wm)
        return JSONResponse({"models": [mfu.model_dump() for mfu in models_for_user]}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting models: {e}"}, status_code=500)


@app.get("/get-lib-models/")
def get_lib_models(api_key: str = Depends(verify_api_key)):
    try:
        library_models: List[CourierModel] = []
        limit = 100
        skip = 0
        while True:
            batch = library.list(limit=limit, skip=skip)
            if not batch:
                break
            library_models.extend(batch)
            skip += limit

        return JSONResponse({"models": [lm.model_dump() for lm in library_models]}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting models: {e}"}, status_code=500)


@app.post('/inference/')
async def inference(request: InferenceRequest, background_tasks: BackgroundTasks,
                    api_key: str = Depends(verify_api_key)):
    """
    Simplified inference endpoint - model manager handles all complexity.
    Supports text, audio, and vision models via vLLM.
    Works for both flex and static models transparently.
    """
    start_time = datetime.now()
    try:
        # Get model from database
        model = workbench.read(key=f"{request.model_id}")

        if model is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)

        # Check permissions
        valid_member = any(f"{m.api_key}" == f"{api_key}" for m in model.memberships)

        if not valid_member and api_key != uce_key:
            return JSONResponse({"error": "You do not have permission to access this model"}, status_code=401)

        # Route to appropriate handler based on model type
        if request.model_type == "audio-text-text":
            result = await create_audio_response(request, model)
        elif request.model_type == "image-text-text":
            result = await create_vision_response(request, model)
        else:
            result = await create_text_response(request, model)

        if isinstance(result, JSONResponse):
            return result

        # result is expected to be a dict if successful
        end_time = datetime.now()

        # result should have 'content', 'prompt_tokens', 'generation_tokens', 'peak_memory'
        response_content = result.get("content", "")
        # Handle cases where result is wrapped (e.g. from create_text_response)
        if "result" in result and isinstance(result["result"], dict):
            prompt_tokens = result["result"].get("prompt_tokens", 0)
            generation_tokens = result["result"].get("generation_tokens", 0)
            peak_memory = result["result"].get("peak_memory", 0.0)
        else:
            prompt_tokens = result.get("prompt_tokens", 0)
            generation_tokens = result.get("generation_tokens", 0)
            peak_memory = result.get("peak_memory", 0.0)

        # Record analytics in background for all model types
        background_tasks.add_task(record_analytics, api_key, model.name, prompt_tokens, generation_tokens, peak_memory,
                                  start_time, end_time)

        return JSONResponse({"content": response_content}, status_code=200)

    except Exception as e:
        print(f"Inference error: {str(e)}")
        return JSONResponse({"error": f"Inference error: {str(e)}"}, status_code=500)


@app.get('/today-analytics/')
def get_today_analytics(api_key: str = Depends(verify_api_key)):
    try:
        analytics_coll = db.collection(f"Analytics-{api_key}", Analytics)
        today = date.today()
        formatted_date = today.strftime("%m-%d-%Y")
        today_analytics = analytics_coll.read(key=f"{formatted_date}")
        if not today_analytics:
            return JSONResponse({"total_entries": 0, "analytics": {"id": formatted_date, "requests": []}},
                                status_code=200)
        return JSONResponse({"total_entries": len(today_analytics.requests), "analytics": today_analytics.model_dump()},
                            status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting analytics: {e}"}, status_code=500)


@app.get('/trend-analytics/')
def get_trend_analytics(request: TrendRequest, api_key: str = Depends(verify_api_key)):
    try:
        analytics_coll = db.collection(f"Analytics-{api_key}", Analytics)
        trend_analytics = analytics_coll.list(limit=request.limit, skip=request.skip)
        total_entries = 0
        for ta in trend_analytics:
            total_entries += len(ta.requests)
        return JSONResponse({"total_entries": total_entries, "analytics": [ta.model_dump() for ta in trend_analytics]},
                            status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting analytics: {e}"}, status_code=500)


@app.get('/all-analytics/')
def all_analytics(request: TrendRequest, api_key: str = Depends(verify_admin_key)):
    try:
        responses = []
        total_entries = 0
        full_total_models = 0
        for collection in db.list_collections():
            if collection.startswith("Analytics-"):
                analytics = db.collection(collection, Analytics)
                trend_analytics = analytics.list(limit=request.limit, skip=request.skip)
                total_requests = 0
                total_models = 0
                for ta in trend_analytics:
                    total_entries += len(ta.requests)
                    total_requests += len(ta.requests)
                    total_models = len(set([ar.model_name for ar in ta.requests]))
                    full_total_models += total_models
                responses.append(
                    {"total_entries": total_requests, "analytics": [ta.model_dump() for ta in trend_analytics],
                     "total_models": total_models})

        return JSONResponse(
            {"analytics": {"total_responses": len(responses), "full_total_models": full_total_models,
                           "total_entries": total_entries, "responses": responses}},
            status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting analytics: {e}"}, status_code=500)


@app.post('/evaluate/')
async def eval_request(request: EvalRequest, api_key: str = Depends(verify_api_key)):
    try:
        create_formatted_dataset(request.model_name, request.dataset_id, val_size=request.val_size,
                                 dataset_name=request.dataset_id)
        loop = asyncio.get_running_loop()
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as pool:
            response = await loop.run_in_executor(
                pool,
                evaluate,
                request.model_name,
                request.adapter_path,
                request.max_seq_length
            )
        return response
    except Exception as e:
        delete_dataset()
        return JSONResponse({"error": f"{e}"}, status_code=500)
    finally:
        delete_dataset()


# @app.post("/update-adapter/")
# async def update_adapter(request: UpdateAdapterRequest):
#     try:
#         model = get_model_helper(models, request.model_name, request.api_key)
#         model_req = ModelRequest(model_name=request.model_name,
#                                  context_window=model.context_window, adapter_path=model.adapter_path,
#                                  api_key=model.api_key)
#         await delete_model(request=model_req)
#         new_model_req = NewModelRequest(model_name=model.name,
#                                         context_window=model.context_window,
#                                         adapter_path=request.adapter_path, api_key=request.api_key,
#                                         model_type=model.model_type)
#         await add_model(request=new_model_req)
#         return JSONResponse({"detail": "adapter updated successfully"}, status_code=200)
#     except Exception as e:
#         return JSONResponse({"error": e}, status_code=500)
#
#
# @app.delete("/delete-adapter-path/")
# async def delete_adapter_path(dar: DeleteAdapterRequest):
#     try:
#         encoded_path = base64.b64encode(dar.adapter_path.encode('utf-8')).decode('utf-8')
#         response = requests.delete(f"{base_url}/delete-adapter-path/{dar.dataset_id}/{encoded_path}", headers=uc_header)
#         if response.status_code != 200:
#             return JSONResponse({"error": f"Error deleting adapter path: {response.json()}"}, status_code=500)
#         import os
#         if os.path.exists(dar.adapter_path):
#             shutil.rmtree(dar.adapter_path)
#         return JSONResponse({"detail": f"Adapter deleted successfully"}, status_code=200)
#     except Exception as e:
#         return JSONResponse({"error": f"Error deleting adapter: {e}"}, status_code=500)


@app.get("/check-validity-status/")
def check_validity_status(api_key: str = Depends(verify_api_key)):
    return JSONResponse({"response": True}, status_code=200)


@app.post("/add-courier-user/")
def add_courier_user(request: CourierUser, api_key: str = Depends(verify_admin_key)):
    try:
        user = CourierUser(id=request.api_key, api_key=request.api_key, valid=request.valid)
        courier_users.upsert(user)
        return JSONResponse({"detail": "User added successfully"}, status_code=201)
    except Exception as e:
        return JSONResponse({"error": f"Error adding user: {e}"}, status_code=500)


@app.post("/add-user/")
def register_credential(request: CourierUser, api_key: str = Depends(verify_admin_key)):
    try:
        user = courier_users.read(key=f"{request.api_key}")
        if not user:
            user = CourierUser(id=request.api_key, api_key=request.api_key, valid=request.valid)
            courier_users.upsert(user)
            return JSONResponse({"detail": "Credential registered successfully"}, status_code=201)
        else:
            return JSONResponse({"error": "Credential already registered"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Error adding user: {e}"}, status_code=500)


COLLECTION_MODELS = {
    "library": CourierModel,
    "workbench": CourierModel,
    "courier_users": CourierUser,
    "default_analytics": Analytics
}


def get_model_for_collection(name: str):
    if name.startswith("Analytics-"):
        return Analytics
    return COLLECTION_MODELS.get(name)


@app.get('/backup-db/')
async def backup_db(api_key: str = Depends(verify_admin_key)):
    try:
        all_data = {}
        collections = db.list_collections()

        for coll_name in collections:
            model = get_model_for_collection(coll_name)
            if not model:
                continue

            coll = db.collection(coll_name, model)
            docs = []
            limit, skip = 100, 0
            while True:
                batch = coll.list(limit=limit, skip=skip)
                if not batch:
                    break
                docs.extend([doc.model_dump() for doc in batch])
                skip += limit
            all_data[coll_name] = docs

        with open("backup.json", "w") as f:
            json.dump(all_data, f, indent=4)

        return JSONResponse({"detail": "Backup created successfully", "collections_backed_up": list(all_data.keys())})
    except Exception as e:
        return JSONResponse({"error": f"Backup failed: {e}"}, status_code=500)


@app.post('/restore-db/')
async def restore_db(api_key: str = Depends(verify_admin_key)):
    try:
        if not os.path.exists("backup.json"):
            return JSONResponse({"error": "backup.json not found"}, status_code=404)

        with open("backup.json", "r") as f:
            data = json.load(f)

        for coll_name, items in data.items():
            model = get_model_for_collection(coll_name)
            if not model:
                continue

            coll = db.collection(coll_name, model)
            for item in items:
                coll.upsert(model(**item))

        return JSONResponse({"detail": "Database restored successfully"})
    except Exception as e:
        return JSONResponse({"error": f"Restore failed: {e}"}, status_code=500)


@app.get("/get-memory/")
def get_memory(api_key: str = Depends(verify_api_key)):
    try:
        if api_key != uce_key:
            return JSONResponse({"error": "Invalid API key"}, status_code=401)

        from src.inference.helpers import get_memory_stats
        stats = get_memory_stats()

        if "error" in stats:
            return JSONResponse(stats, status_code=500)

        return JSONResponse(stats, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Error getting memory: {e}"}, status_code=500)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9100)
    # uvicorn.run(app, host='127.0.0.1', port=9100)
