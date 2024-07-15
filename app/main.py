import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from celery import Celery
from fastapi.responses import RedirectResponse
from tempfile import NamedTemporaryFile
from job import run_load_data_to_snowflake

app = FastAPI()

# Celery configuration using environment variables
celery = Celery(__name__, broker=os.getenv('CELERY_BROKER_URL'), backend=os.getenv('CELERY_RESULT_BACKEND'))

@app.get("/jobs/status/{task_id}")
def get_status(task_id: str):
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "Pending..."}
    elif task.state != 'FAILURE':
        return {"status": task.state, "result": task.result}
    else:
        return {"status": "Failed", "result": str(task.info)}

@app.get("/jobs/active")
def list_active_jobs():
    i = celery.control.inspect()
    active_jobs = i.active()
    return active_jobs

@app.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    table_name: str = Form(...),
    database: str = Form(None),
    schema: str = Form(None),
    warehouse: str = Form(None),
    role: str = Form(None)
):
    try:
        temp_dir = "/tmp/uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with NamedTemporaryFile(delete=False, dir=temp_dir) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Invoke the long-running job
        task = run_load_data_to_snowflake.apply_async(args=[database, schema, warehouse, temp_file_path, table_name, role])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

# Add Swagger UI integration
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')
