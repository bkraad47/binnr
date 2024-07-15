from fastapi import FastAPI
from celery import Celery
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

app = FastAPI()

# Celery configuration
celery = Celery(__name__, broker='redis://redis:6379/0', backend='redis://redis:6379/0')

class JobRequest(BaseModel):
    param1: str
    param2: int

@celery.task
def long_running_job(param1: str, param2: int):
    # Simulate a long-running job
    import time
    time.sleep(param2)
    return f"Processed {param1} with delay of {param2} seconds"

@app.get("/jobs/status/{task_id}")
def get_status(task_id: str):
    task = long_running_job.AsyncResult(task_id)
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

# Add Swagger UI integration
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')
