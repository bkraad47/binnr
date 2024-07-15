import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, Body
from pydantic import BaseModel
from celery import Celery
from fastapi.responses import RedirectResponse
from tempfile import NamedTemporaryFile
from job import (
    run_load_data_to_snowflake,
    run_create_classification_tables,
    run_train_model,
    run_generate_predictions
)
from Snowflake.gbm_classification import get_model_info

app = FastAPI()

# Celery configuration using environment variables
celery = Celery(__name__, broker=os.getenv('CELERY_BROKER_URL'), backend=os.getenv('CELERY_RESULT_BACKEND'))

# Pydantic model for classification request
class ClassificationRequest(BaseModel):
    table_name: str
    source_columns: list[str]
    target_column: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None
    template: str = None
    create_validation: bool = False
    llm_train_source: bool = False

# Pydantic model for training request
class TrainRequest(BaseModel):
    training_table: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None

# Pydantic model for prediction request
class PredictRequest(BaseModel):
    model_name: str
    predict_table: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None

@app.get("/jobs/status/{task_id}")
def get_status(task_id: str):
    """
    Get the status of a Celery task.

    Parameters:
        task_id (str): The ID of the task.

    Returns:
        dict: The status and result of the task.
    """
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "Pending..."}
    elif task.state != 'FAILURE':
        return {"status": task.state, "result": task.result}
    else:
        return {"status": "Failed", "result": str(task.info)}

@app.get("/jobs/active")
def list_active_jobs():
    """
    List all active Celery jobs.

    Returns:
        dict: A dictionary of active jobs.
    """
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
    """
    Upload a file and initiate a long-running job to load data into Snowflake.

    Parameters:
        file (UploadFile): The file to upload.
        table_name (str): The name of the table to create in Snowflake.
        database (str, optional): The name of the database.
        schema (str, optional): The name of the schema.
        warehouse (str, optional): The name of the warehouse.
        role (str, optional): The role to use.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        temp_dir = "/tmp/uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with NamedTemporaryFile(delete=False, dir=temp_dir) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Invoke the long-running job to load data into Snowflake
        task = run_load_data_to_snowflake.apply_async(args=[database, schema, warehouse, temp_file_path, table_name, role])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/data/create/tables/classification")
async def create_classification(request: ClassificationRequest = Body(...)):
    """
    Create classification tables in Snowflake.

    Parameters:
        request (ClassificationRequest): The classification request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to create classification tables
        task = run_create_classification_tables.apply_async(args=[
            request.database,
            request.schema,
            request.warehouse,
            request.table_name,
            request.source_columns,
            request.target_column,
            request.role,
            request.template,
            request.create_validation,
            request.llm_train_source
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/classification/gbm/train")
async def train_classification_model(request: TrainRequest = Body(...)):
    """
    Train a classification model in Snowflake.

    Parameters:
        request (TrainRequest): The training request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to train the model
        task = run_train_model.apply_async(args=[
            request.database,
            request.schema,
            request.role,
            request.warehouse,
            request.training_table
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/classification/gbm/predict")
async def generate_classification_predictions(request: PredictRequest = Body(...)):
    """
    Generate predictions using a trained classification model in Snowflake.

    Parameters:
        request (PredictRequest): The prediction request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to generate predictions
        task = run_generate_predictions.apply_async(args=[
            request.database,
            request.schema,
            request.role,
            request.warehouse,
            request.model_name,
            request.predict_table
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/classification/gbm/info")
async def get_classification_model_info(
    model_name: str,
    database: str = None,
    schema: str = None,
    warehouse: str = None,
    role: str = None
):
    """
    Retrieve model evaluation metrics for a classification model in Snowflake.

    Parameters:
        model_name (str): The name of the model.
        database (str, optional): The name of the database.
        schema (str, optional): The name of the schema.
        warehouse (str, optional): The name of the warehouse.
        role (str, optional): The role to use.

    Returns:
        dict: The model evaluation metrics.
    """
    try:
        # Retrieve model information
        info = get_model_info(database, schema, role, warehouse, model_name)
        return info
    except Exception as e:
        return {"error": str(e)}

# Add Swagger UI integration
@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect to the Swagger UI documentation.
    """
    return RedirectResponse(url='/docs')
