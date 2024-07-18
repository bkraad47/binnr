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
    run_generate_predictions,
    run_train_llm,
    run_predict_llm,
    run_embeddings_cluster
)
from Snowflake.gbm_classification import get_model_info
from Snowflake.llm_classification import check_finetune_job_status, interact_llm

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
    llm_calssification_train_source: bool = False

# Pydantic model for GBM training request
class GBMTrainRequest(BaseModel):
    training_table: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None

# Pydantic model for GBM prediction request
class GBMPredictRequest(BaseModel):
    model_name: str
    predict_table: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None

# Pydantic model for LLM training request
class LLMTrainRequest(BaseModel):
    training_table: str
    validation_table: str
    target_column: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None
    model_type: str = 'llama3-8b'

# Pydantic model for LLM prediction request
class LLMPredictRequest(BaseModel):
    model_name: str
    predict_table: str
    database: str = None
    schema: str = None
    warehouse: str = None
    role: str = None

# Pydantic model for Embeddings cluster request
class EmbeddingsClusterRequest(BaseModel):
    input_table: str
    number_of_iterations: int
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
            request.llm_calssification_train_source
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/classification/gbm/train")
async def train_gbm_model(request: GBMTrainRequest = Body(...)):
    """
    Train a GBM classification model in Snowflake.

    Parameters:
        request (GBMTrainRequest): The training request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to train the GBM model
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
async def predict_gbm_model(request: GBMPredictRequest = Body(...)):
    """
    Generate predictions using a trained GBM classification model in Snowflake.

    Parameters:
        request (GBMPredictRequest): The prediction request data.

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
async def get_gbm_model_info(
    model_name: str,
    database: str = None,
    schema: str = None,
    warehouse: str = None,
    role: str = None
):
    """
    Retrieve model evaluation metrics for a GBM classification model in Snowflake.

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

@app.post("/classification/llm/train")
async def train_llm_model(request: LLMTrainRequest = Body(...)):
    """
    Train an LLM classification model in Snowflake.

    Parameters:
        request (LLMTrainRequest): The training request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to train the LLM model
        task = run_train_llm.apply_async(args=[
            request.training_table,
            request.validation_table,
            request.target_column,
            request.database,
            request.schema,
            request.role,
            request.warehouse,
            request.model_type
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/classification/llm/predict")
async def predict_llm_model(request: LLMPredictRequest = Body(...)):
    """
    Generate predictions using a trained LLM classification model in Snowflake.

    Parameters:
        request (LLMPredictRequest): The prediction request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to generate predictions
        task = run_predict_llm.apply_async(args=[
            request.predict_table,
            request.model_name,
            request.database,
            request.schema,
            request.role,
            request.warehouse
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/classification/llm/train/check")
async def check_llm_finetune_status(
    job_id: str,
    database: str = None,
    schema: str = None,
    warehouse: str = None,
    role: str = None
):
    """
    Check the status of an LLM fine-tuning job in Snowflake.

    Parameters:
        job_id (str): The job ID of the fine-tuning job.
        database (str, optional): The name of the database.
        schema (str, optional): The name of the schema.
        warehouse (str, optional): The name of the warehouse.
        role (str, optional): The role to use.

    Returns:
        dict: The status and progress of the fine-tuning job.
    """
    try:
        # Check the fine-tuning job status
        status = check_finetune_job_status(job_id, database, schema, role, warehouse)
        return status
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/classification/llm/response")
async def interact_llm_async(
    model_name: str,
    llm_prompt: str,
    database: str = None,
    schema: str = None,
    warehouse: str = None,
    role: str = None,
    llm_role: str = 'user',
    llm_temperature: float = 0.1,
    llm_max_tokens: int = 100,
    llm_top_p: float = 0
):
    """
    This function is designed to interact with a Large Language Model (LLM) hosted within a Snowflake environment. It sends a text prompt to the LLM and retrieves the model's response. This can be particularly useful for applications requiring natural language processing capabilities, such as text generation, question answering, or sentiment analysis.

    Parameters:
        model_name (str): The name of the LLM model to query. This should correspond to a model that has been previously trained or deployed within Snowflake.
        llm_prompt (str): The text prompt to send to the LLM. This could be a question, a statement, or any piece of text that the model is expected to process and respond to.
        database (str, optional): The name of the Snowflake database where the LLM model is stored. If not specified, the function will use the default database configured in the Snowflake connection.
        schema (str, optional): The name of the Snowflake schema under the specified database where the LLM model is located. If not provided, the default schema for the connection will be used.
        warehouse (str, optional): The name of the Snowflake warehouse to use for running the query. This parameter allows for specifying the computational resources for the query execution. If omitted, the default warehouse for the Snowflake session will be utilized.
        role (str, optional): The Snowflake role to assume for executing the query. This role should have the necessary permissions to access the model and execute queries. If not specified, the current role of the Snowflake session will be used.
        llm_role (str, optional): An additional role parameter specifically for the LLM query. This can be used to define roles with permissions tailored to LLM operations within Snowflake.
        llm_temperature (float, optional): A parameter controlling the randomness of the LLM's responses. A higher temperature results in more varied and creative responses, while a lower temperature produces more deterministic and predictable outputs. If not set, the model's default temperature setting is used.
        llm_max_tokens (int, optional): The maximum number of tokens (words or pieces of words) that the LLM's response can contain. This limits the length of the generated text. If not specified, the model's default maximum token count is applied.
        llm_top_p (float, optional): This parameter controls the diversity of the LLM's responses by limiting the token selection to those with a cumulative probability above the specified threshold. A higher value increases diversity, while a lower value makes the model's responses more focused. If not provided, the default value for the model is used.

    Returns:
        dict: A dictionary containing the status and progress of the LLM query. This typically includes information about the success of the query, any errors encountered, and the generated response from the LLM.

    This function facilitates the integration of advanced natural language processing capabilities into applications by leveraging the computational power and scalability of Snowflake and the sophistication of Large Language Models.
    """
    try:
        # Check the fine-tuning job status
        response = interact_llm(model_name, llm_prompt, database, schema, role, warehouse, llm_role, llm_temperature, llm_max_tokens, llm_top_p)
        return response
    except Exception as e:
        return {"error": str(e)}

@app.post("/classification/embeddings/cluster")
async def cluster_embeddings(request: EmbeddingsClusterRequest = Body(...)):
    """
    Cluster vector embeddings and run KMeans to identify where unmarked records belong.

    Parameters:
        request (EmbeddingsClusterRequest): The embeddings cluster request data.

    Returns:
        dict: The task ID of the Celery job.
    """
    try:
        # Invoke the long-running job to cluster embeddings
        task = run_embeddings_cluster.apply_async(args=[
            request.database,
            request.schema,
            request.warehouse,
            request.input_table,
            request.number_of_iterations,
            request.role
        ])
        
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

# Add Swagger UI integration
@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect to the Swagger UI documentation.
    """
    return RedirectResponse(url='/docs')
