from celery import Celery
from data_util import load_data_to_snowflake
import os

celery = Celery(__name__, broker=os.getenv('CELERY_BROKER_URL'), backend=os.getenv('CELERY_RESULT_BACKEND'))

@celery.task
def run_load_data_to_snowflake(database, schema, warehouse, file_path, table_name, role):
    return load_data_to_snowflake(database, schema, warehouse, file_path, table_name, role)
