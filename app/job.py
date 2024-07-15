from celery import Celery
from Snowflake.data_util import load_data_to_snowflake, create_classification_tables
from Snowflake.gbm_classification import train_model, generate_predictions
from Snowflake.llm_classification import train_llm, predict_llm
import os

celery = Celery(__name__, broker=os.getenv('CELERY_BROKER_URL'), backend=os.getenv('CELERY_RESULT_BACKEND'))

@celery.task
def run_load_data_to_snowflake(database, schema, warehouse, file_path, table_name, role):
    return load_data_to_snowflake(database, schema, warehouse, file_path, table_name, role)

@celery.task
def run_create_classification_tables(database, schema, warehouse, table_name, source_columns, target_column, role, template=None, create_validation=False, llm_train_source=False):
    return create_classification_tables(database, schema, warehouse, table_name, source_columns, target_column, role, template, create_validation, llm_train_source)

@celery.task
def run_train_model(database, schema, role, warehouse, training_table):
    return train_model(database, schema, role, warehouse, training_table)

@celery.task
def run_generate_predictions(database, schema, role, warehouse, model_name, predict_table):
    return generate_predictions(database, schema, role, warehouse, model_name, predict_table)

@celery.task
def run_train_llm(training_table, validation_table, target_column, database, schema, role, warehouse, model_type):
    return train_llm(training_table, validation_table, target_column, database, schema, role, warehouse, model_type)

@celery.task
def run_predict_llm(predict_table, model_name, database, schema, role, warehouse):
    return predict_llm(predict_table, model_name, database, schema, role, warehouse)
