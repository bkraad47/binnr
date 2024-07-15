import json
import os
import logging
import snowflake.connector
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    # Load configuration from the environment variable
    return json.loads(os.getenv('SNOWFLAKE_CONFIG'))

def establish_connection(config, role=None, warehouse=None, database=None, schema=None):
    return snowflake.connector.connect(
        user=config['user'],
        password=config['password'],
        account=config['account'],
        warehouse=warehouse or config['warehouse'],
        database=database or config['database'],
        schema=schema or config['schema'],
        role=role or config.get('role')
    )

def get_timestamp():
    # Get current timestamp in a specific format
    return datetime.now().strftime('%Y%m%d%H%M%S')

def get_finetune_job_id(cursor, model_type, model_name, training_table, validation_table, target_column):
    finetune_model_sql = f"""
    SELECT SNOWFLAKE.CORTEX.FINETUNE(
      'CREATE',
      '{model_name}',
      '{model_type}',
      'SELECT SOURCE as prompt, "{target_column}" as completion FROM {training_table}',
      'SELECT SOURCE as prompt, "{target_column}" as completion FROM {validation_table}'
    ) AS job_id;
    """
    cursor.execute(finetune_model_sql)
    result = cursor.fetchone()
    job_id = result[0]  # Fetch the job_id from the single value returned
    return job_id

def check_job_status(cursor, job_id):
    status_sql = f"SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', '{job_id}');"
    cursor.execute(status_sql)
    result = cursor.fetchone()
    result_json = json.loads(result[0])  # Parse the JSON string
    job_status = result_json['status']
    progress = result_json['progress']
    return job_status, progress

def train_llm(training_table, validation_table, target_column, database=None, schema=None, role=None, warehouse=None, model_type='llama3-8b'):
    config = load_config()
    conn = establish_connection(config, role, warehouse, database, schema)
    
    model_name = f"{model_type.replace('-', '_')}_{training_table}"

    use_database_sql = f"USE DATABASE {database or config['database']};"
    use_schema_sql = f"USE SCHEMA {schema or config['schema']};"
    use_warehouse_sql = f"USE WAREHOUSE {warehouse or config['warehouse']};"
    use_role_sql = f"USE ROLE {role or config.get('role')};"

    try:
        cursor = conn.cursor()
        
        # Execute the SQL commands to use the role, warehouse, database, and schema
        cursor.execute(use_role_sql)
        cursor.execute(use_warehouse_sql)
        cursor.execute(use_database_sql)
        cursor.execute(use_schema_sql)
        print("Role, warehouse, database, and schema set successfully.")
        
        # Execute the SQL command to fine-tune the model and get the job ID
        job_id = get_finetune_job_id(cursor, model_type, model_name, training_table, validation_table, target_column)
        print(f"Fine-tuning job started with ID: {job_id}")

        return json.dumps({"model_name": model_name, "job_id": job_id})
    
    except snowflake.connector.errors.ProgrammingError as e:
        logging.error(f"An error occurred: {e}")
    
    finally:
        cursor.close()
        conn.close()

def predict_llm(predict_table, model_name, database=None, schema=None, role=None, warehouse=None):
    config = load_config()
    conn = establish_connection(config, role, warehouse, database, schema)
    
    timestamp = get_timestamp()
    result_table_name = f"LLM_{predict_table}_{timestamp}"

    create_classifications_sql_step_1 = f"""
    CREATE OR REPLACE TABLE {result_table_name} AS
    SELECT *, SNOWFLAKE.CORTEX.COMPLETE('{model_name}', [
        {{'role': 'user', 'content': SOURCE}}], 
        {{'temperature': 0.1}}) as llm_response 
    FROM {predict_table};
    """

    try:
        cursor = conn.cursor()
        
        # Execute the subsequent SQL command to create the classification table
        cursor.execute(create_classifications_sql_step_1)
        print(f"Table {result_table_name} created successfully.")
        return json.dumps({"table": result_table_name})
    
    except snowflake.connector.errors.ProgrammingError as e:
        logging.error(f"An error occurred: {e}")
    
    finally:
        cursor.close()
        conn.close()

def check_finetune_job_status(job_id, database=None, schema=None, role=None, warehouse=None):
    config = load_config()
    conn = establish_connection(config, role, warehouse, database, schema)

    use_database_sql = f"USE DATABASE {database or config['database']};"
    use_schema_sql = f"USE SCHEMA {schema or config['schema']};"
    use_warehouse_sql = f"USE WAREHOUSE {warehouse or config['warehouse']};"
    use_role_sql = f"USE ROLE {role or config.get('role')};"

    try:
        cursor = conn.cursor()

        # Execute the SQL commands to use the role, warehouse, database, and schema
        cursor.execute(use_role_sql)
        cursor.execute(use_warehouse_sql)
        cursor.execute(use_database_sql)
        cursor.execute(use_schema_sql)
        print("Role, warehouse, database, and schema set successfully.")

        # Check job status and show progress
        job_status, progress = check_job_status(cursor, job_id)
        return json.dumps({"job_status": job_status, "progress": progress})

    except snowflake.connector.errors.ProgrammingError as e:
        logging.error(f"An error occurred: {e}")

    finally:
        cursor.close()
        conn.close()

