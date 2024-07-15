import json
import os
import snowflake.connector
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    # Load configuration from the environment variable
    return json.loads(os.getenv('SNOWFLAKE_CONFIG'))

def get_timestamp():
    # Get current timestamp in a specific format
    return datetime.now().strftime('%Y%m%d%H%M%S')

def train_model(database, schema, role, warehouse, training_table):
    # Train a classification model in Snowflake
    config = load_config()

    # Use config values if parameters are null or empty
    warehouse = warehouse or config.get('warehouse')
    database = database or config.get('database')
    schema = schema or config.get('schema')
    role = role or config.get('role', 'ACCOUNTADMIN')

    conn = snowflake.connector.connect(
        user=config['user'],
        password=config['password'],
        account=config['account'],
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role,
    )
    
    timestamp = get_timestamp()
    model_name = f"model_{training_table.lower()}_{timestamp}"

    use_role_query = f"USE ROLE {role};"
    use_warehouse_query = f"USE WAREHOUSE {warehouse};"
    use_database_query = f"USE DATABASE {database};"
    use_schema_query = f"USE SCHEMA {schema};"
    create_model_query = f"""
    CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION {model_name}(
        INPUT_DATA => SYSTEM$REFERENCE('TABLE','{training_table}'),
        TARGET_COLNAME => 'TARGET',
        CONFIG_OBJECT => {{ 'ON_ERROR': 'SKIP' }}
    );
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(use_role_query)
        cursor.execute(use_warehouse_query)
        cursor.execute(use_database_query)
        cursor.execute(use_schema_query)
        cursor.execute(create_model_query)
        logger.error(f"Model '{model_name}' created successfully.")
        return json.dumps({"model_name": model_name, "table": training_table})
    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"An error occurred: {e}")
        return json.dumps({"error": str(e)})
    finally:
        cursor.close()
        conn.close()

def generate_predictions(database, schema, role, warehouse, model_name, predict_table):
    # Generate predictions using the trained model
    config = load_config()

    # Use config values if parameters are null or empty
    warehouse = warehouse or config.get('warehouse')
    database = database or config.get('database')
    schema = schema or config.get('schema')
    role = role or config.get('role', 'ACCOUNTADMIN')

    conn = snowflake.connector.connect(
        user=config['user'],
        password=config['password'],
        account=config['account'],
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role,
    )
    
    timestamp = get_timestamp()
    result_table = f"gbm_results_{predict_table}_{timestamp}"
    use_role_query = f"USE ROLE {role};"
    use_warehouse_query = f"USE WAREHOUSE {warehouse};"
    use_database_query = f"USE DATABASE {database};"
    use_schema_query = f"USE SCHEMA {schema};"
    generate_predictions_query = f"""
    CREATE OR REPLACE TABLE {result_table} AS 
    SELECT
        *, 
        {model_name}!PREDICT(
            OBJECT_CONSTRUCT(*),
            {{'ON_ERROR': 'SKIP'}}
        ) as predictions
    FROM {predict_table};
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(use_role_query)
        cursor.execute(use_warehouse_query)
        cursor.execute(use_database_query)
        cursor.execute(use_schema_query)
        cursor.execute(generate_predictions_query)
        logger.error("Predictions generated successfully.")
        return json.dumps({"table": result_table})
    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"An error occurred: {e}")
        return json.dumps({"error": str(e)})
    finally:
        cursor.close()
        conn.close()

def get_model_info(database, schema, role, warehouse, model_name):
    # Retrieve model evaluation metrics
    config = load_config()

    # Use config values if parameters are null or empty
    warehouse = warehouse or config.get('warehouse')
    database = database or config.get('database')
    schema = schema or config.get('schema')
    role = role or config.get('role', 'ACCOUNTADMIN')

    conn = snowflake.connector.connect(
        user=config['user'],
        password=config['password'],
        account=config['account'],
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role,
    )
    
    use_role_query = f"USE ROLE {role};"
    use_warehouse_query = f"USE WAREHOUSE {warehouse};"
    use_database_query = f"USE DATABASE {database};"
    use_schema_query = f"USE SCHEMA {schema};"
    show_evaluation_metrics_query = f"CALL {model_name}!SHOW_EVALUATION_METRICS();"
    
    try:
        cursor = conn.cursor()
        cursor.execute(use_role_query)
        cursor.execute(use_warehouse_query)
        cursor.execute(use_database_query)
        cursor.execute(use_schema_query)
        cursor.execute(show_evaluation_metrics_query)
        metrics = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        metrics_json = [dict(zip(columns, row)) for row in metrics]
        logger.error("Model evaluation metrics retrieved successfully.")
        return json.dumps({"model_name": model_name, "metrics": metrics_json})
    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"An error occurred: {e}")
        return json.dumps({"error": str(e)})
    finally:
        cursor.close()
        conn.close()

