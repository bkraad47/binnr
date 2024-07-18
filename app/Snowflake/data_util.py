import json
import os
import logging
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_to_snowflake(database, schema, warehouse, file_path, table_name, role):
    result = []
    try:
        # Load connection details from the environment variable
        config = json.loads(os.getenv('SNOWFLAKE_CONFIG'))

        # Use config values if parameters are null or empty
        warehouse = warehouse or config.get('warehouse')
        database = database or config.get('database')
        schema = schema or config.get('schema')
        role = role or config.get('role')

        # Create a Snowflake SQLAlchemy engine
        engine = create_engine(URL(
            user=config['user'],
            password=config['password'],
            account=config['account'],
            warehouse=warehouse,
            database=database,
            schema=schema
        ))

        with engine.connect() as connection:
            # Use role, warehouse, database, and schema explicitly
            connection.execute(text(f"USE ROLE {role}"))
            connection.execute(text(f"USE WAREHOUSE {warehouse}"))
            connection.execute(text(f"USE DATABASE {database}"))
            connection.execute(text(f"USE SCHEMA {schema}"))

            # Drop the table if it exists and create a new one
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

            # Create table structure based on the CSV file headers
            with open(file_path, 'r', encoding='utf-8') as file:
                headers = file.readline().strip().split(',')
                columns = ', '.join([f'"{col}" STRING' for col in headers])
                create_table_sql = f"CREATE TABLE {table_name} ({columns})"
                connection.execute(text(create_table_sql))
                result.append({"table": f"{database}.{schema}.{table_name}"})

            # Generate a stage name with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            stage_name = f"{table_name}_stage_{timestamp}"

            # Create a named stage
            connection.execute(text(f"CREATE OR REPLACE STAGE {stage_name}"))

            # Upload the CSV file to the stage
            put_command = f"PUT file://{os.path.abspath(file_path)} @{stage_name}"
            connection.execute(text(put_command))

            # Copy data from the stage into the table
            copy_command = f"""
            COPY INTO {table_name}
            FROM @{stage_name}/{os.path.basename(file_path)}
            FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1)
            ON_ERROR = 'CONTINUE';
            """
            connection.execute(text(copy_command))

            # Remove the stage after loading
            connection.execute(text(f"DROP STAGE IF EXISTS {stage_name}"))

        # Delete the file after loading
        os.remove(file_path)

    except Exception as e:
        logging.error(f"Error loading data to Snowflake: {e}")
        raise e
    return json.dumps(result)

def create_target_table(connection, database, schema, warehouse, table_name, target_column, role):
    result = []
    try:
        target_table_name = f"{target_column.upper()}_TARGET"
        connection.execute(text(f"USE ROLE {role}"))
        connection.execute(text(f"USE WAREHOUSE {warehouse}"))
        connection.execute(text(f"DROP TABLE IF EXISTS {target_table_name}"))
        create_target_sql = f"""
        CREATE OR REPLACE TABLE {target_table_name} AS
        SELECT DISTINCT "{target_column.lower()}", ROW_NUMBER() OVER (ORDER BY "{target_column.lower()}") AS "{target_column.lower()}_number"
        FROM {table_name}
        WHERE "{target_column.lower()}" IS NOT NULL;
        """
        connection.execute(text(create_target_sql))
        
        result.append({"table": f"{database}.{schema}.{target_table_name}"})
    except Exception as e:
        logging.error(f"Error creating target table: {e}")
        raise e
    return result

def create_classification_tables(database, schema, warehouse, table_name, source_columns, target_column, role, template=None, create_validation=False, llm_calssification_train_source=False):
    result = []
    try:
        # Load connection details from the environment variable
        config = json.loads(os.getenv('SNOWFLAKE_CONFIG'))

        # Use config values if parameters are null or empty
        warehouse = warehouse or config.get('warehouse')
        database = database or config.get('database')
        schema = schema or config.get('schema')
        role = role or config.get('role')

        # Create a Snowflake SQLAlchemy engine
        engine = create_engine(URL(
            user=config['user'],
            password=config['password'],
            account=config['account'],
            warehouse=warehouse,
            database=database,
            schema=schema
        ))

        with engine.connect() as connection:
            # Use role, warehouse, database, and schema explicitly
            connection.execute(text(f"USE ROLE {role}"))
            connection.execute(text(f"USE WAREHOUSE {warehouse}"))
            connection.execute(text(f"USE DATABASE {database}"))
            connection.execute(text(f"USE SCHEMA {schema}"))

            # Always create the target table
            target_creation_result = create_target_table(connection, database, schema, warehouse, table_name, target_column, role)
            result.extend(target_creation_result)

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            temp_table_name = f"temp_{table_name}_sources_targets_{timestamp}"

            # Generate the LLM context string
            llm_context = f"'. Respond only with: ', (SELECT LISTAGG(\"{target_column}\", ' or ') WITHIN GROUP (ORDER BY \"{target_column}\") FROM {target_column.upper()}_TARGET), '. **NO ELABORATE SENTENCES. ONLY SPECIFIC WORD(S) FROM THE LIST**'"

            # Generate the source column SQL based on the template and llm_train_source flag
            if template:
                for col in source_columns:
                    template = template.replace(f"{col.lower()}", f"IFNULL(a.\"" + col.lower() + "\", '')")
                formatted_template = template.replace("{", "',").replace("}", ",'")
                if llm_calssification_train_source:
                    source_sql = f"CONCAT('"+formatted_template+"',"+ llm_context +") AS SOURCE"
                else:
                    source_sql = f"CONCAT('"+formatted_template+"') AS SOURCE"
            else:
                source_sql = ",' : ',".join([("IFNULL(a.\""+col.lower()+"\",'')") for col in source_columns])
                if llm_calssification_train_source:
                    source_sql = f"CONCAT({source_sql},{llm_context}) AS SOURCE"
                else:
                    source_sql = f"CONCAT({source_sql}) AS SOURCE"

            # Create the temporary table SQL
            create_temp_sql = f"""
            CREATE OR REPLACE TABLE {temp_table_name} AS
            SELECT a.*,
            {source_sql},
            b."{target_column.lower()}_number" as TARGET
            FROM {table_name} a
            LEFT JOIN {target_column.upper()}_TARGET b USING("{target_column.lower()}");
            """

            connection.execute(text(f"DROP TABLE IF EXISTS {temp_table_name}"))
            connection.execute(text(create_temp_sql))
            result.append({"table": f"{database}.{schema}.{temp_table_name}"})

            # Create the train, validate, and predict tables with timestamp
            train_table_name = f"temp_{table_name}_train_{timestamp}"
            validate_table_name = f"temp_{table_name}_validate_{timestamp}"
            predict_table_name = f"temp_{table_name}_predict_{timestamp}"

            if create_validation:
                create_train_sql = f"""
                CREATE OR REPLACE TABLE {train_table_name} AS
                SELECT *
                FROM {temp_table_name}
                WHERE TARGET IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM()) <= 0.9 * COUNT(*) OVER ();
                """

                create_validate_sql = f"""
                CREATE OR REPLACE TABLE {validate_table_name} AS
                SELECT *
                FROM {temp_table_name}
                WHERE TARGET IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM()) > 0.9 * COUNT(*) OVER ();
                """
            else:
                create_train_sql = f"""
                CREATE OR REPLACE TABLE {train_table_name} AS
                SELECT *
                FROM {temp_table_name}
                WHERE TARGET IS NOT NULL;
                """
                validate_table_name = None

            create_predict_sql = f"""
            CREATE OR REPLACE TABLE {predict_table_name} AS
            SELECT *
            FROM {temp_table_name}
            WHERE TARGET IS NULL;
            """

            connection.execute(text(f"DROP TABLE IF EXISTS {train_table_name}"))
            connection.execute(text(create_train_sql))

            if create_validation:
                connection.execute(text(f"DROP TABLE IF EXISTS {validate_table_name}"))
                connection.execute(text(create_validate_sql))

            connection.execute(text(f"DROP TABLE IF EXISTS {predict_table_name}"))
            connection.execute(text(create_predict_sql))

            result.append({"table": f"{database}.{schema}.{train_table_name}"})
            if create_validation:
                result.append({"table": f"{database}.{schema}.{validate_table_name}"})
            result.append({"table": f"{database}.{schema}.{predict_table_name}"})

    except Exception as e:
        logging.error(f"Error creating temporary table: {e}")
        raise e
    return json.dumps(result)

