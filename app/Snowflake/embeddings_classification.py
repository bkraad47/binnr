import json
import time
import logging
import snowflake.connector
import os
from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf, col, array_construct, sum as snowflake_sum, avg
from snowflake.snowpark.types import ArrayType, FloatType, VectorType
from snowflake.ml.modeling.cluster import KMeans
from snowflake.ml.modeling.decomposition import PCA
import numpy as np
import pandas as pd
import timeit
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embeddings_cluster(database, schema, warehouse, input_table, role, number_of_iterations=2):
    # Load connection details from a JSON file
    config = json.loads(os.getenv('SNOWFLAKE_CONFIG'))

    if database is not None and database != "":
        config['database'] = database
    if schema is not None and schema != "":
        config['schema'] = schema
    if warehouse is not None and warehouse != "":
        config['warehouse'] = warehouse
    if role is not None and role != "":
        config['role'] = role

    # Establish the connection using Snowpark
    logger.info("Establishing Snowpark session")
    session = Session.builder.configs(config).create()

    # Add numpy and pandas packages to the session
    logger.info("Adding required packages to the session")
    session.add_packages("numpy", "pandas", "scikit-learn")

    # Retry decorator
    def retry_on_exception(max_retries=3):
        def decorator(func):
            def wrapper(*args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Exception occurred: {e}. Retrying {retries + 1}/{max_retries}...")
                        retries += 1
                        time.sleep(5)  # Wait for a bit before retrying
                raise Exception(f"Operation failed after {max_retries} retries")
            return wrapper
        return decorator

    start_time = timeit.default_timer()
    logger.info("Starting the k-means clustering process")

    # Step 1: Create embeddings table
    logger.info("Creating embeddings table if it doesn't exist")
    create_embeddings_table_sql = f"""
    CREATE OR REPLACE TABLE embeddings_{input_table.lower()} AS
    SELECT SOURCE, TARGET, 
        SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', SOURCE) AS embeddings 
    FROM {input_table};
    """
    session.sql(create_embeddings_table_sql).collect()

    # Step 2: Load the embeddings data
    logger.info("Loading the embeddings data")
    @retry_on_exception()
    def load_embeddings_data():
        return session.sql(f"SELECT * FROM embeddings_{input_table.lower()}")

    embeddings_df = load_embeddings_data()

    # Define a UDF to extract each float from the embedding vector
    @udf(return_type=ArrayType(FloatType()), input_types=[VectorType(float, 768)])
    def extract_embeddings(vector):
        return [float(x) for x in vector]

    # Apply the UDF to create an array column of embeddings
    logger.info("Extracting embeddings")
    embeddings_df = embeddings_df.with_column("EMBEDDING_ARRAY", extract_embeddings(col("EMBEDDINGS")))

    # Extract each float in the embedding array into separate columns
    for i in range(768):
        embeddings_df = embeddings_df.with_column(f"EMBEDDING_{i}", col("EMBEDDING_ARRAY")[i].cast("float"))

    # Apply PCA to reduce dimensions to 500
    logger.info("Applying PCA to reduce dimensions")
    embedding_cols = [f"EMBEDDING_{i}" for i in range(768)]
    embeddings_np = embeddings_df.select(*embedding_cols).to_pandas()

    # Fill NaN values with the mean of each column
    embeddings_np = embeddings_np.fillna(embeddings_np.mean())

    # Using PCA to reduce the dimensions
    pca = PCA(n_components=500).set_output_cols([f"PCA_{i}" for i in range(500)])
    pca_df = session.create_dataframe(embeddings_np)

    # Fit PCA model
    @retry_on_exception()
    def fit_pca():
        return pca.fit_transform(pca_df)

    reduced_embeddings_df = fit_pca()

    # Convert the resulting Snowpark DataFrame to a Pandas DataFrame for concatenation
    reduced_embeddings_np = reduced_embeddings_df.to_pandas()

    source_df = embeddings_df.select("SOURCE").to_pandas()
    target_df = embeddings_df.select("TARGET").to_pandas()
    embeddings_column_df = embeddings_df.select("EMBEDDINGS").to_pandas()

    # Concatenate SOURCE, TARGET and EMBEDDINGS columns back to the reduced embeddings
    final_df = pd.concat([source_df, target_df, embeddings_column_df, reduced_embeddings_np], axis=1)

    # Convert back to Snowpark DataFrame
    final_snowpark_df = session.create_dataframe(final_df)

    # Split the data into those with target and those without
    with_target_df = final_snowpark_df.filter(col("TARGET").is_not_null())
    without_target_df = final_snowpark_df.filter(col("TARGET").is_null())

    # Ensure both DataFrames have the same columns
    if "TARGET" not in without_target_df.columns:
        without_target_df = without_target_df.with_column("TARGET", col("TARGET"))

    # Ensure both DataFrames have the same order of columns
    common_columns = sorted(set(with_target_df.columns) & set(without_target_df.columns))
    with_target_df = with_target_df.select(*common_columns)
    without_target_df = without_target_df.select(*common_columns)

    # Select initial centroids based on target numbers
    logger.info("Selecting initial centroids based on target numbers")
    initial_centroids = (
        with_target_df.group_by("TARGET")
        .agg(*[avg(f"PCA_{i}").alias(f"PCA_{i}") for i in range(500)])
        .select(*[f"PCA_{i}" for i in range(500)])
        .to_pandas()
        .to_numpy()
    )

    # Perform KMeans clustering with initial centroids
    logger.info("Performing KMeans clustering")
    kmeans = KMeans(n_clusters=len(initial_centroids), max_iter=number_of_iterations, init=initial_centroids)
    input_cols = [f"PCA_{i}" for i in range(500)]

    kmeans = kmeans.set_input_cols(input_cols).set_output_cols(["CLUSTER_ID"])

    # Fit the KMeans model on the data with target numbers
    @retry_on_exception()
    def fit_kmeans():
        return kmeans.fit(with_target_df.select(input_cols))

    kmeans_model = fit_kmeans()

    # Predict the cluster labels for both dataframes
    clustered_with_target_df = kmeans_model.transform(with_target_df)
    clustered_without_target_df = kmeans_model.transform(without_target_df)

    # Combine the data back together
    combined_clustered_df = clustered_with_target_df.union_all(clustered_without_target_df)

    # Extract the centroids directly from the KMeans model
    centroids_np = kmeans_model.to_sklearn().cluster_centers_

    # Define the UDF in Snowpark to use centroids for cluster assignment and similarity calculation
    @udf(return_type=ArrayType(FloatType()), input_types=[ArrayType(FloatType())])
    def extract_cluster_info(vector):
        centroids = np.array(centroids_np)  # Ensure centroids are in the correct format
        vector = np.array(vector[:500])  # Reduce the vector to 500 dimensions
        cluster_id = int(np.argmin([np.linalg.norm(vector - centroid) for centroid in centroids]))
        similarity = round(cosine_similarity([vector], [centroids[cluster_id]])[0][0] * 100, 2)  # Convert similarity to percentage and round to 2 decimal places
        return [cluster_id, similarity]

    # Apply the UDF on the PCA-reduced embeddings
    logger.info("Calculating cluster assignment and similarity")
    for i in range(500):
        combined_clustered_df = combined_clustered_df.with_column(f"PCA_{i}", col(f"PCA_{i}").cast("float"))

    combined_clustered_df = combined_clustered_df.with_column("CLUSTER_INFO", extract_cluster_info(array_construct(*[col(f"PCA_{i}") for i in range(500)])))

    # Split the array into two separate columns: CLUSTER_ID_INT and SIMILARITY
    combined_clustered_df = combined_clustered_df.with_column("CLUSTER_ID_INT", col("CLUSTER_INFO")[0])
    combined_clustered_df = combined_clustered_df.with_column("SIMILARITY", col("CLUSTER_INFO")[1])

    # Rename target column in final_snowpark_df to avoid ambiguity
    final_snowpark_df = final_snowpark_df.with_column_renamed("TARGET", "ORIGINAL_TARGET")

    # Ensure PCA columns are included as string names for the join operation
    pca_column_names = [f"PCA_{i}" for i in range(500)]
    pca_column_names.append("SOURCE") # Add the SOURCE column to the list of column names as well

    # Perform the join operation correctly using the column names as strings
    logger.info("Joining dataframes and calculating weighted targets")
    combined_clustered_df = combined_clustered_df.join(final_snowpark_df, pca_column_names, how="inner")
    
    # Calculate the number of members in each cluster
    cluster_counts = (
        combined_clustered_df.group_by("CLUSTER_ID_INT")
        .agg(snowflake_sum(col("SIMILARITY")).alias("CLUSTER_SIZE"))
        .to_pandas()
    )

    # Merge the cluster counts back with the combined_clustered_df
    combined_clustered_df = combined_clustered_df.join(session.create_dataframe(cluster_counts), "CLUSTER_ID_INT")

    # Adjust weights based on the inverse of the number of members in each cluster
    combined_clustered_df = combined_clustered_df.with_column("ADJUSTED_SIMILARITY", col("SIMILARITY") / col("CLUSTER_SIZE"))

    # Calculate weighted targets with adjusted weights
    weighted_targets_df = (
        combined_clustered_df.group_by("CLUSTER_ID_INT", "ORIGINAL_TARGET")
        .agg(snowflake_sum(col("ORIGINAL_TARGET") * col("ADJUSTED_SIMILARITY")).alias("WEIGHTED_TARGET"))
        .select("CLUSTER_ID_INT", "ORIGINAL_TARGET", col("WEIGHTED_TARGET"))
        .to_pandas()
    )

    # Find the mode target number for each cluster
    centroid_targets = (
        weighted_targets_df
        .groupby("CLUSTER_ID_INT")
        .apply(lambda x: x.loc[x["WEIGHTED_TARGET"].idxmax() if not x["WEIGHTED_TARGET"].isna().all() else x.index[0]]["ORIGINAL_TARGET"])
        .to_dict()
    )

    # Create a mapping DataFrame to join with the combined_clustered_df
    mode_df = pd.DataFrame(list(centroid_targets.items()), columns=["CLUSTER_ID_INT", "MODE_TARGET"])
    mode_snowpark_df = session.create_dataframe(mode_df)

    # Join the combined_clustered_df with mode_snowpark_df to get the predicted target numbers
    combined_clustered_df = combined_clustered_df.join(mode_snowpark_df, "CLUSTER_ID_INT", how="left")
    combined_clustered_df = combined_clustered_df.with_column_renamed("MODE_TARGET", "PREDICTED_TARGET")

    # Define the expected columns and ensure they exist in the DataFrame
    expected_columns = ["SOURCE", "TARGET", "CLUSTER_ID", "SIMILARITY", "PREDICTED_TARGET"]
    available_columns = [col for col in expected_columns if col in combined_clustered_df.columns]

    # Select only the available columns
    result_df = combined_clustered_df.select(*available_columns)

    # Save the result to a new Snowflake table
    logger.info("Saving the result to a new Snowflake table")
    result_df.write.mode("overwrite").save_as_table(f"embeddings_clustered_{input_table.upper()}")
    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    return json.dumps({"table": f"embeddings_clustered_{input_table.upper()}"})
