
# binner (data-binnr)


<p align="center">
  <img src="Binnr.png" alt="Binnr">
</p>

Binnr is a tool designed for textual classification, leveraging the power of Snowflake. It employs a range of machine learning algorithms such as GBM, LLM, and KMeans clustering to analyze and categorize text data. The tool features a FastAPI interface, enabling users to upload data, establish classification tables, train various models, make predictions, and cluster embeddings. Binnr is optimized for high performance (even with large datasets), by utilizing Snowpark capabilities.

## Package Requirements

Ensure you have the following Python packages installed:

- fastapi
- celery
- pydantic
- snowflake-connector-python
- snowflake-snowpark-python
- uvicorn
- numpy
- pandas
- scikit-learn

## How to Run

1. **Clone the repository:**
   ```
   sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up your environment**:

Create a .env file in the root directory and add the following environment variables:

```
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
SNOWFLAKE_CONFIG='{
    "user": "<your-snowflake-username>",
    "password": "<your-snowflake-password>",
    "account": "<your-snowflake-account>",
    "warehouse": "<default-warehouse>",
    "database": "<default-database>",
    "schema": "<default-schema>",
    "role": "<default-role>"
}'
```

**Build and run the Docker containers:**

Copy code
`docker-compose up --build`

## API Endpoints
# API Endpoints Overview

## `/data/upload`
**Description:** Upload a file and initiate a long-running job to load data into Snowflake. This process creates its own stage automatically.

- **Method:** POST
- **Parameters:**
  - `file` (UploadFile): The file to upload.
  - `table_name` (str): The name of the table to create in Snowflake.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The task ID of the Celery job.

## `/data/create/tables/classification`
**Description:** Create classification tables in Snowflake.

- **Method:** POST
- **Parameters:** (ClassificationRequest)
  - `table_name` (str): The name of the table.
  - `source_columns` (list[str]): The source columns for classification.
  - `target_column` (str): The target column for classification.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
  - `template` (str, optional): A template for formatting.
  - `create_validation` (bool, optional): Whether to create validation (split is random 90-10).
  - `llm_train_source` (bool, optional): Whether to train LLM sources. Creates prompts in SOURCE
- **Returns:** The task ID of the Celery job.

## `/classification/gbm/train`
**Description:** Train a GBM classification model in Snowflake.

- **Method:** POST
- **Parameters:** (GBMTrainRequest)
  - `training_table` (str): The training table name.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The task ID of the Celery job.

## `/classification/gbm/predict`
**Description:** Generate predictions using a trained GBM classification model in Snowflake.

- **Method:** POST
- **Parameters:** (GBMPredictRequest)
  - `model_name` (str): The name of the model.
  - `predict_table` (str): The table to predict.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The task ID of the Celery job.

## `/classification/gbm/info`
**Description:** Retrieve model evaluation metrics for a GBM classification model in Snowflake.

- **Method:** GET
- **Parameters:**
  - `model_name` (str): The name of the model.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The model evaluation metrics.

## `/classification/llm/train`
**Description:** Train an LLM classification model in Snowflake.

- **Method:** POST
- **Parameters:** (LLMTrainRequest)
  - `training_table` (str): The training table name.
  - `validation_table` (str): The validation table name.
  - `target_column` (str): The target column.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
  - `model_type` (str, optional): The model type (default: 'llama3-8b').
- **Returns:** The task ID of the Celery job.

## `/classification/llm/predict`
**Description:** Generate predictions using a trained LLM classification model in Snowflake.

- **Method:** POST
- **Parameters:** (LLMPredictRequest)
  - `model_name` (str): The name of the model.
  - `predict_table` (str): The table to predict.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The task ID of the Celery job.

## `/classification/llm/train/check`
**Description:** Check the status of an LLM fine-tuning job in Snowflake.

- **Method:** GET
- **Parameters:**
  - `job_id` (str): The job ID of the fine-tuning job.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The status and progress of the fine-tuning job.

## `/classification/embeddings/cluster`
**Description:** Cluster vector embeddings and run KMeans to identify where unmarked records belong.

- **Method:** POST
- **Parameters:** (EmbeddingsClusterRequest)
  - `input_table` (str): The input table name.
  - `number_of_iterations` (int): The number of iterations for KMeans.
  - `database` (str, optional): The name of the database.
  - `schema` (str, optional): The name of the schema.
  - `warehouse` (str, optional): The name of the warehouse.
  - `role` (str, optional): The role to use.
- **Returns:** The task ID of the Celery job.

# Algorithms Overview

## GBM (Gradient Boosting Machine)
GBM is an ensemble learning method that builds models in a stage-wise fashion. It optimizes for accuracy by combining the strengths of multiple weak models, typically decision trees, to create a strong predictive model. The GBM training process involves sequentially training new models to correct the errors made by previously trained models.

## LLM (Large Language Model)
LLMs are deep learning models that are trained on a massive corpus of text data. They can generate human-like text based on the input they receive. The training involves fine-tuning the model on specific datasets to improve its accuracy in understanding and generating text related to a particular domain.

## KMeans Clustering
KMeans is an unsupervised learning algorithm used for clustering. It partitions data into K clusters, where each data point belongs to the cluster with the nearest mean value. The algorithm iteratively refines the cluster centroids to minimize the within-cluster variance.

## Embeddings
Embeddings are low-dimensional vector representations of high-dimensional data, such as text. They capture the semantic meaning of the data, enabling algorithms to perform operations like clustering and classification more effectively. In this tool, embeddings are used to cluster textual data and identify the most similar target for unmarked records.