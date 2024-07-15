# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./app/requirements.txt /app/requirements.txt

# Install the required libraries
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the code into the container
COPY ./app /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
