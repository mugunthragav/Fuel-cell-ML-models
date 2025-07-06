# Use a lightweight Python base image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY ./app /app

# Copy the models directory
COPY ./models /app/models

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# main:app refers to the 'app' object in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
