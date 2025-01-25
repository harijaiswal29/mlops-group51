# Use an official Python 3.10 runtime
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first for Docker caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code, including model folder and app.py
COPY . /app/

# Expose port 5000
EXPOSE 5000

# Define the entrypoint for the container
CMD ["python", "app.py"]
