# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files and source code
COPY requirements.txt .
COPY . .

# Create and activate virtual environment
RUN python -m venv .venv \
    && . .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy .env file
COPY .env.example .env
# Set environment variables from .env (done at runtime, not build time)
# This way changes to .env don't require rebuilding the image
RUN pip install --no-cache-dir python-dotenv

# Expose port from .env (default fallback to 8080)
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Command to run the app
CMD ["/bin/bash", "-c", ". .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port ${PORT} --reload"]
