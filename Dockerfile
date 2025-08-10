# ---- Builder Stage ----
FROM python:3.10-slim AS builder

# Install build tools and necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file and install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy your application source code and configuration files
COPY . .

# Expose the port; fallback value is 8000 and can be overridden at runtime
ENV PORT=8000
EXPOSE 8000

# Ensure the virtual environment is activated and run uvicorn
CMD ["/venv/bin/uvicorn", "main:app", "--reload"]
