FROM python:3.13-slim

RUN apt-get update && apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install uv
RUN pip install uv

# Copy the entire application
COPY pyproject.toml uv.lock README.md .

# Install dependencies with uv
RUN uv sync --frozen

COPY . .

# Command to run when the container starts
CMD ["uv", "run", "main.py"]