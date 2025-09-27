FROM python:3.13-slim

WORKDIR /

# Install uv
RUN pip install uv

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --frozen

# Copy the entire application
COPY . .

# Command to run when the container starts
CMD ["uv", "run", "main.py"]