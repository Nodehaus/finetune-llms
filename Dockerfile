FROM python:3.13-slim

WORKDIR /

# Install uv
RUN pip install uv

# Copy the entire application
COPY . .

# Install dependencies with uv
RUN uv sync --frozen

# Command to run when the container starts
CMD ["uv", "run", "main.py"]