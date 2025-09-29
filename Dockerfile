FROM ghcr.io/ggml-org/llama.cpp:full-cuda
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /

# Copy the entire application
COPY pyproject.toml uv.lock README.md .

# Install dependencies with uv
RUN uv sync --frozen

COPY . .

# Command to run when the container starts
CMD ["uv", "run", "main.py"]