FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y build-essential cmake curl libcurl4-openssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN git clone https://github.com/ggml-org/llama.cpp && \
    cd llama.cpp && \
    cmake -B build && \
    cmake --build build --config Release --target install && \
    cd .. && \
    rm -rf llama.cpp/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy the entire application
COPY pyproject.toml uv.lock README.md .

# Install dependencies with uv
RUN uv sync --frozen

COPY . .

# Command to run when the container starts
CMD ["uv", "run", "main.py"]