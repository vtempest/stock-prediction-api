FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync 

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["uv", "run", "python", "-m", "uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]