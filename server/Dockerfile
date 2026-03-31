FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY openenv.yaml .
COPY *.py ./
COPY simulation/ simulation/
COPY server/ server/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose the server port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
