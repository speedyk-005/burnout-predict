FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR app/

# Copy only the dependency manifest first to cache the environment
COPY pyproject.toml ./
RUN uv pip install --system .

# Copy code and model weights afterward (changes here won't trigger re-installs)
COPY predict.py pipeline.bin ./

EXPOSE 8000

CMD ["python", "predict.py"]
