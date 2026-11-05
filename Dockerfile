FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR app/

# Copy the manifest and lockfile first so dependency installs stay cached
COPY pyproject.toml uv.lock ./

# Install the exact versions pinned in uv.lock. `--frozen` forbids re-resolving;
# `--no-install-project` skips building this package (nothing imports it at runtime);
# `--no-dev` drops ruff/uv from the [project.optional-dependencies] dev extra.
RUN uv sync --frozen --no-install-project --no-dev

# `uv sync` creates the venv at /app/.venv; put it on PATH so `python` finds it
ENV PATH="/app/.venv/bin:$PATH"

# Copy code and model weights afterward (changes here won't trigger re-installs)
COPY predict.py pipeline.bin ./

EXPOSE 8000

CMD ["python", "predict.py"]
