# syntax=docker/dockerfile:1.7
# One image, two roles: `sisyphus serve` (FastAPI) and `sisyphus ui` (Streamlit).
# A demo match is generated, analysed and narrated (template mode) at build time
# so a fresh container has something to show without any model credentials.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-generate the demo dataset (synthetic, deterministic seed).
ENV PATH="/app/.venv/bin:$PATH" SISYPHUS_FORCE_TEMPLATE=1
RUN sisyphus simulate --scenario press_collapse --seed 7 --out /app/data/generated --analyze \
 && sisyphus narrate /app/data/generated/press_collapse-s7 --language es --language pt-BR \
      --kinds goal,big_chance,fatigue_breakpoint,press_drop,momentum_swing,substitution,card,period_summary,full_time

FROM python:3.12-slim-bookworm AS runtime
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/data /app/data
COPY --chown=app:app pyproject.toml README.md ./
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SISYPHUS_DATA_DIR=/app/data/generated \
    SISYPHUS_API_URL=http://localhost:8000 \
    HEALTH_URL=http://localhost:8000/health
USER app
EXPOSE 8000 8501
# For the UI role set HEALTH_URL=http://localhost:8501/_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD \
    python -c "import httpx,os;httpx.get(os.environ['HEALTH_URL']).raise_for_status()" || exit 1
CMD ["sisyphus", "serve", "--port", "8000"]
