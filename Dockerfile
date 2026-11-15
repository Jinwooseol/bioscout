# The demo page as a container. It serves on port 7860, the port a Hugging Face Docker Space expects.
#
#   docker build -t bioscout-demo .
#   docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=... bioscout-demo
#
# Without a key the page still shows the recorded example answers (app/examples.json).
# The key is given when the container runs; it is never copied into the image (.dockerignore keeps .env out).
FROM python:3.12-slim

# A non-root user with uid 1000, as Hugging Face Spaces requires.
RUN pip install --no-cache-dir uv==0.13.7 \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app
USER user
WORKDIR /home/user/app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/home/user/app/.venv/bin:${PATH}"

# Dependencies first, so this layer is rebuilt only when the lock file changes.
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra demo --no-install-project

COPY --chown=user:user README.md LICENSE ./
COPY --chown=user:user src ./src
COPY --chown=user:user app ./app
RUN uv sync --locked --no-dev --extra demo

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
