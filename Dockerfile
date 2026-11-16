# The demo page and the HTTP API as one image. Both serve on port 7860, the port a Hugging Face Docker
# Space expects.
#
#   docker build -t bioscout-demo .
#   docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=... bioscout-demo                 # the page
#   docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=... bioscout-demo bioscout-api    # the API
#   docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=... -e BIOSCOUT_SERVE=api bioscout-demo
#                                               # the API, on a host that cannot change the command
#
# Without a key both still serve the recorded example answers (app/examples.json).
# The key is given when the container runs; it is never copied into the image (.dockerignore keeps .env out).
# The count of questions asked (the cap on spending) is quota.sqlite in the cache folder
# (/home/user/.cache/bioscout, or BIOSCOUT_CACHE_DIR): mount a volume there, or the count starts again
# with every new container.
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
RUN uv sync --locked --no-dev --extra demo --extra api --no-install-project

COPY --chown=user:user README.md LICENSE ./
COPY --chown=user:user src ./src
COPY --chown=user:user app ./app
RUN uv sync --locked --no-dev --extra demo --extra api

# Where `bioscout-api` listens when it is run instead of the page.
ENV BIOSCOUT_API_HOST=0.0.0.0 BIOSCOUT_API_PORT=7860
EXPOSE 7860
# The page answers on the first address, the API on the second.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health')" \
     || python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
# The start-up (src/bioscout/container.py) serves the page, or the API when told to. It starts as
# root only to hand the cache folder to uid 1000, since a mounted volume is often owned by root, and
# gives up root before anything is served. Hugging Face starts the container as uid 1000 already.
USER root
ENTRYPOINT ["bioscout-serve"]
