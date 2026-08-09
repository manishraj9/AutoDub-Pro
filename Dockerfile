# ============================================================
# Stage 1: Build bgutil PO-token server
# ============================================================
FROM node:25-bookworm-slim AS bgutil-build

WORKDIR /opt/bgutil

COPY bgutil-ytdlp-pot-provider/server/package.json .
COPY bgutil-ytdlp-pot-provider/server/package-lock.json .

RUN npm ci --no-audit --no-fund

COPY bgutil-ytdlp-pot-provider/server/types ./types
COPY bgutil-ytdlp-pot-provider/server/tsconfig.json .
COPY bgutil-ytdlp-pot-provider/server/src ./src

RUN npx tsc


# ============================================================
# Stage 2: AutoDub-Pro
# ============================================================
FROM node:25-bookworm-slim

# Python + FFmpeg + build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    gcc \
    g++ \
    libsndfile1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno
RUN curl -fsSL https://deno.land/install.sh | sh

ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

# Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . .

# Compiled bgutil server
COPY --from=bgutil-build /opt/bgutil/build /opt/bgutil/build

ENV PORT=8000

EXPOSE 8000

CMD ["sh", "./start.sh"]