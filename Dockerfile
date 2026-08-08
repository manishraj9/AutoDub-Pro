FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for yt-dlp YouTube JavaScript challenges
RUN curl -fsSL https://deno.land/install.sh | sh

ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "python main.py --web --port $PORT"]