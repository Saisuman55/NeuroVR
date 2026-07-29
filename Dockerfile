# ── Base ────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System deps (OpenCV needs libGL, libGlib) ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (layer-cached unless requirements change) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p outputs/predictions outputs/plots outputs/uploads \
             models/classifier models/segmenter

# ── HF Spaces runs as non-root user (UID 1000) ───────────────────────────────
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# ── Expose HF Spaces default port ────────────────────────────────────────────
EXPOSE 7860

# ── Download model weights then start Flask ───────────────────────────────────
CMD ["sh", "-c", "python download_models.py && python app.py"]
