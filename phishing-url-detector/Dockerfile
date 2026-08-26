# Phishing URL Detector — production image
#
# Build:
#   docker build -t phishing-url-detector .
# Run:
#   docker run -p 5000:5000 phishing-url-detector
#
# The image ships with the pre-trained model in models/model.pkl baked in
# via the build context. To retrain inside the container instead, run:
#   docker run phishing-url-detector python3 src/train.py --data data/dataset.csv

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-trained artifacts
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY docs/ docs/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=3)" || exit 1

# Production WSGI server (gunicorn), 4 workers
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--chdir", "src", "app:app"]
