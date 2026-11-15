FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# INSTALL_ML=true instala torch (CPU), ultralytics e transformers (YOLOv8 / AST)
ARG INSTALL_ML=false

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
 && if [ "$INSTALL_ML" = "true" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-ml.txt; \
    fi

COPY . .

RUN useradd --create-home --uid 1000 app \
 && mkdir -p logs reports artifacts \
 && chown -R app:app /app
USER app

EXPOSE 8766

CMD ["python", "-m", "src.webapp.server", "--host", "0.0.0.0", "--port", "8766"]
