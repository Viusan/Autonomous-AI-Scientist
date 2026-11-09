FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home agentuser

COPY --chown=root:root . .

#Agent har ikke skriverettigheter på /app
RUN chmod -R a-w /app

#Lager en katalog agenten kan skrive på for output
RUN mkdir -p /app/output \
    && chown agentuser:agentuser /app/output \
    && chmod 700 /app/output

USER agentuser

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
