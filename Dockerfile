FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home agentuser
COPY --chown=agentuser:agentuser . .
USER agentuser

CMD ["python", "langgraph-orc.py"]