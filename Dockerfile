FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home agentuser

COPY --chown=root:root . .

# Application files should not be writable by the agent
RUN chmod -R a-w /app

# Create a dedicated writable directory
RUN mkdir -p /app/output \
    && chown agentuser:agentuser /app/output \
    && chmod 700 /app/output

USER agentuser

CMD ["python", "langgraph-orc.py"]