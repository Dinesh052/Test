FROM python:3.11-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ server/
COPY scenarios/ scenarios/
COPY ui/ ui/
COPY models.py grader.py client.py inference.py openenv.yaml q_network.pt ./
COPY __init__.py .
EXPOSE 7860
ENV ENABLE_WEB_INTERFACE=true
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/ || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
