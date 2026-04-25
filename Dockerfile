FROM python:3.11-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ server/
COPY scenarios/ scenarios/
COPY ui/ ui/
COPY models.py grader.py client.py inference.py openenv.yaml q_network.pt ./
COPY __init__.py .
EXPOSE 8000
ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
