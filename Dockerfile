FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY tests ./tests
COPY data ./data

RUN useradd --create-home runner
USER runner

CMD ["python", "-m", "pytest", "-q", "tests"]
