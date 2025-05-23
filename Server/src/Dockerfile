FROM python:3.9-slim

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    build-essential \ 
    python3-pip \
    libpython3-dev \
    python3-sklearn \
    python3-sklearn-lib \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace 

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

CMD ["python3", "main.py"]

