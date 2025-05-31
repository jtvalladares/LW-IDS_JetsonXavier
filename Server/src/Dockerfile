FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3 

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

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt 

CMD ["python3", "main.py"]