# Imagen base para Jetson Nano con TensorFlow 2.7
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

# Establece el directorio de trabajo
WORKDIR /workspace

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Instala dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia el contenido del proyecto al contenedor
COPY . /workspace

# Instala pip y las dependencias Python necesarias
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Comando por defecto al iniciar el contenedor
CMD ["python3", "main.py"]
