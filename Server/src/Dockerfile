# Use the NVIDIA L4T TensorFlow base image

# jetson xavier:
# FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3 

# jetson nano:
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3 

#FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3
# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    vim git libspdlog-dev dpkg python3-pip libpython3-dev libjpeg-dev zlib1g-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libopenblas-base libopenmpi-dev libopenblas-dev ffmpeg --no-install-recommends 

ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# para error de bind address
#ENV IP_HOST = "0.0.0.0"
#ENV PORT = 5000 #

# Copy application code into the container
COPY . /workspace

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Default command to run your application
CMD ["python3", "main.py"]
