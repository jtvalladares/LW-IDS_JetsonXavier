# Use the specified NVIDIA L4T TensorFlow base image
FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3

# Set the working directory
WORKDIR /workspace

# Install any additional packages or dependencies you need
# For example, if you need to install git
RUN apt-get update &&\
    apt-get install -y vim git libspdlog-dev dpkg python3-pip libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev libopenblas-base libopenmpi-dev libopenblas-dev ffmpeg --no-install-recommends
RUN apt-get update && \
    apt-get install -y git

# Optionally, copy your application code into the container
COPY . /workspace

# Optionally, install Python dependencies
# For example, if you have a requirements.txt file
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# By default, just run bash in interactive mode
CMD ["python3", "main.py"]

