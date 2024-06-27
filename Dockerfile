# base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app

# Install system dependencies
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        git \
        unzip \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

