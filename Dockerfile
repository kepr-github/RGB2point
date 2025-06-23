FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# install additional apt packages if needed
RUN apt-get update && \
    apt-get install -y git libgl1 && \
    rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app
# source will be mounted at runtime via docker-compose

# install python dependencies
RUN pip install --no-cache-dir timm accelerate wandb open3d scikit-learn

CMD ["bash"]
