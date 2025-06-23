FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# install additional apt packages if needed
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy source
COPY . /app

# install python dependencies
RUN pip install --no-cache-dir timm accelerate wandb open3d scikit-learn

CMD ["bash"]
