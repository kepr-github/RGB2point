FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# install additional apt packages if needed
RUN apt-get update && \
    apt-get install -y git libgl1 libglm-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws


# set working directory and copy the project
WORKDIR /app
COPY . .

# install python dependencies
RUN pip install --no-cache-dir timm accelerate wandb scikit-learn gsplat

CMD ["bash"]
