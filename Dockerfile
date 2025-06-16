FROM hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0

# Set the working directory inside the container
WORKDIR /workspace

# Clone lm-evaluation-harness and install it in editable mode
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .

# install conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate base" >> /root/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# SHELL ["/bin/bash", "-c"]

COPY requirements.txt /workspace/requirements.txt


RUN conda create -n verl python=3.10 -y && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate verl && \
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y && \
    pip install flashrag-dev --pre && \
    pip install -r requirements.txt

CMD bash -c "\
    wandb login $WANDB_API_KEY && \
    huggingface-cli login --token $HF_TOKEN && \
    bash"